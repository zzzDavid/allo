# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import allo
import allo.dataflow as df
from allo._mlir.ir import ArrayAttr, IntegerAttr, StringAttr
from allo.ir.builder import _spmw_group_contract
from allo.ir.types import bfloat16, float32
from allo.pim.schedule_promotion import (
    CorrectnessEvidence,
    ExactCyclePolicy,
    MetricMeasurements,
    MetricPromotionCell,
    ObjectiveMetricBridge,
    PromotionEvidence,
    ScheduleEvidence,
    SchedulePromotionGate,
    SemanticScope,
)
from allo.pim.schedule_search import ScheduleCandidate, ScheduleObjectiveDomain
from allo.spmw_autoschedule import MatcherWorkScope, _bucket_for_autoschedule
from allo.spmw_codegen import RunResult, _check_work_grid
from allo.spmw_liveness import MatcherValueId, trace_liveness
from allo.spmw_match import MatchTrace, MatchedOp
from allo.spmw_plan import BufferMetricManifest, _coerce_buffer_metric_manifest


class _FakeCompiled:
    def __init__(self):
        self.execution_graph = object()
        self.host_moves = [
            SimpleNamespace(verb=SimpleNamespace(name="gather"), buffer_role="C")
        ]
        self.schedule_search_result = object()
        self.schedule_activation = object()
        self.fallback_reason = "shadow_only:test"
        self.calls = []

    def run(self, **inputs):
        self.calls.append(inputs)
        return RunResult(
            cycles=17,
            stdout="fake",
            backend="virtual",
            extra={"outputs": {"out": np.asarray(inputs["A"]) + inputs["B"]}},
        )


def _workload(A, B, C):
    del A, B, C


@df.region()
def _repeated_shape_metric_pipeline(
    A: bfloat16[4, 4],
    x: bfloat16[4],
    tmp: bfloat16[4],
    C: bfloat16[4],
):
    @allo.work(mapping=[1], args=[A, x, tmp])
    def first_stage(
        local_A: bfloat16[4, 4],
        local_x: bfloat16[4],
        local_tmp: bfloat16[4],
    ):
        for i in range(4):
            acc: bfloat16 = 0
            for j in range(4):
                acc += local_A[i, j] * local_x[j]
            local_tmp[i] = acc

    @allo.work(mapping=[1], args=[A, tmp, C])
    def second_stage(
        local_A: bfloat16[4, 4],
        local_tmp: bfloat16[4],
        local_C: bfloat16[4],
    ):
        for i in range(4):
            acc: bfloat16 = 0
            for j in range(4):
                acc += local_A[i, j] * local_tmp[j]
            local_C[i] = acc


@df.region()
def _scope_order_original(
    A: float32[4],
    B: float32[4],
    C: float32[4],
    D: float32[4],
):
    @allo.work(mapping=2, args=[A, B])
    def add_stage(source: float32[4], destination: float32[4]):
        destination[0] = source[0] + source[1]

    @allo.work(mapping=[2], args=[C, D])
    def multiply_stage(source: float32[4], destination: float32[4]):
        destination[0] = source[0] * source[1]


@df.region()
def _scope_order_relocated(
    first: float32[4],
    second: float32[4],
    third: float32[4],
    fourth: float32[4],
):
    @allo.work(mapping=[2], args=[third, fourth])
    def renamed_multiply(value: float32[4], result: float32[4]):
        result[0] = value[0] * value[1]

    @allo.work(mapping=2, args=[first, second])
    def renamed_add(value: float32[4], result: float32[4]):
        result[0] = value[0] + value[1]


@df.region()
def _duplicate_groups_original(
    A: float32[4],
    B: float32[4],
    C: float32[4],
    D: float32[4],
):
    @allo.work(mapping=[1], args=[A, B])
    def first_stage(source: float32[4], destination: float32[4]):
        destination[0] = source[0] + source[1]

    @allo.work(mapping=[1], args=[C, D])
    def second_stage(source: float32[4], destination: float32[4]):
        destination[0] = source[0] + source[1]


@df.region()
def _duplicate_groups_reordered(
    first: float32[4],
    second: float32[4],
    third: float32[4],
    fourth: float32[4],
):
    @allo.work(mapping=[1], args=[third, fourth])
    def renamed_second(value: float32[4], result: float32[4]):
        result[0] = value[0] + value[1]

    @allo.work(mapping=[1], args=[first, second])
    def renamed_first(value: float32[4], result: float32[4]):
        result[0] = value[0] + value[1]


def _retained_function(name, location, *, kernel, semantic="generic"):
    attributes = {
        "sym_name": SimpleNamespace(value=name),
        "test.semantic": semantic,
    }
    if kernel:
        attributes["df.kernel"] = "unit"
        attributes["spmw.abi_value_ids"] = ()
    return SimpleNamespace(
        operation=SimpleNamespace(
            name="func.func",
            operands=(),
            results=(),
            regions=(),
        ),
        attributes=attributes,
        arguments=(),
        location=location,
    )


def _stamp_retained_group(functions, work_ids, group_shape, coalesced_axes=()):
    group_id, group_fingerprint, body_fingerprints = _spmw_group_contract(
        group_shape,
        coalesced_axes,
        tuple(zip(work_ids, functions)),
    )
    body_fingerprints = dict(body_fingerprints)
    scopes = []
    for function, work_id in zip(functions, work_ids):
        function.attributes.update(
            {
                "spmw.group_id": group_id,
                "spmw.work_id": work_id,
                "spmw.group_shape": group_shape,
                "spmw.coalesced_axes": coalesced_axes,
                "spmw.group_fingerprint": group_fingerprint,
                "spmw.body_fingerprint": body_fingerprints[work_id],
            }
        )
        scopes.append(MatcherWorkScope(group_id, work_id, group_shape, coalesced_axes))
    return tuple(scopes)


def _matched_function(name, work_id, target_op_name="MAC"):
    return MatchedOp(
        target_op_name=target_op_name,
        func_name=name,
        work_id=work_id,
        enclosing_loops=[],
        operands=[],
        result_memref_name=None,
        op_range=("begin", "end"),
    )


@dataclass(frozen=True)
class _Materialization:
    promotion_materialization_fingerprint: tuple[str, str]
    promotion_platform_fingerprint: tuple[str, str] = ("platform", "test")


def _promotion_evidence():
    platform = ("platform", "test")
    domain = ScheduleObjectiveDomain(
        metric="cycles",
        target="test-target",
        target_revision="revision-1",
        model_fingerprint=("model", 1),
        fidelity="hardware",
        scope="whole_program",
        unit="cycles",
        direction="minimize",
    )

    def candidate(name, incumbent):
        return ScheduleCandidate(
            decisions={"plan": name},
            payload=name,
            materialized=_Materialization(("artifact", name)),
            score=100,
            objective=100,
            objective_domain=domain,
            is_incumbent=incumbent,
            enumeration_index=int(not incumbent),
        )

    recommended = candidate("recommended", False)
    incumbent = candidate("incumbent", True)
    correctness = CorrectnessEvidence.exact_pass(("oracle", 1))
    scope = SemanticScope(("semantics", 1), (("complete", True),), complete=True)

    def schedule_evidence(schedule):
        fingerprint = schedule.materialized.promotion_materialization_fingerprint
        return ScheduleEvidence.from_schedule(
            schedule,
            correctness=correctness,
            semantic_scope=scope,
            scored_fingerprint=fingerprint,
            emitted_fingerprint=fingerprint,
            platform_fingerprint=platform,
        )

    return PromotionEvidence(
        schedule_evidence(recommended),
        schedule_evidence(incumbent),
        (
            MetricPromotionCell(
                MetricMeasurements(domain, (90,), None, platform),
                MetricMeasurements(domain, (100,), None, platform),
                ExactCyclePolicy(),
                ObjectiveMetricBridge(
                    domain,
                    domain,
                    domain.unit,
                    domain.unit,
                    domain.direction,
                    domain.direction,
                    "identity_metric",
                    ("compile-interface-cycle-identity", 1),
                ),
            ),
        ),
    )


def _patch_pipeline(monkeypatch):
    compiler_api = importlib.import_module("allo.compiler")
    fake = _FakeCompiled()
    captured = {}
    monkeypatch.setattr(
        compiler_api,
        "customize",
        lambda workload, enable_tensor=False: SimpleNamespace(module="module"),
    )
    monkeypatch.setattr(
        compiler_api,
        "match_workload",
        lambda target, module: SimpleNamespace(target_name=target.name),
    )

    def compile_fake(target, trace, **kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(compiler_api, "compile_for_target", compile_fake)
    return fake, captured


def test_allo_compile_returns_signature_compatible_numpy_callable(monkeypatch):
    fake, captured = _patch_pipeline(monkeypatch)
    target = SimpleNamespace(name="fake")

    module = allo.compile(_workload, target, backend="virtual")
    A = np.arange(4, dtype=np.float32)
    B = np.ones(4, dtype=np.float32)
    C = np.zeros(4, dtype=np.float32)
    result = module(A, B=B, C=C)

    assert callable(module)
    assert inspect.signature(module) == inspect.signature(_workload)
    assert result.cycles == 17
    assert module.last_result is result
    assert fake.calls[0] == {"A": A, "B": B, "C": C}
    np.testing.assert_array_equal(C, A + B)
    assert captured["backend"] == "virtual"
    assert captured["cost"] is None
    assert module.schedule_search_result is fake.schedule_search_result
    assert module.schedule_activation is fake.schedule_activation
    assert module.fallback_reason == fake.fallback_reason


def test_compile_rejects_missing_numpy_operand_before_backend(monkeypatch):
    _patch_pipeline(monkeypatch)
    target = SimpleNamespace(name="fake")
    module = allo.compile(_workload, target)

    with pytest.raises(TypeError, match="missing a required argument: 'C'"):
        module(np.ones(2), np.ones(2))


def test_compile_binds_executable_cost_spec(monkeypatch):
    _fake, captured = _patch_pipeline(monkeypatch)
    from allo.pim.costs import samsung_cost
    from allo.pim.targets import build_samsung_target

    target = build_samsung_target()
    module = allo.compile(_workload, target, cost=samsung_cost)

    assert module.cost.spec is samsung_cost
    assert module.cost.target is target
    assert captured["cost"] is module.cost


def test_compile_binds_repeated_shape_metrics_by_retained_value(monkeypatch):
    compiler_api = importlib.import_module("allo.compiler")
    fake = _FakeCompiled()
    captured = {}

    def compile_fake(target, trace, **kwargs):
        captured.update(trace=trace, **kwargs)
        return fake

    monkeypatch.setattr(compiler_api, "compile_for_target", compile_fake)
    from allo.pim.costs import aim_cost
    from allo.pim.targets import build_aim_target

    allo.compile(_repeated_shape_metric_pipeline, build_aim_target(), aim_cost)

    trace = captured["trace"]
    manifest = captured["buffer_metrics"]
    assert isinstance(manifest, BufferMetricManifest)
    liveness = trace_liveness(trace)

    def matcher_value_for_source(ordinal):
        source_ref = trace.source_value_refs[ordinal]
        values = {
            liveness.value_id_for_operand(match, index)
            for match in trace.matches
            for index, operand in enumerate(match.operands)
            if operand.value_ref == source_ref
        }
        values.discard(None)
        assert len(values) == 1
        return next(iter(values))

    x_value = matcher_value_for_source(1)
    tmp_value = matcher_value_for_source(2)
    assert x_value != tmp_value
    assert manifest.metrics_for(x_value)["shape"] == (4,)
    assert manifest.metrics_for(tmp_value)["shape"] == (4,)
    assert manifest.metrics_for(MatcherValueId("abi", -1, "argument", 3))["shape"] == (
        4,
    )

    # The same whole-program manifest remains valid while the autoscheduler
    # scores either individual kernel, which is where the legacy name/shape
    # map previously rejected tmp/C as unbound.
    for _scope, matches in _bucket_for_autoschedule(trace):
        sub_trace = MatchTrace(trace.target_name, trace.module_name, matches)
        coerced, _values = _coerce_buffer_metric_manifest(sub_trace, (), manifest)
        assert coerced is manifest


def test_run_backend_accepts_lowered_roles_outside_workload_signature(monkeypatch):
    fake, _captured = _patch_pipeline(monkeypatch)
    target = SimpleNamespace(name="fake")
    module = allo.compile(_workload, target)
    A = np.ones(4, dtype=np.float32)
    B = np.ones(4, dtype=np.float32)

    result = module.run_backend(A=A, B=B)

    assert result.cycles == 17
    assert fake.calls[-1] == {"A": A, "B": B}


def test_compile_forwards_only_evidence_backed_promotion_gate(monkeypatch):
    _fake, captured = _patch_pipeline(monkeypatch)
    from allo.pim.costs import samsung_cost
    from allo.pim.targets import build_samsung_target

    evidence = _promotion_evidence()
    allo.compile(
        _workload,
        build_samsung_target(),
        cost=samsung_cost,
        promotion_evidence=evidence,
    )

    gate = captured["promotion_gate"]
    assert allo.PromotionEvidence is PromotionEvidence
    assert isinstance(gate, SchedulePromotionGate)
    assert gate.evidence is evidence


@pytest.mark.parametrize("invalid", [True, object(), "promote"])
def test_compile_rejects_non_evidence_promotion_requests(monkeypatch, invalid):
    _patch_pipeline(monkeypatch)
    with pytest.raises(TypeError, match="PromotionEvidence"):
        allo.compile(
            _workload,
            SimpleNamespace(name="fake"),
            promotion_evidence=invalid,
        )


def test_compile_rejects_promotion_without_cost_or_with_explicit_layout(monkeypatch):
    _patch_pipeline(monkeypatch)
    evidence = _promotion_evidence()
    target = SimpleNamespace(name="fake")
    with pytest.raises(ValueError, match="executable cost model"):
        allo.compile(_workload, target, promotion_evidence=evidence)

    from allo.pim.costs import samsung_cost
    from allo.pim.targets import build_samsung_target

    with pytest.raises(ValueError, match="explicit layout"):
        allo.compile(
            _workload,
            build_samsung_target(),
            cost=samsung_cost,
            layout=object(),
            promotion_evidence=evidence,
        )


def test_structural_mapping_boundaries_define_relocation_invariant_apu_scopes():
    compiler_api = importlib.import_module("allo.compiler")

    def stamped(names, locations, parsed_work_ids):
        functions = [
            _retained_function(
                name,
                location,
                kernel=True,
                semantic="pair" if index < 2 else "tail",
            )
            for index, (name, location) in enumerate(zip(names, locations))
        ]
        retained_scopes = _stamp_retained_group(
            functions[:2], ((0,), (1,)), (2,), (0,)
        ) + _stamp_retained_group(functions[2:], ((0,),), (1,), (0,))
        matches = [
            _matched_function(name, work_id)
            for name, work_id in zip(names, parsed_work_ids)
        ]
        module = SimpleNamespace(body=SimpleNamespace(operations=functions))
        trace = MatchTrace("apu_v1", "retained_scope", matches)
        compiler_api._stamp_matcher_work_scopes(
            SimpleNamespace(name="apu_v1"),
            module,
            trace,
        )
        search_domains = tuple(
            scope for scope, _matches in _bucket_for_autoschedule(trace)
        )
        return trace, search_domains, retained_scopes

    original, original_domains, expected = stamped(
        ("tile9_0", "tile9_1", "stage_99_0"),
        ("loc(kernel-a)", "loc(kernel-a)", "loc(kernel-b)"),
        ((0,), (1,), (99, 0)),
    )
    relocated, relocated_domains, relocated_expected = stamped(
        ("phase_3000_0", "phase_3000_1", "tail_8080_0"),
        ("loc(new-a)", "loc(new-b)", "loc(new-b)"),
        ((3000, 0), (3000, 1), (8080, 0)),
    )
    assert relocated_expected == expected

    for trace in (original, relocated):
        scopes = tuple(match.extra["spmw_work_scope"] for match in trace.matches)
        assert scopes == expected
        assert tuple(match.work_id for match in trace.matches) == tuple(
            scope.work_id for scope in expected
        )
        assert all("spmw_group_count" not in match.extra for match in trace.matches)
        assert all("coalesced_spmw_axis" not in match.extra for match in trace.matches)
    assert original_domains == relocated_domains
    assert original_domains == ((expected[0].group_id,), (expected[2].group_id,))


def test_structural_mapping_boundaries_keep_distinct_kernel_groups():
    compiler_api = importlib.import_module("allo.compiler")
    names = ("first_0", "second_1")
    functions = [
        _retained_function(
            name,
            "loc(shared)",
            kernel=True,
            semantic=operation,
        )
        for name, operation in zip(names, ("MAC", "ADD"))
    ]
    retained_scopes = tuple(
        _stamp_retained_group((function,), ((0,),), (1,))[0] for function in functions
    )
    matches = [
        _matched_function(names[0], (0,), "MAC"),
        _matched_function(names[1], (1,), "ADD"),
    ]
    module = SimpleNamespace(body=SimpleNamespace(operations=functions))
    trace = MatchTrace("fake", "distinct_structural_groups", matches)

    compiler_api._stamp_matcher_work_scopes(
        SimpleNamespace(name="fake"),
        module,
        trace,
    )

    assert tuple(match.extra["spmw_work_scope"] for match in matches) == retained_scopes
    assert tuple(scope for scope, _matches in _bucket_for_autoschedule(trace)) == (
        (retained_scopes[0].group_id, (0,)),
        (retained_scopes[1].group_id, (0,)),
    )


def test_frontend_retains_stable_scopes_across_rename_relocation_and_reorder():
    compiler_api = importlib.import_module("allo.compiler")

    def retained(workload, customize_fn):
        schedule = customize_fn(workload, enable_tensor=False)
        matches = []
        operation_kinds = []
        for function in schedule.module.body.operations:
            if (
                function.operation.name != "func.func"
                or "df.kernel" not in function.attributes
            ):
                continue
            assert isinstance(function.attributes["spmw.group_id"], IntegerAttr)
            for name in (
                "spmw.work_id",
                "spmw.group_shape",
                "spmw.coalesced_axes",
                "spmw.abi_value_ids",
            ):
                assert isinstance(function.attributes[name], ArrayAttr)
            for name in ("spmw.group_fingerprint", "spmw.body_fingerprint"):
                assert isinstance(function.attributes[name], StringAttr)
            body = str(function)
            operation_kind = "ADD" if "arith.addf" in body else "MUL"
            operation_kinds.append(operation_kind)
            matches.append(
                _matched_function(
                    compiler_api._retained_symbol_name(function),
                    (9999,),
                    operation_kind,
                )
            )
        trace = MatchTrace("fake", "retained_frontend_scopes", matches)
        compiler_api._stamp_matcher_work_scopes(
            SimpleNamespace(name="fake"),
            schedule.module,
            trace,
        )
        by_operation = {"ADD": [], "MUL": []}
        for operation_kind, match in zip(operation_kinds, trace.matches):
            by_operation[operation_kind].append(match.extra["spmw_work_scope"])
        search_domains = {scope for scope, _matches in _bucket_for_autoschedule(trace)}
        assert compiler_api._module_has_retained_matcher_scopes(schedule.module)
        return by_operation, search_domains

    original_metadata = _scope_order_original.mappings
    relocated_metadata = _scope_order_relocated.mappings
    _scope_order_original.mappings = {"mutable_label": [99]}
    _scope_order_relocated.mappings = {
        "other_label": [7, 3],
        "mutable_label": 5,
    }
    try:
        original, original_domains = retained(_scope_order_original, allo.customize)
        relocated, relocated_domains = retained(
            _scope_order_relocated,
            allo.customize,
        )
        rebuilt, rebuilt_domains = retained(_scope_order_original, df.customize)
    finally:
        _scope_order_original.mappings = original_metadata
        _scope_order_relocated.mappings = relocated_metadata

    assert original == relocated == rebuilt
    assert original_domains == relocated_domains == rebuilt_domains
    assert original["ADD"][0].group_id != original["MUL"][0].group_id
    assert tuple(scope.work_id for scope in original["ADD"]) == ((0,), (1,))
    assert tuple(scope.work_id for scope in original["MUL"]) == ((0,), (1,))
    assert all(
        scope.group_shape == (2,) for scopes in original.values() for scope in scopes
    )
    assert all(scope.coalesced_axes == (0,) for scope in original["ADD"])
    assert all(scope.coalesced_axes == () for scope in original["MUL"])


def test_duplicate_isomorphic_groups_use_explicit_abi_identity_not_occurrence():
    compiler_api = importlib.import_module("allo.compiler")

    def retained(workload):
        schedule = allo.customize(workload, enable_tensor=False)
        functions = [
            function
            for function in schedule.module.body.operations
            if function.operation.name == "func.func"
            and "df.kernel" in function.attributes
        ]
        assert (
            len({int(function.attributes["spmw.group_id"]) for function in functions})
            == 1
        )
        assert (
            len(
                {
                    function.attributes["spmw.group_fingerprint"].value
                    for function in functions
                }
            )
            == 2
        )
        matches = [
            _matched_function(compiler_api._retained_symbol_name(function), (999,))
            for function in functions
        ]
        trace = MatchTrace("fake", "duplicate_isomorphic_groups", matches)
        compiler_api._stamp_matcher_work_scopes(
            SimpleNamespace(name="fake"), schedule.module, trace
        )
        return {
            tuple(
                int(value) for value in function.attributes["spmw.abi_value_ids"]
            ): match.extra["spmw_work_scope"].group_id
            for function, match in zip(functions, matches)
        }

    original = retained(_duplicate_groups_original)
    reordered = retained(_duplicate_groups_reordered)

    assert original == reordered
    assert set(original) == {(0, 1), (2, 3)}
    assert len(set(original.values())) == 2


def test_structurally_ambiguous_duplicate_groups_fail_closed():
    compiler_api = importlib.import_module("allo.compiler")
    functions = [
        _retained_function(name, "loc(shared)", kernel=True, semantic="same")
        for name in ("first", "second")
    ]
    for function in functions:
        _stamp_retained_group((function,), ((0,),), (1,))
    module = SimpleNamespace(body=SimpleNamespace(operations=functions))

    with pytest.raises(ValueError, match="complete grid"):
        compiler_api._module_has_retained_matcher_scopes(module)


def test_later_pid_body_mutation_invalidates_complete_group_manifest():
    compiler_api = importlib.import_module("allo.compiler")
    schedule = allo.customize(_scope_order_original, enable_tensor=False)
    later_add = next(
        function
        for function in schedule.module.body.operations
        if function.operation.name == "func.func"
        and "arith.addf" in str(function)
        and tuple(int(value) for value in function.attributes["spmw.work_id"]) == (1,)
    )
    first_operation = next(iter(later_add.entry_block.operations))
    with schedule.module.context:
        first_operation.attributes["audit.semantic_mutation"] = StringAttr.get(
            "changed"
        )

    with pytest.raises(ValueError, match="stale"):
        compiler_api._module_has_retained_matcher_scopes(schedule.module)


def test_incomplete_retained_adapter_contract_fails_closed():
    compiler_api = importlib.import_module("allo.compiler")
    schedule = allo.customize(_scope_order_original, enable_tensor=False)
    first_kernel = next(
        function
        for function in schedule.module.body.operations
        if function.operation.name == "func.func" and "df.kernel" in function.attributes
    )
    del first_kernel.attributes["spmw.body_fingerprint"]

    with pytest.raises(ValueError, match="incomplete"):
        compiler_api._module_has_retained_matcher_scopes(schedule.module)


def test_missing_retained_grid_coordinate_fails_closed():
    compiler_api = importlib.import_module("allo.compiler")
    schedule = allo.customize(_scope_order_original, enable_tensor=False)
    missing = next(
        function
        for function in schedule.module.body.operations
        if function.operation.name == "func.func"
        and "df.kernel" in function.attributes
        and tuple(int(value) for value in function.attributes["spmw.work_id"]) == (1,)
    )
    missing.operation.erase()

    with pytest.raises(ValueError, match="complete grid"):
        compiler_api._module_has_retained_matcher_scopes(schedule.module)


def test_non_kernel_fallback_identity_is_rename_and_reorder_invariant():
    compiler_api = importlib.import_module("allo.compiler")

    def retained(specifications):
        functions = [
            _retained_function(name, location, kernel=False, semantic=semantic)
            for name, location, semantic, _operation in specifications
        ]
        matches = [
            _matched_function(name, (999,), operation)
            for name, _location, _semantic, operation in specifications
        ]
        module = SimpleNamespace(body=SimpleNamespace(operations=functions))
        trace = MatchTrace("fake", "ordinary_reorder", list(reversed(matches)))
        compiler_api._stamp_matcher_work_scopes(
            SimpleNamespace(name="fake"), module, trace
        )
        return {
            match.target_op_name: match.extra["spmw_work_scope"]
            for match in trace.matches
        }

    original = retained(
        (
            ("ordinary_99", "loc(a)", "add-body", "ADD"),
            ("ordinary_42", "loc(b)", "mul-body", "MUL"),
        )
    )
    reordered = retained(
        (
            ("renamed_mul_700", "loc(new-b)", "mul-body", "MUL"),
            ("renamed_add_800", "loc(new-a)", "add-body", "ADD"),
        )
    )

    assert original == reordered
    assert original["ADD"].group_id != original["MUL"].group_id
    assert all(scope.work_id == () for scope in original.values())


def test_public_compile_selects_matcher_adapter_from_retained_ir(monkeypatch):
    compiler_api = importlib.import_module("allo.compiler")
    schedule = allo.customize(_scope_order_original, enable_tensor=False)
    matches = []
    for function in schedule.module.body.operations:
        if (
            function.operation.name == "func.func"
            and "df.kernel" in function.attributes
        ):
            matches.append(
                _matched_function(
                    compiler_api._retained_symbol_name(function),
                    (12345,),
                )
            )
    trace = MatchTrace("apu_v1", "retained_adapter", matches)
    fake = _FakeCompiled()
    monkeypatch.setattr(compiler_api, "customize", lambda *args, **kwargs: schedule)
    monkeypatch.setattr(compiler_api, "match_workload", lambda *args: trace)
    monkeypatch.setattr(
        compiler_api, "compile_for_target", lambda *args, **kwargs: fake
    )
    monkeypatch.setattr(
        compiler_api,
        "compile_apu_v1_vector_workload",
        lambda *args, **kwargs: pytest.fail("retained dataflow IR used vector adapter"),
    )

    original_metadata = _scope_order_original.mappings
    _scope_order_original.mappings = {"misleading": [1024, 1024]}
    try:
        compiled = compiler_api.compile(
            _scope_order_original,
            SimpleNamespace(name="apu_v1"),
        )
    finally:
        _scope_order_original.mappings = original_metadata

    assert compiled.compiled is fake
    assert all("spmw_work_scope" in match.extra for match in trace.matches)
    assert all(match.work_id != (12345,) for match in trace.matches)


def test_non_kernel_digit_suffix_is_not_a_structural_work_id():
    compiler_api = importlib.import_module("allo.compiler")
    name = "ordinary_stage_99"
    module = SimpleNamespace(
        body=SimpleNamespace(
            operations=[_retained_function(name, "loc(ordinary)", kernel=False)]
        )
    )
    match = _matched_function(name, (99,))
    trace = MatchTrace("fake", "ordinary_digit_suffix", [match])

    compiler_api._stamp_matcher_work_scopes(SimpleNamespace(name="fake"), module, trace)

    assert match.work_id == ()
    assert match.extra["spmw_work_scope"].group_id > 0
    assert match.extra["spmw_work_scope"].work_id == ()


def test_apu_work_grid_consumes_typed_coalesced_scope_without_legacy_tags():
    matches = [
        _matched_function("phase_2024", (0,)),
        _matched_function("answer42", (1,)),
    ]
    for work_id, match in enumerate(matches):
        match.extra["spmw_work_scope"] = MatcherWorkScope(
            group_id=0,
            work_id=(work_id,),
            group_shape=(2,),
            coalesced_axes=(0,),
        )
    trace = MatchTrace("apu_v1", "typed_coalesced_scope", matches)
    target = SimpleNamespace(name="apu_v1", work_grid=lambda: ([4], 4))

    assert _check_work_grid(target, trace, auto_fill=False) == 4
    assert all("coalesced_spmw_axis" not in match.extra for match in matches)
