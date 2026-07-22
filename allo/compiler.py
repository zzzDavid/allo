# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public workload + target + cost compilation facade."""

# Public API spelling intentionally matches Python's built-in compiler verb.
# pylint: disable=redefined-builtin

from __future__ import annotations

import inspect
import math

import numpy as np

from .customize import customize
from .ir.builder import _spmw_group_contract
from .perf import BoundCostSpec, CostSpec
from .pim.aim_program import AimProgram, AimProgramCallable, compile_aim_program
from .pim.apu_v1_program import APUv1Program, compile_apu_v1_program
from .pim.apu_v1_vector_program import (
    APUv1VectorCallable,
    compile_apu_v1_vector_workload,
)
from .pim.apu_g2_program import APUG2Callable, APUG2Program, compile_apu_g2_program
from .pim.apu_g2_vector_program import (
    APUG2AtaxCallable,
    APUG2ChunkedGesummvCallable,
    APUG2ColumnBatchedGemmCallable,
    APUG2ContractionChainCallable,
    APUG2CorrelationCallable,
    APUG2CovarianceCallable,
    APUG2GemverCallable,
    APUG2GemvCallable,
    APUG2IndependentContractionsCallable,
    APUG2RankNContractionCallable,
    APUG2StreamingGemvCallable,
    APUG2SymmCallable,
    APUG2TrmmCallable,
    compile_apu_g2_vector_workload,
)
from .pim.schedule_promotion import PromotionEvidence, SchedulePromotionGate
from .pim.upmem_program import (
    UPMEMProgram,
    UPMEMProgramCallable,
    compile_upmem_program,
)
from .spmw_autoschedule import MatcherWorkScope, _retain_matcher_work_scope
from .spmw_codegen import RunResult, compile_for_target
from .spmw_liveness import MatcherValueId, trace_liveness
from .spmw_match_engine import match_workload
from .spmw_plan import BufferMetricManifest


def _materialize_workload(workload):
    if callable(workload):
        return workload
    build = getattr(workload, "build", None)
    if callable(build):
        return build()
    raise TypeError("workload must be callable or expose a callable build()")


def _materialize_target(target):
    if hasattr(target, "name"):
        return target
    if callable(target):
        built = target()
        if hasattr(built, "name"):
            return built
    raise TypeError("target must be a Target or a zero-argument target builder")


def _resolve_cost(target, cost_spec):
    """Bind an executable cost program to this target instance."""
    if cost_spec is None:
        return None
    if isinstance(cost_spec, BoundCostSpec):
        if cost_spec.target is not target:
            raise ValueError("bound cost spec belongs to a different target instance")
        return cost_spec
    if not isinstance(cost_spec, CostSpec):
        raise TypeError("cost must be an executable CostSpec")
    return cost_spec.bind(target)


def _discover_host_moves(workload):
    direct = getattr(workload, "HOST_MOVES", None)
    if direct is not None:
        return list(direct)
    module = inspect.getmodule(workload)
    if module is not None:
        records = getattr(module, "HOST_MOVES", None)
        if records is not None:
            return list(records)
    return None


def _buffer_metrics(workload):
    """Extract static operand geometry for executable cost rules.

    Allo workloads commonly use postponed annotations, so resolve them in the
    defining module before inspecting ``TypeAnnotation.shape`` and dtype bits.
    Unknown annotations are simply omitted; cost programs remain free to use
    other event metrics.
    """
    try:
        annotations = inspect.get_annotations(workload, eval_str=True)
    except (NameError, TypeError):
        annotations = getattr(workload, "__annotations__", {}) or {}

    metrics = {}
    for name, annotation in annotations.items():
        if name == "return":
            continue
        shape = getattr(annotation, "shape", None)
        if shape is None:
            continue
        try:
            shape = tuple(int(extent) for extent in shape)
        except (TypeError, ValueError):
            continue
        dtype = getattr(annotation, "dtype", None)
        bits = int(getattr(dtype, "bits", 0) or 0)
        elements = math.prod(shape)
        entry = {"shape": shape, "elements": elements}
        if bits > 0:
            entry.update(dtype_bits=bits, bytes=(elements * bits + 7) // 8)
        metrics[name] = entry
    return metrics


def _host_move_buffer_role(record):
    """Recover the workload ABI role from a recorded or resolved host move."""
    role = getattr(record, "buffer_role", None)
    if role is not None:
        return role
    from .spmw_target import BufferToken, HandleToken

    for argument in getattr(record, "args", ()):
        if isinstance(argument, BufferToken):
            return argument.name
        if isinstance(argument, str):
            return argument
        if not isinstance(argument, HandleToken):
            name = getattr(argument, "name", None)
            if name is not None:
                return name
    return None


def _buffer_metric_manifest(workload, trace, host_moves):
    """Bind annotation geometry to exact matcher-visible workload values."""
    metrics_by_name = _buffer_metrics(workload)
    source_value_refs = getattr(trace, "source_value_refs", None)
    if not source_value_refs:
        # Preserve the legacy path for externally constructed traces that do
        # not carry frontend source identity.
        return metrics_by_name

    liveness = trace_liveness(trace)
    values_by_ref = {}
    for match in trace.matches:
        for index, operand in enumerate(match.operands):
            value_id = liveness.value_id_for_operand(match, index)
            if operand.value_ref is not None and value_id is not None:
                values_by_ref.setdefault(operand.value_ref, set()).add(value_id)
        value_id = liveness.value_id_for_result(match)
        if match.result_value_ref is not None and value_id is not None:
            values_by_ref.setdefault(match.result_value_ref, set()).add(value_id)

    parameters = tuple(inspect.signature(workload).parameters)
    parameter_ordinals = {name: index for index, name in enumerate(parameters)}
    entries = {}
    parameter_values = {}
    for name, metrics in metrics_by_name.items():
        ordinal = parameter_ordinals[name]
        source_ref = source_value_refs.get(ordinal)
        candidates = values_by_ref.get(source_ref, set())
        if len(candidates) > 1:
            raise ValueError(
                f"workload buffer {name!r} has conflicting structural values"
            )
        value_id = (
            next(iter(candidates))
            if candidates
            else MatcherValueId("abi", -1, "argument", ordinal)
        )
        existing = entries.get(value_id)
        if existing is not None and existing != metrics:
            raise ValueError("one structural value has conflicting buffer metrics")
        entries[value_id] = metrics
        parameter_values[name] = value_id

    host_bindings = {}
    for index, record in enumerate(host_moves or ()):
        role = _host_move_buffer_role(record)
        if role not in parameter_values:
            raise ValueError(f"host transfer {index} lacks an exact ABI metric binding")
        host_bindings[index] = parameter_values[role]
    return BufferMetricManifest.create(entries, host_bindings=host_bindings)


def _retained_symbol_name(function):
    attributes = getattr(function, "attributes", {})
    if "sym_name" not in attributes:
        return None
    attribute = attributes["sym_name"]
    value = getattr(attribute, "value", None)
    if value is not None:
        return str(value)
    text = str(attribute)
    return text[1:-1] if len(text) >= 2 and text[0] == text[-1] == '"' else text


_SPMW_SCOPE_ATTRS = (
    "spmw.group_id",
    "spmw.work_id",
    "spmw.group_shape",
    "spmw.coalesced_axes",
    "spmw.group_fingerprint",
    "spmw.body_fingerprint",
)
_SPMW_ABI_VALUE_IDS_ATTR = "spmw.abi_value_ids"


def _retained_integer_attribute(attributes, name):
    attribute = attributes[name]
    return int(getattr(attribute, "value", attribute))


def _retained_integer_array_attribute(attributes, name):
    return tuple(
        int(getattr(attribute, "value", attribute)) for attribute in attributes[name]
    )


def _retained_string_attribute(attributes, name):
    attribute = attributes[name]
    value = getattr(attribute, "value", attribute)
    return str(value)


def _retained_matcher_work_scope(function):
    attributes = getattr(function, "attributes", {})
    present = tuple(name in attributes for name in _SPMW_SCOPE_ATTRS)
    if not any(present):
        return None
    if not all(present):
        raise ValueError("retained matcher scope attributes are incomplete")
    return MatcherWorkScope(
        group_id=_retained_integer_attribute(attributes, "spmw.group_id"),
        work_id=_retained_integer_array_attribute(attributes, "spmw.work_id"),
        group_shape=_retained_integer_array_attribute(attributes, "spmw.group_shape"),
        coalesced_axes=_retained_integer_array_attribute(
            attributes, "spmw.coalesced_axes"
        ),
    )


def _retained_function_arguments(function):
    arguments = getattr(function, "arguments", None)
    if arguments is not None:
        return tuple(arguments)
    return None


def _validated_retained_matcher_contract(module):
    body = getattr(module, "body", None)
    records = []
    saw_kernel = False
    for function in getattr(body, "operations", ()):
        operation = getattr(function, "operation", None)
        if getattr(operation, "name", None) != "func.func":
            continue
        attributes = getattr(function, "attributes", {})
        is_kernel = "df.kernel" in attributes
        retained_scope = _retained_matcher_work_scope(function)
        if retained_scope is not None and not is_kernel:
            raise ValueError("retained matcher scope is attached to a non-kernel")
        if not is_kernel:
            continue
        saw_kernel = True
        if retained_scope is None:
            raise ValueError("retained matcher adapter contract is incomplete")
        if _SPMW_ABI_VALUE_IDS_ATTR not in attributes:
            raise ValueError("retained matcher adapter lacks typed ABI identities")
        abi_value_ids = _retained_integer_array_attribute(
            attributes, _SPMW_ABI_VALUE_IDS_ATTR
        )
        if any(value < 0 for value in abi_value_ids):
            raise ValueError("retained matcher ABI identities must be non-negative")
        arguments = _retained_function_arguments(function)
        if arguments is not None and len(abi_value_ids) != len(arguments):
            raise ValueError("retained matcher ABI identity arity is inconsistent")
        if tuple(sorted(set(retained_scope.coalesced_axes))) != tuple(
            retained_scope.coalesced_axes
        ):
            raise ValueError("retained matcher coalesced axes are not canonical")
        if any(
            coordinate >= extent
            for coordinate, extent in zip(
                retained_scope.work_id, retained_scope.group_shape
            )
        ):
            raise ValueError("retained matcher work coordinate is outside its grid")
        records.append(
            {
                "function": function,
                "symbol": _retained_symbol_name(function),
                "scope": retained_scope,
                "group_fingerprint": _retained_string_attribute(
                    attributes, "spmw.group_fingerprint"
                ),
                "body_fingerprint": _retained_string_attribute(
                    attributes, "spmw.body_fingerprint"
                ),
                "abi_value_ids": abi_value_ids,
            }
        )

    if not saw_kernel:
        return ()

    groups = {}
    for record in records:
        scope = record["scope"]
        key = (scope.group_id, record["group_fingerprint"])
        groups.setdefault(key, []).append(record)

    for group_records in groups.values():
        first_scope = group_records[0]["scope"]
        for record in group_records[1:]:
            scope = record["scope"]
            if (
                scope.group_shape != first_scope.group_shape
                or scope.coalesced_axes != first_scope.coalesced_axes
            ):
                raise ValueError("retained matcher group has inconsistent topology")
        expected_size = math.prod(first_scope.group_shape)
        if len(group_records) != expected_size:
            raise ValueError("retained matcher group does not cover its complete grid")
        work_ids = tuple(record["scope"].work_id for record in group_records)
        if len(set(work_ids)) != len(work_ids):
            raise ValueError("retained matcher group repeats a work coordinate")
        expected_id, expected_group_fingerprint, expected_bodies = _spmw_group_contract(
            first_scope.group_shape,
            first_scope.coalesced_axes,
            [(record["scope"].work_id, record["function"]) for record in group_records],
        )
        if expected_id != first_scope.group_id:
            raise ValueError("retained matcher group identity is stale")
        if any(
            record["group_fingerprint"] != expected_group_fingerprint
            for record in group_records
        ):
            raise ValueError("retained matcher group fingerprint is stale")
        expected_bodies_by_work_id = dict(expected_bodies)
        if any(
            record["body_fingerprint"]
            != expected_bodies_by_work_id[record["scope"].work_id]
            for record in group_records
        ):
            raise ValueError("retained matcher body fingerprint is stale")
    return tuple(records)


def _module_has_retained_matcher_scopes(module):
    return bool(_validated_retained_matcher_contract(module))


def _effective_group_ids(records):
    fingerprints_by_group_id = {}
    for record in records:
        scope = record["scope"]
        fingerprints_by_group_id.setdefault(scope.group_id, set()).add(
            record["group_fingerprint"]
        )
    effective = {}
    for record in records:
        scope = record["scope"]
        key = (scope.group_id, record["group_fingerprint"])
        if len(fingerprints_by_group_id[scope.group_id]) == 1:
            effective[key] = scope.group_id
        else:
            effective[key] = int(record["group_fingerprint"], 16)
    if len(set(effective.values())) != len(effective):
        raise ValueError("retained matcher identities collide")
    return effective


def _stamp_matcher_work_scopes(
    target,
    module,
    trace,
):
    """Copy typed retained-IR scopes onto matcher records."""
    matches = list(getattr(trace, "matches", ()) or ())
    kernel_records = list(_validated_retained_matcher_contract(module))

    matches_by_symbol = {}
    for match in matches:
        matches_by_symbol.setdefault(match.func_name, []).append(match)

    body = getattr(module, "body", None)
    operations = getattr(body, "operations", ())
    functions_by_symbol = {}
    for function in operations:
        operation = getattr(function, "operation", None)
        if getattr(operation, "name", None) != "func.func":
            continue
        symbol = _retained_symbol_name(function)
        if symbol in functions_by_symbol:
            raise ValueError("retained module repeats a function symbol")
        functions_by_symbol[symbol] = function

    records_by_symbol = {record["symbol"]: dict(record) for record in kernel_records}
    if len(records_by_symbol) != len(kernel_records):
        raise ValueError("retained matcher kernels repeat a function symbol")
    for symbol, function_matches in matches_by_symbol.items():
        function = functions_by_symbol.get(symbol)
        if function is None:
            raise ValueError("matcher trace site has no retained function boundary")
        record = records_by_symbol.get(symbol)
        if record is None:
            group_id, group_fingerprint, body_fingerprints = _spmw_group_contract(
                (), (), [((), function)]
            )
            record = {
                "function": function,
                "symbol": symbol,
                "scope": MatcherWorkScope(group_id, (), (), ()),
                "group_fingerprint": group_fingerprint,
                "body_fingerprint": dict(body_fingerprints)[()],
                "abi_value_ids": (),
            }
            records_by_symbol[symbol] = record
        record["matches"] = function_matches

    if any("matches" not in record for record in records_by_symbol.values()):
        raise ValueError("retained matcher kernel has no matched implementation")

    records = list(records_by_symbol.values())
    non_kernel_keys = set()
    for record in records:
        attributes = getattr(record["function"], "attributes", {})
        if "df.kernel" in attributes:
            continue
        key = (record["scope"].group_id, record["group_fingerprint"])
        if key in non_kernel_keys:
            raise ValueError("non-kernel matcher boundaries are structurally ambiguous")
        non_kernel_keys.add(key)

    assigned = {id(match) for record in records for match in record["matches"]}
    if len(assigned) != len(matches):
        raise ValueError("matcher trace site has no retained function boundary")

    del target
    effective_group_ids = _effective_group_ids(records)
    for record in records:
        scope = record["scope"]
        effective_scope = MatcherWorkScope(
            effective_group_ids[(scope.group_id, record["group_fingerprint"])],
            scope.work_id,
            scope.group_shape,
            scope.coalesced_axes,
        )
        for match in record["matches"]:
            match.work_id = effective_scope.work_id
            _retain_matcher_work_scope(match, effective_scope)


class CompiledCallable:
    """An Allo-style callable backed by a compiled PIM artifact.

    Calls accept positional or keyword NumPy operands according to the original
    workload signature and return a :class:`RunResult`. When a backend returns
    output arrays, explicitly gathered output arguments are updated in place,
    matching the mutation behavior of Allo's LLVM callable modules.
    """

    def __init__(
        self,
        workload,
        target,
        schedule,
        trace,
        compiled,
        cost=None,
    ):
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.trace = trace
        self.compiled = compiled
        self.cost = cost
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "compiled_workload")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.schedule_search_result = getattr(
            compiled,
            "schedule_search_result",
            None,
        )
        self.schedule_activation = getattr(compiled, "schedule_activation", None)
        self.fallback_reason = getattr(compiled, "fallback_reason", None)
        self.last_result: RunResult | None = None

    def __call__(self, *args, **kwargs) -> RunResult:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return self.run_backend(**bound.arguments)

    def run_backend(self, **inputs) -> RunResult:
        """Invoke with explicit backend-role arrays.

        This compiler-testing escape hatch supports backend ABIs whose lowered
        operand roles differ from the source workload signature. Applications
        should normally call the object directly.
        """
        result = self.compiled.run(**inputs)
        self._copy_outputs(result, inputs)
        self.last_result = result
        return result

    run = __call__

    @property
    def execution_graph(self):
        return self.compiled.execution_graph

    def estimate(self):
        """Evaluate the retained candidate graph without invoking a backend."""
        if self.cost is None:
            raise RuntimeError("compiled workload has no executable cost spec")
        if self.execution_graph is None:
            raise RuntimeError("compiled workload has no retained execution graph")
        return self.cost.evaluate(self.execution_graph)

    def _copy_outputs(self, result, arguments):
        outputs = result.extra.get("outputs", {}) if result.extra else {}
        if not outputs:
            return
        gather_roles = []
        for resolved in getattr(self.compiled, "host_moves", ()):
            verb = getattr(getattr(resolved, "verb", None), "name", None)
            if verb == "gather" and resolved.buffer_role in arguments:
                gather_roles.append(resolved.buffer_role)

        assignments = {}
        lower_names = {name.lower(): name for name in arguments}
        for output_name, value in outputs.items():
            if output_name in arguments:
                assignments[output_name] = value
            elif output_name.lower() in lower_names:
                assignments[lower_names[output_name.lower()]] = value
        if len(outputs) == 1 and len(gather_roles) == 1:
            assignments.setdefault(gather_roles[0], next(iter(outputs.values())))

        for name, value in assignments.items():
            destination = arguments[name]
            if not isinstance(destination, np.ndarray):
                continue
            source = np.asarray(value)
            if source.shape != destination.shape:
                if source.size != destination.size:
                    raise ValueError(
                        f"backend output {name!r} has shape {source.shape}, "
                        f"but destination has shape {destination.shape}"
                    )
                source = source.reshape(destination.shape)
            np.copyto(destination, source, casting="same_kind")


def compile(
    workload,
    target,
    cost=None,
    *,
    backend=None,
    host_moves=None,
    layout=None,
    promotion_evidence=None,
) -> (
    CompiledCallable
    | AimProgramCallable
    | UPMEMProgramCallable
    | APUv1VectorCallable
    | APUG2Callable
    | APUG2ChunkedGesummvCallable
    | APUG2GemvCallable
    | APUG2GemverCallable
    | APUG2ContractionChainCallable
    | APUG2CorrelationCallable
    | APUG2CovarianceCallable
    | APUG2RankNContractionCallable
    | APUG2SymmCallable
    | APUG2TrmmCallable
):
    """Compile ``workload`` for ``target`` and return a NumPy-callable object.

    Parameters
    ----------
    workload : callable or object
        An Allo workload callable, or a module/object exposing ``build()``.
    target : Target or callable
        A built Tenon target or a zero-argument target builder.
    cost : CostSpec
        Executable cost program used by autoscheduling and virtual execution.
    backend : str or None
        ``None`` selects the target's normal simulator/device runner;
        ``"virtual"`` evaluates only the analytical performance graph.
    host_moves : sequence or None
        Optional explicit host-transfer records. If omitted, ``HOST_MOVES`` is
        discovered beside the workload.
    layout : object
        Optional preselected placement or placement list.
    promotion_evidence : PromotionEvidence or None
        Exact correctness, materialization, and repeated-performance evidence
        for an already inspected search challenger. ``None`` always retains
        the incumbent when the search recommendation differs.
    """
    workload = _materialize_workload(workload)
    target = _materialize_target(target)
    bound_cost = _resolve_cost(target, cost)
    if promotion_evidence is not None and not isinstance(
        promotion_evidence,
        PromotionEvidence,
    ):
        raise TypeError("promotion_evidence must be a PromotionEvidence record")
    promotion_gate = (
        None
        if promotion_evidence is None
        else SchedulePromotionGate(promotion_evidence)
    )
    if promotion_gate is not None and bound_cost is None:
        raise ValueError("promotion evidence requires an executable cost model")
    if promotion_gate is not None and layout is not None:
        raise ValueError("promotion evidence cannot override an explicit layout")

    if isinstance(workload, AimProgram):
        if promotion_gate is not None:
            raise ValueError("AimProgram has no schedule-search activation")
        if host_moves is not None or layout is not None:
            raise ValueError("AimProgram owns its ordered trace and placement")
        return compile_aim_program(
            workload,
            target,
            cost=bound_cost,
            backend=backend,
        )

    if isinstance(workload, APUv1Program):
        if promotion_gate is not None:
            raise ValueError("APUv1Program has no schedule-search activation")
        if backend not in (None, "virtual", "functional"):
            raise ValueError(
                "APUv1Program supports the device, virtual, or functional backend"
            )
        if host_moves is not None or layout is not None:
            raise ValueError("APUv1Program owns its scalar L4 ABI")
        return compile_apu_v1_program(
            workload, target, cost=bound_cost, backend=backend
        )

    if isinstance(workload, UPMEMProgram):
        if backend not in (None, "virtual", "functional"):
            raise ValueError(
                "MLIR-driven UPMEMProgram currently supports only the functional "
                "portable-C runtime (backend=None, 'virtual', or 'functional')"
            )
        if host_moves is not None or layout is not None:
            raise ValueError(
                "UPMEMProgram owns its phased ABI; host_moves/layout are not accepted"
            )
        return compile_upmem_program(
            workload,
            target,
            cost=bound_cost,
            promotion_gate=promotion_gate,
        )

    if isinstance(workload, APUG2Program):
        if promotion_gate is not None:
            raise ValueError("APUG2Program has no schedule-search activation")
        if host_moves is not None or layout is not None:
            raise ValueError("APUG2Program owns its VL64 layout and hardware ABI")
        return compile_apu_g2_program(
            workload,
            target,
            cost=bound_cost,
            backend=backend,
        )

    if target.name == "upmem":
        raise TypeError(
            "the contraction-only UPMEM matcher backend was removed; wrap one "
            "or more MLIR callables in allo.UPMEMProgram"
        )

    if host_moves is None:
        host_moves = _discover_host_moves(workload)

    schedule = customize(workload, enable_tensor=False)
    retained_matcher_dataflow = _module_has_retained_matcher_scopes(schedule.module)
    # Ordinary contractions use the new MLIR -> layout-plan -> vector path.
    # Dataflow regions retain their matcher/group implementation through typed
    # structural scope attributes stamped on retained IR kernel functions.
    if target.name == "apu_v1" and not retained_matcher_dataflow:
        from .pim.contraction_analysis import NoContractionError

        try:
            return compile_apu_v1_vector_workload(
                workload,
                target,
                schedule,
                cost=bound_cost,
                layout=layout,
                backend=backend,
                promotion_gate=promotion_gate,
            )
        except NoContractionError:
            # Non-contraction APU workloads continue through the existing
            # matcher path; this dispatch is deliberately additive.
            pass
    if target.name == "apu_v2" and not retained_matcher_dataflow:
        from .pim.contraction_analysis import NoContractionError

        try:
            if host_moves is not None or layout is not None:
                raise ValueError(
                    "MLIR-driven APUg2 GEMV owns its reduction layout and hardware ABI"
                )
            return compile_apu_g2_vector_workload(
                workload,
                target,
                schedule,
                cost=bound_cost,
                backend=backend,
                promotion_gate=promotion_gate,
            )
        except NoContractionError:
            # Non-contractions retain the established matcher path.
            pass
    trace = match_workload(target, schedule.module)
    _stamp_matcher_work_scopes(target, schedule.module, trace)
    compiled = compile_for_target(
        target,
        trace,
        layout=layout,
        backend=backend,
        host_moves=host_moves,
        buffer_metrics=_buffer_metric_manifest(workload, trace, host_moves),
        cost=bound_cost,
        promotion_gate=promotion_gate,
    )
    return CompiledCallable(
        workload,
        target,
        schedule,
        trace,
        compiled,
        cost=bound_cost,
    )
