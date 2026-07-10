# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical matcher value identity and metric-manifest tests."""

from types import SimpleNamespace

import allo
import allo.dataflow as df
import pytest

from allo.compiler import _stamp_matcher_work_scopes
from allo.ir.types import float32
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import MatcherWorkScope, Placement
from allo.spmw_knobs import KnobCtx, _residency_candidates
from allo.spmw_liveness import trace_liveness
from allo.spmw_match import IRValueRef, MatchTrace, MatchedOp, OperandBinding
from allo.spmw_match_engine import match_workload
from allo.spmw_plan import BufferMetricManifest, build_execution_graph


@df.region()
def _retained_non_adjacent_handoff(
    source: float32[4],
    middle: float32[4],
    unrelated_source: float32[4],
    unrelated_result: float32[4],
    final: float32[4],
):
    @allo.work(mapping=1, args=[source, middle])
    def producer(value: float32[4], result: float32[4]):
        result[0] = value[0] + value[1]

    @allo.work(mapping=1, args=[unrelated_source, unrelated_result])
    def unrelated(value: float32[4], result: float32[4]):
        result[1] = value[1] + value[2]

    @allo.work(mapping=1, args=[middle, final])
    def consumer(renamed_input: float32[4], result: float32[4]):
        result[2] = renamed_input[2] + renamed_input[3]


@df.region()
def _retained_local_handoffs(source: float32[4], final: float32[4]):
    first: float32[4] = 0.0
    second: float32[4] = 0.0

    @allo.work(mapping=1, args=[source, first])
    def first_stage(value: float32[4], result: float32[4]):
        result[0] = value[0] + value[1]

    @allo.work(mapping=1, args=[first, second])
    def second_stage(value: float32[4], result: float32[4]):
        result[1] = value[1] + value[2]

    @allo.work(mapping=1, args=[second, final])
    def final_stage(value: float32[4], result: float32[4]):
        result[2] = value[2] + value[3]


def _mac(
    group_id,
    work_id,
    names,
    *,
    group_shape=(1,),
    operand_types=(None, None, None),
    value_refs=(None, None, None),
):
    weight, vector, output = names
    match = MatchedOp(
        target_op_name="MAC",
        func_name=f"diagnostic_{group_id}_{work_id}",
        work_id=work_id,
        enclosing_loops=[("k", "0", "128", 1)],
        operands=[
            OperandBinding(
                "x",
                weight,
                memref_type=operand_types[0],
                value_ref=value_refs[0],
            ),
            OperandBinding(
                "y",
                vector,
                memref_type=operand_types[1],
                value_ref=value_refs[1],
            ),
            OperandBinding(
                "acc",
                output,
                is_loop_carried=True,
                memref_type=operand_types[2],
                value_ref=value_refs[2],
            ),
        ],
        result_memref_name=output,
        op_range=("begin", "end"),
        extra={
            "spmw_work_scope": MatcherWorkScope(
                group_id,
                work_id,
                group_shape,
                (),
            )
        },
        result_value_ref=value_refs[2],
    )
    return match


def _value_ref(identity):
    return IRValueRef("test", (identity,))


def _liveness_signature(liveness):
    return tuple(
        (
            value_id,
            span.group_ids,
            span.work_ids,
            span.roles,
            span.crosses_kernel,
            span.crosses_workid,
        )
        for value_id, span in liveness.items()
    )


def test_memref_renames_preserve_value_ids_and_residency_domain():
    target = build_samsung_target()
    value_refs = tuple(_value_ref(index) for index in range(3))
    original_matches = [
        _mac(
            0,
            (0,),
            ("weight", "vector", "output"),
            group_shape=(2,),
            value_refs=value_refs,
        ),
        _mac(
            0,
            (1,),
            ("weight", "vector", "output"),
            group_shape=(2,),
            value_refs=value_refs,
        ),
    ]
    renamed_matches = [
        _mac(
            0,
            (0,),
            ("alpha", "beta", "gamma"),
            group_shape=(2,),
            value_refs=value_refs,
        ),
        _mac(
            0,
            (1,),
            ("alpha", "beta", "gamma"),
            group_shape=(2,),
            value_refs=value_refs,
        ),
    ]
    original = trace_liveness(MatchTrace(target.name, "original", original_matches))
    renamed = trace_liveness(MatchTrace(target.name, "renamed", renamed_matches))

    assert _liveness_signature(renamed) == _liveness_signature(original)
    original_ctx = KnobCtx(
        target=target,
        matches=[original_matches[0]],
        base=Placement(placements={"vector": target.grf_a}),
        liveness=original,
    )
    renamed_ctx = KnobCtx(
        target=target,
        matches=[renamed_matches[0]],
        base=Placement(placements={"beta": target.grf_a}),
        liveness=renamed,
    )
    assert _residency_candidates(original_ctx) == ["restage", "resident"]
    assert _residency_candidates(renamed_ctx) == _residency_candidates(original_ctx)


def test_same_named_local_results_do_not_merge_across_groups():
    first = _mac(0, (0,), ("weight_a", "input_a", "temporary"))
    second = _mac(1, (0,), ("weight_b", "input_b", "temporary"))
    liveness = trace_liveness(
        MatchTrace("samsung_hbm_pim", "collision", [first, second])
    )

    value_ids = liveness.value_ids_for_memref("temporary")
    assert len(value_ids) == 2
    assert all(not liveness[value_id].crosses_kernel for value_id in value_ids)
    assert liveness.span_for_memref("temporary") is None


def test_adjacent_same_type_buffers_do_not_create_a_handoff():
    producer = _mac(
        0,
        (0,),
        ("producer_weight", "producer_input", "producer_output"),
        operand_types=(
            "memref<32x16xf32>",
            "memref<16xf32>",
            "memref<32xf32>",
        ),
    )
    consumer = _mac(
        1,
        (0,),
        ("consumer_weight", "unrelated_spelling", "consumer_output"),
        operand_types=(
            "memref<8x32xf32>",
            "memref<32xf32>",
            "memref<8xf32>",
        ),
    )
    liveness = trace_liveness(
        MatchTrace("samsung_hbm_pim", "typed_handoff", [producer, consumer])
    )
    producer_id = liveness.value_id_for_result(producer)
    consumer_id = liveness.value_id_for_operand(consumer, "y")

    assert consumer_id != producer_id
    assert not liveness[producer_id].crosses_kernel
    assert not liveness[consumer_id].crosses_kernel


def test_exact_non_adjacent_handoff_ignores_intervening_same_type_buffer():
    shared = _value_ref(20)
    producer = _mac(
        0,
        (0,),
        ("producer_weight", "producer_input", "producer_output"),
        operand_types=("memref<8x32xf32>", "memref<32xf32>", "memref<8xf32>"),
        value_refs=(_value_ref(1), _value_ref(2), shared),
    )
    unrelated = _mac(
        1,
        (0,),
        ("other_weight", "other_input", "other_output"),
        operand_types=("memref<8x32xf32>", "memref<8xf32>", "memref<8xf32>"),
        value_refs=(_value_ref(3), _value_ref(4), _value_ref(5)),
    )
    consumer = _mac(
        2,
        (0,),
        ("consumer_weight", "renamed_handoff", "consumer_output"),
        operand_types=("memref<8x8xf32>", "memref<8xf32>", "memref<8xf32>"),
        value_refs=(_value_ref(6), shared, _value_ref(7)),
    )

    liveness = trace_liveness(
        MatchTrace(
            "samsung_hbm_pim",
            "non_adjacent_handoff",
            [producer, unrelated, consumer],
        )
    )
    producer_id = liveness.value_id_for_result(producer)

    assert liveness.value_id_for_operand(consumer, "y") == producer_id
    assert liveness.value_id_for_result(unrelated) != producer_id
    assert liveness[producer_id].crosses_kernel
    assert liveness[producer_id].producer_func == (0, (0,))
    assert liveness[producer_id].consumer_func == (2, (0,))


def test_frontend_retains_exact_non_adjacent_call_and_abi_edge():
    target = build_samsung_target()
    schedule = allo.customize(_retained_non_adjacent_handoff, enable_tensor=False)
    trace = match_workload(target, schedule.module)
    _stamp_matcher_work_scopes(target, schedule.module, trace)

    assert len(trace.matches) == 3
    producer, unrelated, consumer = trace.matches
    producer_ref = producer.result_value_ref
    consumer_refs = {operand.value_ref for operand in consumer.operands}
    assert producer_ref is not None
    assert producer_ref in consumer_refs
    assert unrelated.result_value_ref != producer_ref
    assert all(
        operand.value_ref is not None
        for match in trace.matches
        for operand in match.operands
    )

    liveness = trace_liveness(trace)
    producer_id = liveness.value_id_for_result(producer)
    assert producer_id in {
        liveness.value_id_for_operand(consumer, index)
        for index in range(len(consumer.operands))
    }
    assert liveness[producer_id].crosses_kernel


def test_frontend_retains_distinct_local_buffer_call_edges():
    target = build_samsung_target()
    schedule = allo.customize(_retained_local_handoffs, enable_tensor=False)
    trace = match_workload(target, schedule.module)
    _stamp_matcher_work_scopes(target, schedule.module, trace)

    assert len(trace.matches) == 3
    first, second, final = trace.matches
    second_inputs = {operand.value_ref for operand in second.operands}
    final_inputs = {operand.value_ref for operand in final.operands}
    assert first.result_value_ref in second_inputs
    assert second.result_value_ref in final_inputs
    assert first.result_value_ref != second.result_value_ref

    liveness = trace_liveness(trace)
    assert liveness[liveness.value_id_for_result(first)].crosses_kernel
    assert liveness[liveness.value_id_for_result(second)].crosses_kernel


def test_multiple_same_type_values_follow_only_their_exact_edges():
    first_ref = _value_ref(30)
    second_ref = _value_ref(31)
    memref_types = ("memref<8x8xf32>", "memref<8xf32>", "memref<8xf32>")
    first_producer = _mac(
        0,
        (0,),
        ("w0", "x0", "out0"),
        operand_types=memref_types,
        value_refs=(_value_ref(32), _value_ref(33), first_ref),
    )
    second_producer = _mac(
        1,
        (0,),
        ("w1", "x1", "out1"),
        operand_types=memref_types,
        value_refs=(_value_ref(34), _value_ref(35), second_ref),
    )
    second_consumer = _mac(
        2,
        (0,),
        ("w2", "second_alias", "out2"),
        operand_types=memref_types,
        value_refs=(_value_ref(36), second_ref, _value_ref(37)),
    )
    first_consumer = _mac(
        3,
        (0,),
        ("w3", "first_alias", "out3"),
        operand_types=memref_types,
        value_refs=(_value_ref(38), first_ref, _value_ref(39)),
    )
    liveness = trace_liveness(
        MatchTrace(
            "samsung_hbm_pim",
            "multiple_same_type",
            [first_producer, second_producer, second_consumer, first_consumer],
        )
    )
    first_id = liveness.value_id_for_result(first_producer)
    second_id = liveness.value_id_for_result(second_producer)

    assert first_id != second_id
    assert liveness.value_id_for_operand(first_consumer, "y") == first_id
    assert liveness.value_id_for_operand(second_consumer, "y") == second_id
    assert liveness[first_id].crosses_kernel
    assert liveness[second_id].crosses_kernel


def test_ambiguous_exact_handoff_with_multiple_producers_fails_closed():
    shared = _value_ref(50)
    first = _mac(
        0,
        (0,),
        ("w0", "x0", "out0"),
        value_refs=(_value_ref(51), _value_ref(52), shared),
    )
    second = _mac(
        1,
        (0,),
        ("w1", "x1", "out1"),
        value_refs=(_value_ref(53), _value_ref(54), shared),
    )
    consumer = _mac(
        2,
        (0,),
        ("w2", "input", "out2"),
        value_refs=(_value_ref(55), shared, _value_ref(56)),
    )
    liveness = trace_liveness(
        MatchTrace("samsung_hbm_pim", "ambiguous_handoff", [first, second, consumer])
    )
    shared_id = liveness.value_id_for_result(first)

    assert liveness.value_id_for_result(second) == shared_id
    assert liveness.value_id_for_operand(consumer, "y") == shared_id
    assert not liveness[shared_id].crosses_kernel
    target = build_samsung_target()
    assert _residency_candidates(
        KnobCtx(
            target=target,
            matches=[consumer],
            base=Placement(placements={"input": target.grf_a}),
            liveness=liveness,
        )
    ) == ["restage"]


def test_structural_manifest_scores_identically_after_all_buffer_renames():
    target = build_samsung_target()
    types = ("memref<128xf32>", "memref<128xf32>", "memref<8xf32>")
    original_match = _mac(0, (0,), ("weight", "vector", "output"), operand_types=types)
    renamed_match = _mac(0, (0,), ("one", "two", "three"), operand_types=types)
    original_trace = MatchTrace(target.name, "original_metrics", [original_match])
    renamed_trace = MatchTrace(target.name, "renamed_metrics", [renamed_match])
    original_values = trace_liveness(original_trace)
    renamed_values = trace_liveness(renamed_trace)
    vector_id = original_values.value_id_for_operand(original_match, "y")
    output_id = original_values.value_id_for_result(original_match)

    assert vector_id == renamed_values.value_id_for_operand(renamed_match, "y")
    assert output_id == renamed_values.value_id_for_result(renamed_match)
    manifest = BufferMetricManifest.create(
        {
            vector_id: {"shape": (128,), "elements": 128, "bytes": 64},
            output_id: {"shape": (8,), "elements": 8, "bytes": 32},
        },
        host_bindings={0: vector_id, 1: output_id},
    )
    changed_manifest = BufferMetricManifest.create(
        {
            vector_id: {"shape": (128,), "elements": 128, "bytes": 96},
            output_id: {"shape": (8,), "elements": 8, "bytes": 32},
        },
        host_bindings={0: vector_id, 1: output_id},
    )
    assert changed_manifest.fingerprint != manifest.fingerprint

    def host_moves(vector_name, output_name):
        return [
            SimpleNamespace(
                verb=SimpleNamespace(name="scatter"),
                move=target.move("SCATTER_BANKS"),
                buffer_role=vector_name,
            ),
            SimpleNamespace(
                verb=SimpleNamespace(name="gather"),
                move=target.move("GATHER_BANKS"),
                buffer_role=output_name,
            ),
        ]

    bound = samsung_cost.bind(target)
    original_graph = build_execution_graph(
        target,
        original_trace,
        Placement(placements={}, extra={"n_fibers": 1}),
        bound,
        host_moves=host_moves("vector", "output"),
        buffer_metrics=manifest,
    )
    renamed_graph = build_execution_graph(
        target,
        renamed_trace,
        Placement(placements={}, extra={"n_fibers": 1}),
        bound,
        host_moves=host_moves("arbitrary_input", "arbitrary_output"),
        buffer_metrics=manifest,
    )

    assert bound.evaluate(original_graph).cycles == 4 + 65 + 3
    assert bound.evaluate(renamed_graph).cycles == bound.evaluate(original_graph).cycles
    assert (
        renamed_graph.metadata["buffer_metric_fingerprint"]
        == original_graph.metadata["buffer_metric_fingerprint"]
        == manifest.fingerprint
    )


def test_ambiguous_name_mapping_requires_typed_manifest():
    target = build_samsung_target()
    types = ("memref<16xf32>", "memref<32xf32>", "memref<8xf32>")
    first = _mac(
        0,
        (0,),
        ("weight_a", "input_a", "temporary"),
        operand_types=types,
    )
    second = _mac(
        1,
        (0,),
        ("weight_b", "input_b", "temporary"),
        operand_types=types,
    )
    trace = MatchTrace(target.name, "ambiguous_metrics", [first, second])

    with pytest.raises(ValueError, match="no structural value binding"):
        build_execution_graph(
            target,
            trace,
            [Placement(placements={}), Placement(placements={})],
            samsung_cost.bind(target),
            buffer_metrics={"temporary": {"shape": (8,), "bytes": 32}},
        )


@pytest.mark.parametrize("legacy_name", ["weight", "LOCAL_WEIGHT"])
def test_missing_structural_metric_binding_fails_closed(legacy_name):
    target = build_samsung_target()
    match = _mac(0, (0,), ("local_weight", "local_vector", "local_output"))
    trace = MatchTrace(target.name, "missing_manifest", [match])

    with pytest.raises(ValueError, match="no structural value binding"):
        build_execution_graph(
            target,
            trace,
            Placement(placements={}),
            samsung_cost.bind(target),
            buffer_metrics={legacy_name: {"shape": (17,), "bytes": 68}},
        )
