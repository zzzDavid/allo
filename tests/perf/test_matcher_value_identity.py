# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical matcher value identity and metric-manifest tests."""

from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import MatcherWorkScope, Placement
from allo.spmw_knobs import KnobCtx, _residency_candidates
from allo.spmw_liveness import trace_liveness
from allo.spmw_match import IRValueRef, MatchTrace, MatchedOp, OperandBinding


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
