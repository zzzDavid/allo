"""APU v1 inter-VR / intra-VR move-cost mechanism (task 015, design 01 §4).

These tests pin the data-movement decision that design 01 surfaced:

  * the enumerator emits {sv, sv_lookup} x {intra, inter} vr_dma candidates;
  * the cost model prices the inter-stage retile + L4 DMA so argmin sees it;
  * a single-stage GEMV keeps the intra-VR pick (zero stage boundaries),
    re-proving the -0.17% board floor layout;
  * a two-stage FFN flips to inter-VR organically (the per-tile replication
    penalty intra pays at the stage boundary is what inter avoids);
  * every term traces to operand shape + `target.move(...).cycles` -- no
    256/64/intra/inter literal keyed on the workload.

No simulator is booted; these are static cost/argmin assertions (the board
run is the verifier's domain).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import allo  # noqa: E402
from allo.spmw_autoschedule import autoschedule, _apu_v1_enumerate  # noqa: E402
from allo.spmw_cost_models import (  # noqa: E402
    _apu_v1_vr_tiling,
    _apu_v1_move_cycles,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding  # noqa: E402
from _fixtures import build_apu_v1_target  # noqa: E402


def _gemv_trace() -> MatchTrace:
    """Single-stage GEMV-shaped trace: one MAC match, K=1024."""
    return MatchTrace(
        target_name="apu_v1",
        module_name="synthetic_gemv",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%i", "0", "16", 1),
                    ("%k", "0", "1024", 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(role="acc", memref_name="acc",
                                   is_loop_carried=True),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )


def _ffn_trace() -> MatchTrace:
    """Fused two-stage FFN-shaped trace (64-256-64): one `@allo.work`
    kernel (matching the board's single fused `ffn-cinm-opt` task) with
    two MAC matches, so one producer->consumer stage boundary. Layer 1
    output rows = 256 (the hidden dim replicated across consumer column
    groups); layer 2 output rows = 64. Shapes are loop bounds read by the
    cost model; no literal reaches the enumerator/cost code path."""
    return MatchTrace(
        target_name="apu_v1",
        module_name="synthetic_ffn",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="ffn_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%i1", "0", "256", 1),  # hidden dim (layer-1 output)
                    ("%k1", "0", "64", 1),   # input dim (contraction)
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W1"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(role="acc", memref_name="local_h",
                                   is_loop_carried=True),
                ],
                result_memref_name="local_h",
                op_range=("%a", "%b"),
            ),
            MatchedOp(
                target_op_name="MAC",
                func_name="ffn_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%i2", "0", "64", 1),   # output dim
                    ("%k2", "0", "256", 1),  # hidden dim (contraction)
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W2"),
                    OperandBinding(role="y", memref_name="local_h"),
                    OperandBinding(role="acc", memref_name="local_y",
                                   is_loop_carried=True),
                ],
                result_memref_name="local_y",
                op_range=("%c", "%d"),
            ),
        ],
    )


# --------------------------------------------------------------------- #
# Shape arithmetic provenance
# --------------------------------------------------------------------- #


def test_vr_tiling_is_shape_derived():
    """Tile counts come from loop bounds + target.vrs.width; the stage
    boundary count is (#MAC matches - 1)."""
    target = build_apu_v1_target()

    n_out_g, n_wt_g, n_bnd_g = _apu_v1_vr_tiling(target, _gemv_trace())
    # GEMV: one MAC match -> zero boundaries.
    assert n_bnd_g == 0
    # Output rows 16 and weight 16*1024 both fit in one 32K lane fold.
    assert n_out_g == 1
    assert n_wt_g == 1

    n_out_f, n_wt_f, n_bnd_f = _apu_v1_vr_tiling(target, _ffn_trace())
    # FFN: two MAC matches in one fused kernel -> one boundary.
    assert n_bnd_f == 1
    # Tiny weights (64*256 + 256*64 = 32768 elems) fold into one tile;
    # widest output (256) also folds into one tile.
    assert n_out_f == 1
    assert n_wt_f == 1


def test_move_cost_traces_to_target_move_cycles():
    """The per-move cost is exactly the declared DMA + LD_VR move-op
    cycles on the spec -- not a magic constant. The move COUNT is the
    documented closed form over the shape-derived tile counts."""
    target = build_apu_v1_target()
    # Per design 04 the per-move cost lives on the bound CostModel.
    from allo.spmw_cost_model import MoveCostCtx, get_cost_model
    _m = get_cost_model("apu_v1", "faithful")
    per_move = (
        _m.move_cost("DMA_L4_L1", MoveCostCtx("DMA_L4_L1"))
        + _m.move_cost("LD_VR", MoveCostCtx("LD_VR"))
    )

    n_out, n_wt, n_bnd = _apu_v1_vr_tiling(target, _ffn_trace())
    layout_intra = allo.Placement(extra={"vr_dma": "intra"})
    layout_inter = allo.Placement(extra={"vr_dma": "inter"})

    # intra pays the weight stream + per-boundary activation replication;
    # inter pays only the weight stream (no replication).
    expect_intra = (n_wt + n_bnd * n_out) * per_move
    expect_inter = n_wt * per_move
    assert _apu_v1_move_cycles(target, _ffn_trace(), layout_intra) == expect_intra
    assert _apu_v1_move_cycles(target, _ffn_trace(), layout_inter) == expect_inter
    # The boundary makes intra strictly costlier for the fused FFN.
    assert expect_intra > expect_inter


def test_move_cost_zero_without_vr_dma_key():
    """Back-compat / rollback: a placement with no vr_dma key prices 0
    move cycles, so the compute-only ranking is preserved (design 01 §7)."""
    target = build_apu_v1_target()
    bare = allo.Placement(mode="sv_lookup")  # no extra["vr_dma"]
    assert _apu_v1_move_cycles(target, _ffn_trace(), bare) == 0
    assert _apu_v1_move_cycles(target, _gemv_trace(), bare) == 0


def test_gemv_move_cost_ties_across_vr_dma():
    """Single-stage GEMV has zero stage boundaries, so intra and inter
    cost the same on the move term -- the replication penalty only exists
    at a producer->consumer boundary. Both pay the unavoidable weight
    stream, so they tie (not zero) and the enumerator-order tie-break
    keeps intra."""
    target = build_apu_v1_target()
    intra = allo.Placement(extra={"vr_dma": "intra"})
    inter = allo.Placement(extra={"vr_dma": "inter"})
    c_intra = _apu_v1_move_cycles(target, _gemv_trace(), intra)
    c_inter = _apu_v1_move_cycles(target, _gemv_trace(), inter)
    assert c_intra == c_inter


# --------------------------------------------------------------------- #
# argmin behaviour
# --------------------------------------------------------------------- #


def test_argmin_keeps_intra_sv_lookup_for_gemv():
    """GEMV non-regression gate: argmin must pick sv_lookup + intra-VR
    (the board's -0.17%-floor layout)."""
    target = build_apu_v1_target()
    layouts = autoschedule(target, _gemv_trace())
    assert len(layouts) == 1
    chosen = layouts[0]
    assert chosen.mode == "sv_lookup", chosen.mode
    assert chosen.extra.get("vr_dma") == "intra", chosen.extra


def test_argmin_flips_to_inter_for_fused_ffn():
    """FFN win gate (model side): for the fused two-stage FFN kernel the
    inter-stage replication makes intra-VR strictly costlier than inter-VR
    (kernel_cycles term), so the cheapest candidate is sv_lookup + inter.
    This is the decision design 01 moved *inside* Tenon; the board
    re-measure proves the real crun drop.

    NOTE: this scores the candidates on the FULL fused trace directly,
    because the per-bucket `autoschedule` loop groups by func_name and a
    fused FFN's two MACs bind role `x` to different weights (W1 vs W2),
    which `_trace_memrefs_by_role` rejects in one bucket. The cross-stage
    retile is therefore only visible to the cost model on the whole-trace
    call, not the per-bucket `autoschedule` path -- the open seam flagged
    to the architect in design 01 §8."""
    target = build_apu_v1_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    trace = _ffn_trace()
    candidates = [
        allo.Placement(mode=m, extra={"vr_dma": d})
        for m in ("sv", "sv_lookup")
        for d in ("intra", "inter")
    ]
    scored = sorted(
        (cost_fn(trace, c), i, c) for i, c in enumerate(candidates)
    )
    best = scored[0][2]
    assert best.mode == "sv_lookup", best.mode
    assert best.extra["vr_dma"] == "inter", best.extra


def test_enumerator_emits_four_distinct_candidates():
    """{sv, sv_lookup} x {intra, inter} = 4 candidates, each carrying the
    shape-derived tile counts. Uses a single-stage layer trace (one bucket
    = uniform role->memref, the shape the per-bucket enumerator sees)."""
    target = build_apu_v1_target()
    cands = _apu_v1_enumerate(target, _gemv_trace().matches)
    # The {sv, sv_lookup} x {intra, inter} = 4 distinct (mode, vr_dma) keys are
    # unchanged. APU v1 also carries the SPEC-023 D3 double_buffer 2x fan (it
    # has the overlap-fold DMA-hide DOF), so the depth-1 subset is exactly
    # those 4; depth does not change the (mode, vr_dma) key set.
    depth1 = [c for c in cands if "double_buffer" not in c.extra]
    assert len(depth1) == 4
    keys = {(c.mode, c.extra["vr_dma"]) for c in cands}
    assert keys == {
        ("sv", "intra"), ("sv", "inter"),
        ("sv_lookup", "intra"), ("sv_lookup", "inter"),
    }


if __name__ == "__main__":
    test_vr_tiling_is_shape_derived()
    test_move_cost_traces_to_target_move_cycles()
    test_move_cost_zero_without_vr_dma_key()
    test_gemv_move_cost_ties_across_vr_dma()
    test_argmin_keeps_intra_sv_lookup_for_gemv()
    test_argmin_flips_to_inter_for_fused_ffn()
    test_enumerator_emits_four_distinct_candidates()
    print("ALL PASSED")
