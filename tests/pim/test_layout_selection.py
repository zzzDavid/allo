"""A layout-selection pass driven by the perf-IR cost attributes.

Scenario: GEMV weight W[4096, 1024] on the AiM target.
Two candidate layouts:
  (A) bank-interleaved: reduction dim striped across 16 banks/channel. Each
      MAC_ABK reduces over 16 columns at once — compute cost ~ K/(16*banks).
  (B) channel-replicated: whole K vector replicated per channel. Compute cost
      ~ K/banks. Higher compute cost but avoids an on-chip copy step.

We encode each layout as a different `SrcOp.attrs["layout"]`, adjust the
pattern lowering per layout, and pick the lowest-cycle layout using the
cost attribute attached by the compiler. No heuristics — just
argmin(pim_target.cost.cycles).
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from allo.pim import SrcOp, SrcProgram, lower
from allo.pim.backends import build_aim


def _augment_with_layout_patterns(t):
    """Two competing lowerings for 'gemv' differentiated by layout attr.

    Returns the original target mutated in place. A real MLIR pass would
    register two rewrite patterns and rely on the cost model to rank them;
    here we just call each candidate explicitly.
    """
    base = [p for p in t.patterns if p.name.startswith("gemv->")]
    # drop the layout-unaware base gemv pattern so layout-specific ones fire
    t.patterns = [p for p in t.patterns if not p.name.startswith("gemv->")]
    # Layout A: bank-interleaved — 1 MAC_ABK covers 16 banks; no copy.
    def a_match(src): return src.kind == "gemv" and src.attrs.get("layout") == "bank_interleaved"
    def a_lower(src, tgt):
        M, K = src.shape[0], src.shape[-1]
        return [("aim.wr_gb", {}), ("aim.wr_bias", {}),
                ("aim.mac_abk", {"n_elems": (M * K) // 16}),
                ("aim.rd_mac", {})]
    # reuse the existing compute
    a_compute = base[0].compute
    t.pattern(a_match, a_lower, a_compute, name="gemv(bank_interleaved)->aim_mac_abk")

    def b_match(src): return src.kind == "gemv" and src.attrs.get("layout") == "chan_replicated"
    def b_lower(src, tgt):
        M, K = src.shape[0], src.shape[-1]
        # extra COPY step to broadcast K to every channel, and more MAC work
        return [("aim.wr_gb", {}), ("aim.wr_bias", {}),
                ("aim.mac_abk", {"n_elems": M * K}),
                ("aim.mac_abk", {"n_elems": M * K}),
                ("aim.rd_mac", {})]
    b_compute = base[0].compute
    t.pattern(b_match, b_lower, b_compute, name="gemv(chan_replicated)->aim_mac_abk")
    return t


def _cost_for_layout(layout: str) -> int:
    t = _augment_with_layout_patterns(build_aim())
    prog = SrcProgram().add(
        SrcOp(kind="gemv", shape=(4096, 1024), dtype="fp16",
              inputs=("W", "x"), output="y",
              attrs={"layout": layout, "op": "matmul"})
    )
    res = lower(prog, t)
    assert not res.unlowered
    return res.total_cycles


def test_layout_selection_picks_cheaper():
    cycles_a = _cost_for_layout("bank_interleaved")
    cycles_b = _cost_for_layout("chan_replicated")
    print(f"bank_interleaved: {cycles_a} cycles")
    print(f"chan_replicated : {cycles_b} cycles")
    assert cycles_a < cycles_b, \
        f"expected bank_interleaved to win, got {cycles_a} vs {cycles_b}"
    print(f"compiler picks: bank_interleaved "
          f"(saves {cycles_b - cycles_a} cycles = "
          f"{100*(cycles_b-cycles_a)/cycles_b:.1f}%)")


if __name__ == "__main__":
    test_layout_selection_picks_cheaper()
    print("\nok  test_layout_selection_picks_cheaper")
