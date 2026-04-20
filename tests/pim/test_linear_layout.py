"""Unit tests for pimdsl.linear_layout."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from allo.pim.linear_layout import LinearLayout


def test_identity1d_roundtrip():
    L = LinearLayout.identity1d("element", 16)
    # every input maps to itself
    for i in range(16):
        assert L.apply(element=i) == (i,), (i, L.apply(element=i))


def test_product_blockdiagonal():
    """A layout over lane+bank should place lane on low bits, bank on high."""
    L_lane = LinearLayout(bases={"lane": [(1,), (2,), (4,), (8,)]},
                          out_dims=("offset",))
    L_bank = LinearLayout(bases={"bank": [(16,), (32,), (64,), (128,)]},
                          out_dims=("offset",))
    # union (same out dim) - simulated by merging bases
    L = LinearLayout(bases={**L_lane.bases, **L_bank.bases},
                     out_dims=("offset",))
    assert L.apply(lane=5, bank=3) == (5 ^ (16 | 32),)    # = 5 ^ 48 = 53


def test_compose_matches_matrix_product():
    # L1: element -> offset  (identity on 4 bits)
    L1 = LinearLayout.identity1d("element", 16)
    # L2: offset -> addr     (reverse bit order)
    L2 = LinearLayout(bases={"offset": [(8,), (4,), (2,), (1,)]},
                      out_dims=("addr",))
    L = L1.compose(L2)
    # L(element=i) = reverse-bits of i  (actual XOR fold is reverse-bits)
    # element=1 -> L1 -> offset=1 -> L2 -> (8,)
    assert L.apply(element=1) == (8,)
    assert L.apply(element=4) == (2,)
    assert L.apply(element=15) == (15,)   # all bits set -> all reversed bits set


def test_sublayout_projects():
    """Strip away 'channel' dim from a 3-input layout."""
    L = LinearLayout(
        bases={"lane": [(1,), (2,)],
               "bank": [(4,), (8,)],
               "channel": [(16,), (32,)]},
        out_dims=("offset",))
    proj = L.sublayout(input_dims=("lane", "bank"))
    assert proj.in_dims() == ("lane", "bank")
    # projection evaluates without channel bit
    assert proj.apply(lane=3, bank=2) == (3 ^ 8,)


def test_block_interleaved_no_bank_conflict():
    """Canonical PIM layout: lane in low bits, bank in middle, channel in
    high.  Setting the bank field to a different value moves to a different
    bit-region, so no bank conflict along lane/channel axes."""
    L = LinearLayout(
        bases={"lane":    [(1,), (2,), (4,), (8,)],
               "bank":    [(16,), (32,), (64,), (128,)],
               "channel": [(256,), (512,), (1024,), (2048,)]},
        out_dims=("offset",))
    assert L.describes_conflict_free("bank")


def test_surjective_simple():
    L = LinearLayout.identity1d("x", 8)    # 3-bit input -> 3-bit output
    assert L.is_surjective()


def test_not_surjective_when_image_collapses():
    # image covers {0, 2} but out_size includes bit 0 too (we declare 4-pos
    # space by adding a no-op basis that reaches bit 1 but not bit 0).
    # Simpler: max value = 2 -> out_size = 4, but image rank = 1 -> only 2
    # of 4 positions reachable. Force this via a single basis (2,).
    L = LinearLayout(bases={"x": [(2,)]}, out_dims=("offset",))
    # out_size = next pow2 of max(2) = 4, rank = 1 -> NOT surjective
    assert not L.is_surjective()


def test_samsung_gemv_swizzle_matches_report03():
    """Report 03 §5.4 derives a 7×8 layout for Samsung GEMV weights with one
    off-diagonal entry (bank bit 0 XOR tile-parity).  Verify the XOR-basis
    form.  Dims: [grf_0..2, bank_0..3] -> offset bits (7 output bits)."""
    # Naïve layout: identity on 7 bits (grf + bank), no swizzle.
    naive = LinearLayout(
        bases={
            "grf":  [(1,), (2,), (4,)],      # grf bits 0..2 -> offset 0..2
            "bank": [(8,), (16,), (32,), (64,)],  # bank bits 0..3 -> offset 3..6
            "tile_parity": [(0,)],           # no effect
        },
        out_dims=("offset",))
    # Swizzled: bank_0 XORs with tile_parity
    swizzled = LinearLayout(
        bases={
            "grf":  [(1,), (2,), (4,)],
            "bank": [(8,), (16,), (32,), (64,)],
            # XOR bank bit 0 into the same position under tile_parity:
            # tile_parity=1 flips bank-bit-0, giving the odd/even alternation.
            "tile_parity": [(8,)],
        },
        out_dims=("offset",))
    # apply(tile_parity=0, bank=5, grf=2) -> grf bits + bank bits with no flip
    assert naive.apply(tile_parity=0, bank=5, grf=2) == (2 ^ 8 ^ 32,)
    # swizzle flips bank bit 0 when tile_parity=1
    assert swizzled.apply(tile_parity=0, bank=5, grf=2) == (2 ^ 8 ^ 32,)
    assert swizzled.apply(tile_parity=1, bank=5, grf=2) == (2 ^ 8 ^ 32 ^ 8,)
    # swizzled layout is the naive layout composed with a single XOR:
    # no bank collisions along tile_parity (each parity maps to a disjoint bank
    # for a fixed input tile).  Both are surjective over their 7-bit cover.
    assert naive.is_surjective()
    assert swizzled.is_surjective()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall linear_layout tests passed")
