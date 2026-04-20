"""Make the 'A/B is a LinearLayout choice' claim literal.

For each benchmark suite, construct the two LinearLayout objects and print
their basis matrices + surjectivity / conflict-freeness properties."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from allo.pim.linear_layout import LinearLayout


def aim_layouts(D: int = 16):
    """AiM dot product with M=16 queries of length D.

    no-bank: layout over (element, row) only — computation serialized per
             bank since the bank-axis is not used.
    all-bank: layout adds a 4-bit bank axis → 16 banks in parallel.
    """
    rows = D // 16
    el_bits = 4                       # 16 fp16 lanes per burst
    bank_bits = 4                     # 16 banks
    row_bits = max(0, (rows - 1).bit_length())

    A = LinearLayout(
        bases={
            "element": [(1 << i,) for i in range(el_bits)],
            "row":     [(1 << (el_bits + i),) for i in range(row_bits)],
        },
        out_dims=("offset",))
    B = LinearLayout(
        bases={
            "element": [(1 << i,) for i in range(el_bits)],
            "row":     [(1 << (el_bits + i),) for i in range(row_bits)],
            "bank":    [(1 << (el_bits + row_bits + i),)
                        for i in range(bank_bits)],
        },
        out_dims=("offset",))
    return {"no-bank": A, "all-bank": B}


def upmem_layouts(N: int = 1024):
    """Vector-add element-addressing inside one DPU.

    scalar:    layout uses only the element axis.
    tasklet16: layout adds a 4-bit tasklet axis as parallelism dim.
    """
    el_bits = (N - 1).bit_length()
    tk_bits = 4                       # log2(16)

    A = LinearLayout(
        bases={"element": [(1 << i,) for i in range(el_bits)]},
        out_dims=("offset",))
    B = LinearLayout(
        bases={
            "element": [(1 << i,) for i in range(el_bits - tk_bits)],
            "tasklet": [(1 << (el_bits - tk_bits + i),)
                        for i in range(tk_bits)],
        },
        out_dims=("offset",))
    return {"scalar": A, "tasklet16": B}


if __name__ == "__main__":
    print("=" * 70)
    print("AiM A/B linear layouts (D=16 dot product, 16 banks, 1 row)")
    print("=" * 70)
    for name, L in aim_layouts(D=16).items():
        print(f"\n-- {name} --")
        print(L.pretty())
        print(f"  surjective over its output cover : {L.is_surjective()}")
        print(f"  conflict-free along 'bank' axis  : {L.describes_conflict_free('bank')}")

    print("\n" + "=" * 70)
    print("UPMEM A/B linear layouts (N=1024 vector-add, 1 DPU)")
    print("=" * 70)
    for name, L in upmem_layouts(N=1024).items():
        print(f"\n-- {name} --")
        print(L.pretty())
        print(f"  surjective       : {L.is_surjective()}")
        if name == "tasklet16":
            print(f"  conflict-free along 'tasklet' axis : "
                  f"{L.describes_conflict_free('tasklet')}")
