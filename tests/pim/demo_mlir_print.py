"""Print the target description + cost-annotated attention kernel as MLIR-like IR."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from allo.pim import lower
from allo.pim.mlir_print import target_to_mlir, program_to_mlir
from allo.pim.backends import build_samsung
from test_self_attention import make_attention


def main():
    prog = make_attention(8, 16, 8)
    t = build_samsung()
    res = lower(prog, t)
    print("// ==== target description ====")
    print(target_to_mlir(t))
    print("\n// ==== source program with pim_target.cost attrs ====")
    print(program_to_mlir(prog, res))


if __name__ == "__main__":
    main()
