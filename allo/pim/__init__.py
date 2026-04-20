"""allo.pim — a Python mini-language for describing PIM targets.

Public API:

  from allo.pim import (
      Target, Memory, Op, Pattern,          # target description primitives
      SrcOp, SrcProgram,                    # source IR
      Add, Mul, Matmul, Relu, Scale, Softmax,  # typed ops (carry semantics)
      lower, execute,                       # compile + interpret
  )

Backends:
  from allo.pim.backends import build_samsung, build_aim, build_upmem

Runtime glue for real simulators:
  from allo.pim.runtime import aim_shadow, pimsim_driver, upmem_codegen
"""
from .ops import SrcOp, SrcProgram, Add, Mul, Matmul, Relu, Scale, Softmax
from .target import (Target, Memory, Op, Pattern,
                     Grid, Leaf, grid, build_from_grid)
from .lowering import lower, execute
from . import target as spmw  # alias: `spmw.grid(...)`, `spmw.Leaf(...)`.

# NOTE: the decorator surface previously lived here as ``pimdsl.tn`` but
# has moved into Allo proper; use ``from allo import target, unit,
# memory, op, stream, cost`` (or ``from allo.unit import ...``).

__all__ = [
    "Target", "Memory", "Op", "Pattern",
    "Grid", "Leaf", "grid", "build_from_grid", "spmw",
    "SrcOp", "SrcProgram",
    "Add", "Mul", "Matmul", "Relu", "Scale", "Softmax",
    "lower", "execute",
]
