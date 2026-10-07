# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW match-result contract.

This file defines the *interface* between Step E's matcher (Q2) and
codegen (Q3). The matcher's job is to scan the workload's MLIR and
produce a list of `MatchedOp`s, each pointing at a contiguous chunk of
IR that implements one target Op (e.g. MAC). Codegen consumes the list
and emits backend-specific instructions.

Both sides import from this module. It also hosts the pattern
vocabulary (``exp``, ``select``, ...) that target ``fn`` lambdas call.
"""

import math
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------- #
# Pattern vocabulary for target ``fn`` lambdas.  The matcher recognizes a
# call by the callee name.  Bodies are scalar and use only ``math`` and
# builtins, because the cost fingerprint rejects NumPy ufuncs (ruling 028).
# --------------------------------------------------------------------- #


def exp(x):
    return math.exp(x)


def log(x):
    return math.log(x)


def sqrt(x):
    return math.sqrt(x)


def rsqrt(x):
    return 1.0 / math.sqrt(x)


def tanh(x):
    return math.tanh(x)


def erf(x):
    return math.erf(x)


def abs_(x):
    return abs(x)


def select(cond, a, b):
    return a if cond else b


@dataclass(frozen=True, order=True)
class IRValueRef:
    """Exact retained-IR identity for one matcher-visible SSA value."""

    namespace: str
    path: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("IR value namespace must be nonempty")
        if (
            not isinstance(self.path, tuple)
            or not self.path
            or any(
                not isinstance(component, int)
                or isinstance(component, bool)
                or component < 0
                for component in self.path
            )
        ):
            raise ValueError("IR value path must contain non-negative components")

    def manifest(self) -> tuple[str, tuple[int, ...]]:
        return self.namespace, self.path


@dataclass
class OperandBinding:
    """One operand of a MatchedOp, traced back to its memref-load origin.

    role:
      - position name from the Op fn signature (e.g. "x", "y", "acc"
        for `lambda x, y, acc: acc + x*y`).
    memref_name:
      - allo's frontend annotates loads/stores with `from`/`to` attrs
        (visible in the MLIR as `{from = "local_W"}`); this is that
        string. None for SSA values that don't trace back to a memref.
    indices:
      - list of MLIR SSA-value names or affine expressions used to
        index the memref. Strings, copied verbatim from `ast.unparse`-
        style printing — codegen interprets them.
    is_loop_carried:
      - True if this operand is a reduction accumulator (e.g. `acc`).
    """

    role: str
    memref_name: str | None
    indices: list[str] = field(default_factory=list)
    is_loop_carried: bool = False
    # Retained shaped type of the source memref (for example
    # ``memref<1900x2100xbf16>``).  This is part of the programming
    # abstraction: a target lowering may need the logical tensor extents to
    # materialise a compact matched loop nest as a native command stream.
    # Older/synthetic matches may leave it unset.
    memref_type: str | None = None
    # Exact source SSA/storage identity retained from MLIR def-use and call/ABI
    # edges. Diagnostic memref names and textual types never substitute for it.
    value_ref: IRValueRef | None = None
    # ``affine.load`` access-map text (None for ``memref.load``), so a family
    # can read constant coordinates such as a label column ``S[s, 8]``.
    index_map: str | None = None


@dataclass
class MatchedOp:
    """One workload IR site identified as an instance of a target op.

    target_op_name:
      - the name registered in the target spec, e.g. "MAC", "MUL", "ADD".
    func_name:
      - the MLIR func.func that contains the matched site (typically
        `gemv_<pid>_<uid>` after allo's unroll pass).
    work_id:
      - tuple of work-item coordinates extracted from func_name (e.g.
        (0, 3) for gemv_0_3); empty tuple if not unrolled.
    enclosing_loops:
      - list of (loop_var_name, lower_bound, upper_bound, step) for
        every affine.for nesting the matched site, outer→inner. Bounds
        are strings (affine map text) so codegen can re-parse if needed.
    operands:
      - list of OperandBinding, one per fn parameter, in fn signature
        order.
    result_memref_name:
      - name of the memref this op writes to (None for pure-reg results).
    op_range:
      - `(begin_op_name, end_op_name)` — string handles into the
        containing block, naming the first and last MLIR op covered by
        this match. For codegen this is just an audit trail; lowering
        should not need to reach back into the IR.
    """

    target_op_name: str
    func_name: str
    work_id: tuple[int, ...]
    enclosing_loops: list[tuple[str, str, str, int]]
    operands: list[OperandBinding]
    result_memref_name: str | None
    op_range: tuple[str, str]
    extra: dict[str, Any] = field(default_factory=dict)
    # Exact SSA/storage identity of the store destination. Synthetic or legacy
    # traces may leave it unset; cross-boundary analysis then fails closed.
    result_value_ref: IRValueRef | None = None
    # Number of consecutive innermost-loop iterations one issue of the target
    # op consumes.  It is the op's declared ``vector_width`` only when the
    # innermost loop walks some non-accumulator operand contiguously.
    vector_width: int = 1


@dataclass
class MatchTrace:
    """Top-level container — what `match_workload(target, module)` returns."""

    target_name: str
    module_name: str
    matches: list[MatchedOp] = field(default_factory=list)
    # Exact retained frontend-source identities. Keys are the module-local
    # source IDs stamped on dataflow kernel arguments (ordinary single-function
    # workloads use their ABI argument ordinal); values are the canonical IR
    # references already carried by matcher operands/results.
    source_value_refs: dict[int, IRValueRef] = field(default_factory=dict)

    def by_target_op(self, name: str) -> list[MatchedOp]:
        return [m for m in self.matches if m.target_op_name == name]

    def by_work_id(self, wid: tuple[int, ...]) -> list[MatchedOp]:
        return [m for m in self.matches if m.work_id == wid]
