"""Typed source-IR ops with shared tensor semantics.

Instead of every backend re-implementing `_compute_matmul`, `_compute_softmax`,
etc. (identical NumPy under the hood), the semantics live here *once* on the
op class, and backends only provide the lowering + emit + cost.

Usage:
    from allo.pim.ops import Matmul, Add, Softmax, Scale, Relu, Mul
    prog = SrcProgram().add(
        Matmul(shape=(M, S, D), inputs=("Q", "K"), output="scores",
               attrs={"op": "Q@K.T"})
    )

Each typed op sets its own `.kind` so existing `match=lambda s: s.kind == ...`
pattern predicates keep working without change. The interpreter calls
`src.compute(state)` automatically when a pattern doesn't supply one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SrcOp:
    """Base class. Holds name, shape, dtype, inputs/output/attrs.

    Subclasses override `compute(state)` to supply tensor semantics. Plain
    SrcOp instances (used by backward-compat code) raise when `compute` is
    called — in that case the pattern's own compute closure must run.
    """
    kind: str = ""
    shape: Tuple[int, ...] = ()
    dtype: str = "fp16"
    name: str = ""
    inputs: Tuple[str, ...] = ()
    output: str = ""
    attrs: dict = field(default_factory=dict)

    def compute(self, state: Dict) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}(kind={self.kind!r}) has no built-in "
            f"compute; either use a typed subclass (Add, Matmul, Softmax, …) "
            f"or pass `compute=...` on the pattern"
        )


# ---- typed ops ----------------------------------------------------------- #

@dataclass
class Add(SrcOp):
    def __post_init__(self):
        if not self.kind:
            self.kind = "add"

    def compute(self, state):
        a, b = self.inputs
        state[self.output] = state[a] + state[b]


@dataclass
class Mul(SrcOp):
    def __post_init__(self):
        if not self.kind:
            self.kind = "mul"

    def compute(self, state):
        a, b = self.inputs
        state[self.output] = state[a] * state[b]


@dataclass
class Matmul(SrcOp):
    """Covers gemv + matmul. `attrs["op"]` disambiguates transpositions:
       "matmul" | "Q@K.T" | "S@V" (plain-matmul is the default)."""
    def __post_init__(self):
        if not self.kind:
            self.kind = "gemv"

    def compute(self, state):
        a, b = self.inputs
        A, B = state[a], state[b]
        op = self.attrs.get("op", "matmul")
        if op == "Q@K.T":
            state[self.output] = A @ B.T
        else:
            state[self.output] = A @ B


@dataclass
class Relu(SrcOp):
    def __post_init__(self):
        if not self.kind:
            self.kind = "relu"

    def compute(self, state):
        state[self.output] = np.maximum(state[self.inputs[0]], 0)


@dataclass
class Scale(SrcOp):
    """Scalar * tensor. `attrs["scale"]` is the scalar."""
    def __post_init__(self):
        if not self.kind:
            self.kind = "scale"

    def compute(self, state):
        state[self.output] = state[self.inputs[0]] * self.attrs["scale"]


@dataclass
class Softmax(SrcOp):
    """Softmax over the last axis."""
    def __post_init__(self):
        if not self.kind:
            self.kind = "softmax"

    def compute(self, state):
        x = state[self.inputs[0]]
        shifted = x - x.max(axis=-1, keepdims=True)
        ex = np.exp(shifted)
        state[self.output] = ex / ex.sum(axis=-1, keepdims=True)


# ---- program container (unchanged from ir.py) --------------------------- #

@dataclass
class SrcProgram:
    ops: List[SrcOp] = field(default_factory=list)

    def add(self, op: SrcOp) -> "SrcProgram":
        self.ops.append(op)
        return self
