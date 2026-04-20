# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""allo.work -- decorator surface for Tenon workload description.

Companion to ``allo.unit``. ``@allo.unit`` names a level of the target
grid; ``@allo.work`` names a workload kernel. In Report 11 terms, this
is the workload-side decorator that the same SPMW vocabulary serves.

The decorator is deliberately MINIMAL: it captures the Python function
(plus user-supplied shape / dtype / mapping hints) and returns a
``Work`` handle. The actual compilation -- AST walking, SrcProgram
construction, lowering against a target -- lives in ``allo.compile``
(which is invoked by ``allo.compile(work, target=...)``). Splitting
capture from compile keeps ``@allo.work`` side-effect free at decoration
time, which mirrors how ``@allo.unit`` builds a Target but does not
lower anything by itself.

Example:

    @allo.work(shapes={"A": (1024,), "B": (1024,), "C": (1024,)},
               dtype="fp16")
    def vadd(A, B, C):
        C[:] = A + B

The decorated name (``vadd``) becomes a ``Work`` object carrying:

  * ``func``   -- the original Python function (unexecuted)
  * ``name``   -- ``func.__name__``
  * ``shapes`` -- dict[str, tuple[int, ...]] of named tensors
  * ``dtype``  -- default dtype for emitted SrcOps
  * ``mapping``-- optional work-item mapping hint (ignored for MVP)

Phase-1 scope notes
-------------------

The MVP this decorator supports is SINGLE-FUNCTION vadd-shaped kernels:
one output tensor written as a slice assignment of a combinator over
input tensors (``A + B``, ``A * B``, ``np.maximum(A, 0)``, ``A @ B``,
etc.). Multi-kernel programs, grid-fitting, and named axes are flagged
in ``allo/compile.py`` as follow-up work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


_Shape = Tuple[int, ...]


@dataclass
class Work:
    """A captured workload kernel. Produced by ``@allo.work(...)``."""
    func: Callable
    name: str
    shapes: Dict[str, _Shape] = field(default_factory=dict)
    dtype: str = "fp16"
    mapping: Optional[Union[int, List[int], Dict[str, int]]] = None

    # Lazily populated by allo.compile(work, target=...) so callers can
    # introspect the recognized SrcProgram without re-running the AST
    # walker. Not load-bearing; debug-only.
    _last_program: Any = None

    def __repr__(self) -> str:
        shp = ", ".join(f"{k}={v}" for k, v in self.shapes.items())
        return (f"Work(name={self.name!r}, shapes=[{shp}], "
                f"dtype={self.dtype!r})")


def work(*,
         shapes: Optional[Dict[str, _Shape]] = None,
         dtype: str = "fp16",
         mapping: Optional[Union[int, List[int], Dict[str, int]]] = None
         ) -> Callable[[Callable], Work]:
    """Decorator. Capture a Python kernel function as a ``Work`` handle.

    ``shapes``  : optional dict mapping each tensor argument name to its
                  shape. Required for the MVP compile path because the
                  AST walker emits ``SrcOp(shape=...)`` and has to get
                  shapes from somewhere. If omitted, the compile path
                  raises with a clear error.
    ``dtype``   : default dtype for emitted source ops ("fp16" by
                  default; matches what the five backends expect).
    ``mapping`` : optional work-item mapping hint. Unused by the MVP
                  compile path (vadd needs no grid-fitting). Preserved
                  so the future grid-fitting pass can read it.
    """

    def decorator(func: Callable) -> Work:
        name = getattr(func, "__name__", "") or ""
        if not name or name == "<lambda>":
            raise TypeError(
                "@allo.work requires a named function, not a lambda.")
        return Work(
            func=func,
            name=name,
            shapes=dict(shapes) if shapes else {},
            dtype=dtype,
            mapping=mapping,
        )

    return decorator


__all__ = ["work", "Work"]
