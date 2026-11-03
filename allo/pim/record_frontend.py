# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the typed program-record frontends (release gate G9).

A program record (a Gemini-II ``APUG2TypedProgram``, an ``APUv1Program``)
has no source the matcher can run on.  Its frontend states the record's
semantics in the shared program contract instead: synthetic ``MatchedOp``s
carrying the structural facts the matcher stamps for the equivalent
source, and ``ProgramFacts`` whose ``frontend`` is the record.  These
helpers build those synthetic facts and the callable stand-in whose
signature the compiled artifact exposes.
"""

from __future__ import annotations

import inspect

from ..spmw_match import IRValueRef, OperandBinding
from ..spmw_semantics import BufferAccess, ElementType, LoopAxis


class RecordWorkload:
    """Callable stand-in for a program record: the signature of its call
    convention and its name.  It never executes; the realized program
    does.  ``optional`` names parameters that default to None (a Gemini-II
    record's ``out``)."""

    def __init__(self, record, names, *, optional=()):
        self.record = record
        name = getattr(record, "name", type(record).__name__)
        self.__name__ = name
        self.__qualname__ = name
        parameters = []
        for parameter in names:
            if parameter in optional:
                parameters.append(
                    inspect.Parameter(
                        parameter,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                    )
                )
            else:
                parameters.append(
                    inspect.Parameter(
                        parameter, inspect.Parameter.POSITIONAL_OR_KEYWORD
                    )
                )
        self.__signature__ = inspect.Signature(parameters=parameters)
        self.__annotations__ = {}

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.__name__} is a device program record; call the compiled "
            "artifact returned by allo.compile"
        )


def record_ref(ordinal: int) -> IRValueRef:
    """The identity of a record parameter (its ordinal in the call)."""
    return IRValueRef("record", (int(ordinal),))


def loop_axis(var: str, extent: int) -> LoopAxis:
    return LoopAxis(var=var, name=var, lower=0, upper=int(extent), step=1)


def loop_tuple(var: str, extent: int):
    return (var, "0", str(int(extent)), 1)


def buffer_access(role, name, ordinal, shape, element, dims, coords, local=False):
    return BufferAccess(
        role=role,
        memref_name=name,
        parameter=None if local else name,
        local=local,
        value_ref=record_ref(ordinal),
        shape=tuple(shape),
        element=element,
        coords=coords,
        dims=tuple(dims),
        indices=tuple(d if d is not None else "?" for d in dims),
        index_map=None,
    )


def operand_binding(role, name, ordinal, indices, coords, memref_type, carried=False):
    return OperandBinding(
        role=role,
        memref_name=name,
        indices=list(indices),
        is_loop_carried=carried,
        memref_type=memref_type,
        value_ref=record_ref(ordinal),
        index_map=None,
        coords=coords,
    )


def memref_type_text(shape, element: ElementType) -> str:
    dims = "x".join(str(int(d)) for d in shape)
    return f"memref<{dims}x{element.text}>" if dims else f"memref<{element.text}>"


__all__ = [
    "RecordWorkload",
    "buffer_access",
    "loop_axis",
    "loop_tuple",
    "memref_type_text",
    "operand_binding",
    "record_ref",
]
