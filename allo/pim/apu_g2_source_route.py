# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordinary ``@allo`` integer contractions on Gemini-II (composed-dot route).

``allo.compile(source, build_apu_g2_target(), apu_g2_cost)`` reaches this
module for a plain (non-dataflow) source function.  The route

1. matches the function against the Gemini-II contraction idioms
   (:func:`~allo.pim.targets.build_apu_g2_contraction_patterns`: ``MAC`` and
   the literal-coefficient ``AXPBY`` epilogue) and requires full store
   coverage;
2. derives, from the matches alone, each contraction stream: its result
   coordinates (in output-buffer dimension order), its single reduction loop,
   each operand's access (which loop indexes which buffer dimension, i.e. any
   transposition or broadcast), the accumulator initialization (``zero`` or
   ``carried``) and the epilogue (identity, accumulate, or pair-affine);
3. checks that the source's declared integer widths make the source result
   equal to the exact integer result the device computes;
4. enumerates launch plans (fused/interleaved versus separate streams, one or
   two reduction chunks, output tiles over carriers), builds one
   :class:`APUG2ComposedContractionProgram` per launch, prices every launch's
   issued recipe with the bound cost program, and selects the cheapest legal
   plan (``layout=<plan name>`` forces one);
5. at call time packs operands by broadcast-gather from the source arrays,
   runs each launch on the card (or evaluates the cost graph for
   ``backend="virtual"``), scatters results to the output parameters, and
   checks them against an independent ``einsum`` oracle of the matched
   source contractions.

Nothing here consults benchmark names or shapes; see
``docs/release/design/apu-g2-source-route.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import importlib
import math
import string

import numpy as np

from ..perf import BoundCostSpec, ExecutionGraph
from ..spmw_codegen import RunResult
from .apu_g2_composed_contraction import (
    APUG2ComposedContractionProgram,
    APUG2DotEpilogue,
    APUG2DotEpilogueMode,
)
from .apu_g2_composed_layout import (
    gather_apu_g2_composed_output,
    interleave_apu_g2_dot_streams,
)
from .apu_g2_composed_program import build_apu_g2_composed_recipe
from .apu_g2_layout import APUG2DotTiling
from .apu_g2_recipe import apu_g2_cost_tick_model, emit_apu_g2_priced_recipe
from .apu_g2_typed_program import APUG2ScalarType

APUG2_SOURCE_SUPPORTED_FAMILY = (
    "The Gemini-II source route lowers one plain (non-dataflow) @allo function "
    "whose only effects are one or two integer contractions "
    "'acc += X[...] * Y[...]' over a single power-of-two reduction loop "
    "(2..256 per launch; 512 as two accumulated launches for identity "
    "epilogues), with operands indexed by plain loop variables, each result "
    "written to a function-argument output either by accumulating into it "
    "(zero-initialized, or carried from the caller) or through a "
    "zero-initialized local copied to it, or the two results combined as "
    "'alpha*t + beta*u' with literal coefficients; all operands share one "
    "signedness and widths of 2..24 bits. Other programs compile as "
    "APUG2TypedProgram or APUG2ComposedContractionProgram records."
)

# One launch of the composed task: zero-padded reduction blocks of at most
# 256 values (``APUG2DotTilePlan``); exact-width accumulation can chain two.
_MAX_LAUNCH_REDUCTION = 256
_MAX_REDUCTION_CHUNKS = 2


class APUG2SourceRouteError(NotImplementedError):
    """A source workload outside the Gemini-II composed-dot family."""

    def __init__(self, reason: str):
        super().__init__(
            f"Gemini-II (apu_v2) cannot lower this source workload: {reason}. "
            f"{APUG2_SOURCE_SUPPORTED_FAMILY}"
        )
        self.reason = reason


# --------------------------------------------------------------------------- #
# Structural contract derived from the matches
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class APUG2SourceIntType:
    """A source buffer or accumulator integer type (1..64 bits).

    Contraction operands must also fit a VL64 lane (2..24 bits, an
    :class:`APUG2ScalarType`); outputs and accumulators may be wider
    containers of the exact device result.
    """

    bits: int
    signed: bool

    @property
    def name(self) -> str:
        return f"{'int' if self.signed else 'uint'}{self.bits}"

    @property
    def minimum(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def maximum(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1


@dataclass(frozen=True)
class APUG2OperandAccess:
    """One contraction input: which loop variable indexes each dimension."""

    parameter: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    scalar_type: APUG2ScalarType

    def manifest(self, names: dict[str, str]) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "dims": [names[var] for var in self.dims],
            "shape": list(self.shape),
            "type": self.scalar_type.name,
        }


@dataclass(frozen=True)
class APUG2SourceStream:
    """One matched dot-product reduction ``out[r] (+)= sum_k lhs * rhs``."""

    lhs: APUG2OperandAccess
    rhs: APUG2OperandAccess
    result_vars: tuple[str, ...]
    result_extents: tuple[int, ...]
    reduction_var: str
    reduction_extent: int
    accumulator: str
    accumulator_type: APUG2SourceIntType
    output: str | None
    output_type: APUG2SourceIntType | None

    @property
    def coordinate_count(self) -> int:
        return math.prod(self.result_extents)

    @property
    def exact_dot_type(self) -> APUG2ScalarType:
        return APUG2ScalarType(
            self.lhs.scalar_type.bits
            + self.rhs.scalar_type.bits
            + _log2(self.reduction_extent),
            self.lhs.scalar_type.signed,
        )

    @property
    def packing_form(self) -> str:
        """Which host packing helper this stream's gather coincides with."""

        k = self.reduction_var
        result = self.result_vars
        lhs, rhs = self.lhs.dims, self.rhs.dims
        if len(result) == 1 and rhs == (k,) and set(lhs) == {result[0], k}:
            transposed = lhs == (k, result[0])
            return (
                "pack_apu_g2_matrix_vector_dots"
                f"(transpose={'True' if transposed else 'False'})"
            )
        if (
            len(result) >= 2
            and lhs == (*result[:-2], result[-2], k)
            and rhs == (*result[:-2], k, result[-1])
        ):
            return "pack_apu_g2_batched_gemm_dots"
        return "broadcast_gather"

    def manifest(self, names: dict[str, str]) -> dict[str, object]:
        return {
            "lhs": self.lhs.manifest(names),
            "rhs": self.rhs.manifest(names),
            "result_vars": [names[var] for var in self.result_vars],
            "result_extents": list(self.result_extents),
            "reduction_var": names[self.reduction_var],
            "reduction_extent": self.reduction_extent,
            "accumulator": self.accumulator,
            "accumulator_type": self.accumulator_type.name,
            "output": self.output,
            "output_type": (
                None if self.output_type is None else self.output_type.name
            ),
            "exact_dot_type": self.exact_dot_type.name,
            "packing_form": self.packing_form,
        }


@dataclass(frozen=True)
class APUG2SourceContract:
    """Everything the lowering needs, derived from one matched function."""

    function: str
    parameters: tuple[str, ...]
    parameter_shapes: tuple[tuple[int, ...], ...]
    parameter_types: tuple[APUG2SourceIntType, ...]
    streams: tuple[APUG2SourceStream, ...]
    coefficients: tuple[int, int] | None = None
    combined_output: str | None = None
    combined_output_type: APUG2SourceIntType | None = None
    loop_names: tuple[tuple[str, str], ...] = ()

    @property
    def pair_affine(self) -> bool:
        return self.coefficients is not None

    def parameter_type(self, name: str) -> APUG2SourceIntType:
        return self.parameter_types[self.parameters.index(name)]

    def parameter_shape(self, name: str) -> tuple[int, ...]:
        return self.parameter_shapes[self.parameters.index(name)]

    @property
    def outputs(self) -> tuple[str, ...]:
        if self.pair_affine:
            return (self.combined_output,)
        return tuple(stream.output for stream in self.streams)

    def manifest(self) -> dict[str, object]:
        names = dict(self.loop_names)
        return {
            "function": self.function,
            "parameters": [
                {"name": name, "shape": list(shape), "type": scalar.name}
                for name, shape, scalar in zip(
                    self.parameters, self.parameter_shapes, self.parameter_types
                )
            ],
            "streams": [stream.manifest(names) for stream in self.streams],
            "combine": (
                None
                if not self.pair_affine
                else {
                    "kind": "pair_affine",
                    "coefficients": list(self.coefficients),
                    "output": self.combined_output,
                    "output_type": self.combined_output_type.name,
                }
            ),
        }


def _log2(value: int) -> int:
    return int(value).bit_length() - 1


def _source_int(bits: int, signed: bool, what: str) -> APUG2SourceIntType:
    if not 1 <= bits <= 64:
        raise APUG2SourceRouteError(f"{what} is a {bits}-bit integer")
    return APUG2SourceIntType(bits, signed)


def _scalar(bits: int, signed: bool, what: str) -> APUG2ScalarType:
    if not 2 <= bits <= 24:
        raise APUG2SourceRouteError(
            f"{what} is a {bits}-bit integer; Gemini-II VL64 lanes hold 2..24 bits"
        )
    return APUG2ScalarType(bits, signed)


def _int_type(element, what: str) -> APUG2SourceIntType:
    """The source integer type of a buffer from its shared element fact."""

    if element is None:
        raise APUG2SourceRouteError(f"{what} has no retained element type")
    if not element.is_integer:
        raise APUG2SourceRouteError(
            f"{what} has element type {element.text}; only integers are lowered"
        )
    if element.signed is None:
        raise APUG2SourceRouteError(f"{what} is not an integer buffer")
    return _source_int(element.bits, element.signed, what)


def _plain_dims(access, what: str) -> tuple[str, ...]:
    """Loop variable of every accessed dimension (plain subscripts only)."""

    if access.shape is None:
        raise APUG2SourceRouteError(f"{what} has no retained memref type")
    if len(access.dims) != len(access.shape):
        raise APUG2SourceRouteError(f"{what} is not indexed in every dimension")
    if any(dim is None for dim in access.dims):
        raise APUG2SourceRouteError(
            f"{what} is indexed by an expression; every subscript must be one "
            "plain loop variable"
        )
    return tuple(access.dims)


def _source_loop_names(items) -> dict[str, str]:
    """Stable, name-free labels for loop variables (``l0``, ``l1``...)."""

    names: dict[str, str] = {}
    for item in items:
        for var in item.loop_vars:
            names.setdefault(var, f"l{len(names)}")
    return names


def analyze_apu_g2_source(program) -> APUG2SourceContract:
    """Derive the Gemini-II contraction contract from the shared facts.

    ``program`` is the :class:`allo.spmw_semantics.ProgramFacts` of the
    matched source (one maintained semantic analysis for every target).  This
    function applies the composed-dot family's legality to those facts:
    integer buffers of 1..64 bits with operands in 2..24-bit lanes, at most
    two unguarded ``acc += x * y`` reductions over ``range(N)`` loops with
    plain loop-variable subscripts, each accumulating into a function
    argument or into a local scalar copied once to an argument (or combined
    by one ``alpha * x + beta * y`` epilogue with literal coefficients), and
    results covering their output buffers.  It refines the facts; it never
    reinterprets them.
    """

    facts = program
    if facts.foreign_functions:
        raise APUG2SourceRouteError(
            f"matched code lives in {list(facts.foreign_functions)}, not the top "
            f"function '{facts.function}' (dataflow regions and work grids are not "
            "lowered)"
        )
    macs = [c for c in facts.contractions if c.target_op_name == "MAC"]
    combines = [e for e in facts.epilogues if e.target_op_name == "AXPBY"]
    extra = [c.target_op_name for c in facts.contractions if c.target_op_name != "MAC"]
    extra += [e.target_op_name for e in facts.epilogues if e.target_op_name != "AXPBY"]
    extra += [m.target_op_name for m in facts.other_matches]
    if extra:
        raise APUG2SourceRouteError(
            f"the function computes {extra}, outside the contraction family"
        )
    if not macs:
        raise APUG2SourceRouteError("no contraction was matched")
    if len(macs) > 2:
        raise APUG2SourceRouteError(
            f"{len(macs)} contractions were matched; at most two products "
            "(one interleaved stream pair) are lowered"
        )
    if len(combines) > 1:
        raise APUG2SourceRouteError("more than one combining epilogue was matched")

    parameters = facts.parameter_names
    shapes = [parameter.shape for parameter in facts.parameters]
    types = [
        _int_type(parameter.element, f"argument '{parameter.name}'")
        for parameter in facts.parameters
    ]

    def parameter_type(name):
        return types[parameters.index(name)]

    def parameter_shape(name):
        return shapes[parameters.index(name)]

    def argument_name(access, what):
        if access.local or access.parameter is None:
            raise APUG2SourceRouteError(f"{what} is not a function argument")
        return access.parameter

    combine_inputs = set()
    if combines:
        combine_inputs = {operand.value_ref for operand in combines[0].operands}
        combine_inputs.discard(None)

    streams = []
    for index, contraction in enumerate(macs):
        what = f"contraction {index}"
        if contraction.guards:
            raise APUG2SourceRouteError(f"{what} is guarded by a condition")
        for loop in contraction.loops:
            if loop.trip is None or not loop.normalized:
                raise APUG2SourceRouteError(
                    f"{what} is nested in a loop that is not 'range(N)' with a "
                    "constant N"
                )
        trips = {loop.var: loop.trip for loop in contraction.loops}
        if contraction.roles_unresolved:
            raise APUG2SourceRouteError(f"{what}: {contraction.roles_unresolved}")
        reduction = contraction.reduction_vars
        if len(reduction) != 1:
            raise APUG2SourceRouteError(
                f"{what} reduces over {len(reduction)} loops; exactly one "
                "reduction loop is lowered"
            )
        (k,) = reduction
        loop_vars = list(contraction.loop_vars)
        inputs = {access.role: access for access in contraction.inputs}
        acc = contraction.accumulator
        if set(inputs) != {"x", "y"}:
            raise APUG2SourceRouteError(f"{what} is not 'acc += x * y'")

        accesses = {}
        for role in ("x", "y"):
            access = inputs[role]
            parameter = argument_name(access, f"{what} input")
            dims = _plain_dims(access, f"{what} input '{parameter}'")
            shape = parameter_shape(parameter)
            if len(set(dims)) != len(dims) or any(v not in loop_vars for v in dims):
                raise APUG2SourceRouteError(
                    f"{what} input '{parameter}' must use distinct enclosing "
                    "loop variables as subscripts"
                )
            for var, extent in zip(dims, shape):
                if trips[var] > extent:
                    raise APUG2SourceRouteError(
                        f"{what} reads '{parameter}' out of bounds"
                    )
            source = parameter_type(parameter)
            accesses[role] = APUG2OperandAccess(
                parameter,
                dims,
                shape,
                _scalar(source.bits, source.signed, f"operand '{parameter}'"),
            )
        matrix_role = contraction.matrix.role if contraction.matrix else "x"
        lhs = accesses[matrix_role]
        rhs = accesses["y" if matrix_role == "x" else "x"]

        init = contraction.accumulator_init
        sinks = list(contraction.sinks)
        if not acc.local:
            # Accumulates directly into a buffer: the result coordinates are
            # its subscripts, in dimension order.
            output = argument_name(acc, f"{what} accumulator")
            result_vars = _plain_dims(acc, f"{what} accumulator '{output}'")
            if sinks:
                raise APUG2SourceRouteError(
                    f"{what} accumulates into '{output}' and also copies it out"
                )
            if init not in ("zero", "carried"):
                raise APUG2SourceRouteError(
                    f"{what} accumulator '{output}' is initialized as {init!r}"
                )
            accumulator_type = parameter_type(output)
            output_type = accumulator_type
            accumulator = init
        else:
            # A local accumulator: either copied to one output buffer, or
            # combined with the other stream by the AXPBY epilogue.
            if init != "zero":
                raise APUG2SourceRouteError(
                    f"{what} local accumulator is initialized as {init!r}"
                )
            local = acc.memref_name
            if acc.shape is None or acc.element is None:
                raise APUG2SourceRouteError(
                    f"{what} accumulator '{local}' has no retained memref type"
                )
            if acc.shape:
                raise APUG2SourceRouteError(
                    f"{what} accumulator '{local}' must be a scalar local"
                )
            if not acc.element.is_integer:
                raise APUG2SourceRouteError(
                    f"{what} accumulator '{local}' has element type "
                    f"{acc.element.text}; only integers are lowered"
                )
            if acc.element.signed is None:
                raise APUG2SourceRouteError(
                    f"local accumulator '{local}' has no unique declaration"
                )
            accumulator_type = _source_int(
                acc.element.bits, acc.element.signed, f"accumulator '{local}'"
            )
            result_vars = tuple(var for var in loop_vars if var != k)
            accumulator = "zero"
            if acc.value_ref is not None and acc.value_ref in combine_inputs:
                if sinks:
                    raise APUG2SourceRouteError(
                        f"{what} result is both combined and copied out"
                    )
                output, output_type = None, None
            else:
                if len(sinks) != 1:
                    raise APUG2SourceRouteError(
                        f"{what} result is copied to {len(sinks)} buffers; "
                        "exactly one output per contraction is lowered"
                    )
                sink = sinks[0]
                output = sink.parameter
                if sink.local or output is None or output not in parameters:
                    raise APUG2SourceRouteError(
                        f"{what} result is copied to '{sink.memref_name}', which "
                        "is not a function argument"
                    )
                subscripts = tuple(sink.dims)
                index_text = ", ".join(sink.indices)
                if (
                    any(dim is None for dim in subscripts)
                    or sorted(subscripts) != sorted(result_vars)
                    or len(set(subscripts)) != len(subscripts)
                ):
                    raise APUG2SourceRouteError(
                        f"{what} result is stored to '{output}[{index_text}]'; "
                        "the subscripts must be the result loop variables"
                    )
                result_vars = subscripts
                output_type = parameter_type(output)
        if k in result_vars or any(var not in loop_vars for var in result_vars):
            raise APUG2SourceRouteError(f"{what} result coordinates are not loops")
        if set(result_vars) | {k} != set(loop_vars):
            raise APUG2SourceRouteError(
                f"{what} is nested in a loop that is neither a result nor the "
                "reduction loop"
            )
        used = set(lhs.dims) | set(rhs.dims)
        unused = [var for var in (*result_vars, k) if var not in used]
        if unused:
            raise APUG2SourceRouteError(
                f"{what} has a result or reduction loop that indexes no operand"
            )
        extents = tuple(trips[var] for var in result_vars)
        if output is not None and parameter_shape(output) != extents:
            raise APUG2SourceRouteError(
                f"{what} writes {extents} elements of '{output}' "
                f"{parameter_shape(output)}; the result must cover the output"
            )
        streams.append(
            APUG2SourceStream(
                lhs=lhs,
                rhs=rhs,
                result_vars=result_vars,
                result_extents=extents,
                reduction_var=k,
                reduction_extent=trips[k],
                accumulator=accumulator,
                accumulator_type=accumulator_type,
                output=output,
                output_type=output_type,
            )
        )

    coefficients = combined_output = combined_type = None
    if combines:
        combine = combines[0]
        operands = {operand.role: operand for operand in combine.operands}
        if set(operands) != {"x", "y"}:
            raise APUG2SourceRouteError(
                "the 'alpha*x + beta*y' epilogue must combine two accumulators"
            )
        refs = [operands[role].value_ref for role in ("x", "y")]
        by_ref = {}
        for contraction, stream in zip(macs, streams):
            by_ref[contraction.accumulator.value_ref] = stream
        if len(streams) != 2 or any(ref is None or ref not in by_ref for ref in refs):
            raise APUG2SourceRouteError(
                "the 'alpha*x + beta*y' epilogue must combine the two "
                "contraction accumulators"
            )
        if refs[0] == refs[1]:
            raise APUG2SourceRouteError("the epilogue combines one stream twice")
        ordered = (by_ref[refs[0]], by_ref[refs[1]])
        constants = combine.constants
        try:
            coefficients = (int(constants["alpha"]), int(constants["beta"]))
        except (KeyError, TypeError, ValueError) as error:
            raise APUG2SourceRouteError(
                "the epilogue coefficients are not integer literals"
            ) from error
        if combine.result.local or combine.result.parameter is None:
            raise APUG2SourceRouteError(
                "the combined result is not a function argument"
            )
        combined_output = combine.result.parameter
        loops = list(combine.loop_vars)
        store = tuple(combine.result.dims)
        first, second = ordered
        if (
            first.result_vars != second.result_vars
            or tuple(loops) != tuple(v for v in loops if v in first.result_vars)
            or any(dim is None for dim in store)
            or sorted(store) != sorted(first.result_vars)
            or len(set(store)) != len(store)
        ):
            raise APUG2SourceRouteError(
                "the combined result must be stored at the contractions' "
                "result coordinates"
            )
        combined_type = parameter_type(combined_output)
        if parameter_shape(combined_output) != first.result_extents:
            raise APUG2SourceRouteError(
                "the combined result must cover its output buffer"
            )
        reordered = tuple(
            APUG2SourceStream(**{**stream.__dict__, "result_vars": store})
            for stream in ordered
        )
        streams = list(reordered)

    outputs = {stream.output for stream in streams if stream.output is not None}
    if combined_output is not None:
        outputs.add(combined_output)
    read = {
        access.parameter for stream in streams for access in (stream.lhs, stream.rhs)
    }
    if outputs & read:
        raise APUG2SourceRouteError(
            f"output(s) {sorted(outputs & read)} are also contraction operands"
        )
    if len(outputs) != (1 if combined_output is not None else len(streams)):
        raise APUG2SourceRouteError("two contractions write the same output")

    contract = APUG2SourceContract(
        function=facts.function,
        parameters=parameters,
        parameter_shapes=tuple(shapes),
        parameter_types=tuple(types),
        streams=tuple(streams),
        coefficients=coefficients,
        combined_output=combined_output,
        combined_output_type=combined_type,
        loop_names=tuple(_source_loop_names(macs + combines).items()),
    )
    _check_widths(contract)
    return contract


def _check_widths(contract: APUG2SourceContract) -> None:
    """The source's declared widths must not wrap the exact device result."""

    signed = {stream.lhs.scalar_type.signed for stream in contract.streams}
    signed |= {stream.rhs.scalar_type.signed for stream in contract.streams}
    signed |= {stream.accumulator_type.signed for stream in contract.streams}
    signed |= {
        stream.output_type.signed
        for stream in contract.streams
        if stream.output_type is not None
    }
    if contract.pair_affine:
        signed.add(contract.combined_output_type.signed)
    if len(signed) != 1:
        raise APUG2SourceRouteError(
            "operands, accumulators and outputs mix signed and unsigned types"
        )
    for index, stream in enumerate(contract.streams):
        extent = stream.reduction_extent
        if extent < 2 or extent & (extent - 1):
            raise APUG2SourceRouteError(
                f"contraction {index} reduces over {extent} values; the "
                "reduction extent must be a power of two (the device sums "
                "aligned power-of-two blocks)"
            )
        if extent > _MAX_LAUNCH_REDUCTION * _MAX_REDUCTION_CHUNKS:
            raise APUG2SourceRouteError(
                f"contraction {index} reduces over {extent} values; one launch "
                f"sums at most {_MAX_LAUNCH_REDUCTION} and exact-width "
                f"accumulation chains at most {_MAX_REDUCTION_CHUNKS} launches"
            )
        dot = stream.exact_dot_type
        if dot.bits > 24:
            raise APUG2SourceRouteError(
                f"contraction {index} needs {dot.bits}-bit sums; one MMB "
                "segment holds 24"
            )
        if stream.accumulator_type.bits < dot.bits:
            raise APUG2SourceRouteError(
                f"contraction {index} accumulates in "
                f"{stream.accumulator_type.name}, narrower than the exact "
                f"{dot.name} sum; the device computes exact sums"
            )
        if stream.output_type is not None:
            needed = dot.bits + (1 if stream.accumulator == "carried" else 0)
            if stream.output_type.bits < needed:
                raise APUG2SourceRouteError(
                    f"output '{stream.output}' is {stream.output_type.name}; "
                    f"the exact result needs {needed} bits"
                )
    if contract.pair_affine:
        first, second = contract.streams
        if (first.lhs.scalar_type, first.rhs.scalar_type) != (
            second.lhs.scalar_type,
            second.rhs.scalar_type,
        ) or first.reduction_extent != second.reduction_extent:
            raise APUG2SourceRouteError(
                "combined contractions have mixed operand types or reduction " "extents"
            )
        try:
            epilogue = APUG2DotEpilogue(coefficients=contract.coefficients)
            coefficient = epilogue.coefficient_type(signed=first.lhs.scalar_type.signed)
        except ValueError as error:
            raise APUG2SourceRouteError(str(error)) from error
        needed = first.exact_dot_type.bits + coefficient.bits + 1
        if contract.combined_output_type.bits < needed:
            raise APUG2SourceRouteError(
                f"output '{contract.combined_output}' is "
                f"{contract.combined_output_type.name}; the exact combined "
                f"result needs {needed} bits"
            )
        if first.reduction_extent > _MAX_LAUNCH_REDUCTION:
            raise APUG2SourceRouteError(
                "a combined epilogue needs the whole reduction in one launch "
                f"(at most {_MAX_LAUNCH_REDUCTION} values)"
            )
    for index, stream in enumerate(contract.streams):
        if (
            stream.accumulator == "carried"
            and stream.reduction_extent > _MAX_LAUNCH_REDUCTION
        ):
            raise APUG2SourceRouteError(
                f"contraction {index} carries its accumulator, which needs the "
                f"whole reduction in one launch (at most {_MAX_LAUNCH_REDUCTION})"
            )


# --------------------------------------------------------------------------- #
# Launch plans
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class APUG2Launch:
    """One composed-dot task: a tile of result coordinates and a K chunk."""

    program: APUG2ComposedContractionProgram
    streams: tuple[int, ...]
    coordinates: tuple[int, int]
    reduction: tuple[int, int]
    accumulator: str
    final: bool

    @property
    def layout_fingerprint(self) -> str:
        return self.program.layout_manifest()["fingerprint"]

    def manifest(self) -> dict[str, object]:
        return {
            "streams": list(self.streams),
            "coordinates": list(self.coordinates),
            "reduction": list(self.reduction),
            "accumulator": self.accumulator,
            "final": self.final,
            "program": self.program.manifest(),
            "program_fingerprint": self.program.structural_fingerprint,
            "layout_fingerprint": self.layout_fingerprint,
        }


@dataclass
class APUG2LaunchPlan:
    """A complete realization of the contract as a sequence of launches."""

    name: str
    fusion: str
    reduction_chunks: int
    launches: tuple[APUG2Launch, ...]
    estimate_cycles: int | None = None
    estimate_validity: str | None = None
    execution_graph: ExecutionGraph | None = field(default=None, repr=False)

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "fusion": self.fusion,
            "reduction_chunks": self.reduction_chunks,
            "launch_count": len(self.launches),
            "estimate_cycles": self.estimate_cycles,
            "estimate_validity": self.estimate_validity,
            "launches": [launch.manifest() for launch in self.launches],
        }


def launch_program_for(contract, group, coordinates, reduction, *, repetitions):
    """The device recipe, accumulator mode and finality that the retained
    contract implies for one launch partition: the streams ``group`` (stream
    indices), the result-coordinate tile ``coordinates`` (begin, end) and the
    reduction range ``reduction`` (begin, end).

    This is the single derivation of a launch's intrinsic recipe from the
    contract: the enumerator builds every launch with it and
    ``APUG2SourceCtx.check_realization`` re-derives each realized launch's
    recipe from its partition and refuses a launch whose recipe differs
    (reduction extent, operand and output types, output geometry, epilogue
    with its coefficients in the launch's stream order, accumulator
    chaining or final-result routing).  Returns
    ``(program, accumulator, final)``; raises ``ValueError`` when the
    partition is not one the contract admits."""

    streams = [contract.streams[index] for index in group]
    if not streams:
        raise ValueError("a launch must realize at least one stream")
    if len(set(group)) != len(group) or any(
        not 0 <= int(index) < len(contract.streams) for index in group
    ):
        raise ValueError(
            f"streams {tuple(group)} are not distinct streams of the contract"
        )
    if contract.pair_affine and sorted(group) != list(range(len(contract.streams))):
        raise ValueError(
            f"a pair-affine epilogue needs both streams in one launch, got {tuple(group)}"
        )
    first = streams[0]
    for stream in streams[1:]:
        if (
            (stream.lhs.scalar_type, stream.rhs.scalar_type)
            != (first.lhs.scalar_type, first.rhs.scalar_type)
            or stream.reduction_extent != first.reduction_extent
            or stream.result_extents != first.result_extents
            or stream.accumulator != first.accumulator
        ):
            raise ValueError(f"streams {tuple(group)} cannot share one launch")
    lhs_type, rhs_type = first.lhs.scalar_type, first.rhs.scalar_type
    extent = first.reduction_extent
    begin_k, end_k = (int(v) for v in reduction)
    chunk = end_k - begin_k
    if chunk <= 0 or extent % chunk or begin_k % chunk or end_k > extent:
        raise ValueError(
            f"reduction range {tuple(reduction)} is not a chunk of the "
            f"contract's reduction extent {extent}"
        )
    chunks = extent // chunk
    index = begin_k // chunk
    terms = len(group)
    pair = contract.pair_affine
    carried = first.accumulator == "carried"
    tiling = APUG2DotTiling(first.coordinate_count * terms, chunk)
    if tiling.reduction_tile_count != 1:
        raise ValueError(f"reduction chunk {chunk} exceeds one launch")
    begin, end = (int(v) for v in coordinates)
    tiles = {
        (row_begin // terms, row_end // terms)
        for row_begin, row_end in (
            tiling.tile_bounds(t) for t in range(tiling.tile_count)
        )
    }
    if (begin, end) not in tiles:
        raise ValueError(
            f"coordinates {(begin, end)} are not a result tile of the contract "
            f"({sorted(tiles)})"
        )
    if tiling.tile_count == 1:
        shape = first.result_extents
    else:
        shape = (end - begin,)
    if terms == 2 and not pair:
        shape = (*shape, 2)
    dot_bits = lhs_type.bits + rhs_type.bits + _log2(chunk)
    if pair:
        # The epilogue applies coefficient j to interleaved term j, and the
        # packing interleaves the operands in the launch's stream order, so
        # the coefficients are bound to the stream identities: the
        # contract's coefficient of stream s travels with s wherever the
        # launch places it.
        coefficients = tuple(contract.coefficients[int(index)] for index in group)
        epilogue = APUG2DotEpilogue(coefficients=coefficients)
        coefficient = epilogue.coefficient_type(signed=lhs_type.signed)
        out_bits = dot_bits + coefficient.bits + 1
        accumulator = "none"
    elif carried:
        epilogue = APUG2DotEpilogue(
            accumulator_type=APUG2ScalarType(dot_bits, lhs_type.signed)
        )
        out_bits = dot_bits + 1
        accumulator = "carried"
    elif index == 0:
        epilogue = APUG2DotEpilogue()
        out_bits = dot_bits
        accumulator = "none"
    else:
        epilogue = APUG2DotEpilogue(
            accumulator_type=APUG2ScalarType(dot_bits, lhs_type.signed)
        )
        out_bits = dot_bits + 1
        accumulator = "previous"
    program = APUG2ComposedContractionProgram(
        (lhs_type, rhs_type),
        APUG2ScalarType(out_bits, lhs_type.signed),
        shape,
        chunk,
        epilogue,
        repetitions=repetitions,
    )
    return program, accumulator, index == chunks - 1


def _launch_programs(contract, group, chunks, repetitions):
    """Launches realizing ``group`` (stream indices) with ``chunks`` K chunks."""

    first = contract.streams[group[0]]
    extent = first.reduction_extent
    chunk = extent // chunks
    terms = len(group)
    tiling = APUG2DotTiling(first.coordinate_count * terms, chunk)
    if tiling.reduction_tile_count != 1:
        raise ValueError(f"reduction chunk {chunk} exceeds one launch")
    launches = []
    for tile in range(tiling.tile_count):
        row_begin, row_end = tiling.tile_bounds(tile)
        coordinates = (row_begin // terms, row_end // terms)
        for index in range(chunks):
            reduction = (index * chunk, (index + 1) * chunk)
            program, accumulator, final = launch_program_for(
                contract, group, coordinates, reduction, repetitions=repetitions
            )
            launches.append(
                APUG2Launch(
                    program=program,
                    streams=tuple(group),
                    coordinates=coordinates,
                    reduction=reduction,
                    accumulator=accumulator,
                    final=final,
                )
            )
    return launches


def _fusable(contract) -> str | None:
    """None when the two streams can share one interleaved program."""

    first, second = contract.streams
    if contract.pair_affine:
        return None
    for name, a, b in (
        (
            "operand types",
            (first.lhs.scalar_type, first.rhs.scalar_type),
            (second.lhs.scalar_type, second.rhs.scalar_type),
        ),
        ("reduction extents", first.reduction_extent, second.reduction_extent),
        ("result extents", first.result_extents, second.result_extents),
        ("accumulator initialization", first.accumulator, second.accumulator),
    ):
        if a != b:
            return f"the streams differ in {name}"
    return None


def enumerate_apu_g2_launch_plans(contract, *, repetitions=256):
    """All structurally legal plans, and the rejected ones with reasons."""

    if len(contract.streams) == 1:
        fusions = [("single", [(0,)])]
    elif contract.pair_affine:
        fusions = [("fused", [(0, 1)])]
    else:
        fusions = [("separate", [(0,), (1,)])]
        if _fusable(contract) is None:
            fusions.insert(0, ("fused", [(0, 1)]))
    plans, rejected = [], []
    for fusion, groups in fusions:
        for chunks in range(1, _MAX_REDUCTION_CHUNKS + 1):
            name = f"{fusion}/k{chunks}"
            launches = []
            try:
                for group in groups:
                    streams = [contract.streams[index] for index in group]
                    extent = streams[0].reduction_extent
                    if extent // chunks < 2:
                        raise ValueError(f"K={extent} has no {chunks}-way split")
                    if chunks > 1 and (
                        contract.pair_affine
                        or any(s.accumulator == "carried" for s in streams)
                    ):
                        raise ValueError(
                            "split reductions chain an identity launch into an "
                            "accumulating one; this epilogue cannot be chained"
                        )
                    if extent // chunks > _MAX_LAUNCH_REDUCTION:
                        raise ValueError(
                            f"a {extent // chunks}-value chunk exceeds the "
                            f"{_MAX_LAUNCH_REDUCTION}-value launch capacity"
                        )
                    launches.extend(
                        _launch_programs(contract, group, chunks, repetitions)
                    )
            except ValueError as error:
                rejected.append((name, str(error)))
                continue
            plans.append(APUG2LaunchPlan(name, fusion, chunks, tuple(launches)))
    if not plans:
        raise APUG2SourceRouteError(
            "no launch plan is legal: "
            + "; ".join(f"{name}: {reason}" for name, reason in rejected)
        )
    return plans, rejected


def price_apu_g2_launch_plan(plan, target, cost: BoundCostSpec) -> ExecutionGraph:
    """Price every launch's issued recipe; launches run back to back."""

    graph = ExecutionGraph(
        f"apu_g2_source_{plan.name.replace('/', '_')}",
        metadata={
            "target": "apu_v2",
            "latency_unit": "device_ticks_per_call",
            "cost": cost.spec.name,
            "cost_fingerprint": cost.fingerprint,
            "plan": plan.name,
            "launch_count": len(plan.launches),
            "host_roundtrips_between_launches": max(0, len(plan.launches) - 1),
            "timed_scope": (
                "sum over launches of one pipeline body each; host packing, "
                "transfers and inter-launch readback are excluded"
            ),
        },
    )
    terminals: tuple[str, ...] = ()
    validity = "calibrated"
    for index, launch in enumerate(plan.launches):
        recipe = build_apu_g2_composed_recipe(launch.program)
        terminals, summary = emit_apu_g2_priced_recipe(
            graph,
            recipe,
            target,
            cost,
            prefix=f"launch{index:03d}:",
            dependencies=terminals,
        )
        if summary["validity"] != "calibrated":
            validity = "extrapolated"
    graph.metadata["estimate_validity"] = validity
    graph.metadata["tick_model_fingerprint"] = apu_g2_cost_tick_model(cost).fingerprint
    return graph


def rank_apu_g2_launch_plans(plans, target, cost):
    """Price and order plans: estimate, then fewer launches, then catalog order."""

    for plan in plans:
        graph = price_apu_g2_launch_plan(plan, target, cost)
        estimate = cost.evaluate(graph)
        plan.execution_graph = graph
        plan.estimate_cycles = int(estimate.cycles)
        plan.estimate_validity = graph.metadata["estimate_validity"]
    order = sorted(
        range(len(plans)),
        key=lambda index: (
            plans[index].estimate_cycles,
            len(plans[index].launches),
            index,
        ),
    )
    return [plans[index] for index in order]


# --------------------------------------------------------------------------- #
# Host packing, execution, and the source oracle
# --------------------------------------------------------------------------- #


def _source_operand(contract, name, value) -> np.ndarray:
    scalar = contract.parameter_type(name)
    shape = contract.parameter_shape(name)
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise TypeError(f"argument '{name}' must be an integer NumPy array")
    if array.shape != shape:
        raise ValueError(
            f"argument '{name}' must have shape {shape}, got {array.shape}"
        )
    wide = array.astype(np.int64)
    if np.any(wide < scalar.minimum) or np.any(wide > scalar.maximum):
        raise ValueError(f"argument '{name}' has values outside {scalar.name}")
    return wide


def apu_g2_stream_dots(
    stream: APUG2SourceStream, arrays
) -> tuple[np.ndarray, np.ndarray]:
    """Logical ``(coordinates, K)`` dot operands of one stream (int64).

    Each operand is sliced to its loop extents, its dimensions transposed into
    (result coordinates..., reduction) order, and broadcast along the loops it
    does not use; rows are row-major over the output buffer's dimensions.
    """

    axes = (*stream.result_vars, stream.reduction_var)
    extents = (*stream.result_extents, stream.reduction_extent)
    extent_of = dict(zip(axes, extents))

    def gather(access):
        array = arrays[access.parameter]
        array = array[tuple(slice(0, extent_of[var]) for var in access.dims)]
        present = [var for var in axes if var in access.dims]
        array = np.transpose(array, [access.dims.index(var) for var in present])
        array = array.reshape(
            [extent_of[var] if var in access.dims else 1 for var in axes]
        )
        return np.broadcast_to(array, extents).reshape(-1, stream.reduction_extent)

    return gather(stream.lhs), gather(stream.rhs)


def apu_g2_source_oracle(contract, arrays, initial) -> dict[str, np.ndarray]:
    """Exact int64 result of the matched source contractions (``einsum``)."""

    letters = {}
    for stream in contract.streams:
        for var in (
            *stream.result_vars,
            stream.reduction_var,
            *stream.lhs.dims,
            *stream.rhs.dims,
        ):
            letters.setdefault(var, string.ascii_letters[len(letters)])
    values = []
    for stream in contract.streams:
        operands = []
        subscripts = []
        for access in (stream.lhs, stream.rhs):
            extent_of = dict(
                zip(
                    (*stream.result_vars, stream.reduction_var),
                    (*stream.result_extents, stream.reduction_extent),
                )
            )
            array = arrays[access.parameter][
                tuple(slice(0, extent_of[var]) for var in access.dims)
            ]
            operands.append(array)
            subscripts.append("".join(letters[var] for var in access.dims))
        output = "".join(letters[var] for var in stream.result_vars)
        value = np.einsum(
            f"{subscripts[0]},{subscripts[1]}->{output}", *operands, dtype=np.int64
        )
        if stream.accumulator == "carried":
            value = value + initial[stream.output]
        values.append(value)
    if contract.pair_affine:
        alpha, beta = contract.coefficients
        return {contract.combined_output: alpha * values[0] + beta * values[1]}
    return {stream.output: value for stream, value in zip(contract.streams, values)}


def _launch_operands(contract, launch, dots, partial):
    begin, end = launch.coordinates
    k_begin, k_end = launch.reduction
    program = launch.program
    lhs_dtype = program.input_types[0].numpy_dtype
    rhs_dtype = program.input_types[1].numpy_dtype
    pieces = [
        (
            dots[index][0][begin:end, k_begin:k_end],
            dots[index][1][begin:end, k_begin:k_end],
        )
        for index in launch.streams
    ]
    if len(pieces) == 1:
        lhs, rhs = pieces[0]
    else:
        lhs, rhs = interleave_apu_g2_dot_streams(*pieces)
    lhs = np.ascontiguousarray(lhs, dtype=lhs_dtype)
    rhs = np.ascontiguousarray(rhs, dtype=rhs_dtype)
    accumulator = None
    if launch.accumulator != "none":
        scalar = program.epilogue.accumulator_type
        if launch.accumulator == "carried":
            columns = [
                partial[contract.streams[index].output].reshape(-1)[begin:end]
                for index in launch.streams
            ]
            if len(columns) == 1:
                logical = columns[0]
            else:
                logical = np.stack(columns, axis=1)
            if np.any(logical < scalar.minimum) or np.any(logical > scalar.maximum):
                raise ValueError(
                    "carried accumulator values must fit "
                    f"{scalar.name}, the exact dot width the device adds them at"
                )
        else:
            logical = partial[("previous", launch.streams, launch.coordinates)]
        accumulator = np.ascontiguousarray(
            np.asarray(logical).reshape(program.output_shape), dtype=scalar.numpy_dtype
        )
    return lhs, rhs, accumulator


def _scatter(contract, launch, gathered, results):
    begin, end = launch.coordinates
    values = np.asarray(gathered, dtype=np.int64)
    if contract.pair_affine:
        results[contract.combined_output][begin:end] = values.reshape(-1)
        return
    columns = values.reshape(end - begin, -1)
    for position, index in enumerate(launch.streams):
        results[contract.streams[index].output][begin:end] = columns[:, position]


class APUG2SourceProgram:
    """The realized program of one selected Gemini-II launch plan.

    Instantiated by the backend context (:mod:`allo.spmw_apu_g2`) for the
    candidate the shared ranking selected; it packs the host carriers, runs
    the launches on the card (or reports the virtual estimate) and checks
    the device result against the source oracle.  ``candidates`` are the
    priced launch plans the ranking chose among (for inspection:
    ``ranking``, ``placement_ranking``); ``selection`` is ``"cost"`` or
    ``"forced"`` (``layout=`` named the plan).
    """

    INTROSPECTION = (
        "plan",
        "contract",
        "ranking",
        "rejected",
        "selection",
        "candidates",
        "repetitions",
        "pack",
        "manifest",
    )

    def __init__(
        self,
        contract,
        target,
        plan,
        *,
        candidates=(),
        rejected=(),
        selection="cost",
        cost=None,
        backend=None,
    ):
        if backend not in (None, "device", "virtual"):
            raise ValueError("the Gemini-II source route supports device or virtual")
        self.contract = contract
        self.target = target
        self.plan = plan
        self.candidates = tuple(candidates) or (plan,)
        self.rejected = tuple(rejected)
        self.selection = selection
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.host_moves = ()
        self.compiled = None
        self._repetitions = _plan_repetitions(plan)

    def attach(self, compiled) -> None:
        self.compiled = compiled
        if compiled.cost is not None:
            self.cost = compiled.cost
        self.backend = "device" if compiled.backend is None else compiled.backend

    @property
    def ranking(self):
        """The priced candidates in estimate order (then fewer launches)."""
        priced = [plan for plan in self.candidates if plan.estimate_cycles is not None]
        return tuple(
            sorted(
                priced,
                key=lambda plan: (plan.estimate_cycles, len(plan.launches)),
            )
        )

    @property
    def repetitions(self) -> int:
        return self._repetitions

    @repetitions.setter
    def repetitions(self, value):
        """Rebuild the selected plan's launches with another timed-loop
        repetition count (the plan, its coordinates and chunks are unchanged;
        the estimate is re-priced per call)."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("repetitions must be an integer")
        if not 1 <= int(value) <= np.iinfo(np.uint32).max:
            raise ValueError("repetitions must fit a nonzero uint32")
        repetitions = int(value)
        if repetitions == self._repetitions:
            return
        # Derive, validate and price first; install only when everything
        # succeeded, so a failure leaves the previous plan, graph and
        # repetition count in place (the validated artifact).
        rebuilt = rebuild_apu_g2_launch_plan(
            self.contract, self.plan, repetitions=repetitions
        )
        graph = None
        if self.cost is not None:
            graph = price_apu_g2_launch_plan(rebuilt, self.target, self.cost)
            rebuilt.execution_graph = graph
            rebuilt.estimate_cycles = int(self.cost.evaluate(graph).cycles)
            rebuilt.estimate_validity = graph.metadata["estimate_validity"]
            # ``attach`` receives the codegen artifact itself (its ``layout`` is
            # the selected placement, or a list of them); the priced graph
            # must describe that placement's declared launches before it
            # can replace the installed one.
            layout = getattr(self.compiled, "layout", None)
            if layout is not None:
                from ..spmw_candidate import check_graph_consistency

                for placement in (
                    layout if isinstance(layout, (list, tuple)) else [layout]
                ):
                    check_graph_consistency(placement, graph)
        self._repetitions = repetitions
        self.plan = rebuilt
        if graph is not None and self.compiled is not None:
            self.compiled.execution_graph = graph

    @property
    def execution_graph(self):
        return self.plan.execution_graph

    @property
    def placement_ranking(self):
        return [
            {
                "plan": plan.name,
                "estimate_cycles": plan.estimate_cycles,
                "launches": len(plan.launches),
                "estimate_validity": plan.estimate_validity,
            }
            for plan in self.ranking
        ]

    def manifest(self) -> dict[str, object]:
        return {
            "route": "apu_g2_source_composed_dot",
            "contract": self.contract.manifest(),
            "selection": self.selection,
            "selected": self.plan.manifest(),
            "ranking": self.placement_ranking,
            "rejected": [list(item) for item in self.rejected],
            "cost_model": None if self.cost is None else self.cost.spec.name,
            "cost_fingerprint": None if self.cost is None else self.cost.fingerprint,
        }

    def virtual_extra(self) -> dict[str, object]:
        return {
            "latency_unit": "device_ticks_per_call",
            "estimate_validity": self.plan.estimate_validity,
            "plan": self.manifest(),
        }

    def _validated_inputs(self, inputs):
        contract = self.contract
        missing = [name for name in contract.parameters if name not in inputs]
        if missing:
            raise TypeError(f"missing arguments {missing}")
        outputs = set(contract.outputs)
        arrays = {}
        for name in contract.parameters:
            value = inputs[name]
            if name in outputs and not any(
                s.accumulator == "carried" and s.output == name
                for s in contract.streams
            ):
                array = np.asarray(value)
                if array.shape != contract.parameter_shape(name):
                    raise ValueError(
                        f"output '{name}' must have shape "
                        f"{contract.parameter_shape(name)}"
                    )
                continue
            arrays[name] = _source_operand(contract, name, value)
        initial = {
            stream.output: arrays[stream.output]
            for stream in contract.streams
            if stream.accumulator == "carried"
        }
        return arrays, initial

    def pack(self, **inputs):
        """Host packing of every launch that does not consume a previous one.

        Returns one ``{"lhs", "rhs", "auxiliary"}`` dict of physical
        ``(4, 65536)`` carriers per launch (``None`` for a launch whose
        accumulator is a previous launch's device result).
        """

        from .apu_g2_composed_layout import (
            pack_apu_g2_composed_auxiliary,
            pack_apu_g2_composed_operand,
        )

        arrays, initial = self._validated_inputs(inputs)
        dots = [apu_g2_stream_dots(stream, arrays) for stream in self.contract.streams]
        partial = {name: value.reshape(-1) for name, value in initial.items()}
        packed = []
        for launch in self.plan.launches:
            if launch.accumulator == "previous":
                packed.append(None)
                continue
            lhs, rhs, accumulator = _launch_operands(
                self.contract, launch, dots, partial
            )
            packed.append(
                {
                    "lhs": pack_apu_g2_composed_operand(launch.program, lhs, 0),
                    "rhs": pack_apu_g2_composed_operand(launch.program, rhs, 1),
                    "auxiliary": pack_apu_g2_composed_auxiliary(
                        launch.program, accumulator
                    ),
                }
            )
        return packed

    def run(self, compiled, inputs) -> RunResult:
        """Execute the selected plan: ``backend=None`` on the card
        (``kind="device"``); the virtual backend never reaches here."""
        self.backend = "device" if compiled.backend is None else compiled.backend
        if self.backend != "device":
            raise ValueError(
                "the Gemini-II source route supports backend=None (card) or "
                f"'virtual'; got {self.backend!r}"
            )
        contract = self.contract
        inputs = {
            name: value for name, value in inputs.items() if name in contract.parameters
        }
        arrays, initial = self._validated_inputs(inputs)
        oracle = apu_g2_source_oracle(contract, arrays, initial)
        for name in oracle:
            container = np.asarray(inputs[name])
            scalar = contract.parameter_type(name)
            kind = container.dtype.kind
            bits = container.dtype.itemsize * 8
            fits = (kind == "i" and bits >= scalar.bits + (not scalar.signed)) or (
                kind == "u" and not scalar.signed and bits >= scalar.bits
            )
            if not fits:
                raise TypeError(
                    f"output '{name}' container {container.dtype} cannot hold "
                    f"every {scalar.name} value"
                )
        return self._run_device(arrays, initial, oracle, inputs)

    def _run_device(self, arrays, initial, oracle, inputs):
        contract = self.contract
        runtime = importlib.import_module(
            ".apu_g2_composed_runtime", package=__package__
        )
        dots = [apu_g2_stream_dots(stream, arrays) for stream in contract.streams]
        results = {
            name: np.zeros(math.prod(contract.parameter_shape(name)), dtype=np.int64)
            for name in contract.outputs
        }
        partial = {name: value.reshape(-1) for name, value in initial.items()}
        launches = []
        total = 0
        stdout = []
        for index, launch in enumerate(self.plan.launches):
            lhs, rhs, accumulator = _launch_operands(contract, launch, dots, partial)
            result = runtime.run_apu_g2_composed(
                launch.program,
                lhs,
                rhs,
                accumulator=accumulator,
                expected_layout_fingerprint=launch.layout_fingerprint,
            )
            gathered = result.extra["outputs"]["out"]
            if launch.final:
                _scatter(contract, launch, gathered, results)
            else:
                partial[("previous", launch.streams, launch.coordinates)] = gathered
            total += int(result.cycles)
            stdout.append(result.stdout)
            launches.append(
                {
                    "index": index,
                    "program_fingerprint": launch.program.structural_fingerprint,
                    "layout_fingerprint": result.extra["layout_fingerprint"],
                    "packing_attestation": result.extra.get("packing_attestation"),
                    "cycles": result.cycles,
                    "raw_ticks": result.extra["raw_ticks"],
                    "final_pipeline_ticks": result.extra["final_pipeline_ticks"],
                    "host_timings_us": result.extra["host_timings_us"],
                    "artifact_sha256": result.extra["project"]["artifact_sha256"],
                }
            )
        written = {}
        for name, flat in results.items():
            value = flat.reshape(contract.parameter_shape(name))
            expected = oracle[name]
            if not np.array_equal(value, expected):
                where = tuple(int(i) for i in np.argwhere(value != expected)[0])
                raise RuntimeError(
                    f"Gemini-II source result '{name}' differs from the source "
                    f"oracle at {where}: device={int(value[where])}, "
                    f"oracle={int(expected[where])}"
                )
            written[name] = value.astype(np.asarray(inputs[name]).dtype)
        return RunResult(
            kind="device",
            cycles=total,
            stdout="\n".join(stdout),
            backend="apu_v2",
            extra={
                "outputs": written,
                "launches": launches,
                "plan": self.manifest(),
                "estimate_cycles": self.plan.estimate_cycles,
                "cost_model": None if self.cost is None else self.cost.spec.name,
                "repetitions": self._repetitions,
                "cycles_scope": (
                    "sum over launches of round(device pipeline ticks / "
                    "repetitions); host transfers excluded"
                ),
            },
        )


def _plan_repetitions(plan) -> int:
    for launch in plan.launches:
        value = getattr(launch.program, "repetitions", None)
        if value is not None:
            return int(value)
    return 256


def check_launch_recipes(contract, plan) -> None:
    """Bind every launch's intrinsic recipe to the retained contract and its
    own partition (streams, coordinate tile, reduction range): the recipe
    the contract implies for that partition is derived again
    (``launch_program_for``) and must equal the launch's in reduction
    extent, operand and output types, output geometry, epilogue and
    structural fingerprint; the accumulator mode and the final flag must be
    the derived ones; and the final launches of every stream must tile its
    result coordinates exactly once, each preceded by the partial launches
    of its tile in reduction order.  A recipe that does not match is a
    realization that costing and selection must not accept."""
    finals: dict = {}
    chains: dict = {}
    for index, launch in enumerate(plan.launches):
        try:
            expected, accumulator, final = launch_program_for(
                contract,
                tuple(launch.streams),
                tuple(launch.coordinates),
                tuple(launch.reduction),
                repetitions=getattr(launch.program, "repetitions", 256),
            )
        except ValueError as error:
            raise RuntimeError(
                f"Gemini-II launch {index} partition is not one the contract admits: {error}"
            ) from error
        program = launch.program
        fields = {
            "reduction_extent": (program.reduction_extent, expected.reduction_extent),
            "input_types": (tuple(program.input_types), tuple(expected.input_types)),
            "output_type": (program.output_type, expected.output_type),
            "output_shape": (tuple(program.output_shape), tuple(expected.output_shape)),
            "epilogue": (program.epilogue, expected.epilogue),
            "structural_fingerprint": (
                program.structural_fingerprint,
                expected.structural_fingerprint,
            ),
            "accumulator": (launch.accumulator, accumulator),
            "final": (bool(launch.final), bool(final)),
        }
        wrong = {name: pair for name, pair in fields.items() if pair[0] != pair[1]}
        if wrong:
            raise RuntimeError(
                f"Gemini-II launch {index} (streams {tuple(launch.streams)}, coordinates "
                f"{tuple(launch.coordinates)}, reduction {tuple(launch.reduction)}) carries "
                "a recipe the contract does not imply for that partition: "
                + "; ".join(
                    f"{name} realized {a!r} vs derived {b!r}"
                    for name, (a, b) in wrong.items()
                )
            )
        key = (tuple(launch.streams), tuple(launch.coordinates))
        chains.setdefault(key, []).append(tuple(launch.reduction))
        if launch.final:
            for stream in launch.streams:
                finals.setdefault(int(stream), []).append(tuple(launch.coordinates))
    for key, ranges in chains.items():
        expected_ranges = sorted(ranges)
        if (
            ranges != expected_ranges
            or any(ranges[i][1] != ranges[i + 1][0] for i in range(len(ranges) - 1))
            or ranges[0][0] != 0
            or ranges[-1][1] != contract.streams[key[0][0]].reduction_extent
        ):
            raise RuntimeError(
                f"Gemini-II launches of tile {key} do not chain the reduction in order "
                f"from 0 to the contract's extent: {ranges}"
            )
    for index, stream in enumerate(contract.streams):
        tiles = sorted(finals.get(index, []))
        covered = 0
        for begin, end in tiles:
            if begin != covered:
                raise RuntimeError(
                    f"Gemini-II final launches of stream {index} do not tile its result "
                    f"coordinates exactly once: {tiles}"
                )
            covered = end
        if covered != stream.coordinate_count:
            raise RuntimeError(
                f"Gemini-II final launches of stream {index} cover coordinates "
                f"[0, {covered}) of {stream.coordinate_count}: {tiles}"
            )


_RECIPE_FIELDS = (
    "reduction_extent",
    "input_types",
    "output_type",
    "output_shape",
    "epilogue",
)


def rebuild_apu_g2_launch_plan(contract, plan, *, repetitions):
    """The same selected plan with another profiling repetition count.

    Each launch's recipe is derived again from the retained contract and the
    launch's own partition (``launch_program_for``: streams, coordinate tile,
    reduction range) at the new repetition count; the enumerator is not
    consulted and no plan is looked up by name.  The rebuilt launch must
    equal the validated one in everything but ``repetitions`` (streams,
    coordinates, reduction, accumulator, final flag, reduction extent,
    operand and output types, output geometry, epilogue), and the rebuilt
    plan passes ``check_launch_recipes`` again; otherwise ``ValueError`` is
    raised and nothing is installed."""
    launches = []
    for index, launch in enumerate(plan.launches):
        program, accumulator, final = launch_program_for(
            contract,
            tuple(launch.streams),
            tuple(launch.coordinates),
            tuple(launch.reduction),
            repetitions=repetitions,
        )
        drift = {
            name: (getattr(launch.program, name), getattr(program, name))
            for name in _RECIPE_FIELDS
            if getattr(launch.program, name) != getattr(program, name)
        }
        if accumulator != launch.accumulator:
            drift["accumulator"] = (launch.accumulator, accumulator)
        if bool(final) != bool(launch.final):
            drift["final"] = (launch.final, final)
        if drift:
            raise ValueError(
                f"repetitions update would change launch {index} of plan {plan.name!r} "
                "beyond its repetition count: "
                + "; ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in drift.items())
            )
        launches.append(
            APUG2Launch(
                program=program,
                streams=tuple(launch.streams),
                coordinates=tuple(launch.coordinates),
                reduction=tuple(launch.reduction),
                accumulator=launch.accumulator,
                final=launch.final,
            )
        )
    rebuilt = APUG2LaunchPlan(
        plan.name, plan.fusion, plan.reduction_chunks, tuple(launches)
    )
    try:
        check_launch_recipes(contract, rebuilt)
    except RuntimeError as error:
        raise ValueError(f"repetitions update rejected: {error}") from error
    return rebuilt


@dataclass
class APUG2SourceCandidate:
    """The payload of one launch-plan candidate: the contract, the plan and
    the whole enumerated catalog (shared by every candidate of the kernel)."""

    contract: APUG2SourceContract
    plan: APUG2LaunchPlan
    catalog: tuple
    rejected: tuple
    facts: Any = None

    @property
    def name(self) -> str:
        return self.plan.name

    def decision(self) -> tuple:
        """Name-free physical identity: fusion, chunking and every launch's
        stream set, coordinates, reduction range and accumulator mode."""
        return (
            self.plan.fusion,
            int(self.plan.reduction_chunks),
            tuple(
                (
                    tuple(launch.streams),
                    tuple(launch.coordinates),
                    tuple(launch.reduction),
                    launch.accumulator,
                    launch.program.structural_fingerprint,
                )
                for launch in self.plan.launches
            ),
        )

    def matches_request(self, request) -> bool:
        return isinstance(request, str) and request == self.plan.name

    def effects(self):
        """Scheduling effects in the common candidate contract.

        ``facts`` (the shared `ProgramFacts`) supplies the loop labels the
        value layouts are stated over; stream ``i`` of the contract is the
        work item's contraction ``i`` (the pair-affine epilogue keeps the
        contraction order).
        """
        from ..spmw_candidate import (
            CandidateCoverage,
            CandidateEffects,
            CandidateLaunch,
            CandidateTransfer,
            CandidateValueLayout,
        )

        contract = self.contract
        labels = {}
        for contraction in getattr(self.facts, "contractions", ()):
            labels.update(contraction.loop_labels)
        inputs = tuple(
            dict.fromkeys(
                [
                    access.parameter
                    for stream in contract.streams
                    for access in (stream.lhs, stream.rhs)
                ]
                # A carried accumulator's buffer seeds the reduction: a read.
                + [
                    stream.output
                    for stream in contract.streams
                    if stream.accumulator == "carried" and stream.output is not None
                ]
            )
        )
        operations = ["MAC"] * len(contract.streams)
        if contract.pair_affine:
            operations.append("AXPBY")
        layouts = {}
        for stream in contract.streams:
            for access in (stream.lhs, stream.rhs):
                layouts.setdefault(
                    access.parameter,
                    CandidateValueLayout(
                        access.parameter,
                        tuple(labels.get(d, d) for d in access.dims),
                        "mmb",
                        stream.packing_form,
                    ),
                )
        for stream in contract.streams:
            if stream.output is not None:
                layouts.setdefault(
                    stream.output,
                    CandidateValueLayout(
                        stream.output,
                        tuple(labels.get(v, v) for v in stream.result_vars),
                        "mmb",
                        "device_readback",
                    ),
                )
        if contract.combined_output is not None:
            layouts.setdefault(
                contract.combined_output,
                CandidateValueLayout(
                    contract.combined_output,
                    tuple(labels.get(v, v) for v in contract.streams[0].result_vars),
                    "mmb",
                    "device_readback",
                ),
            )
        epilogue_op = len(contract.streams) if contract.pair_affine else None
        launches = []
        for index, launch in enumerate(self.plan.launches):
            ops = tuple(dict.fromkeys(int(s) for s in launch.streams))
            if epilogue_op is not None and launch.final:
                ops = ops + (epilogue_op,)
            launches.append(
                CandidateLaunch(
                    index=index,
                    kind="composed_dot",
                    operations=ops,
                    instances=tuple(int(v) for v in launch.coordinates),
                    reduction=tuple(int(v) for v in launch.reduction),
                    accumulator=launch.accumulator,
                    covers=(self.plan.name, tuple(launch.streams)),
                )
            )
        return CandidateEffects(
            coverage=CandidateCoverage(
                function=contract.function,
                operations=tuple(dict.fromkeys(operations)),
                inputs=inputs,
                outputs=tuple(contract.outputs),
            ),
            value_layouts=tuple(layouts.values()),
            transfers=tuple(
                [
                    CandidateTransfer(name, "in", ("host_pack", "l1"))
                    for name in inputs
                    if name not in contract.outputs
                ]
                + [
                    (
                        # A carried accumulator is packed in and read back.
                        CandidateTransfer(
                            name, "inout", ("host_pack", "l1", "device_readback")
                        )
                        if name in inputs
                        else CandidateTransfer(name, "out", ("device_readback",))
                    )
                    for name in contract.outputs
                ]
            ),
            launches=tuple(launches),
            abi_inputs=inputs,
            abi_outputs=tuple(contract.outputs),
            device_scope="one Gemini-II card, launches back to back",
        )


def apu_g2_source_candidates(target, matches) -> list:
    """``autoschedule`` enumerator for ``apu_v2``: the composed-dot launch
    plans of the matched contraction family, derived from the shared
    semantic facts (one ``Placement`` per structurally legal plan).

    ``target`` is the matcher view; its ``mmb`` segments are the storage the
    legality stage checks the operands against (inputs in ``seg0``,
    accumulators in ``seg1``).
    """
    from ..spmw_autoschedule import Placement

    facts = matches[0].extra.get("program_facts")
    if facts is None:
        # The facts are stamped on the entry function's matches only; a
        # match without them lives in a dataflow kernel (or came from a
        # hand-built trace).
        raise APUG2SourceRouteError(
            f"matched code lives in '{matches[0].func_name}', not the entry "
            "function (dataflow regions and work grids are not lowered; the "
            "shared semantic facts of the entry function are required)"
        )
    frontend = getattr(facts, "frontend", None)
    if frontend is not None:
        from .apu_g2_record_frontend import apu_g2_record_candidates, is_apu_g2_record

        if is_apu_g2_record(frontend):
            return apu_g2_record_candidates(target, facts, frontend)
    contract = analyze_apu_g2_source(facts)
    plans, rejected = enumerate_apu_g2_launch_plans(contract)
    catalog = tuple(plans)
    seg0, seg1 = target.mmb["seg0"], target.mmb["seg1"]
    registers = {}
    for contraction in facts.contractions:
        for access in contraction.inputs:
            if access.memref_name is not None:
                registers.setdefault(access.memref_name, seg0)
        if contraction.accumulator.memref_name is not None:
            registers[contraction.accumulator.memref_name] = seg1
    for epilogue in facts.epilogues:
        if epilogue.result.memref_name is not None:
            registers[epilogue.result.memref_name] = seg1
    placements = []
    for plan in plans:
        payload = APUG2SourceCandidate(contract, plan, catalog, tuple(rejected), facts)
        placements.append(
            Placement(
                placements=dict(registers),
                mode=plan.name,
                extra={
                    "candidate_plan": plan.name,
                    "candidate_decision": payload.decision(),
                    "payload": payload,
                    "whole_device_launch": True,
                    "launch_count": len(plan.launches),
                    "effects": payload.effects(),
                },
            )
        )
    return placements


__all__ = [
    "APUG2_SOURCE_SUPPORTED_FAMILY",
    "APUG2SourceIntType",
    "APUG2Launch",
    "APUG2LaunchPlan",
    "APUG2OperandAccess",
    "APUG2SourceContract",
    "APUG2SourceProgram",
    "APUG2SourceRouteError",
    "APUG2SourceStream",
    "APUG2SourceCandidate",
    "analyze_apu_g2_source",
    "apu_g2_source_candidates",
    "apu_g2_source_oracle",
    "apu_g2_stream_dots",
    "enumerate_apu_g2_launch_plans",
    "price_apu_g2_launch_plan",
    "rank_apu_g2_launch_plans",
    "rebuild_apu_g2_launch_plan",
]


def _register_route_steps():
    """Route steps the Gemini-II runtime names beyond the view's handles:
    the card's L1 the packed carriers are loaded into."""
    from ..spmw_candidate import ROUTE_VOCABULARY

    ROUTE_VOCABULARY.setdefault("apu_v2", set()).update({"l1"})


_register_route_steps()
