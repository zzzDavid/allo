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
import importlib
import inspect
import math
import re
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


_ELEMENT_RE = re.compile(r"^memref<(?:(?:\d+x)*)(i(\d+)|[a-z]+\d*)>$")
_SHAPE_RE = re.compile(r"^memref<((?:\d+x)*)")
_CONST_MAP_RE = re.compile(r"\(\s*(-?\d+)\s*\)>?\s*$")


def _memref_shape_and_bits(text: str | None, what: str) -> tuple[tuple[int, ...], int]:
    if text is None:
        raise APUG2SourceRouteError(f"{what} has no retained memref type")
    text = text.strip()
    match = _ELEMENT_RE.match(text)
    if match is None:
        raise APUG2SourceRouteError(f"{what} has an unsupported memref type {text}")
    if match.group(2) is None:
        raise APUG2SourceRouteError(
            f"{what} has element type {match.group(1)}; only integers are lowered"
        )
    dims = _SHAPE_RE.match(text).group(1)
    shape = tuple(int(item) for item in dims.split("x") if item)
    return shape, int(match.group(2))


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


def _constant_bound(text: str) -> int | None:
    text = str(text).strip()
    try:
        return int(text)
    except ValueError:
        pass
    match = _CONST_MAP_RE.search(text)
    return None if match is None else int(match.group(1))


def _map_dims(index_map: str | None, indices, what: str) -> tuple[str, ...]:
    """Loop variable of every accessed dimension, through an affine map."""

    indices = list(indices)
    if index_map is None:
        return tuple(indices)
    text = str(index_map).strip()
    match = re.match(r"^affine_map<\((.*?)\)(?:\[.*?\])?\s*->\s*\((.*)\)>$", text)
    if match is None:
        raise APUG2SourceRouteError(f"{what} uses an unparsable access map {text}")
    dims = [item.strip() for item in match.group(1).split(",") if item.strip()]
    results = [item.strip() for item in match.group(2).split(",") if item.strip()]
    if len(dims) != len(indices):
        raise APUG2SourceRouteError(f"{what} access map does not match its indices")
    resolved = []
    for result in results:
        if result not in dims:
            raise APUG2SourceRouteError(
                f"{what} is indexed by '{result}'; every subscript must be one "
                "plain loop variable"
            )
        resolved.append(indices[dims.index(result)])
    return tuple(resolved)


def _walk_operations(operation):
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child
                yield from _walk_operations(child)


def _string_attribute(operation, name: str) -> str | None:
    if name not in operation.attributes:
        return None
    text = str(operation.attributes[name])
    return text[1:-1] if text.startswith('"') and text.endswith('"') else text


class _FunctionFacts:
    """Signedness and store maps read from the retained top function."""

    def __init__(self, schedule, parameters):
        self.function = schedule.top_func
        self.name = schedule.top_func_name
        itypes = _string_attribute(self.function, "itypes") or ""
        if len(itypes) != len(parameters):
            raise APUG2SourceRouteError(
                "the function signature carries no per-argument signedness"
            )
        self.argument_signed = {}
        for name, code in zip(parameters, itypes):
            if code not in ("s", "u"):
                raise APUG2SourceRouteError(
                    f"argument '{name}' is not an integer buffer"
                )
            self.argument_signed[name] = code == "s"
        self.local_signed: dict[str, set] = {}
        self.stores: dict[str, list] = {}
        for operation in _walk_operations(self.function):
            name = operation.operation.name
            if name in ("memref.alloc", "memref.alloca"):
                local = _string_attribute(operation, "name")
                if local is not None:
                    self.local_signed.setdefault(local, set()).add(
                        "unsigned" not in operation.attributes
                    )
            elif name in ("affine.store", "memref.store"):
                target = _string_attribute(operation, "to")
                if target is not None:
                    self.stores.setdefault(target, []).append(operation)

    def local_is_signed(self, name: str) -> bool:
        flags = self.local_signed.get(name)
        if not flags or len(flags) != 1:
            raise APUG2SourceRouteError(
                f"local accumulator '{name}' has no unique declaration"
            )
        return next(iter(flags))

    def identity_store(self, buffer: str) -> bool:
        """Every store into ``buffer`` uses an identity access map."""

        for operation in self.stores.get(buffer, ()):
            if operation.operation.name != "affine.store":
                continue
            if "map" not in operation.attributes:
                continue
            text = str(operation.attributes["map"])
            match = re.match(r"^affine_map<\((.*?)\)\s*->\s*\((.*)\)>$", text)
            if match is None or match.group(1).replace(" ", "") != match.group(
                2
            ).replace(" ", ""):
                return False
        return True


def _argument_parameter(binding, parameters, what: str) -> str:
    return _ref_parameter(getattr(binding, "value_ref", None), parameters, what)


def _ref_parameter(ref, parameters, what: str) -> str:
    if ref is not None and ref.namespace == "mlir" and len(ref.path) == 3:
        if ref.path[1] == 0 and 0 <= ref.path[2] < len(parameters):
            return parameters[ref.path[2]]
    raise APUG2SourceRouteError(f"{what} is not a function argument")


def _loop_trips(match, what: str) -> dict[str, int]:
    trips = {}
    for var, lower, upper, step in match.enclosing_loops:
        low = _constant_bound(lower)
        high = _constant_bound(upper)
        if low is None or high is None or int(step) != 1 or low != 0:
            raise APUG2SourceRouteError(
                f"{what} is nested in a loop that is not 'range(N)' with a "
                "constant N"
            )
        trips[var] = high
    return trips


def _source_loop_names(trace_matches) -> dict[str, str]:
    """Stable, name-free labels for loop variables (``l0``, ``l1``...)."""

    names: dict[str, str] = {}
    for match in trace_matches:
        for var, *_rest in match.enclosing_loops:
            names.setdefault(var, f"l{len(names)}")
    return names


def analyze_apu_g2_source(workload, schedule, trace) -> APUG2SourceContract:
    """Derive the contraction contract of one matched source function."""

    parameters = tuple(inspect.signature(workload).parameters)
    facts = _FunctionFacts(schedule, parameters)
    matches = list(trace.matches)
    if not matches:
        raise APUG2SourceRouteError("no contraction was matched")
    foreign = sorted({m.func_name for m in matches if m.func_name != facts.name})
    if foreign:
        raise APUG2SourceRouteError(
            f"matched code lives in {foreign}, not the top function "
            f"'{facts.name}' (dataflow regions and work grids are not lowered)"
        )
    macs = [m for m in matches if m.target_op_name == "MAC"]
    combines = [m for m in matches if m.target_op_name == "AXPBY"]
    if len(macs) > 2:
        raise APUG2SourceRouteError(
            f"{len(macs)} contractions were matched; at most two products "
            "(one interleaved stream pair) are lowered"
        )
    if len(combines) > 1:
        raise APUG2SourceRouteError("more than one combining epilogue was matched")

    shapes, types = [], []
    argument_types = [str(argument.type) for argument in facts.function.arguments]
    for name, text in zip(parameters, argument_types):
        shape, bits = _memref_shape_and_bits(text, f"argument '{name}'")
        shapes.append(shape)
        types.append(
            _source_int(bits, facts.argument_signed[name], f"argument '{name}'")
        )

    def parameter_type(name):
        return types[parameters.index(name)]

    def parameter_shape(name):
        return shapes[parameters.index(name)]

    combine_inputs = set()
    if combines:
        combine = combines[0]
        for operand in combine.operands:
            if operand.role in ("x", "y"):
                combine_inputs.add(operand.value_ref)

    streams = []
    for index, match in enumerate(macs):
        what = f"contraction {index}"
        if match.extra.get("guards"):
            raise APUG2SourceRouteError(f"{what} is guarded by a condition")
        trips = _loop_trips(match, what)
        roles = match.extra.get("axis_roles") or {}
        if roles.get("unresolved"):
            raise APUG2SourceRouteError(f"{what}: {roles['unresolved']}")
        reduction = tuple(roles.get("reduction", ()))
        if len(reduction) != 1:
            raise APUG2SourceRouteError(
                f"{what} reduces over {len(reduction)} loops; exactly one "
                "reduction loop is lowered"
            )
        (k,) = reduction
        loop_vars = [loop[0] for loop in match.enclosing_loops]
        accumulators = [op for op in match.operands if op.is_loop_carried]
        inputs = {op.role: op for op in match.operands if not op.is_loop_carried}
        if len(accumulators) != 1 or set(inputs) != {"x", "y"}:
            raise APUG2SourceRouteError(f"{what} is not 'acc += x * y'")
        (acc,) = accumulators

        accesses = {}
        for role in ("x", "y"):
            binding = inputs[role]
            parameter = _argument_parameter(binding, parameters, f"{what} input")
            dims = _map_dims(
                binding.index_map, binding.indices, f"{what} input '{parameter}'"
            )
            shape = parameter_shape(parameter)
            if len(dims) != len(shape):
                raise APUG2SourceRouteError(
                    f"{what} input '{parameter}' is not indexed in every dimension"
                )
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
        matrix_role = roles.get("matrix") or "x"
        lhs = accesses[matrix_role]
        rhs = accesses["y" if matrix_role == "x" else "x"]

        init = match.extra.get("accumulator_init")
        # A carried accumulator's own store is recorded as its output sink
        # (the matcher's coverage analysis); only copies to other buffers
        # count as "copying it out".
        sinks = [
            sink
            for sink in (match.extra.get("output_sinks") or ())
            if sink[0] != acc.memref_name
        ]
        store_indices = list(match.extra.get("store_indices") or ())
        if store_indices:
            # Accumulates directly into a buffer: the result coordinates are
            # its subscripts, in dimension order.
            output = _argument_parameter(acc, parameters, f"{what} accumulator")
            result_vars = _map_dims(
                acc.index_map, acc.indices, f"{what} accumulator '{output}'"
            )
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
            acc_shape, acc_bits = _memref_shape_and_bits(
                acc.memref_type, f"{what} accumulator '{local}'"
            )
            if acc_shape:
                raise APUG2SourceRouteError(
                    f"{what} accumulator '{local}' must be a scalar local"
                )
            accumulator_type = _source_int(
                acc_bits, facts.local_is_signed(local), f"accumulator '{local}'"
            )
            result_vars = tuple(var for var in loop_vars if var in set(loop_vars) - {k})
            accumulator = "zero"
            if acc.value_ref in combine_inputs:
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
                output, index_text = sinks[0]
                if output not in parameters:
                    raise APUG2SourceRouteError(
                        f"{what} result is copied to '{output}', which is not "
                        "a function argument"
                    )
                subscripts = tuple(item.strip() for item in index_text.split(","))
                if sorted(subscripts) != sorted(result_vars) or len(
                    set(subscripts)
                ) != len(subscripts):
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
        operands = {op.role: op for op in combine.operands}
        refs = [operands[role].value_ref for role in ("x", "y")]
        by_ref = {}
        for match, stream in zip(macs, streams):
            acc = next(op for op in match.operands if op.is_loop_carried)
            by_ref[acc.value_ref] = stream
        if len(streams) != 2 or any(ref not in by_ref for ref in refs):
            raise APUG2SourceRouteError(
                "the 'alpha*x + beta*y' epilogue must combine the two "
                "contraction accumulators"
            )
        if refs[0] == refs[1]:
            raise APUG2SourceRouteError("the epilogue combines one stream twice")
        ordered = (by_ref[refs[0]], by_ref[refs[1]])
        constants = combine.extra.get("constants") or {}
        try:
            coefficients = (int(constants["alpha"]), int(constants["beta"]))
        except (KeyError, TypeError, ValueError) as error:
            raise APUG2SourceRouteError(
                "the epilogue coefficients are not integer literals"
            ) from error
        combined_output = _ref_parameter(
            combine.result_value_ref, parameters, "the combined result"
        )
        loops = [loop[0] for loop in combine.enclosing_loops]
        store = tuple(combine.extra.get("store_indices") or ())
        first, second = ordered
        if (
            first.result_vars != second.result_vars
            or tuple(loops) != tuple(v for v in loops if v in first.result_vars)
            or sorted(store) != sorted(first.result_vars)
            or not facts.identity_store(combined_output)
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
        function=facts.name,
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


def _launch_programs(contract, group, chunks, repetitions):
    """Launches realizing ``group`` (stream indices) with ``chunks`` K chunks."""

    streams = [contract.streams[index] for index in group]
    first = streams[0]
    lhs_type, rhs_type = first.lhs.scalar_type, first.rhs.scalar_type
    extent = first.reduction_extent
    chunk = extent // chunks
    terms = len(group)
    coordinates = first.coordinate_count
    pair = contract.pair_affine
    carried = first.accumulator == "carried"
    tiling = APUG2DotTiling(coordinates * terms, chunk)
    if tiling.reduction_tile_count != 1:
        raise ValueError(f"reduction chunk {chunk} exceeds one launch")
    launches = []
    for tile in range(tiling.tile_count):
        row_begin, row_end = tiling.tile_bounds(tile)
        begin, end = row_begin // terms, row_end // terms
        if tiling.tile_count == 1:
            shape = first.result_extents
        else:
            shape = (end - begin,)
        if terms == 2 and not pair:
            shape = (*shape, 2)
        for index in range(chunks):
            dot_bits = lhs_type.bits + rhs_type.bits + _log2(chunk)
            if pair:
                epilogue = APUG2DotEpilogue(coefficients=contract.coefficients)
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
            launches.append(
                APUG2Launch(
                    program=program,
                    streams=tuple(group),
                    coordinates=(begin, end),
                    reduction=(index * chunk, (index + 1) * chunk),
                    accumulator=accumulator,
                    final=index == chunks - 1,
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
    """The compiled backend object of one Gemini-II source workload."""

    def __init__(self, contract, target, cost, *, backend=None, layout=None):
        if backend not in (None, "device", "virtual"):
            raise ValueError("the Gemini-II source route supports device or virtual")
        self.contract = contract
        self.target = target
        self.cost = cost
        self.backend = "device" if backend is None else backend
        self.host_moves = ()
        self._repetitions = 256
        self._forced = layout
        self._select()

    def _select(self):
        plans, rejected = enumerate_apu_g2_launch_plans(
            self.contract, repetitions=self._repetitions
        )
        ranking = rank_apu_g2_launch_plans(plans, self.target, self.cost)
        self.rejected = tuple(rejected)
        self.ranking = tuple(ranking)
        if self._forced is None:
            self.plan = ranking[0]
            self.selection = "cost"
        else:
            chosen = [plan for plan in ranking if plan.name == self._forced]
            if not chosen:
                raise ValueError(
                    f"layout={self._forced!r} names no legal plan; legal plans: "
                    f"{[plan.name for plan in ranking]}"
                )
            self.plan = chosen[0]
            self.selection = "forced"

    @property
    def repetitions(self) -> int:
        return self._repetitions

    @repetitions.setter
    def repetitions(self, value):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("repetitions must be an integer")
        if not 1 <= int(value) <= np.iinfo(np.uint32).max:
            raise ValueError("repetitions must fit a nonzero uint32")
        self._repetitions = int(value)
        self._select()

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
            "cost_model": self.cost.spec.name,
            "cost_fingerprint": self.cost.fingerprint,
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

    def run(self, **inputs) -> RunResult:
        contract = self.contract
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
        if self.backend == "virtual":
            return RunResult(
                kind="virtual",
                cycles=int(self.plan.estimate_cycles),
                stdout=(
                    f"virtual Gemini-II source route: plan {self.plan.name!r} "
                    f"priced by {self.cost.spec.name!r}; nothing executed, "
                    "outputs not written"
                ),
                backend="virtual",
                extra={
                    "cost_model": self.cost.spec.name,
                    "latency_unit": "device_ticks_per_call",
                    "estimate_validity": self.plan.estimate_validity,
                    "plan": self.manifest(),
                },
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
                "cost_model": self.cost.spec.name,
                "repetitions": self._repetitions,
                "cycles_scope": (
                    "sum over launches of round(device pipeline ticks / "
                    "repetitions); host transfers excluded"
                ),
            },
        )


def compile_apu_g2_source_workload(
    workload, target, *, cost, backend=None, host_moves=None, layout=None
):
    """``allo.compile`` route for ordinary source workloads on ``apu_v2``."""

    from ..customize import customize
    from ..spmw_match_engine import match_workload, require_match_coverage
    from .targets import build_apu_g2_contraction_patterns

    if host_moves is not None:
        raise ValueError(
            "the Gemini-II source route derives its host packing; host_moves "
            "is not accepted"
        )
    if layout is not None and not isinstance(layout, str):
        raise TypeError("layout= on Gemini-II names a launch plan (a string)")
    if not isinstance(cost, BoundCostSpec) or cost.target is not target:
        raise TypeError("the Gemini-II source route requires a target-bound cost")
    schedule = customize(workload, enable_tensor=False)
    patterns = build_apu_g2_contraction_patterns()
    trace = match_workload(patterns, schedule.module)
    try:
        require_match_coverage(trace, target=patterns, defer_unknown=True)
    except ValueError as error:
        raise APUG2SourceRouteError(
            f"the function has effects outside the contraction family ({error})"
        ) from error
    contract = analyze_apu_g2_source(workload, schedule, trace)
    program = APUG2SourceProgram(contract, target, cost, backend=backend, layout=layout)
    return schedule, trace, program


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
    "analyze_apu_g2_source",
    "apu_g2_source_oracle",
    "apu_g2_stream_dots",
    "compile_apu_g2_source_workload",
    "enumerate_apu_g2_launch_plans",
    "price_apu_g2_launch_plan",
    "rank_apu_g2_launch_plans",
]
