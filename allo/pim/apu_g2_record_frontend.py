# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed Gemini-II program records as frontends of the shared pipeline.

``APUG2TypedProgram`` (exact-width ``add``/``mul``/``div``/``block_sum``/``dot``)
and ``APUG2ComposedContractionProgram`` (composed dots with an identity,
pair-affine or accumulate epilogue) are program records without source.
Since release round 15 they are *frontends*: :func:`normalize_apu_g2_record`
states the record's semantics in the shared program contract (a synthetic
``MatchTrace`` whose matches carry the same structural facts the matcher
stamps, and the `ProgramFacts` derived for it) and the record's own
realization enters the common candidate pipeline as one candidate placement
(:func:`apu_g2_record_candidates`, called by the ``apu_v2`` enumerator):
shared legality against the matcher view, the candidate contract
(`allo.spmw_candidate`), pricing through the context's
``candidate_execution_graph`` (the record's recipe graph), ranking,
materialization (:class:`APUG2RecordProgram`) and the shared artifact
(``Compiled`` wrapped by ``CompiledCallable``).  The typed VL64 recipe and
the composed-dot recipe remain target-specific realizations.

The record's call convention (``lhs, rhs, out=None`` and friends) is the
signature of :class:`RecordWorkload`, the callable stand-in the artifact is
built around; ``out`` may be omitted, in which case the realized program
allocates it and returns it in ``RunResult.extra["outputs"]``.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..spmw_codegen import RunResult
from ..spmw_match import MatchedOp, MatchTrace
from ..spmw_match_engine import coordinate_forms
from ..spmw_reduction_policy import ReductionPolicy, normalize_reduction_order
from ..spmw_semantics import (
    Contraction,
    ElementType,
    Epilogue,
    Parameter,
    ProgramFacts,
    stamp_program_facts,
)
from .apu_g2_composed_contraction import (
    APUG2ComposedContractionProgram,
    APUG2DotEpilogueMode,
)
from .apu_g2_typed_program import APUG2TypedOperation, APUG2TypedProgram
from .record_frontend import (
    RecordWorkload as _RecordWorkload,
    buffer_access,
    loop_axis,
    loop_tuple,
    memref_type_text,
    operand_binding,
    record_ref,
)

RECORD_TYPES = (APUG2TypedProgram, APUG2ComposedContractionProgram)


def is_apu_g2_record(workload) -> bool:
    return isinstance(workload, RECORD_TYPES)


def record_fingerprint(record) -> str:
    """Structural identity of a record: the composed program's own structural
    fingerprint, or a digest of the typed program's manifest."""

    fingerprint = getattr(record, "structural_fingerprint", None)
    if fingerprint is not None:
        return str(fingerprint)
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(record.manifest(), sort_keys=True, default=str).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# The record's call convention
# --------------------------------------------------------------------------- #


def record_parameter_names(record) -> tuple[str, ...]:
    """Input names then ``out``, in the record's documented call order."""

    if isinstance(record, APUG2TypedProgram):
        inputs = (
            ("src",)
            if record.operation is APUG2TypedOperation.BLOCK_SUM
            else ("lhs", "rhs")
        )
        return inputs + ("out",)
    inputs = ("lhs", "rhs")
    if record.epilogue.mode is APUG2DotEpilogueMode.ACCUMULATE:
        inputs = inputs + ("accumulator",)
    return inputs + ("out",)


def record_parameter_shapes(record) -> dict[str, tuple[int, ...]]:
    names = record_parameter_names(record)
    shapes = list(record.input_shapes) + [record.output_shape]
    return dict(zip(names, shapes))


def record_parameter_types(record) -> dict[str, Any]:
    names = record_parameter_names(record)
    types = list(record.input_types)
    if isinstance(record, APUG2ComposedContractionProgram) and (
        record.epilogue.mode is APUG2DotEpilogueMode.ACCUMULATE
    ):
        types.append(record.epilogue.accumulator_type)
    types.append(record.output_type)
    return dict(zip(names, types))


class RecordWorkload(_RecordWorkload):
    """Callable stand-in for a Gemini-II record (``out`` optional)."""

    def __init__(self, record):
        super().__init__(record, record_parameter_names(record), optional=("out",))


# --------------------------------------------------------------------------- #
# Semantic contract of a record: synthetic trace and facts
# --------------------------------------------------------------------------- #


def _element(scalar) -> ElementType:
    return ElementType(
        f"i{int(scalar.bits)}", int(scalar.bits), "int", bool(scalar.signed)
    )


_ref = record_ref
_loop = loop_axis
_loop_tuple = loop_tuple
_access = buffer_access
_binding = operand_binding
_memref_type = memref_type_text


def _policy(order, element) -> ReductionPolicy:
    return ReductionPolicy(order, element.text)


@dataclass
class _Synthesis:
    matches: list
    contractions: list
    epilogues: list
    loop_names: dict


def _typed_semantics(record, order, params, shapes, types):
    """Synthetic matches and facts of an ``APUG2TypedProgram``."""

    op = record.operation
    out_element = _element(types["out"])
    out_shape = shapes["out"]
    ordinals = {name: index for index, name in enumerate(params)}
    syn = _Synthesis([], [], [], {})
    if op is APUG2TypedOperation.DOT:
        n, k = record.shape
        loops = (_loop("n", n), _loop("k", k))
        lhs_e, rhs_e = _element(types["lhs"]), _element(types["rhs"])
        x = _binding(
            "x",
            "lhs",
            ordinals["lhs"],
            ["n", "k"],
            coordinate_forms("n", "k"),
            _memref_type((n, k), lhs_e),
        )
        y = _binding(
            "y",
            "rhs",
            ordinals["rhs"],
            ["n", "k"],
            coordinate_forms("n", "k"),
            _memref_type((n, k), rhs_e),
        )
        acc = _binding(
            "acc",
            "out",
            ordinals["out"],
            ["n"],
            coordinate_forms("n"),
            _memref_type(out_shape, out_element),
            carried=True,
        )
        match = MatchedOp(
            target_op_name="MAC",
            func_name=record.name,
            work_id=(),
            enclosing_loops=[_loop_tuple("n", n), _loop_tuple("k", k)],
            operands=[x, y, acc],
            result_memref_name="out",
            op_range=("record", "record"),
            extra={
                "store_indices": ["n"],
                "store_coords": coordinate_forms("n"),
                "result_memref_type": _memref_type(out_shape, out_element),
                "accumulator_init": "zero",
                "result_is_local": False,
                "axis_roles": {
                    "reduction": ("k",),
                    "output": ("n",),
                    "batch": (),
                    "trips": {"n": n, "k": k},
                    "matrix": "x",
                    "vector": "y",
                    "same_buffer": False,
                    "unresolved": None,
                },
                "loop_names": {"n": "n", "k": "k"},
                "reduction_policy": _policy(order, out_element),
            },
            result_value_ref=_ref(ordinals["out"]),
        )
        contraction = Contraction(
            match=match,
            function=record.name,
            target_op_name="MAC",
            loops=loops,
            reduction_vars=("k",),
            output_vars=("n",),
            batch_vars=(),
            roles_unresolved=None,
            inputs=(
                _access(
                    "x",
                    "lhs",
                    ordinals["lhs"],
                    (n, k),
                    lhs_e,
                    ("n", "k"),
                    coordinate_forms("n", "k"),
                ),
                _access(
                    "y",
                    "rhs",
                    ordinals["rhs"],
                    (n, k),
                    rhs_e,
                    ("n", "k"),
                    coordinate_forms("n", "k"),
                ),
            ),
            matrix=None,
            vector=None,
            accumulator=_access(
                "acc",
                "out",
                ordinals["out"],
                out_shape,
                out_element,
                ("n",),
                coordinate_forms("n"),
            ),
            accumulator_init="zero",
            sinks=(),
            store_coords=coordinate_forms("n"),
            guards=(),
            product_kind="mul",
            accumulate_kind="add",
            product_operation="arith.muli",
            accumulate_operation="arith.addi",
            reduction_policy=_policy(order, out_element),
            same_buffer=False,
            vector_width=1,
        )
        syn.matches.append(match)
        syn.contractions.append(contraction)
        return syn
    if op is APUG2TypedOperation.BLOCK_SUM:
        k = int(record.reduction_extent)
        blocks = math.prod(record.shape) // k
        src_e = _element(types["src"])
        src_coords = (((("b", k), ("k", 1)), 0),)
        x = _binding(
            "x",
            "src",
            ordinals["src"],
            ["b*K+k"],
            src_coords,
            _memref_type(record.shape, src_e),
        )
        acc = _binding(
            "acc",
            "out",
            ordinals["out"],
            ["b"],
            coordinate_forms("b"),
            _memref_type(out_shape, out_element),
            carried=True,
        )
        match = MatchedOp(
            target_op_name="SUM",
            func_name=record.name,
            work_id=(),
            enclosing_loops=[_loop_tuple("b", blocks), _loop_tuple("k", k)],
            operands=[x, acc],
            result_memref_name="out",
            op_range=("record", "record"),
            extra={
                "store_indices": ["b"],
                "store_coords": coordinate_forms("b"),
                "result_memref_type": _memref_type(out_shape, out_element),
                "accumulator_init": "zero",
                "result_is_local": False,
                "axis_roles": {
                    "reduction": ("k",),
                    "output": ("b",),
                    "batch": (),
                    "trips": {"b": blocks, "k": k},
                    "matrix": "x",
                    "vector": "x",
                    "same_buffer": False,
                    "unresolved": None,
                },
                "loop_names": {"b": "b", "k": "k"},
                "reduction_policy": _policy(order, out_element),
            },
            result_value_ref=_ref(ordinals["out"]),
        )
        contraction = Contraction(
            match=match,
            function=record.name,
            target_op_name="SUM",
            loops=(_loop("b", blocks), _loop("k", k)),
            reduction_vars=("k",),
            output_vars=("b",),
            batch_vars=(),
            roles_unresolved=None,
            inputs=(
                _access(
                    "x",
                    "src",
                    ordinals["src"],
                    record.shape,
                    src_e,
                    (None,) * len(record.shape),
                    src_coords,
                ),
            ),
            matrix=None,
            vector=None,
            accumulator=_access(
                "acc",
                "out",
                ordinals["out"],
                out_shape,
                out_element,
                ("b",),
                coordinate_forms("b"),
            ),
            accumulator_init="zero",
            sinks=(),
            store_coords=coordinate_forms("b"),
            guards=(),
            product_kind="identity",
            accumulate_kind="add",
            product_operation="identity",
            accumulate_operation="arith.addi",
            reduction_policy=_policy(order, out_element),
            same_buffer=False,
            vector_width=1,
        )
        syn.matches.append(match)
        syn.contractions.append(contraction)
        return syn
    # Elementwise add / mul / div over the whole shape.
    names = {
        APUG2TypedOperation.ADD: "ADD",
        APUG2TypedOperation.MUL: "MUL",
        APUG2TypedOperation.DIV: "DIV",
    }
    loop_vars = tuple(f"i{d}" for d in range(len(record.shape)))
    loops = tuple(_loop(v, e) for v, e in zip(loop_vars, record.shape))
    coords = coordinate_forms(*loop_vars)
    lhs_e, rhs_e = _element(types["lhs"]), _element(types["rhs"])
    x = _binding(
        "x",
        "lhs",
        ordinals["lhs"],
        list(loop_vars),
        coords,
        _memref_type(record.shape, lhs_e),
    )
    y = _binding(
        "y",
        "rhs",
        ordinals["rhs"],
        list(loop_vars),
        coords,
        _memref_type(record.shape, rhs_e),
    )
    match = MatchedOp(
        target_op_name=names[op],
        func_name=record.name,
        work_id=(),
        enclosing_loops=[_loop_tuple(v, e) for v, e in zip(loop_vars, record.shape)],
        operands=[x, y],
        result_memref_name="out",
        op_range=("record", "record"),
        extra={
            "store_indices": list(loop_vars),
            "store_coords": coords,
            "result_memref_type": _memref_type(out_shape, out_element),
            "result_is_local": False,
            "loop_names": {v: v for v in loop_vars},
        },
        result_value_ref=_ref(ordinals["out"]),
    )
    epilogue = Epilogue(
        match=match,
        function=record.name,
        target_op_name=names[op],
        loops=loops,
        operands=(
            _access(
                "x", "lhs", ordinals["lhs"], record.shape, lhs_e, loop_vars, coords
            ),
            _access(
                "y", "rhs", ordinals["rhs"], record.shape, rhs_e, loop_vars, coords
            ),
        ),
        constants={},
        result=_access(
            "result", "out", ordinals["out"], out_shape, out_element, loop_vars, coords
        ),
        store_coords=coords,
        guards=(),
    )
    syn.matches.append(match)
    syn.epilogues.append(epilogue)
    return syn


def _composed_semantics(record, order, params, shapes, types):
    """Synthetic matches and facts of an ``APUG2ComposedContractionProgram``:
    ``out.flat[d] = epilogue(sum_k lhs[d, k] * rhs[d, k])`` over the dot rows."""

    ordinals = {name: index for index, name in enumerate(params)}
    syn = _Synthesis([], [], [], {})
    dots, k = record.dot_shape
    lhs_e, rhs_e = _element(types["lhs"]), _element(types["rhs"])
    out_element = _element(types["out"])
    out_shape = shapes["out"]
    outputs = math.prod(out_shape)
    mode = record.epilogue.mode

    def mac(
        row_coords,
        row_label,
        acc_binding,
        acc_access,
        init,
        result_name,
        result_ordinal,
    ):
        x = _binding(
            "x",
            "lhs",
            ordinals["lhs"],
            [row_label, "k"],
            row_coords,
            _memref_type((dots, k), lhs_e),
        )
        y = _binding(
            "y",
            "rhs",
            ordinals["rhs"],
            [row_label, "k"],
            row_coords,
            _memref_type((dots, k), rhs_e),
        )
        extra = {
            "store_indices": [] if result_name is None else ["d"],
            "store_coords": coordinate_forms("d") if result_name is not None else (),
            "result_memref_type": (
                acc_binding.memref_type
                if result_name is None
                else _memref_type(out_shape, out_element)
            ),
            "accumulator_init": init,
            "result_is_local": result_name is None,
            "axis_roles": {
                "reduction": ("k",),
                "output": ("d",),
                "batch": (),
                "trips": {
                    "d": outputs if mode is APUG2DotEpilogueMode.PAIR_AFFINE else dots,
                    "k": k,
                },
                "matrix": "x",
                "vector": "y",
                "same_buffer": False,
                "unresolved": None,
            },
            "loop_names": {"d": "d", "k": "k"},
            "reduction_policy": _policy(order, out_element),
        }
        match = MatchedOp(
            target_op_name="MAC",
            func_name=record.name,
            work_id=(),
            enclosing_loops=[
                _loop_tuple(
                    "d", outputs if mode is APUG2DotEpilogueMode.PAIR_AFFINE else dots
                ),
                _loop_tuple("k", k),
            ],
            operands=[x, y, acc_binding],
            result_memref_name=(
                acc_binding.memref_name if result_name is None else result_name
            ),
            op_range=("record", "record"),
            extra=extra,
            result_value_ref=(
                acc_binding.value_ref if result_name is None else _ref(result_ordinal)
            ),
        )
        contraction = Contraction(
            match=match,
            function=record.name,
            target_op_name="MAC",
            loops=(
                _loop(
                    "d", outputs if mode is APUG2DotEpilogueMode.PAIR_AFFINE else dots
                ),
                _loop("k", k),
            ),
            reduction_vars=("k",),
            output_vars=("d",),
            batch_vars=(),
            roles_unresolved=None,
            inputs=(
                _access(
                    "x",
                    "lhs",
                    ordinals["lhs"],
                    (dots, k),
                    lhs_e,
                    (None, "k"),
                    row_coords,
                ),
                _access(
                    "y",
                    "rhs",
                    ordinals["rhs"],
                    (dots, k),
                    rhs_e,
                    (None, "k"),
                    row_coords,
                ),
            ),
            matrix=None,
            vector=None,
            accumulator=acc_access,
            accumulator_init=init,
            sinks=(),
            store_coords=extra["store_coords"] or None,
            guards=(),
            product_kind="mul",
            accumulate_kind="add",
            product_operation="arith.muli",
            accumulate_operation="arith.addi",
            reduction_policy=_policy(order, out_element),
            same_buffer=False,
            vector_width=1,
        )
        syn.matches.append(match)
        syn.contractions.append(contraction)
        return match

    if mode is APUG2DotEpilogueMode.PAIR_AFFINE:
        # Two interleaved streams (rows 2d and 2d + 1) combined by AXPBY.
        locals_ = []
        for term in range(len(record.epilogue.coefficients)):
            name = f"term{term}"
            ordinal = len(params) + term
            acc_type = record.dot_type
            acc_element = _element(acc_type)
            acc_binding = _binding(
                "acc",
                name,
                ordinal,
                [],
                (),
                _memref_type((), acc_element),
                carried=True,
            )
            acc_access = _access(
                "acc", name, ordinal, (), acc_element, (), (), local=True
            )
            row_coords = (((("d", 2),), term), ((("k", 1),), 0))
            stream = mac(
                row_coords, f"2*d+{term}", acc_binding, acc_access, "zero", None, None
            )
            # Stream `term` reads the interleaved rows 2*d + term of the
            # (dots, K) operands.  The composed layout places every dot row
            # and sums each adjacent pair itself (`apu_g2_composed_layout`,
            # "pair_affine_result_rows: even logical dot rows"), so the access
            # is realized by construction; it is declared so for the shared
            # X-42 rule, which otherwise reads the coefficient 2 as a stride
            # no lowering stages (release ledger G2-16).
            why = (
                f"pair-affine stream {term} reads rows 2*d + {term} of the "
                "interleaved (dots, K) operand; the composed layout realizes "
                "the pair directly"
            )
            stream.extra["record_realized_accesses"] = {
                ("x", "lhs", 0): why,
                ("y", "rhs", 0): why,
            }
            locals_.append((name, ordinal, acc_binding, acc_element))
        alpha, beta = record.epilogue.coefficients
        operands = [
            _binding(
                "x", locals_[0][0], locals_[0][1], [], (), locals_[0][2].memref_type
            ),
            _binding(
                "y", locals_[1][0], locals_[1][1], [], (), locals_[1][2].memref_type
            ),
        ]
        match = MatchedOp(
            target_op_name="AXPBY",
            func_name=record.name,
            work_id=(),
            enclosing_loops=[_loop_tuple("d", outputs)],
            operands=operands,
            result_memref_name="out",
            op_range=("record", "record"),
            extra={
                "store_indices": ["d"],
                "store_coords": coordinate_forms("d"),
                "result_memref_type": _memref_type(out_shape, out_element),
                "result_is_local": False,
                "constants": {"alpha": alpha, "beta": beta},
                "loop_names": {"d": "d"},
            },
            result_value_ref=_ref(ordinals["out"]),
        )
        epilogue = Epilogue(
            match=match,
            function=record.name,
            target_op_name="AXPBY",
            loops=(_loop("d", outputs),),
            operands=tuple(
                _access(role, name, ordinal, (), element, (), (), local=True)
                for role, (name, ordinal, _b, element) in zip(("x", "y"), locals_)
            ),
            constants={"alpha": alpha, "beta": beta},
            result=_access(
                "result",
                "out",
                ordinals["out"],
                out_shape,
                out_element,
                ("d",),
                coordinate_forms("d"),
            ),
            store_coords=coordinate_forms("d"),
            guards=(),
        )
        syn.matches.append(match)
        syn.epilogues.append(epilogue)
        return syn
    row_coords = coordinate_forms("d", "k")
    if mode is APUG2DotEpilogueMode.ACCUMULATE:
        # out.flat[d] = accumulator.flat[d] + dot(d): a zero-initialized
        # local dot followed by an ADD with the accumulator input.
        acc_element = _element(record.dot_type)
        local_ordinal = len(params)
        acc_binding = _binding(
            "acc",
            "acc",
            local_ordinal,
            [],
            (),
            _memref_type((), acc_element),
            carried=True,
        )
        acc_access = _access(
            "acc", "acc", local_ordinal, (), acc_element, (), (), local=True
        )
        mac(row_coords, "d", acc_binding, acc_access, "zero", None, None)
        seed_element = _element(types["accumulator"])
        operands = [
            _binding(
                "x",
                "accumulator",
                ordinals["accumulator"],
                ["d"],
                coordinate_forms("d"),
                _memref_type(out_shape, seed_element),
            ),
            _binding("y", "acc", local_ordinal, [], (), _memref_type((), acc_element)),
        ]
        match = MatchedOp(
            target_op_name="ADD",
            func_name=record.name,
            work_id=(),
            enclosing_loops=[_loop_tuple("d", dots)],
            operands=operands,
            result_memref_name="out",
            op_range=("record", "record"),
            extra={
                "store_indices": ["d"],
                "store_coords": coordinate_forms("d"),
                "result_memref_type": _memref_type(out_shape, out_element),
                "result_is_local": False,
                "loop_names": {"d": "d"},
            },
            result_value_ref=_ref(ordinals["out"]),
        )
        epilogue = Epilogue(
            match=match,
            function=record.name,
            target_op_name="ADD",
            loops=(_loop("d", dots),),
            operands=(
                _access(
                    "x",
                    "accumulator",
                    ordinals["accumulator"],
                    out_shape,
                    seed_element,
                    ("d",),
                    coordinate_forms("d"),
                ),
                _access("y", "acc", local_ordinal, (), acc_element, (), (), local=True),
            ),
            constants={},
            result=_access(
                "result",
                "out",
                ordinals["out"],
                out_shape,
                out_element,
                ("d",),
                coordinate_forms("d"),
            ),
            store_coords=coordinate_forms("d"),
            guards=(),
        )
        syn.matches.append(match)
        syn.epilogues.append(epilogue)
        return syn
    acc_binding = _binding(
        "acc",
        "out",
        ordinals["out"],
        ["d"],
        coordinate_forms("d"),
        _memref_type(out_shape, out_element),
        carried=True,
    )
    acc_access = _access(
        "acc",
        "out",
        ordinals["out"],
        out_shape,
        out_element,
        ("d",),
        coordinate_forms("d"),
    )
    mac(row_coords, "d", acc_binding, acc_access, "zero", "out", ordinals["out"])
    return syn


def normalize_apu_g2_record(record, target, *, reduction_order=None):
    """State a record's semantics in the shared program contract.

    Returns an ``allo.spmw_pipeline.SourceProgram`` whose trace holds the
    record's synthetic matches (the structural facts the matcher would stamp
    for the equivalent source) and whose facts carry the record as
    ``frontend``; the ``apu_v2`` enumerator turns that into the record's
    candidate.
    """

    from ..spmw_pipeline import SourceProgram, matcher_view

    if not is_apu_g2_record(record):
        raise TypeError("normalize_apu_g2_record takes an APUG2 program record")
    if getattr(target, "name", None) != "apu_v2":
        raise ValueError(f"{type(record).__name__} requires the Gemini-II target")
    order = normalize_reduction_order(reduction_order)
    params = record_parameter_names(record)
    shapes = record_parameter_shapes(record)
    types = record_parameter_types(record)
    if isinstance(record, APUG2TypedProgram):
        syn = _typed_semantics(record, order, params, shapes, types)
    else:
        syn = _composed_semantics(record, order, params, shapes, types)
    trace = MatchTrace(
        target_name="apu_v2",
        module_name=record.name,
        matches=list(syn.matches),
        source_value_refs={index: _ref(index) for index in range(len(params))},
        coverage=None,
    )
    parameters = tuple(
        Parameter(name, index, tuple(shapes[name]), _element(types[name]), _ref(index))
        for index, name in enumerate(params)
    )
    facts = ProgramFacts(
        function=record.name,
        parameters=parameters,
        contractions=tuple(syn.contractions),
        epilogues=tuple(syn.epilogues),
        other_matches=(),
        foreign_functions=(),
        trace=trace,
        frontend=record,
    )
    stamp_program_facts(trace, facts)
    program = SourceProgram(
        workload=RecordWorkload(record),
        target=target,
        match_target=matcher_view(target),
        schedule=None,
        trace=trace,
        facts=facts,
        reduction_order=order,
        parameter_names=params,
    )
    return program


# --------------------------------------------------------------------------- #
# The record's candidate and realized program
# --------------------------------------------------------------------------- #


@dataclass
class APUG2RecordCandidate:
    """The payload of a record candidate: the record and the facts."""

    record: Any
    facts: Any

    @property
    def name(self) -> str:
        return self.record.name

    @property
    def kind(self) -> str:
        return (
            "typed_vl64"
            if isinstance(self.record, APUG2TypedProgram)
            else "composed_dot"
        )

    def decision(self) -> tuple:
        return (self.kind, record_fingerprint(self.record))

    def matches_request(self, request) -> bool:
        return isinstance(request, str) and request in (self.record.name, self.kind)

    def effects(self):
        from ..spmw_candidate import (
            CandidateCoverage,
            CandidateEffects,
            CandidateLaunch,
            CandidateTransfer,
            CandidateValueLayout,
        )

        facts = self.facts
        record = self.record
        params = facts.parameter_names
        inputs = tuple(name for name in params if name != "out")
        operations = tuple(dict.fromkeys(m.target_op_name for m in facts.trace.matches))
        layouts = []
        for contraction in facts.contractions:
            labels = contraction.loop_labels
            for access in contraction.inputs:
                layouts.append(
                    CandidateValueLayout(
                        access.parameter or access.memref_name,
                        tuple(labels[v] for v in contraction.loop_vars),
                        "mmb",
                        self.kind,
                    )
                )
            if not contraction.accumulator.local:
                layouts.append(
                    CandidateValueLayout(
                        contraction.accumulator.parameter,
                        tuple(labels[v] for v in contraction.result_vars),
                        "mmb",
                        self.kind,
                    )
                )
            for sink in contraction.sinks:
                layouts.append(
                    CandidateValueLayout(
                        sink.parameter,
                        tuple(labels[v] for v in contraction.result_vars),
                        "mmb",
                        self.kind,
                    )
                )
        for epilogue in facts.epilogues:
            labels = {loop.var: loop.label for loop in epilogue.loops}
            for access in epilogue.operands:
                if not access.local:
                    layouts.append(
                        CandidateValueLayout(
                            access.parameter,
                            tuple(labels[v] for v in epilogue.loop_vars),
                            "mmb",
                            self.kind,
                        )
                    )
            if not epilogue.result.local:
                layouts.append(
                    CandidateValueLayout(
                        epilogue.result.parameter,
                        tuple(labels[v] for v in epilogue.loop_vars),
                        "mmb",
                        self.kind,
                    )
                )
        seen = {}
        for layout in layouts:
            seen.setdefault(layout.value, layout)
        if facts.contractions:
            first = facts.contractions[0]
            trips = first.trips
            instances = 1
            for var in first.result_vars:
                instances *= int(trips[var])
            reduction = 1
            for var in first.reduction_vars:
                reduction *= int(trips[var])
            accumulator = first.accumulator_init
            reduction_range = (0, reduction)
        else:
            instances = math.prod(record.output_shape)
            accumulator = None
            reduction_range = None
        launch = CandidateLaunch(
            index=0,
            kind=self.kind,
            operations=tuple(range(len(facts.trace.matches))),
            instances=(0, instances),
            reduction=reduction_range,
            accumulator=accumulator,
            covers=(record.name, record_fingerprint(record)),
        )
        return CandidateEffects(
            coverage=CandidateCoverage(record.name, operations, inputs, ("out",)),
            value_layouts=tuple(seen.values()),
            transfers=tuple(
                [CandidateTransfer(name, "in", ("host_pack", "l1")) for name in inputs]
                + [CandidateTransfer("out", "out", ("device_readback",))]
            ),
            launches=(launch,),
            abi_inputs=inputs,
            abi_outputs=("out",),
            device_scope="one Gemini-II card, one launch of the whole carrier",
        )


def apu_g2_record_candidates(view, facts, record) -> list:
    """The record's one candidate placement (its own fixed realization)."""

    from ..spmw_autoschedule import Placement

    seg0, seg1 = view.mmb["seg0"], view.mmb["seg1"]
    registers = {}
    for match in facts.trace.matches:
        for operand in match.operands:
            if operand.memref_name is None:
                continue
            registers[operand.memref_name] = seg1 if operand.is_loop_carried else seg0
        if match.result_memref_name is not None:
            registers[match.result_memref_name] = seg1
    payload = APUG2RecordCandidate(record, facts)
    return [
        Placement(
            placements=registers,
            mode=record.name,
            extra={
                "candidate_plan": record.name,
                "candidate_decision": payload.decision(),
                "candidate_families": (payload.kind,),
                "payload": payload,
                "whole_device_launch": True,
                "launch_count": 1,
                "effects": payload.effects(),
            },
        )
    ]


def record_execution_graph(record, target, bound_cost):
    if isinstance(record, APUG2TypedProgram):
        from .apu_g2_typed_program import build_apu_g2_typed_execution_graph

        return build_apu_g2_typed_execution_graph(record, target, bound_cost)
    from .apu_g2_composed_program import build_apu_g2_composed_execution_graph

    return build_apu_g2_composed_execution_graph(record, target, bound_cost)


def _checked_array(value, name, shape, scalar_type):
    array = np.asarray(value)
    if array.dtype != scalar_type.numpy_dtype:
        raise TypeError(
            f"APUg2 operand {name!r} must have storage dtype "
            f"{scalar_type.numpy_dtype.name}"
        )
    if array.shape != tuple(shape):
        raise ValueError(
            f"APUg2 operand {name!r} must have shape {tuple(shape)}, got {array.shape}"
        )
    if np.any(array < scalar_type.minimum) or np.any(array > scalar_type.maximum):
        raise ValueError(
            f"APUg2 operand {name!r} has values outside {scalar_type.name}"
        )
    return np.ascontiguousarray(array)


class APUG2RecordProgram:
    """The realized program of a record candidate: validates the operands of
    a call against the record, runs the typed or composed runtime on the
    card, and returns the gathered output (allocating ``out`` when the
    caller omitted it)."""

    INTROSPECTION = ("record", "program", "selection", "contract")

    def __init__(self, record, facts, selection="cost"):
        self.record = record
        self.program = record  # the record is the program (one fixed plan)
        self.facts = facts
        self.selection = selection
        self.compiled = None
        self.contract = facts

    def attach(self, compiled) -> None:
        self.compiled = compiled

    def virtual_extra(self) -> dict:
        graph = None if self.compiled is None else self.compiled.execution_graph
        metadata = getattr(graph, "metadata", {}) or {}
        return {
            "outputs": {},
            "latency_unit": "device_ticks_per_pipeline_body",
            "estimate_validity": metadata.get("estimate_validity"),
            "program": self.record.manifest(),
            "recipe": metadata.get("recipe"),
        }

    def _operands(self, inputs):
        names = record_parameter_names(self.record)
        shapes = record_parameter_shapes(self.record)
        types = record_parameter_types(self.record)
        arrays = []
        for name in names[:-1]:
            if name not in inputs or inputs[name] is None:
                raise TypeError(f"{self.record.name} needs operand {name!r}")
            arrays.append(_checked_array(inputs[name], name, shapes[name], types[name]))
        out = inputs.get("out")
        if out is not None:
            if not isinstance(out, np.ndarray) or out.dtype != types["out"].numpy_dtype:
                raise TypeError(
                    f"APUg2 output must be a NumPy {types['out'].numpy_dtype.name} array"
                )
            if out.shape != tuple(shapes["out"]):
                raise ValueError(f"APUg2 output must have shape {tuple(shapes['out'])}")
            if not out.flags.writeable:
                raise ValueError("APUg2 output must be writable")
        return arrays, out

    def run(self, compiled, inputs) -> RunResult:
        if compiled.backend not in (None, "device"):
            raise ValueError(
                f"{type(self.record).__name__} supports backend=None (card) or "
                f"'virtual'; got {compiled.backend!r}"
            )
        operands, out = self._operands(inputs)
        record = self.record
        if isinstance(record, APUG2TypedProgram):
            if record.operation is APUG2TypedOperation.DIV and np.any(operands[1] == 0):
                raise ValueError("APUg2 division requires nonzero divisors")
            runtime = importlib.import_module(".apu_g2_typed_runtime", __package__)
            result = runtime.run_apu_g2_typed(record, *operands)
        else:
            from .apu_g2_composed_layout import expected_apu_g2_composed_output

            accumulator = operands[2] if len(operands) == 3 else None
            # The oracle helper performs the complete exact shape/type/range gate.
            expected_apu_g2_composed_output(
                record, operands[0], operands[1], accumulator=accumulator
            )
            runtime = importlib.import_module(".apu_g2_composed_runtime", __package__)
            result = runtime.run_apu_g2_composed(
                record, operands[0], operands[1], accumulator=accumulator
            )
        if out is not None:
            np.copyto(out, result.extra["outputs"]["out"])
        return result


__all__ = [
    "APUG2RecordCandidate",
    "APUG2RecordProgram",
    "RecordWorkload",
    "apu_g2_record_candidates",
    "is_apu_g2_record",
    "normalize_apu_g2_record",
    "record_execution_graph",
    "record_fingerprint",
    "record_parameter_names",
]
