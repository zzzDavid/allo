"""Workload-name-free semantic plans recovered from structured MLIR."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import TypeAlias

from .structured_program import (
    ArgumentExpression,
    BinaryExpression,
    CallResultExpression,
    CastExpression,
    ComparisonExpression,
    ConstantExpression,
    FunctionCall,
    LoadExpression,
    LoopIndexExpression,
    MemoryFill,
    MemoryReference,
    OrderedStore,
    SelectExpression,
    StructuredFunction,
    StructuredProgram,
    UnaryExpression,
    analyze_structured_program,
)


class StructuredAPUG2PlanError(ValueError):
    """Base error for structural APUg2 plan recognition."""


class NoStructuredAPUG2Plan(StructuredAPUG2PlanError):
    """Raised when no supported semantic plan is proven."""


class AmbiguousStructuredAPUG2Plan(StructuredAPUG2PlanError):
    """Raised when more than one semantic plan is proven."""


@dataclass(frozen=True)
class StructuredEffect:
    """One proven memory effect in total program order."""

    order: int
    operation: str
    target_kind: str
    target_position: int
    input_arguments: tuple[int, ...] = ()
    input_allocations: tuple[int, ...] = ()
    loop_extents: tuple[int, ...] = ()
    reduction_extents: tuple[int, ...] = ()
    predicates: tuple[tuple[str, bool], ...] = ()
    coefficient: int | None = None
    divisor: int | None = None


@dataclass(frozen=True)
class RankTwoUpdateGemvChainPlan:
    matrix_argument: int
    first_row_factor_argument: int
    second_row_factor_argument: int
    first_column_factor_argument: int
    second_column_factor_argument: int
    state_argument: int
    projection_argument: int
    output_argument: int
    bias_argument: int
    extent: int
    rank_update_coefficients: tuple[int, int]
    accumulator_coefficients: tuple[int, int, int, int]
    effects: tuple[StructuredEffect, ...]


@dataclass(frozen=True)
class SymmetricContractionPlan:
    symmetric_matrix_argument: int
    rhs_argument: int
    output_argument: int
    rows: int
    columns: int
    reduction_extent: int
    alpha_coefficient: int
    beta_coefficient: int
    predicate_operation: str
    branch_polarities: tuple[bool, bool]
    effects: tuple[StructuredEffect, ...]


@dataclass(frozen=True)
class UnitDiagonalTriangularContractionPlan:
    triangular_matrix_argument: int
    state_argument: int
    rows: int
    columns: int
    reduction_extent: int
    predicate_operation: str
    implicit_diagonal_coefficient: int
    post_scale_coefficient: int
    reads_original_off_diagonal_state: bool
    effects: tuple[StructuredEffect, ...]


@dataclass(frozen=True)
class CenteredGramStatisticsPlan:
    data_argument: int
    mean_argument: int
    gram_argument: int
    sample_count_allocation: int
    gram_divisor_allocation: int
    sample_extent: int
    feature_extent: int
    mean_divisor: int
    gram_divisor: int
    effects: tuple[StructuredEffect, ...]


@dataclass(frozen=True)
class NormalizedGramStatisticsPlan:
    mean_input_argument: int
    variance_input_argument: int
    center_input_argument: int
    gram_argument: int
    sample_count_allocation: int
    root_sample_count_allocation: int
    unit_allocation: int
    mean_allocation: int
    variance_centered_allocation: int
    variance_allocation: int
    standard_deviation_allocation: int
    centered_allocation: int
    denominator_allocation: int
    normalized_allocation: int
    sample_extent: int
    feature_extent: int
    mean_divisor: int
    variance_divisor: int
    root_sample_factor: int
    zero_replacement: int
    diagonal_value: int
    effects: tuple[StructuredEffect, ...]


StructuredAPUG2Plan: TypeAlias = (
    RankTwoUpdateGemvChainPlan
    | SymmetricContractionPlan
    | UnitDiagonalTriangularContractionPlan
    | CenteredGramStatisticsPlan
    | NormalizedGramStatisticsPlan
)


class _Mismatch(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Mismatch


def _integer_width(dtype: str) -> int | None:
    match = re.fullmatch(r"i(\d+)", dtype)
    return int(match.group(1)) if match is not None else None


def _is_float(dtype: str) -> bool:
    return dtype == "bf16" or dtype.startswith("f")


def _validate_cast(expression: CastExpression) -> None:
    operand = expression.operand
    source_width = _integer_width(operand.dtype)
    result_width = _integer_width(expression.dtype)
    operation = expression.operation
    if operation == "arith.extui":
        _require(
            source_width is not None
            and result_width is not None
            and source_width < result_width
            and operand.signedness == "unsigned"
            and expression.signedness == "unsigned"
        )
        return
    if operation == "arith.extsi":
        _require(
            source_width is not None
            and result_width is not None
            and source_width < result_width
            and operand.signedness == "signed"
            and expression.signedness == "signed"
        )
        return
    if operation == "arith.trunci":
        _require(
            source_width is not None
            and result_width is not None
            and source_width > result_width
        )
        return
    if operation == "arith.uitofp":
        _require(
            source_width is not None
            and operand.signedness == "unsigned"
            and _is_float(expression.dtype)
        )
        return
    if operation == "arith.fptoui":
        _require(
            _is_float(operand.dtype)
            and result_width is not None
            and expression.signedness == "unsigned"
        )
        return
    raise _Mismatch


def _normalize(expression):
    if isinstance(expression, CastExpression):
        _validate_cast(expression)
        return _normalize(expression.operand)
    if isinstance(expression, ConstantExpression):
        return ("constant", expression.value)
    if isinstance(expression, LoopIndexExpression):
        return ("loop", expression.axis_position)
    if isinstance(expression, LoadExpression):
        access = expression.access
        memory = access.memory
        _require(access.mode == "read" and memory.result_position == 0)
        return (
            "load",
            memory.source_kind,
            memory.source_position,
            memory.shape,
            memory.dtype,
            memory.signedness,
            access.signedness,
            tuple(_normalize(index) for index in access.indices),
        )
    if isinstance(expression, BinaryExpression):
        return (
            "binary",
            expression.operation,
            _normalize(expression.lhs),
            _normalize(expression.rhs),
        )
    if isinstance(expression, UnaryExpression):
        return ("unary", expression.operation, _normalize(expression.operand))
    if isinstance(expression, ComparisonExpression):
        return (
            "comparison",
            expression.operation,
            expression.predicate,
            _normalize(expression.lhs),
            _normalize(expression.rhs),
        )
    if isinstance(
        expression,
        (ArgumentExpression, CallResultExpression, SelectExpression),
    ):
        raise _Mismatch
    raise _Mismatch


def _loop(position: int):
    return ("loop", position)


def _constant(value: int):
    return ("constant", value)


def _load(
    kind: str,
    position: int,
    shape: tuple[int, ...],
    *indices,
):
    return (
        "load",
        kind,
        position,
        shape,
        "i16",
        "unsigned",
        "unsigned",
        tuple(indices),
    )


def _binary(expression, operation: str):
    _require(expression[0:2] == ("binary", operation))
    return expression[2], expression[3]


def _flatten(expression, operation: str) -> list[tuple]:
    if expression[0:2] != ("binary", operation):
        return [expression]
    return _flatten(expression[2], operation) + _flatten(expression[3], operation)


def _expect_terms(expression, operation: str, expected: tuple[tuple, ...]) -> None:
    actual = _flatten(expression, operation)
    remaining = list(actual)
    for term in expected:
        try:
            remaining.remove(term)
        except ValueError as error:
            raise _Mismatch from error
    _require(not remaining)


def _expect_product(expression, *expected) -> None:
    _expect_terms(expression, "arith.muli", tuple(expected))


def _product_coefficient(expression, *expected) -> int:
    terms = _flatten(expression, "arith.muli")
    for term in expected:
        try:
            terms.remove(term)
        except ValueError as error:
            raise _Mismatch from error
    _require(len(terms) == 1 and terms[0][0] == "constant")
    value = terms[0][1]
    _require(type(value) is int)
    return value


def _expect_u16_arguments(
    function: StructuredFunction, shapes: tuple[tuple[int, ...], ...]
) -> None:
    _require(len(function.arguments) == len(shapes))
    for position, (argument, shape) in enumerate(zip(function.arguments, shapes)):
        _require(
            argument.position == position
            and argument.shape == shape
            and argument.dtype == "i16"
            and argument.signedness == "unsigned"
        )


def _expect_u16_allocations(
    function: StructuredFunction, shapes: tuple[tuple[int, ...], ...]
) -> None:
    _require(len(function.allocations) == len(shapes))
    for position, (allocation, shape) in enumerate(zip(function.allocations, shapes)):
        _require(
            allocation.position == position
            and allocation.type.shape == shape
            and allocation.type.dtype == "i16"
            and allocation.type.signedness == "unsigned"
        )


def _expect_axes(
    function: StructuredFunction, specs: tuple[tuple[int, bool], ...]
) -> None:
    _require(len(function.axes) == len(specs))
    for position, (axis, (extent, reduction)) in enumerate(zip(function.axes, specs)):
        _require(
            axis.position == position
            and axis.lower_bound == 0
            and axis.upper_bound == extent
            and axis.step == 1
            and axis.extent == extent
            and axis.reduction is reduction
        )


def _expect_regions(
    function: StructuredFunction, axes: tuple[tuple[int, ...], ...]
) -> tuple[OrderedStore | MemoryFill, ...]:
    _require(len(function.regions) == len(axes))
    operations = []
    for order, (region, expected_axes) in enumerate(zip(function.regions, axes)):
        _require(
            region.order == order
            and tuple(axis.position for axis in region.axes) == expected_axes
        )
        for operation in region.operations:
            _require(not isinstance(operation, FunctionCall))
            operations.append(operation)
    _require(
        tuple(operation.order for operation in operations)
        == tuple(range(len(operations)))
    )
    return tuple(operations)


def _expect_function(function: StructuredFunction) -> None:
    _require(
        not function.opaque
        and not function.result_types
        and len(function.returns) == 1
        and not function.returns[0].values
    )


def _expect_store(
    operation,
    order: int,
    kind: str,
    position: int,
    indices: tuple,
    axes: tuple[int, ...],
) -> OrderedStore:
    _require(isinstance(operation, OrderedStore) and operation.order == order)
    memory = operation.target.memory
    _require(
        operation.target.mode == "write"
        and memory.source_kind == kind
        and memory.source_position == position
        and memory.result_position == 0
        and memory.dtype == "i16"
        and memory.signedness == "unsigned"
        and operation.target.signedness == "unsigned"
        and tuple(_normalize(index) for index in operation.target.indices) == indices
        and tuple(axis.position for axis in operation.axes) == axes
    )
    return operation


def _expect_fill(operation, order: int, position: int) -> MemoryFill:
    _require(
        isinstance(operation, MemoryFill)
        and operation.order == order
        and operation.target.source_kind == "allocation"
        and operation.target.source_position == position
        and operation.target.result_position == 0
        and operation.target.dtype == "i16"
        and operation.target.signedness == "unsigned"
        and not operation.axes
        and not operation.predicates
        and _normalize(operation.value) == _constant(0)
    )
    return operation


def _expect_no_predicate(store: OrderedStore) -> None:
    _require(not store.predicates)


def _expect_predicate(
    store: OrderedStore,
    operation: str,
    polarity: bool,
    lhs,
    rhs,
) -> None:
    _require(len(store.predicates) == 1)
    predicate = store.predicates[0]
    _require(
        predicate.condition.operation == "arith.cmpi"
        and predicate.operation == operation
        and predicate.polarity is polarity
        and _normalize(predicate.lhs) == lhs
        and _normalize(predicate.rhs) == rhs
    )


def _effect(
    operation: OrderedStore | MemoryFill,
    semantic_operation: str,
    *,
    arguments: tuple[int, ...] = (),
    allocations: tuple[int, ...] = (),
    coefficient: int | None = None,
    divisor: int | None = None,
) -> StructuredEffect:
    target: MemoryReference
    if isinstance(operation, OrderedStore):
        target = operation.target.memory
    else:
        target = operation.target
    return StructuredEffect(
        operation.order,
        semantic_operation,
        target.source_kind,
        target.source_position,
        arguments,
        allocations,
        tuple(axis.extent for axis in operation.axes),
        tuple(axis.extent for axis in operation.axes if axis.reduction),
        tuple(
            (predicate.operation, predicate.polarity)
            for predicate in operation.predicates
        ),
        coefficient,
        divisor,
    )


def _match_rank_two_chain(function: StructuredFunction):
    _expect_function(function)
    _require(len(function.arguments) == 9)
    matrix_shape = function.arguments[0].shape
    _require(
        matrix_shape is not None
        and len(matrix_shape) == 2
        and matrix_shape[0] == matrix_shape[1]
        and matrix_shape[0] > 0
    )
    extent = matrix_shape[0]
    vector = (extent,)
    _expect_u16_arguments(function, (matrix_shape,) + (vector,) * 8)
    _expect_u16_allocations(function, ())
    _expect_axes(function, ((extent, False),) * 7)
    operations = _expect_regions(function, ((0, 1), (2, 3), (4,), (5, 6)))
    _require(len(operations) == 4)
    stores = tuple(
        _expect_store(operation, order, "argument", target, indices, axes)
        for operation, order, target, indices, axes in zip(
            operations,
            range(4),
            (0, 5, 5, 7),
            (
                (_loop(0), _loop(1)),
                (_loop(2),),
                (_loop(4),),
                (_loop(5),),
            ),
            ((0, 1), (2, 3), (4,), (5, 6)),
        )
    )
    for store in stores:
        _expect_no_predicate(store)

    matrix = _load("argument", 0, matrix_shape, _loop(0), _loop(1))
    first_product = (
        "binary",
        "arith.muli",
        _load("argument", 1, vector, _loop(0)),
        _load("argument", 3, vector, _loop(1)),
    )
    second_product = (
        "binary",
        "arith.muli",
        _load("argument", 2, vector, _loop(0)),
        _load("argument", 4, vector, _loop(1)),
    )
    _expect_terms(
        _normalize(stores[0].value),
        "arith.addi",
        (matrix, first_product, second_product),
    )

    state = _load("argument", 5, vector, _loop(2))
    transposed_product = (
        "binary",
        "arith.muli",
        _load("argument", 0, matrix_shape, _loop(3), _loop(2)),
        _load("argument", 6, vector, _loop(3)),
    )
    _expect_terms(
        _normalize(stores[1].value),
        "arith.addi",
        (state, transposed_product),
    )
    _expect_terms(
        _normalize(stores[2].value),
        "arith.addi",
        (
            _load("argument", 5, vector, _loop(4)),
            _load("argument", 8, vector, _loop(4)),
        ),
    )
    output_product = (
        "binary",
        "arith.muli",
        _load("argument", 0, matrix_shape, _loop(5), _loop(6)),
        _load("argument", 5, vector, _loop(6)),
    )
    _expect_terms(
        _normalize(stores[3].value),
        "arith.addi",
        (_load("argument", 7, vector, _loop(5)), output_product),
    )
    effects = (
        _effect(
            stores[0],
            "rank_two_update",
            arguments=(0, 1, 3, 2, 4),
        ),
        _effect(
            stores[1],
            "transposed_matrix_vector_update",
            arguments=(5, 0, 6),
        ),
        _effect(stores[2], "vector_bias_update", arguments=(5, 8)),
        _effect(
            stores[3],
            "matrix_vector_update",
            arguments=(7, 0, 5),
        ),
    )
    return RankTwoUpdateGemvChainPlan(
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        extent,
        (1, 1),
        (1, 1, 1, 1),
        effects,
    )


def _match_symmetric_contraction(function: StructuredFunction):
    _expect_function(function)
    _require(len(function.arguments) == 3)
    matrix_shape = function.arguments[0].shape
    rhs_shape = function.arguments[1].shape
    output_shape = function.arguments[2].shape
    _require(
        matrix_shape is not None
        and rhs_shape is not None
        and output_shape is not None
        and len(matrix_shape) == 2
        and len(rhs_shape) == 2
        and matrix_shape[0] == matrix_shape[1]
        and rhs_shape == output_shape
        and rhs_shape[0] == matrix_shape[0]
    )
    rows, columns = rhs_shape
    _expect_u16_arguments(function, (matrix_shape, rhs_shape, output_shape))
    _expect_u16_allocations(function, ())
    _expect_axes(
        function,
        ((rows, False), (columns, False), (rows, True)),
    )
    operations = _expect_regions(function, ((0, 1, 2),))
    _require(len(operations) == 3)
    stores = tuple(
        _expect_store(
            operation,
            order,
            "argument",
            2,
            (_loop(0), _loop(1)),
            axes,
        )
        for operation, order, axes in zip(
            operations,
            range(3),
            ((0, 1), (0, 1, 2), (0, 1, 2)),
        )
    )
    _expect_no_predicate(stores[0])
    _expect_predicate(stores[1], "sle", True, _loop(2), _loop(0))
    _expect_predicate(stores[2], "sle", False, _loop(2), _loop(0))
    output = _load("argument", 2, output_shape, _loop(0), _loop(1))
    beta = _product_coefficient(_normalize(stores[0].value), output)
    rhs = _load("argument", 1, rhs_shape, _loop(2), _loop(1))
    lower = _load("argument", 0, matrix_shape, _loop(0), _loop(2))
    upper = _load("argument", 0, matrix_shape, _loop(2), _loop(0))
    lower_sum = _flatten(_normalize(stores[1].value), "arith.addi")
    upper_sum = _flatten(_normalize(stores[2].value), "arith.addi")
    _require(len(lower_sum) == 2 and output in lower_sum)
    _require(len(upper_sum) == 2 and output in upper_sum)
    lower_sum.remove(output)
    upper_sum.remove(output)
    alpha = _product_coefficient(lower_sum[0], lower, rhs)
    _require(_product_coefficient(upper_sum[0], upper, rhs) == alpha)
    effects = (
        _effect(
            stores[0],
            "output_prescale",
            arguments=(2,),
            coefficient=beta,
        ),
        _effect(
            stores[1],
            "symmetric_lower_update",
            arguments=(2, 0, 1),
            coefficient=alpha,
        ),
        _effect(
            stores[2],
            "symmetric_upper_update",
            arguments=(2, 0, 1),
            coefficient=alpha,
        ),
    )
    return SymmetricContractionPlan(
        0,
        1,
        2,
        rows,
        columns,
        rows,
        alpha,
        beta,
        "sle",
        (True, False),
        effects,
    )


def _match_unit_diagonal_contraction(function: StructuredFunction):
    _expect_function(function)
    _require(len(function.arguments) == 2)
    matrix_shape = function.arguments[0].shape
    state_shape = function.arguments[1].shape
    _require(
        matrix_shape is not None
        and state_shape is not None
        and len(matrix_shape) == 2
        and len(state_shape) == 2
        and matrix_shape[0] == matrix_shape[1]
        and state_shape[0] == matrix_shape[0]
    )
    rows, columns = state_shape
    _expect_u16_arguments(function, (matrix_shape, state_shape))
    _expect_u16_allocations(function, ())
    _expect_axes(
        function,
        ((rows, False), (columns, False), (rows, True)),
    )
    operations = _expect_regions(function, ((0, 1, 2),))
    _require(len(operations) == 2)
    update = _expect_store(
        operations[0],
        0,
        "argument",
        1,
        (_loop(0), _loop(1)),
        (0, 1, 2),
    )
    scale = _expect_store(
        operations[1],
        1,
        "argument",
        1,
        (_loop(0), _loop(1)),
        (0, 1),
    )
    _expect_predicate(update, "sgt", True, _loop(2), _loop(0))
    _expect_no_predicate(scale)
    state = _load("argument", 1, state_shape, _loop(0), _loop(1))
    off_diagonal = (
        "binary",
        "arith.muli",
        _load("argument", 0, matrix_shape, _loop(2), _loop(0)),
        _load("argument", 1, state_shape, _loop(2), _loop(1)),
    )
    _expect_terms(
        _normalize(update.value),
        "arith.addi",
        (state, off_diagonal),
    )
    coefficient = _product_coefficient(_normalize(scale.value), state)
    effects = (
        _effect(
            update,
            "strict_triangular_update",
            arguments=(1, 0, 1),
        ),
        _effect(
            scale,
            "postscale",
            arguments=(1,),
            coefficient=coefficient,
        ),
    )
    return UnitDiagonalTriangularContractionPlan(
        0,
        1,
        rows,
        columns,
        rows,
        "sgt",
        1,
        coefficient,
        True,
        effects,
    )


def _match_centered_statistics(function: StructuredFunction):
    _expect_function(function)
    _require(len(function.arguments) == 3)
    data_shape = function.arguments[0].shape
    mean_shape = function.arguments[1].shape
    gram_shape = function.arguments[2].shape
    _require(
        data_shape is not None
        and mean_shape is not None
        and gram_shape is not None
        and len(data_shape) == 2
        and mean_shape == (data_shape[1],)
        and gram_shape == (data_shape[1], data_shape[1])
        and data_shape[0] > 1
    )
    samples, features = data_shape
    _expect_u16_arguments(function, (data_shape, mean_shape, gram_shape))
    _expect_u16_allocations(function, ((), ()))
    _expect_axes(
        function,
        (
            (features, False),
            (samples, False),
            (features, False),
            (features, False),
            (samples, False),
            (features, False),
            (features, False),
            (features, False),
            (samples, True),
        ),
    )
    operations = _expect_regions(
        function,
        ((), (0,), (1, 2), (3,), (4, 5), (6, 7, 8)),
    )
    _require(
        len(operations) == 9
        and all(isinstance(operation, OrderedStore) for operation in operations)
    )
    stores = (
        _expect_store(operations[0], 0, "allocation", 0, (), ()),
        _expect_store(operations[1], 1, "allocation", 1, (), ()),
        _expect_store(operations[2], 2, "argument", 1, (_loop(0),), (0,)),
        _expect_store(operations[3], 3, "argument", 1, (_loop(2),), (1, 2)),
        _expect_store(operations[4], 4, "argument", 1, (_loop(3),), (3,)),
        _expect_store(
            operations[5],
            5,
            "argument",
            0,
            (_loop(4), _loop(5)),
            (4, 5),
        ),
        _expect_store(
            operations[6],
            6,
            "argument",
            2,
            (_loop(6), _loop(7)),
            (6, 7),
        ),
        _expect_store(
            operations[7],
            7,
            "argument",
            2,
            (_loop(6), _loop(7)),
            (6, 7, 8),
        ),
        _expect_store(
            operations[8],
            8,
            "argument",
            2,
            (_loop(6), _loop(7)),
            (6, 7),
        ),
    )
    for store in stores:
        _expect_no_predicate(store)
    scalar = ()
    _require(_normalize(stores[0].value) == _constant(samples))
    lhs, rhs = _binary(_normalize(stores[1].value), "arith.subi")
    _require(lhs == _constant(samples) and rhs == _constant(1))
    _require(_normalize(stores[2].value) == _constant(0))
    mean = _load("argument", 1, mean_shape, _loop(2))
    data = _load("argument", 0, data_shape, _loop(1), _loop(2))
    _expect_terms(_normalize(stores[3].value), "arith.addi", (mean, data))
    dividend, divisor = _binary(_normalize(stores[4].value), "arith.divui")
    _require(
        dividend == _load("argument", 1, mean_shape, _loop(3))
        and divisor == _load("allocation", 0, scalar)
    )
    centered_data = _load("argument", 0, data_shape, _loop(4), _loop(5))
    center_mean = _load("argument", 1, mean_shape, _loop(5))
    _require(
        _binary(_normalize(stores[5].value), "arith.subi")
        == (centered_data, center_mean)
    )
    _require(_normalize(stores[6].value) == _constant(0))
    gram = _load("argument", 2, gram_shape, _loop(6), _loop(7))
    product = (
        "binary",
        "arith.muli",
        _load("argument", 0, data_shape, _loop(8), _loop(6)),
        _load("argument", 0, data_shape, _loop(8), _loop(7)),
    )
    _expect_terms(_normalize(stores[7].value), "arith.addi", (gram, product))
    gram_dividend, gram_divisor = _binary(_normalize(stores[8].value), "arith.divui")
    _require(gram_dividend == gram and gram_divisor == _load("allocation", 1, scalar))
    effects = (
        _effect(stores[0], "sample_count_materialize", coefficient=samples),
        _effect(
            stores[1],
            "gram_divisor_materialize",
            coefficient=samples - 1,
        ),
        _effect(stores[2], "mean_initialize"),
        _effect(stores[3], "mean_accumulate", arguments=(1, 0)),
        _effect(
            stores[4],
            "mean_divide",
            arguments=(1,),
            allocations=(0,),
            divisor=samples,
        ),
        _effect(stores[5], "center_in_place", arguments=(0, 1)),
        _effect(stores[6], "gram_initialize"),
        _effect(stores[7], "gram_accumulate", arguments=(2, 0)),
        _effect(
            stores[8],
            "gram_divide",
            arguments=(2,),
            allocations=(1,),
            divisor=samples - 1,
        ),
    )
    return CenteredGramStatisticsPlan(
        0,
        1,
        2,
        0,
        1,
        samples,
        features,
        samples,
        samples - 1,
        effects,
    )


def _match_normalized_statistics(function: StructuredFunction):
    _expect_function(function)
    _require(len(function.arguments) == 4)
    input_shapes = tuple(argument.shape for argument in function.arguments[:3])
    gram_shape = function.arguments[3].shape
    _require(
        input_shapes[0] is not None
        and input_shapes[0] == input_shapes[1] == input_shapes[2]
        and len(input_shapes[0]) == 2
        and gram_shape == (input_shapes[0][1], input_shapes[0][1])
    )
    data_shape = input_shapes[0]
    samples, features = data_shape
    column = (features,)
    _expect_u16_arguments(function, (data_shape, data_shape, data_shape, gram_shape))
    _expect_u16_allocations(
        function,
        (
            (),
            (),
            (),
            column,
            data_shape,
            column,
            column,
            data_shape,
            column,
            data_shape,
        ),
    )
    _expect_axes(
        function,
        (
            (samples, False),
            (features, False),
            (features, False),
            (samples, False),
            (features, False),
            (features, False),
            (samples, True),
            (features, False),
            (samples, False),
            (features, False),
            (features, False),
            (features, False),
            (samples, True),
        ),
    )
    operations = _expect_regions(
        function,
        (
            (),
            (0, 1),
            (2,),
            (3, 4),
            (5, 6),
            (7,),
            (8, 9),
            (10, 11, 12),
        ),
    )
    _require(len(operations) == 23)
    setup = tuple(
        _expect_store(operations[index], index, "allocation", index, (), ())
        for index in range(3)
    )
    fills = tuple(
        _expect_fill(operations[order], order, order) for order in range(3, 10)
    )
    stores = (
        _expect_store(operations[10], 10, "allocation", 3, (_loop(1),), (0, 1)),
        _expect_store(operations[11], 11, "allocation", 3, (_loop(2),), (2,)),
        _expect_store(
            operations[12],
            12,
            "allocation",
            4,
            (_loop(3), _loop(4)),
            (3, 4),
        ),
        _expect_store(operations[13], 13, "allocation", 5, (_loop(5),), (5, 6)),
        _expect_store(operations[14], 14, "allocation", 5, (_loop(5),), (5,)),
        _expect_store(operations[15], 15, "allocation", 6, (_loop(7),), (7,)),
        _expect_store(operations[16], 16, "allocation", 6, (_loop(7),), (7,)),
        _expect_store(operations[17], 17, "allocation", 8, (_loop(7),), (7,)),
        _expect_store(
            operations[18],
            18,
            "allocation",
            7,
            (_loop(8), _loop(9)),
            (8, 9),
        ),
        _expect_store(
            operations[19],
            19,
            "allocation",
            9,
            (_loop(8), _loop(9)),
            (8, 9),
        ),
        _expect_store(
            operations[20],
            20,
            "argument",
            3,
            (_loop(10), _loop(11)),
            (10, 11),
        ),
        _expect_store(
            operations[21],
            21,
            "argument",
            3,
            (_loop(10), _loop(11)),
            (10, 11, 12),
        ),
        _expect_store(
            operations[22],
            22,
            "argument",
            3,
            (_loop(10), _loop(11)),
            (10, 11),
        ),
    )
    for store in setup + stores:
        if store is not stores[6] and store is not stores[12]:
            _expect_no_predicate(store)
    scalar = ()
    _require(_normalize(setup[0].value) == _constant(samples))
    root_factor = _normalize(setup[1].value)
    _require(root_factor[0] == "constant" and type(root_factor[1]) is int)
    root_factor = root_factor[1]
    _require(root_factor == math.isqrt(samples))
    _require(_normalize(setup[2].value) == _constant(1))

    mean = _load("allocation", 3, column, _loop(1))
    mean_input = _load("argument", 0, data_shape, _loop(0), _loop(1))
    _expect_terms(_normalize(stores[0].value), "arith.addi", (mean, mean_input))
    _require(
        _binary(_normalize(stores[1].value), "arith.divui")
        == (
            _load("allocation", 3, column, _loop(2)),
            _load("allocation", 0, scalar),
        )
    )
    _require(
        _binary(_normalize(stores[2].value), "arith.subi")
        == (
            _load("argument", 1, data_shape, _loop(3), _loop(4)),
            _load("allocation", 3, column, _loop(4)),
        )
    )
    variance = _load("allocation", 5, column, _loop(5))
    variance_value = _load("allocation", 4, data_shape, _loop(6), _loop(5))
    variance_product = ("binary", "arith.muli", variance_value, variance_value)
    _expect_terms(
        _normalize(stores[3].value),
        "arith.addi",
        (variance, variance_product),
    )
    _require(
        _binary(_normalize(stores[4].value), "arith.divui")
        == (variance, _load("allocation", 0, scalar))
    )
    _require(
        _normalize(stores[5].value)
        == (
            "unary",
            "math.sqrt",
            _load("allocation", 5, column, _loop(7)),
        )
    )
    _expect_predicate(
        stores[6],
        "eq",
        True,
        _load("allocation", 6, column, _loop(7)),
        _constant(0),
    )
    _require(_normalize(stores[6].value) == _load("allocation", 2, scalar))
    _expect_product(
        _normalize(stores[7].value),
        _load("allocation", 1, scalar),
        _load("allocation", 6, column, _loop(7)),
    )
    _require(
        _binary(_normalize(stores[8].value), "arith.subi")
        == (
            _load("argument", 2, data_shape, _loop(8), _loop(9)),
            _load("allocation", 3, column, _loop(9)),
        )
    )
    _require(
        _binary(_normalize(stores[9].value), "arith.divui")
        == (
            _load("allocation", 7, data_shape, _loop(8), _loop(9)),
            _load("allocation", 8, column, _loop(9)),
        )
    )
    _require(_normalize(stores[10].value) == _constant(0))
    gram = _load("argument", 3, gram_shape, _loop(10), _loop(11))
    gram_product = (
        "binary",
        "arith.muli",
        _load("allocation", 9, data_shape, _loop(12), _loop(10)),
        _load("allocation", 9, data_shape, _loop(12), _loop(11)),
    )
    _expect_terms(
        _normalize(stores[11].value),
        "arith.addi",
        (gram, gram_product),
    )
    _expect_predicate(stores[12], "eq", True, _loop(10), _loop(11))
    _require(_normalize(stores[12].value) == _constant(1))

    effects = (
        _effect(setup[0], "sample_count_materialize", coefficient=samples),
        _effect(
            setup[1],
            "root_sample_count_materialize",
            coefficient=root_factor,
        ),
        _effect(setup[2], "unit_materialize", coefficient=1),
        *tuple(
            _effect(fill, operation)
            for fill, operation in zip(
                fills,
                (
                    "mean_initialize",
                    "variance_center_initialize",
                    "variance_initialize",
                    "standard_deviation_initialize",
                    "center_initialize",
                    "denominator_initialize",
                    "normalized_initialize",
                ),
            )
        ),
        _effect(stores[0], "mean_accumulate", arguments=(0,), allocations=(3,)),
        _effect(
            stores[1],
            "mean_divide",
            allocations=(3, 0),
            divisor=samples,
        ),
        _effect(
            stores[2],
            "variance_center",
            arguments=(1,),
            allocations=(3, 4),
        ),
        _effect(
            stores[3],
            "variance_accumulate",
            allocations=(4, 5),
        ),
        _effect(
            stores[4],
            "variance_divide",
            allocations=(5, 0),
            divisor=samples,
        ),
        _effect(stores[5], "standard_deviation", allocations=(5, 6)),
        _effect(
            stores[6],
            "zero_clamp",
            allocations=(6, 2),
            coefficient=1,
        ),
        _effect(
            stores[7],
            "denominator_form",
            allocations=(1, 6, 8),
            coefficient=root_factor,
        ),
        _effect(
            stores[8],
            "center",
            arguments=(2,),
            allocations=(3, 7),
        ),
        _effect(
            stores[9],
            "normalize",
            allocations=(7, 8, 9),
        ),
        _effect(stores[10], "gram_initialize", arguments=(3,)),
        _effect(
            stores[11],
            "gram_accumulate",
            arguments=(3,),
            allocations=(9,),
        ),
        _effect(
            stores[12],
            "diagonal_override",
            arguments=(3,),
            coefficient=1,
        ),
    )
    return NormalizedGramStatisticsPlan(
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        samples,
        features,
        samples,
        samples,
        root_factor,
        1,
        1,
        effects,
    )


_MATCHERS = (
    _match_rank_two_chain,
    _match_symmetric_contraction,
    _match_unit_diagonal_contraction,
    _match_centered_statistics,
    _match_normalized_statistics,
)


def _candidates(program: StructuredProgram) -> tuple[StructuredAPUG2Plan, ...]:
    candidates = []
    for function in program.functions:
        for matcher in _MATCHERS:
            try:
                candidates.append(matcher(function))
            except _Mismatch:
                pass
    return tuple(candidates)


def analyze_apu_g2_structured_plan(module_or_program) -> StructuredAPUG2Plan:
    """Return the unique proven semantic plan for retained static MLIR."""

    program = (
        module_or_program
        if isinstance(module_or_program, StructuredProgram)
        else analyze_structured_program(module_or_program)
    )
    candidates = _candidates(program)
    if not candidates:
        raise NoStructuredAPUG2Plan("no supported structural APUg2 plan was proven")
    if len(candidates) != 1:
        raise AmbiguousStructuredAPUG2Plan(
            f"retained MLIR proves {len(candidates)} structured APUg2 plans"
        )
    return candidates[0]


__all__ = [
    "AmbiguousStructuredAPUG2Plan",
    "CenteredGramStatisticsPlan",
    "NoStructuredAPUG2Plan",
    "NormalizedGramStatisticsPlan",
    "RankTwoUpdateGemvChainPlan",
    "StructuredAPUG2Plan",
    "StructuredAPUG2PlanError",
    "StructuredEffect",
    "SymmetricContractionPlan",
    "UnitDiagonalTriangularContractionPlan",
    "analyze_apu_g2_structured_plan",
]
