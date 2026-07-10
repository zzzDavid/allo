"""Focused tests for target-neutral retained-MLIR structural analysis."""

from dataclasses import FrozenInstanceError

import pytest

from allo.pim.structured_program import (
    BinaryExpression,
    CallResultExpression,
    FunctionCall,
    LoadExpression,
    LoopIndexExpression,
    MemoryFill,
    StructuredProgramAnalysisError,
    UnaryExpression,
    analyze_structured_program,
    canonical_signature,
    manifest,
)


_CONTRACTION = r"""
module {
  func.func @first_name(%arg_a: memref<4x8xi16>, %arg_b: memref<8x6xi16>, %arg_c: memref<4x6xi16>) attributes {itypes = "uuu", otypes = ""} {
    affine.for %row = 0 to 4 {
      affine.for %col = 0 to 6 {
        affine.for %depth = 0 to 8 {
          %left = affine.load %arg_a[%row, %depth] {from = "left_source", unsigned} : memref<4x8xi16>
          %right = affine.load %arg_b[%depth, %col] {from = "right_source", unsigned} : memref<8x6xi16>
          %product = arith.muli %left, %right {unsigned} : i16
          affine.store %product, %arg_c[%row, %col] {to = "destination"} : memref<4x6xi16>
        } {loop_name = "depth_alias", op_name = "stage_one", reduction}
      } {loop_name = "column_alias"}
    } {loop_name = "row_alias", op_name = "outer_stage"}
    return
  }
}
"""


_RENAMED_CONTRACTION = r"""
module {
  func.func @unrelated(%memory_x: memref<4x8xi16>, %memory_y: memref<8x6xi16>, %memory_z: memref<4x6xi16>) attributes {itypes = "uuu", otypes = ""} {
    affine.for %iv0 = 0 to 4 {
      affine.for %iv1 = 0 to 6 {
        affine.for %iv2 = 0 to 8 {
          %v7 = affine.load %memory_x[%iv0, %iv2] {from = "renamed_a", unsigned} : memref<4x8xi16>
          %v2 = affine.load %memory_y[%iv2, %iv1] {from = "renamed_b", unsigned} : memref<8x6xi16>
          %v99 = arith.muli %v7, %v2 {unsigned} : i16
          affine.store %v99, %memory_z[%iv0, %iv1] {to = "renamed_c"} : memref<4x6xi16>
        } {loop_name = "k", op_name = "different_inner", reduction}
      } {loop_name = "j"}
    } {loop_name = "i", op_name = "different_outer"}
    return
  }
}
"""


def test_alpha_renaming_does_not_change_canonical_signature_or_manifest():
    class RetainedModule:
        def __str__(self):
            return _RENAMED_CONTRACTION

    original = analyze_structured_program(_CONTRACTION)
    renamed = analyze_structured_program(RetainedModule())

    assert original.canonical_signature == renamed.canonical_signature
    assert canonical_signature(_CONTRACTION) == canonical_signature(renamed)
    assert original.canonical_manifest() == manifest(renamed, canonical=True)
    assert original.manifest() != renamed.manifest()
    with pytest.raises(FrozenInstanceError):
        original.functions[0].name = "mutated"


def test_loop_access_arithmetic_and_reduction_changes_alter_signature():
    baseline = canonical_signature(_CONTRACTION)
    variants = (
        _CONTRACTION.replace("%depth = 0 to 8", "%depth = 0 to 7"),
        _CONTRACTION.replace("arith.muli", "arith.addi"),
        _CONTRACTION.replace(
            "%arg_b[%depth, %col]", "%arg_b[%row, %col]"
        ),
        _CONTRACTION.replace(", reduction}", "}"),
    )

    signatures = {canonical_signature(text) for text in variants}
    assert baseline not in signatures
    assert len(signatures) == len(variants)


def test_symmetric_branches_keep_loop_scope_accesses_and_predicate_polarity():
    program = analyze_structured_program(
        r"""
        func.func @symmetric(%a: memref<5x5xi16>, %out: memref<5x5xi16>) attributes {itypes = "uu", otypes = ""} {
          affine.for %i = 0 to 5 {
            affine.for %j = 0 to 5 {
              affine.for %k = 0 to 5 {
                %condition = arith.cmpi sle, %k, %i : index
                scf.if %condition {
                  %lower = affine.load %a[%i, %k] {from = "a", unsigned} : memref<5x5xi16>
                  affine.store %lower, %out[%i, %j] {to = "out"} : memref<5x5xi16>
                } else {
                  %upper = affine.load %a[%k, %i] {from = "a", unsigned} : memref<5x5xi16>
                  affine.store %upper, %out[%i, %j] {to = "out"} : memref<5x5xi16>
                }
              } {loop_name = "reduction", reduction}
            } {loop_name = "column"}
          } {loop_name = "row"}
          return
        }
        """
    )

    function = program.functions[0]
    then_store, else_store = function.stores
    assert len(function.axes) == 3
    assert [axis.position for axis in then_store.axes] == [0, 1, 2]
    assert [axis.position for axis in else_store.axes] == [0, 1, 2]
    assert then_store.predicates[0].operation == "sle"
    assert then_store.predicates[0].polarity is True
    assert else_store.predicates[0].operation == "sle"
    assert else_store.predicates[0].polarity is False
    assert isinstance(then_store.value, LoadExpression)
    assert isinstance(else_store.value, LoadExpression)
    assert [index.axis_position for index in then_store.value.access.indices] == [0, 2]
    assert [index.axis_position for index in else_store.value.access.indices] == [2, 0]


def test_sequential_regions_and_stores_remain_ordered():
    program = analyze_structured_program(
        r"""
        func.func @stages(%source: memref<4xi32>, %middle: memref<4xi32>, %result: memref<4xi32>) {
          %two = arith.constant 2 : i32
          affine.for %first_axis = 0 to 4 {
            %first_load = affine.load %source[%first_axis] : memref<4xi32>
            %first_value = arith.addi %first_load, %two : i32
            affine.store %first_value, %middle[%first_axis] : memref<4xi32>
          } {loop_name = "first"}
          affine.for %second_axis = 0 to 4 {
            %second_load = affine.load %middle[%second_axis] : memref<4xi32>
            %second_value = arith.muli %second_load, %two : i32
            affine.store %second_value, %result[%second_axis] : memref<4xi32>
          } {loop_name = "second"}
          return
        }
        """
    )

    function = program.functions[0]
    assert [region.order for region in function.regions] == [0, 1]
    assert [store.order for store in function.stores] == [0, 1]
    assert [store.target.source_position for store in function.stores] == [1, 2]
    assert [store.value.operation for store in function.stores] == [
        "arith.addi",
        "arith.muli",
    ]


def test_static_constants_math_divisions_and_alloc_backed_values_are_explicit():
    program = analyze_structured_program(
        r"""
        func.func @numeric(%unsigned_in: memref<4xi16>, %signed_in: memref<4xi16>, %float_in: memref<4xf32>, %unsigned_out: memref<4xi16>, %signed_out: memref<4xi16>, %float_out: memref<4xf32>) attributes {itypes = "u__u__", otypes = ""} {
          %begin = arith.constant 0 : index
          %end = arith.constant 4 : index
          %stride = arith.constant 1 : index
          %two_u = arith.constant 2 {unsigned} : i16
          %two_s = arith.constant 2 : i16
          %two_f = arith.constant 2.0 : f32
          %zero = arith.constant 0.0 : f32
          %scratch = memref.alloc() {name = "temporary"} : memref<4xf32>
          linalg.fill ins(%zero : f32) outs(%scratch : memref<4xf32>)
          scf.for %axis = %begin to %end step %stride {
            %u = memref.load %unsigned_in[%axis] : memref<4xi16>
            %s = memref.load %signed_in[%axis] : memref<4xi16>
            %f = memref.load %float_in[%axis] : memref<4xf32>
            %uq = arith.divui %u, %two_u : i16
            %sq = arith.divsi %s, %two_s : i16
            %root = math.sqrt %f : f32
            %fq = arith.divf %root, %two_f : f32
            memref.store %uq, %unsigned_out[%axis] : memref<4xi16>
            memref.store %sq, %signed_out[%axis] : memref<4xi16>
            memref.store %fq, %scratch[%axis] : memref<4xf32>
            memref.store %fq, %float_out[%axis] : memref<4xf32>
          }
          return
        }
        """
    )

    function = program.functions[0]
    assert function.axes[0].extent == 4
    assert function.allocations[0].type.shape == (4,)
    assert isinstance(function.regions[0].operations[0], MemoryFill)
    assert [store.value.operation for store in function.stores[:2]] == [
        "arith.divui",
        "arith.divsi",
    ]
    float_division = function.stores[2].value
    assert isinstance(float_division, BinaryExpression)
    assert float_division.operation == "arith.divf"
    assert isinstance(float_division.lhs, UnaryExpression)
    assert float_division.lhs.operation == "math.sqrt"


def test_direct_calls_are_retained_and_function_names_are_alpha_renamed():
    source = r"""
    module {
      func.func private @helper(%value: i32) -> i32
      func.func @caller(%input: i32, %output: memref<i32>) {
        %called = call @helper(%input) : (i32) -> i32
        affine.store %called, %output[] : memref<i32>
        return
      }
    }
    """
    renamed = source.replace("@helper", "@opaque_fn").replace(
        "@caller", "@entry_fn"
    )

    program = analyze_structured_program(source)
    caller = program.functions[1]
    assert program.functions[0].opaque is True
    assert isinstance(caller.regions[0].operations[0], FunctionCall)
    assert caller.regions[0].operations[0].callee_position == 0
    assert isinstance(caller.stores[0].value, CallResultExpression)
    assert canonical_signature(source) == canonical_signature(renamed)


@pytest.mark.parametrize(
    "text",
    [
        "func.func @dynamic_shape(%arg: memref<?xi32>) { return }",
        r"""
        func.func @dynamic_bound(%size: index, %out: memref<4xi32>) {
          %zero = arith.constant 0 : index
          %one = arith.constant 1 : index
          scf.for %axis = %zero to %size step %one {
          }
          return
        }
        """,
        r"""
        func.func @unsupported_math(%value: f32) -> f32 {
          %result = math.powf %value : f32
          return %result : f32
        }
        """,
        r"""
        func.func @undefined_value(%out: memref<i32>) {
          affine.store %missing, %out[] : memref<i32>
          return
        }
        """,
        "func.func @unclosed(%out: memref<i32>) {",
    ],
)
def test_dynamic_malformed_and_unsupported_constructs_fail_closed(text):
    with pytest.raises(StructuredProgramAnalysisError):
        analyze_structured_program(text)
