# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for retained-MLIR UPMEM cost summarization."""

import pytest

from allo.pim.upmem_analysis import (
    UnsupportedDynamicBoundError,
    analyze_upmem_mlir,
)


def test_nested_static_loops_scale_operations_memory_and_loop_control():
    summary = analyze_upmem_mlir(
        """
        module {
          func.func @kernel(%a: memref<2x4xf32>, %b: memref<2x4xf32>) {
            affine.for %i = 0 to 2 {
              affine.for %j = 1 to 5 step 2 {
                %0 = affine.load %a[%i, %j] : memref<2x4xf32>
                %1 = arith.addf %0, %0 : f32
                affine.store %1, %b[%i, %j] : memref<2x4xf32>
              }
            }
            return
          }
        }
        """
    )

    assert summary.count("ADD", "float") == 4
    # Synthesized induction updates: outer 2 + two activations * inner 2.
    assert summary.count("ADD", "integer") == 6
    # Loop condition/branches: outer 3 + two activations * inner 3.
    assert summary.count("BRANCH") == 9
    assert summary.memory.load_instructions == 4
    assert summary.memory.store_instructions == 4
    assert summary.memory.read_bytes == 16
    assert summary.memory.written_bytes == 16
    records = dict(summary.cost_records())
    assert records["LD_WRAM"]["iterations"] == 4
    assert records["ST_WRAM"]["iterations"] == 4


def test_scf_loop_resolves_constant_ssa_bounds_and_module_objects():
    class RetainedModule:
        def __str__(self):
            return """
            module {
              %c0 = arith.constant 0 : index
              %c7 = arith.constant 7 : index
              %c2 = arith.constant 2 : index
              scf.for %i = %c0 to %c7 step %c2 {
                %0 = arith.muli %i, %i : index
              }
            }
            """

    summary = analyze_upmem_mlir(RetainedModule())

    assert summary.count("MUL", "integer") == 4
    assert summary.count("ADD", "integer") == 4
    assert summary.count("BRANCH", "integer") == 5
    assert not summary.unclassified_operations


def test_dynamic_bounds_fail_closed_or_record_explicit_fallback():
    text = """
    func.func @dynamic(%n: index) {
      scf.for %i = %c0 to %n step %c1 {
        %0 = arith.addi %i, %i : index
      }
    }
    """
    with pytest.raises(UnsupportedDynamicBoundError, match="cannot prove loop trip"):
        analyze_upmem_mlir(text)

    bounded = analyze_upmem_mlir(
        text,
        symbol_values={"c0": 0, "c1": 1},
        dynamic_trip_count=32,
    )
    assert bounded.count("ADD", "integer") == 64  # 32 body + 32 induction
    assert bounded.count("BRANCH") == 33
    assert "dynamic_trip_count=32" in bounded.diagnostics[0]


def test_general_arithmetic_and_control_are_grouped_by_numeric_kind():
    summary = analyze_upmem_mlir(
        """
        func.func @ops(%a: f32, %b: f32, %i: i32, %j: i32) {
          %0 = arith.subf %a, %b : f32
          %1 = arith.divsi %i, %j : i32
          %2 = math.sqrt %a : f32
          %3 = arith.cmpf olt, %a, %b : f32
          %4 = arith.select %3, %i, %j : i32
          %5 = arith.minimumf %a, %b : f32
          %6 = arith.maxsi %i, %j : i32
          cf.cond_br %3, ^yes, ^no
        }
        """
    )

    assert summary.count("SUB", "float") == 1
    assert summary.count("DIV", "integer") == 1
    assert summary.count("SQRT", "float") == 1
    assert summary.count("CMP", "float") == 1
    assert summary.count("SELECT", "integer") == 1
    assert summary.count("MIN", "float") == 1
    assert summary.count("MAX", "integer") == 1
    assert summary.count("BRANCH", "integer") == 1


def test_vector_memory_and_static_copy_report_instruction_and_byte_counts():
    summary = analyze_upmem_mlir(
        """
        func.func @memory(%a: memref<4x8xi16>, %b: memref<4x8xi16>) {
          %0 = vector.load %a[%c0, %c0] : memref<4x8xi16>, vector<4xi16>
          vector.store %0, %b[%c0, %c0] : memref<4x8xi16>, vector<4xi16>
          memref.copy %a, %b : memref<4x8xi16> to memref<4x8xi16>
        }
        """
    )

    assert summary.memory.load_instructions == 4 + 32
    assert summary.memory.store_instructions == 4 + 32
    assert summary.memory.read_bytes == 8 + 64
    assert summary.memory.written_bytes == 8 + 64
    assert summary.memory.unknown_byte_accesses == 0


def test_conditionals_are_explicit_conservative_upper_bounds():
    summary = analyze_upmem_mlir(
        """
        func.func @conditional(%p: i1, %a: i32, %b: i32) {
          scf.if %p {
            %0 = arith.addi %a, %b : i32
          } else {
            %1 = arith.subi %a, %b : i32
          }
        }
        """
    )

    assert summary.count("BRANCH") == 1
    assert summary.count("ADD") == 1
    assert summary.count("SUB") == 1
    assert "summed mutually exclusive" in summary.diagnostics[0]
