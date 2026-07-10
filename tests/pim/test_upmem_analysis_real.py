# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression checks over canonical PolyBench modules retained by Allo."""

import allo

from allo.pim.upmem_analysis import analyze_upmem_mlir
from lib.upmem_polybench import get_case


def _summary(name):
    case = get_case(name)
    kernel, instantiate = case.kernel_and_instantiate()
    schedule = allo.customize(kernel, instantiate=instantiate)
    return case, analyze_upmem_mlir(schedule.module)


def test_deriche_sequential_affine_phases_do_not_leak_loop_multipliers():
    case, summary = _summary("deriche")
    assert case.dims == {"W": 192, "H": 128}

    # Six sequential 2-D sweeps all visit W*H elements.  Four recurrence
    # sweeps perform three adds/four multiplies and two combine sweeps perform
    # one each: 4*(3*W*H)+2*(W*H), 4*(4*W*H)+2*(W*H).
    assert summary.count("ADD", "float") == 344_064
    assert summary.count("MUL", "float") == 442_368
    # The two reverse sweeps in each dimension materialize two index subs.
    assert summary.count("SUB", "integer") == 98_304

    # Exact static loop-control counts across four W-outer/H-inner and two
    # H-outer/W-inner nests.
    assert summary.count("ADD", "integer") == 148_480
    assert summary.count("BRANCH", "integer") == 149_510
    assert summary.memory.load_instructions == 983_040
    assert summary.memory.store_instructions == 542_912
    assert summary.memory.read_bytes == 3_932_160
    assert summary.memory.written_bytes == 2_171_648
    assert not summary.diagnostics

    # The historical leaked-region failure produced O(1e17) counts.
    assert summary.total_instruction_count < 10_000_000


def test_gemm_real_nested_loops_match_static_small_shape_products():
    case, summary = _summary("gemm")
    assert case.dims == {"P": 60, "R": 70, "Q": 80}

    # mm1: 60*70*80 MAC source operations; ele_add: 60*70 mul/add.
    assert summary.count("ADD", "float") == 340_200
    assert summary.count("MUL", "float") == 340_200

    # Loop controls include all three mm1 levels and both ele_add levels.
    assert summary.count("ADD", "integer") == 344_520
    assert summary.count("BRANCH", "integer") == 348_842
    assert summary.memory.load_instructions == 1_016_400
    assert summary.memory.store_instructions == 340_200
    assert summary.memory.read_bytes == 4_065_600
    assert summary.memory.written_bytes == 1_360_800
    assert not summary.diagnostics
