# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression checks over canonical PolyBench modules retained by Allo."""

import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import allo
from allo.ir.types import float32

from allo.pim.upmem_analysis import analyze_upmem_mlir

_PSIZE = json.loads(
    (Path(__file__).resolve().parents[2] / "examples" / "polybench" / "psize.json")
    .read_text()
)


def _deriche_bindings():
    alpha = 0.25
    exp = math.exp
    k = ((1.0 - exp(-alpha)) ** 2) / (
        1.0 + 2.0 * alpha * exp(-alpha) - exp(2.0 * alpha)
    )
    return {
        "a1": k,
        "a2": k * exp(-alpha) * (alpha - 1.0),
        "a3": k * exp(-alpha) * (alpha + 1.0),
        "a4": -k * exp(-2.0 * alpha),
        "a5": k,
        "a6": k * exp(-alpha) * (alpha - 1.0),
        "a7": k * exp(-alpha) * (alpha + 1.0),
        "a8": -k * exp(-2.0 * alpha),
        "b1": 2.0 ** (-alpha),
        "b2": -exp(-2.0 * alpha),
        "c1": 1.0,
        "c2": 1.0,
    }


# name -> (kernel symbol, instantiate dimension order, ambient bindings)
_CASES = {
    "deriche": ("kernel_deriche", ("W", "H"), _deriche_bindings),
    "gemm": ("kernel_gemm", ("P", "Q", "R"), lambda: {"beta": 0.1}),
}


def _summary(name):
    kernel_name, order, bindings = _CASES[name]
    dims = {key: int(value) for key, value in _PSIZE[name]["small"].items()}
    module = importlib.import_module(f"examples.polybench.{name}")
    for key, value in bindings().items():
        setattr(module, key, value)
    instantiate = [float32, *(dims[key] for key in order)]
    schedule = allo.customize(getattr(module, kernel_name), instantiate=instantiate)
    return SimpleNamespace(dims=dims), analyze_upmem_mlir(schedule.module)


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
