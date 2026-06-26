# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""UPMEM FFN bespoke MRAM-selector host port (task 026).

The Exo (256-1024-256) and Cinnamon (64-256-64) FFN reference kernels are
single binaries that read a phase selector word from a fixed MRAM heap offset
and dispatch to one of several leg bodies. The generic TENON drop-slot fills the
heap with random VA-shape data and cannot place that selector word, so which leg
runs is undefined -- the BLOCKED-ON-HARNESS state.

Task 026 ports the bespoke hosts into our uPIMulator tree as two benchmark slots
(EXO_FFN_1PD, CINM_FFN), each with a DPU kernel + a Go assemblable whose data prep
writes the selector word at the kernel's fixed offset so the chosen phase is
pinned deterministically. These tests statically assert that contract -- the
kernel's selector read offset and the host's selector write offset agree, the
slots are registered, and the run mechanism is in place. They do NOT run the
simulator (the verifier re-runs the live cycle measurement); they pin the
harness wiring so a regression is caught without a multi-minute Docker build.

No `allo/spmw_*.py` is touched by this task -- it is simulator-tree driver code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_UPMEM = (
    Path(__file__).resolve().parents[3]
    / "simulators"
    / "uPIMulator"
    / "golang"
    / "uPIMulator"
)
_BENCH = _UPMEM / "benchmark"
_PRIM = _UPMEM / "src" / "assembler" / "prim"

_HAVE_TREE = _UPMEM.is_dir()
pytestmark = pytest.mark.skipif(
    not _HAVE_TREE, reason="uPIMulator tree not present in this checkout"
)


def _read(path: Path) -> str:
    return path.read_text()


# --- slot registration ----------------------------------------------------


def test_assembler_registers_both_ffn_slots():
    src = _read(_PRIM.parent / "assembler.go")
    assert 'this.assemblables["EXO_FFN_1PD"] = new(prim.ExoFfn1Pd)' in src
    assert 'this.assemblables["CINM_FFN"] = new(prim.CinmFfn)' in src


def test_cmake_registers_both_ffn_slots():
    top = _read(_BENCH / "CMakeLists.txt")
    assert "add_subdirectory(EXO_FFN_1PD)" in top
    assert "add_subdirectory(CINM_FFN)" in top
    # The linker locates build/<B>/dpu/CMakeFiles/<B>_device.dir/task.c.o, so the
    # executable target must live in dpu/ and be named <B>_device.
    assert "add_executable(EXO_FFN_1PD_device" in _read(
        _BENCH / "EXO_FFN_1PD" / "dpu" / "CMakeLists.txt"
    )
    assert "add_executable(CINM_FFN_device" in _read(
        _BENCH / "CINM_FFN" / "dpu" / "CMakeLists.txt"
    )


# --- the selector-offset contract: kernel read == host write --------------


def _selector_read_offset(task_c: str) -> int:
    """Byte offset the kernel reads its phase selector from (heap + N)."""
    m = re.search(r"DPU_MRAM_HEAP_POINTER\s*\+\s*(\d+)\s*\)\s*,\s*\(int32_t\s*\*\)\s*sel", task_c)
    if m is None:
        # Exo reads into selbuf via an intermediate `heap` variable.
        m = re.search(r"heap\s*\+\s*(\d+)\s*\)\s*,\s*\(int32_t\s*\*\)\s*selbuf", task_c)
    assert m is not None, "could not locate the selector read offset in the kernel"
    return int(m.group(1))


def _host_selector_offset_words(prim_go: str, const_name: str) -> int:
    m = re.search(rf"{const_name}\s*=\s*(\d+)", prim_go)
    assert m is not None, f"const {const_name} not found"
    return int(m.group(1))


def test_exo_selector_offset_matches():
    task_c = _read(_BENCH / "EXO_FFN_1PD" / "dpu" / "task.c")
    read_off = _selector_read_offset(task_c)
    assert read_off == 32768, "Exo kernel must read the selector at heap+32768"

    prim = _read(_PRIM / "exo_ffn_1pd.go")
    sel_words = _host_selector_offset_words(prim, "exoFfnSelOffsetWords")
    assert sel_words * 4 == read_off, "host write offset must equal kernel read offset"
    # The host must actually write the selector value there and source it from
    # data_prep_params (deterministic phase selection, not random fill).
    assert "img[exoFfnSelOffsetWords] = this.selector" in prim
    assert "this.selector = int64(command_line_parser.DataPrepParams()[0])" in prim


def test_cinm_selector_offset_matches():
    task_c = _read(_BENCH / "CINM_FFN" / "dpu" / "task.c")
    read_off = _selector_read_offset(task_c)
    assert read_off == 4096, "Cinnamon kernel must read the selector at heap+4096"

    prim = _read(_PRIM / "cinm_ffn.go")
    sel_words = _host_selector_offset_words(prim, "cinmFfnSelOffsetWords")
    assert sel_words * 4 == read_off, "host write offset must equal kernel read offset"
    assert "img[cinmFfnSelOffsetWords] = this.selector" in prim
    assert "this.selector = int64(command_line_parser.DataPrepParams()[0])" in prim


# --- host-loader symbol the kernels need to be loadable --------------------


def test_kernels_declare_host_arg_symbol():
    # ChannelTransferInputDpuHost resolves a transfer address via the
    # __host DPU_INPUT_ARGUMENTS symbol; without it the host transfer panics.
    for bench in ("EXO_FFN_1PD", "CINM_FFN"):
        task_c = _read(_BENCH / bench / "dpu" / "task.c")
        assert "__host dpu_arguments_t DPU_INPUT_ARGUMENTS;" in task_c


# --- the image must span through the selector so a single contiguous heap --
# --- stream (loaded at offset 0) actually reaches the selector word --------


def test_image_spans_through_selector():
    exo = _read(_PRIM / "exo_ffn_1pd.go")
    assert "exoFfnImageWords = exoFfnSelOffsetWords + 2" in exo
    cinm = _read(_PRIM / "cinm_ffn.go")
    assert "cinmFfnImageWords     = 1024 + 2" in cinm
