# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the APU v1 build harness (`gen_apu_v1_low_mode_project`).

Verifies project-directory structure and emitted file contents without
invoking the ARC GNU toolchain. Skips cleanly when the example-gvml
template directory is missing -- that's the same skip condition used by
`_apu_v1_unavailable_reason`, restricted to the template piece.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import allo
from allo.spmw_apu_v1_build import (
    _DEFAULT_TEMPLATE_DIR,
    gen_apu_v1_low_mode_project,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import build_apu_v1_target


def _trace() -> MatchTrace:
    return MatchTrace(
        target_name="apu_v1",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0",
                work_id=(0,),
                enclosing_loops=[
                    ("%arg0", "0", "16", 1),
                    ("%arg1", "0", "1024", 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )


_TEMPLATE_DIR = Path(_DEFAULT_TEMPLATE_DIR)


@pytest.mark.skipif(
    not _TEMPLATE_DIR.exists(),
    reason=f"APU v1 template dir missing at {_TEMPLATE_DIR}",
)
def test_emit_project_dir_offline(tmp_path):
    """gen_apu_v1_low_mode_project emits a buildable project tree.

    The ARC toolchain is not invoked; we only assert structure +
    contents of the generated files. This is the spec 017 offline
    guard test.
    """
    target = build_apu_v1_target()
    compiled = allo.compile_for_target(target, _trace())

    inputs = {
        "local_W": np.zeros(_32k(), dtype=np.uint16),
        "local_x": np.zeros(_32k(), dtype=np.uint16),
    }
    output_specs = {"acc": ((_32k(),), np.dtype("uint16"))}

    project_dir = gen_apu_v1_low_mode_project(
        dst_dir=tmp_path / "proj",
        compiled=compiled,
        inputs=inputs,
        output_specs=output_specs,
        lab_name="tenon-kernel",
    )

    # Structure
    assert project_dir.is_dir()
    for required in (
        "Makefile",
        "struct.h",
        "device.c",
        "host.c",
        "Common",
        "gsi_dma.c",
        "gsi_dma.h",
        "gsi_device_profiling.h",
    ):
        assert (project_dir / required).exists(), (
            f"missing {required} in {project_dir}"
        )
    assert (project_dir / "Common").is_dir()

    # Makefile: lab_name threaded through.
    mk_text = (project_dir / "Makefile").read_text()
    assert "lab_name := tenon-kernel" in mk_text
    assert "Common/common.mk" in mk_text

    # struct.h: one field per role, inputs first (sorted alpha) then outputs.
    sh_text = (project_dir / "struct.h").read_text()
    assert "mem_hndl_local_W" in sh_text
    assert "mem_hndl_local_x" in sh_text
    assert "mem_hndl_acc" in sh_text
    # Inputs (alphabetical: local_W < local_x) before outputs (acc).
    pos_w = sh_text.index("mem_hndl_local_W")
    pos_x = sh_text.index("mem_hndl_local_x")
    pos_acc = sh_text.index("mem_hndl_acc")
    assert pos_w < pos_x < pos_acc
    # SPEC-018: inline MAC popcount LUT field in program_data.
    assert "uint16_t mac_lut[256]" in sh_text, sh_text

    # device.c: PROF_VAR(total) + the canonical VR alias fallback +
    # the emitted body lines.
    dev_text = (project_dir / "device.c").read_text()
    assert "PROF_VAR(total)" in dev_text
    assert "PROF_START(total)" in dev_text
    assert "PROF_END(total)" in dev_text
    assert "enum gvml_vr16 vrs = GVML_VR16_0;" in dev_text or (
        "enum gvml_vr16" in dev_text
    )
    # The compiled body's GVML calls must end up inside my_kernel.
    # SPEC-018: the lookup call must carry the canonical LUT pointer
    # name and the 256-entry length, and the decl must be present.
    assert "gvml_lookup_16(" in dev_text, dev_text
    assert "mac_lut_ptr, 256" in dev_text, dev_text
    assert "const uint16_t *mac_lut_ptr" in dev_text, dev_text
    assert "GAL_TASK_ENTRY_POINT(apu_kernel_task" in dev_text

    # host.c: argv-driven file IO + gdl_run_task_timeout.
    host_text = (project_dir / "host.c").read_text()
    assert "GDL_TASK_DECLARE(apu_kernel_task);" in host_text
    assert "gdl_run_task_timeout" in host_text
    # Argv path variables for each role.
    assert "path_local_W" in host_text
    assert "path_local_x" in host_text
    assert "path_acc" in host_text
    # SPEC-018: host.c populates the inline popcount LUT before DMA.
    assert "cmd.data.mac_lut[k]" in host_text, host_text


def _32k() -> int:
    return 32 * 1024


# --------------------------------------------------------------------- #
# SPEC-001 guards
# --------------------------------------------------------------------- #


def test_device_c_does_not_include_libgvml_logical():
    """SPEC-001 §3.2: `_emit_device_c` must not emit `#include
    <gsi/libgvml_logical.h>`; that header is not shipped with the GVML
    release on this host, and the logical bit ops (xor/or/and/not_16)
    live in `libgvml_element_wise.h` which is already included.
    """
    from allo.spmw_apu_v1_build import _emit_device_c

    class _FakeCompiled:
        cmds = ["gvml_xor_16(vrs, vrs, vrs);"]
        layout_ctx = None

    src = _emit_device_c(_FakeCompiled(), in_roles=["x"], out_roles=["y"])
    assert "libgvml_logical.h" not in src, (
        "device.c must not include the non-existent libgvml_logical.h header"
    )
    # The element_wise header (which actually declares the logical ops)
    # must still be present.
    assert "libgvml_element_wise.h" in src


def test_gvml_include_root_default_and_env_override(monkeypatch):
    """SPEC-001 §3.1: `_gvml_include_root` returns the env override when
    set, otherwise the stock `/usr/local/include` path that GSI's
    `Common/common.mk` already adds via $(GSI_USR_LOCAL_INCLUDE).
    """
    from allo.spmw_apu_v1_build import (
        _DEFAULT_GVML_INCLUDE_ROOT,
        _gvml_include_root,
    )

    monkeypatch.delenv("TENON_APU_V1_GVML_INCLUDE_ROOT", raising=False)
    assert _gvml_include_root() == _DEFAULT_GVML_INCLUDE_ROOT

    monkeypatch.setenv("TENON_APU_V1_GVML_INCLUDE_ROOT", "/opt/custom/inc")
    assert _gvml_include_root() == "/opt/custom/inc"


def test_gvml_sdk_probe_raising_and_nonraising(tmp_path, monkeypatch):
    """SPEC-001 §3.1 + task-015: `_assert_gvml_sdk_present` raises
    FileNotFoundError when the canary header is missing;
    `_gvml_sdk_available` returns False in the same condition and True
    when every canary header is present. Task 001 narrowed
    `_GVML_CANARY_HEADERS` to a single entry (libgvml_element_wise.h)
    because libgvml_logical.h is not shipped with the GVML release;
    this test asserts exactly one canary and covers absent + present.
    """
    from allo.spmw_apu_v1_build import (
        _GVML_CANARY_HEADERS,
        _assert_gvml_sdk_present,
        _gvml_sdk_available,
    )

    # Task 001 reduced the canary set to a single header.
    assert len(_GVML_CANARY_HEADERS) == 1, (
        f"expected one canary header post task-001, got "
        f"{_GVML_CANARY_HEADERS!r}"
    )

    # Point the probe at an empty dir -- canary absent.
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    monkeypatch.setenv("TENON_APU_V1_GVML_INCLUDE_ROOT", str(empty_root))
    assert not _gvml_sdk_available()
    with pytest.raises(FileNotFoundError, match="GVML SDK not found"):
        _assert_gvml_sdk_present()

    # Plant every canary and confirm both probes flip.
    for rel in _GVML_CANARY_HEADERS:
        h = empty_root / rel
        h.parent.mkdir(parents=True, exist_ok=True)
        h.write_text("/* fake header for test */\n")
    assert _gvml_sdk_available()
    _assert_gvml_sdk_present()  # must not raise


def test_emit_makefile_unchanged_no_extra_include():
    """SPEC-001 §3.1: the Makefile emission must NOT be extended with a
    `-I<gvml_include_root>` flag. Common.mk already supplies the include
    path through $(GSI_USR_LOCAL_INCLUDE).
    """
    from allo.spmw_apu_v1_build import _emit_makefile

    mk = _emit_makefile("tenon-kernel")
    assert "GNU_TOOLCHAIN_FOR_ARC_BASE" in mk
    assert "lab_name := tenon-kernel" in mk
    assert "Common/common.mk" in mk
    # The spec explicitly forbids adding -I<gvml> to the Tenon Makefile.
    assert "-I" not in mk, (
        "Tenon Makefile must not carry a -I flag; common.mk handles "
        "includes (see SPEC-001 §3.1)."
    )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_emit_project_dir_offline(Path(td))
        print("ALL PASSED")
