# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-device runtime for a realized APU v1 vector plan.

The vector lowering deliberately stops at a reusable GVML function fragment.
This module supplies the GDL task/host harness, translates layout-defined NumPy
images into per-APUC L4 buffers, and gathers the returned VR images.  It is kept
separate from planning and costing so those abstractions remain executable on a
machine without the proprietary SDK.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

from ..spmw_codegen import RunResult
from .apu_v1_program import _generate_project, _ledag_log
from .apu_v1_vector_codegen import PointerOffsetRef, PointerRef, VR_LANES


def _pointer_values(realization):
    return tuple(
        sorted(
            {
                (argument.name, argument.level)
                for instruction in realization.instructions
                for argument in instruction.args
                if isinstance(argument, (PointerRef, PointerOffsetRef))
            }
        )
    )


def _parse_vector_profile(text: str) -> int | None:
    """Return the newest device ``total`` CRUN counter."""

    totals = re.findall(r"\btotal\b[^\n]*?\bcrun\s*[:=]\s*(\d+)", text)
    return int(totals[-1]) if totals else None


def _full_device_source(realization, role_bytes: dict[str, int]) -> str:
    pointers = _pointer_values(realization)
    function = re.sub(r"\W", "_", str(getattr(realization.plan, "name", "apu_vector")))
    declarations = []
    arguments = []
    l3_copies = []
    for name, level in pointers:
        if level == "L4":
            declarations.append(
                f"    uint16_t *{name}_L4ptr = (uint16_t *)"
                f"gal_mem_handle_to_apu_ptr(data->mem_hndl_{name});"
            )
        else:
            nbytes = int(role_bytes[name])
            if nbytes > (1 << 20):
                raise ValueError(
                    f"APU v1 L3 image {name!r} needs {nbytes} bytes; per-APUC L3 is 1 MiB"
                )
            declarations.append(
                f"    uint16_t *{name}_L3ptr = (uint16_t *)"
                f"gal_fast_malloc_cache_aligned({nbytes}, true);"
            )
            declarations.append(
                f"    uint16_t *{name}_L4_source = (uint16_t *)"
                f"gal_mem_handle_to_apu_ptr(data->mem_hndl_{name});"
            )
            l3_copies.append(
                f"    allo_l4_to_l3({name}_L3ptr, {name}_L4_source, {nbytes});"
            )
        arguments.append(f"{name}_{level}ptr")

    return (
        "#include <gsi/libsys/assert.h>\n"
        "#include <gsi/libsys.h>\n"
        "#include <gsi/libgal.h>\n"
        "#include <gsi/gal-fast-funcs.h>\n"
        '#include "struct.h"\n'
        "#include <gsi_device_profiling.h>\n\n"
        + realization.device_source()
        + "\nPROF_VAR(total);\n\n"
        "static void allo_l4_to_l3(void *destination, void *source, uint32_t bytes) {\n"
        "    uint8_t *dst = (uint8_t *)destination;\n"
        "    uint8_t *src = (uint8_t *)source;\n"
        "    gal_fast_l2dma_async_memcpy_init(GAL_L2DMA_APC_ID_0);\n"
        "    for (uint32_t offset = 0; offset < bytes; offset += 512)\n"
        "        gal_fast_l2dma_mem_to_mem_512(dst + offset, src + offset, "
        "GAL_L2DMA_APC_ID_0);\n"
        "    gal_fast_l2dma_async_memcpy_end(GAL_L2DMA_APC_ID_0);\n"
        "}\n\n"
        "static int my_kernel(struct program_data *data) {\n"
        "    arc_counters_init();\n"
        "    PROF_INIT(total);\n"
        + "\n".join(declarations)
        + "\n"
        + "\n".join(l3_copies)
        + "\n    PROF_START(total);\n"
        f"    int ret = {function}_vector({', '.join(arguments)});\n"
        "    PROF_END(total);\n"
        "    PROF_PRINT(total);\n"
        "    return ret;\n"
        "}\n\n"
        "GAL_TASK_ENTRY_POINT(apu_kernel_task, in, out) {\n"
        "    struct program_cmd *cmd = (struct program_cmd *)in;\n"
        "    return my_kernel(&cmd->data);\n"
        "}\n"
    )


def run_apu_v1_vector(compiled, arrays, *, lab_name: str = "tenon-vector"):
    """Build and execute one vector realization on the installed APU v1 board.

    This reference mapping intentionally launches APUC 0, matching the
    MICRO'25 layout study.  Explicit transfer routes compact lookup tables,
    retain RHS banks, and materialize compute replicas on device; ``vr_batch``
    handles outputs larger than one VR.  An ``apuc`` layout dimension is the
    independent multicore extension and does not change this ABI.
    """

    from ..spmw_codegen import _apu_v1_unavailable_reason

    realization = compiled.realization
    if realization is None:
        raise RuntimeError(compiled.realization_error or "plan has no GVML realization")
    reason = _apu_v1_unavailable_reason()
    if reason:
        return RunResult(None, reason, "apu_v1")

    if realization.compute_tiles_per_output > 1:
        for value in realization.values:
            if value.intent in {"out", "inout"} and np.any(arrays[value.name] != 0):
                raise ValueError(
                    "spatial APU group-reduction scatter currently requires a "
                    "zero-initialized output; nonzero inout accumulation has no "
                    "layout-safe group-head ingress"
                )

    output_names = tuple(
        value.name for value in realization.values if value.intent in {"out", "inout"}
    )
    packed = realization.abi.pack(arrays, values=output_names)
    routed_inputs = realization.abi.transfer_input_images(arrays)
    input_images = {}
    output_specs = {}
    output_batches = {}
    for value in realization.values:
        if value.intent == "in":
            input_images[value.name] = routed_inputs[value.name]
        elif value.intent in {"out", "inout"}:
            image = packed[value.name]
            # The current contraction plans initialize the carried output to
            # zero.  Keeping it as the output buffer avoids an unnecessary
            # host round trip and matches the MICRO kernels.
            batches = int(image.data.shape[0])
            output_batches[value.name] = batches
            output_specs[value.name] = ((batches, VR_LANES), np.uint16)

    referenced = {name for name, _level in _pointer_values(realization)}
    absent = referenced - set(input_images) - set(output_specs)
    if absent:
        raise ValueError(
            f"vector device source references unbound roles: {sorted(absent)}"
        )
    role_bytes = {
        **{name: value.nbytes for name, value in input_images.items()},
        **{
            name: math.prod(shape) * np.dtype(dtype).itemsize
            for name, (shape, dtype) in output_specs.items()
        },
    }

    root = tempfile.mkdtemp(prefix="tenon-apu-v1-vector-")
    try:
        project = _generate_project(
            Path(root) / "project",
            _full_device_source(realization, role_bytes),
            input_images,
            output_specs,
            lab_name,
        )
        input_paths = {}
        for role, value in input_images.items():
            path = Path(root) / f"in_{role}.bin"
            value.tofile(path)
            input_paths[role] = path
        output_paths = {role: Path(root) / f"out_{role}.bin" for role in output_specs}
        build = subprocess.run(
            ["make"], cwd=project, capture_output=True, timeout=600, check=False
        )
        if build.returncode:
            raise RuntimeError(
                "APU v1 vector build failed:\n"
                + build.stderr.decode(errors="replace")[-8000:]
            )
        binary = project / "build" / "debug" / lab_name
        argv = [str(binary)]
        argv.extend(str(input_paths[name]) for name in sorted(input_paths))
        argv.extend(str(output_paths[name]) for name in sorted(output_paths))
        process = None
        text = ""
        for attempt in range(3):
            process = subprocess.run(
                argv, cwd=project, capture_output=True, timeout=900, check=False
            )
            text = process.stdout.decode(errors="replace") + process.stderr.decode(
                errors="replace"
            )
            if process.returncode == 0 or "no valid context" not in text:
                break
            time.sleep(attempt + 1)
        if process is None or process.returncode:
            code = None if process is None else process.returncode
            raise RuntimeError(
                f"APU v1 vector device run failed ({code}):\n{text[-6000:]}"
            )

        raw_outputs = {}
        packed_outputs = dict(packed)
        for role, batches in output_batches.items():
            raw = np.fromfile(output_paths[role], dtype=np.uint16).reshape(
                batches, VR_LANES
            )
            raw_outputs[role] = raw
            image = packed_outputs[role]
            data = image.data.copy()
            data[:, 0, :] = raw
            packed_outputs[role] = data
        outputs = realization.abi.gather(packed_outputs, outputs=tuple(output_specs))
        for name, value in outputs.items():
            if name in arrays:
                np.copyto(arrays[name], value, casting="same_kind")
        log = text + "\n" + _ledag_log()
        return RunResult(
            _parse_vector_profile(log),
            log,
            "apu_v1",
            extra={
                "outputs": outputs,
                "raw_outputs": raw_outputs,
                "selected_plan": realization.plan.name,
                "device_source": _full_device_source(realization, role_bytes),
                "single_apuc": True,
            },
        )
    finally:
        if os.environ.get("TENON_APU_V1_KEEP_TMP") != "1":
            shutil.rmtree(root, ignore_errors=True)


__all__ = ["run_apu_v1_vector"]
