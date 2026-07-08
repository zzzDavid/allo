# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent-L4 GDL runner for uint16 APU v1/GVML hybrids.

The first supported physical program is canonical GEMM: four shard-local GVML
realizations execute as one batch and the retained MLIR scalar epilogue runs on
APUC 0.  All ingress images, packed vector results, gather maps, the dense
intermediate, and the final result occupy one stitched L4 allocation for the
whole program.  Host packing is an ingress ABI operation; no intermediate is
copied back to the host.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from .apu_v1_hybrid import realize_apu_v1_region_shards
from .apu_v1_program import _generate_project, _ledag_log
from .apu_v1_vector_codegen import PointerOffsetRef, PointerRef, VR_LANES


@dataclass(frozen=True)
class APUv1HybridProject:
    path: Path
    inputs: dict[str, np.ndarray]
    output_specs: dict[str, tuple[tuple[int, ...], np.dtype]]
    output_roles: dict[str, str]
    device_source: str
    host_source: str
    launch_plan: tuple[tuple[int, int], ...]


def _field(name):
    if not name.isidentifier():
        raise ValueError(f"hybrid L4 role {name!r} is not a C identifier")
    return f"mem_hndl_{name}"


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


def _rename_vector_function(source, old, new):
    marker = f"static int {old}("
    if source.count(marker) != 1:
        raise RuntimeError(f"cannot find unique vector entry {old!r}")
    return source.replace(marker, f"static int {new}(", 1)


def _parse_hybrid_profile(text, launch_plan=((0, 4), (1, 1))):
    """Return critical-path cycles and the newest vector/scalar phase split."""

    records = [
        (int(core), int(cycles))
        for core, cycles in re.findall(
            r"ARCT\[(\d+)\]:.*?\btotal\b[^\n]*?\bcrun:(\d+)", text
        )
    ]
    expected = sum(count for _phase_id, count in launch_plan)
    if len(records) < expected:
        return None, None
    newest = records[-expected:]
    offset = 0
    batches = []
    total = 0
    for phase_id, count in launch_plan:
        batch = newest[offset : offset + count]
        offset += count
        if {core for core, _cycles in batch} != set(range(count)):
            return None, None
        per_apuc = {core: cycles for core, cycles in batch}
        critical = max(per_apuc.values())
        total += critical
        batches.append(
            {"phase_id": phase_id, "per_apuc": per_apuc, "critical": critical}
        )
    if tuple(launch_plan) == ((0, 4), (1, 1)):
        phases = {
            "vector_per_apuc": batches[0]["per_apuc"],
            "vector_critical": batches[0]["critical"],
            "scalar_core0": batches[1]["critical"],
        }
    else:
        phases = {"batches": batches}
    return total, phases


def _emit_struct(roles):
    handles = "\n".join(f"        uint64_t {_field(role)};" for role in roles)
    return f"""#ifndef DATA_STRUCT_H
#define DATA_STRUCT_H
#include <stdint.h>
struct program_data {{
{handles}
        uint32_t phase_id;
        uint32_t core_id;
}} __attribute__((packed));
struct program_cmd {{
        char buffer[64];
        union {{ struct program_data data; }} __attribute__((packed));
}} __attribute__((packed));
#endif
"""


def _emit_host(input_roles, zero_roles, output_roles, role_sizes, launch_plan):
    roles = tuple(dict.fromkeys(input_roles + zero_roles + output_roles))
    argv_roles = tuple(input_roles) + tuple(output_roles)
    argv = "\n".join(
        f"    const char *path_{role} = argv[{index + 1}];"
        for index, role in enumerate(argv_roles)
    )
    sizes = "\n".join(
        f"    const uint64_t sz_{role} = {int(role_sizes[role])}ULL;" for role in roles
    )
    total = " + ".join(f"sz_{role}" for role in roles)
    handles = [
        f"    struct program_cmd base_cmd = {{ .data.{_field(roles[0])} = arena }};"
    ]
    for previous, role in zip(roles, roles[1:]):
        handles.extend(
            [
                f"    gdl_mem_handle_t handle_{role};",
                f"    ret = gdl_add_to_mem_handle(&handle_{role}, "
                f"base_cmd.data.{_field(previous)}, sz_{previous});",
                "    if (ret) goto CLEAN_UP;",
                f"    base_cmd.data.{_field(role)} = handle_{role};",
            ]
        )
    upload = []
    for role in input_roles:
        upload.append(
            f"""    {{
        FILE *file = fopen(path_{role}, "rb");
        if (!file) {{ ret = -1; goto CLEAN_UP; }}
        void *buffer = malloc(sz_{role});
        if (!buffer || fread(buffer, 1, sz_{role}, file) != sz_{role}) {{
            if (buffer) {{ free(buffer); }}
            fclose(file); ret = -1; goto CLEAN_UP;
        }}
        fclose(file);
        ret = gdl_mem_cpy_to_dev(base_cmd.data.{_field(role)}, buffer, sz_{role});
        free(buffer);
        if (ret) goto CLEAN_UP;
    }}"""
        )
    zero = []
    for role in tuple(zero_roles) + tuple(output_roles):
        zero.append(
            f"""    {{
        void *buffer = calloc(1, sz_{role});
        if (!buffer) {{ ret = gsi_status(ENOMEM); goto CLEAN_UP; }}
        ret = gdl_mem_cpy_to_dev(base_cmd.data.{_field(role)}, buffer, sz_{role});
        free(buffer);
        if (ret) goto CLEAN_UP;
    }}"""
        )
    download = []
    for role in output_roles:
        download.append(
            f"""    {{
        void *buffer = malloc(sz_{role});
        if (!buffer) {{ ret = gsi_status(ENOMEM); goto CLEAN_UP; }}
        ret = gdl_mem_cpy_from_dev(buffer, base_cmd.data.{_field(role)}, sz_{role});
        if (ret) {{ free(buffer); goto CLEAN_UP; }}
        FILE *file = fopen(path_{role}, "wb");
        if (!file || fwrite(buffer, 1, sz_{role}, file) != sz_{role}) {{
            if (file) {{ fclose(file); }}
            free(buffer); ret = -1; goto CLEAN_UP;
        }}
        fclose(file); free(buffer);
    }}"""
        )
    launches = []
    for phase_id, apuc_count in launch_plan:
        launches.extend(
            [
                f"    ret = schedule_phase(ctx, base_cmd, {phase_id}, "
                f"{apuc_count}, command_buffers);",
                "    if (ret) goto CLEAN_UP;",
            ]
        )
    return f"""#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <gsi/libgdl.h>
#include <gsi/libsys.h>
#include <gsi/gsi_sim_config.h>
GDL_TASK_DECLARE(apu_kernel_task);
#include "struct.h"
enum {{ NUM_APUC = 4 }};

static int schedule_phase(gdl_context_handle_t ctx, struct program_cmd base,
                          uint32_t phase_id, uint32_t count,
                          gdl_mem_handle_t command_buffers[NUM_APUC]) {{
    int ret = 0;
    struct program_cmd commands[NUM_APUC];
    struct gsi_task_desc tasks[NUM_APUC];
    memset(tasks, 0, sizeof(tasks));
    for (uint32_t core = 0; core < count; ++core) {{
        commands[core] = base;
        commands[core].data.phase_id = phase_id;
        commands[core].data.core_id = core;
        ret = gdl_mem_cpy_to_dev(command_buffers[core], &commands[core],
                                 sizeof(struct program_cmd));
        if (ret) return ret;
        if (GSI_IS_ERR_PTR_OR_NULL(gdl_task_desc_init(
                ctx, &tasks[core], GDL_TASK(apu_kernel_task),
                command_buffers[core], GDL_MEM_HANDLE_NULL, 0, core))) return -1;
    }}
    return gdl_schedule_batch_timeout(
        tasks, count, GDL_TEMPORARY_DEFAULT_MEM_BUF,
        GDL_TEMPORARY_DEFAULT_MEM_BUF_SIZE, NULL, 0, GDL_USER_MAPPING);
}}

static int run_hybrid(gdl_context_handle_t ctx, int argc, char *argv[]) {{
    int ret = 0;
    (void)argc;
    gdl_mem_handle_t arena = GDL_MEM_HANDLE_NULL;
    gdl_mem_handle_t command_buffers[NUM_APUC] = {{0}};
{argv}
{sizes}
    const uint64_t arena_size = {total};
    arena = gdl_mem_alloc_aligned(ctx, arena_size, GDL_CONST_MAPPED_POOL,
                                  GDL_ALIGN_32);
    if (gdl_mem_handle_is_null(arena)) {{ ret = gsi_status(ENOMEM); goto CLEAN_UP; }}
{chr(10).join(handles)}
{chr(10).join(upload)}
{chr(10).join(zero)}
    for (uint32_t core = 0; core < NUM_APUC; ++core) {{
        command_buffers[core] = gdl_mem_alloc_aligned(
            ctx, sizeof(struct program_cmd), GDL_CONST_MAPPED_POOL, GDL_ALIGN_32);
        if (gdl_mem_handle_is_null(command_buffers[core])) {{
            ret = gsi_status(ENOMEM); goto CLEAN_UP;
        }}
    }}
{chr(10).join(launches)}
{chr(10).join(download)}
CLEAN_UP:
    for (uint32_t core = 0; core < NUM_APUC; ++core)
        if (!gdl_mem_handle_is_null(command_buffers[core]))
            gdl_mem_free(command_buffers[core]);
    if (!gdl_mem_handle_is_null(arena)) gdl_mem_free(arena);
    return ret;
}}

enum {{ NUM_CTXS = 1 }};
static struct gsi_sim_contexts contexts[NUM_CTXS] = {{
    {{ .apu_count = 1, .apucs_per_apu = 4, .mem_size = 0x40000000 }}
}};
int main(int argc, char *argv[]) {{
    uint32_t count = 0;
    struct gdl_context_desc descriptions[GDL_MAX_NUM_CONTEXTS];
    int ret = gsi_libsys_init("tenon hybrid", true);
    if (ret) return ret;
    gsi_sim_create_simulator(NUM_CTXS, contexts);
    if ((ret = gdl_init())) return ret;
    if ((ret = gdl_context_count_get(&count))) return ret;
    if ((ret = gdl_context_desc_get(descriptions, count))) return ret;
    gdl_context_handle_t context = 0;
    uint32_t index = 0;
    for (; index < count; ++index)
        if (descriptions[index].status == GDL_CONTEXT_READY) {{
            context = descriptions[index].ctx_id; break;
        }}
    if (index == count) gsi_fatal("no valid context");
    long long unsigned int constant_size = 0, dynamic_size = 0;
    ret = gdl_context_alloc(context, 3LL * 1024LL * 1024LL * 1024LL,
                            &constant_size, &dynamic_size);
    if (!ret) ret = run_hybrid(context, argc, argv);
    gdl_context_free(context); gdl_exit(); gsi_libsys_exit();
    return ret;
}}
"""


def _pointer_cast(dtype, shape, expression):
    ctype = {
        "ui16": "uint16_t",
        "i16": "int16_t",
        "f32": "float",
        # GVML transports f16 as its raw 16-bit representation.
        "f16": "uint16_t",
    }[dtype]
    if len(shape) == 1:
        return f"({ctype} *){expression}"
    tail = "".join(f"[{extent}]" for extent in shape[1:])
    return f"({ctype} (*){tail}){expression}"


def _emit_device(physical, scalar_region, shards, role_by_shard, dense_roles):
    fragments = []
    calls = []
    for item in shards:
        artifact = item.artifact
        old = re.sub(r"\W", "_", artifact.plan.name) + "_vector"
        new = f"{old}_apuc{item.shard.apuc}"
        fragments.append(_rename_vector_function(artifact.device_source(), old, new))
        arguments = []
        for name, level in _pointer_values(artifact):
            if level != "L4":
                raise NotImplementedError(
                    "hybrid runner currently requires shard-local L4 vector pointers"
                )
            role = role_by_shard[item.shard.apuc][name]
            arguments.append(
                f"(uint16_t *)gal_mem_handle_to_apu_ptr(data->{_field(role)})"
            )
        calls.append(
            f"        case {item.shard.apuc}: return {new}({', '.join(arguments)});"
        )

    scalar_source = scalar_region.executable.device_source
    scalar_function = scalar_region.executable.device_function
    bindings = dict(scalar_region.bindings)
    scalar_arguments = []
    # The physical binding order is retained from the MLIR function ABI.
    logical_region = next(
        region
        for phase in physical.logical.phases
        for region in phase.regions
        if region.id == scalar_region.name
    )
    for operand in logical_region.operands:
        role = bindings[operand.value]
        scalar_arguments.append(
            _pointer_cast(
                operand.storage_dtype,
                operand.shape,
                f"gal_mem_handle_to_apu_ptr(data->{_field(role)})",
            )
        )

    gather = []
    for item in shards:
        roles = role_by_shard[item.shard.apuc]
        rows = item.shard.extent
        columns = item.analysis.output.shape[1]
        global_offset = item.shard.start * columns
        gather.append(
            f"""    {{
        uint16_t *packed = (uint16_t *)gal_mem_handle_to_apu_ptr(
            data->{_field(roles[item.analysis.output.value])});
        uint32_t *map = (uint32_t *)gal_mem_handle_to_apu_ptr(
            data->{_field(roles['__map__'])});
        for (uint32_t index = 0; index < {rows * columns}; ++index)
            dense[{global_offset} + index] = packed[map[index]];
    }}"""
        )
    return f"""#include <stdint.h>
#include <stddef.h>
#include <gsi/libsys.h>
#include <gsi/libgal.h>
#include <gsi_device_profiling.h>
#include "struct.h"
{chr(10).join(fragments)}
{scalar_source}
PROF_VAR(total);
static int run_vector(struct program_data *data) {{
    switch (data->core_id) {{
{chr(10).join(calls)}
        default: return -1;
    }}
}}
static int run_scalar(struct program_data *data) {{
    if (data->core_id != 0) return 0;
    uint16_t *dense = (uint16_t *)gal_mem_handle_to_apu_ptr(
        data->{_field(dense_roles['intermediate'])});
{chr(10).join(gather)}
    {scalar_function}({', '.join(scalar_arguments)});
    return 0;
}}
GAL_TASK_ENTRY_POINT(apu_kernel_task, in, out) {{
    struct program_cmd *command = (struct program_cmd *)in;
    arc_counters_init(); PROF_INIT(total); PROF_START(total);
    int ret = command->data.phase_id == 0
        ? run_vector(&command->data) : run_scalar(&command->data);
    PROF_END(total); PROF_PRINT(total); return ret;
}}
"""


def emit_apu_v1_hybrid_gemm_project(
    compiled, inputs, dst, *, lab_name="tenon-hybrid-gemm"
):
    """Emit the buildable, persistent-L4 project for canonical hybrid GEMM."""

    program = getattr(compiled, "compiled", compiled)
    hybrid = (
        compiled
        if hasattr(compiled, "physical_manifest")
        else getattr(program, "hybrid_callable", None)
    )
    if hybrid is None:
        raise ValueError("compiled program has no physical hybrid callable")
    physical = hybrid.physical_manifest
    logical_regions = [
        region for phase in physical.logical.phases for region in phase.regions
    ]
    vector_regions = [region for region in logical_regions if region.kind == "vector"]
    scalar_regions = [region for region in logical_regions if region.kind == "scalar"]
    if len(vector_regions) != 1 or len(scalar_regions) != 1:
        raise NotImplementedError("physical runner milestone currently supports GEMM")
    vector_region = vector_regions[0]
    vector_phase = next(
        phase for phase in physical.phases if phase.name == vector_region.id
    )
    scalar_phase = next(
        phase for phase in physical.phases if phase.name == scalar_regions[0].id
    )
    shards = realize_apu_v1_region_shards(
        vector_region, vector_phase.executable.partition
    )

    ingress = {}
    zero_specs = {}
    role_sizes = {}
    role_by_shard = {}
    output_value = vector_region.compute_analysis.output.value
    for item in shards:
        arrays = {}
        for value in item.artifact.values:
            if value.intent in {"out", "inout"}:
                arrays[value.name] = np.zeros(value.shape, dtype=value.dtype)
            else:
                source = np.asarray(inputs[value.name])[item.value_slice(value.name)]
                arrays[value.name] = np.asarray(source, dtype=value.dtype)
        routed = item.artifact.abi.transfer_input_images(arrays)
        roles = {}
        for name, image in routed.items():
            role = f"region0_apuc{item.shard.apuc}_{name}"
            ingress[role] = np.ascontiguousarray(image)
            role_sizes[role] = ingress[role].nbytes
            roles[name] = role
        packed = item.artifact.abi.pack(arrays, values=(output_value,))[output_value]
        raw_output = np.ascontiguousarray(packed.data[:, 0, :])
        output_role = f"region0_apuc{item.shard.apuc}_{output_value}"
        zero_specs[output_role] = (raw_output.shape, raw_output.dtype)
        role_sizes[output_role] = raw_output.nbytes
        roles[output_value] = output_role
        value = next(v for v in item.artifact.values if v.name == output_value)
        mapping = np.empty(math.prod(value.shape), dtype=np.uint32)
        for linear, index in enumerate(np.ndindex(value.shape)):
            coordinates = item.artifact.abi._coordinates(value, index)
            if len(coordinates) != 1 or coordinates[0][1] != 0:
                raise NotImplementedError("hybrid gather needs one APUC-local owner")
            batch, _apuc, lane = coordinates[0]
            mapping[linear] = batch * VR_LANES + lane
        map_role = f"region0_apuc{item.shard.apuc}_map"
        ingress[map_role] = mapping
        role_sizes[map_role] = mapping.nbytes
        roles["__map__"] = map_role
        role_by_shard[item.shard.apuc] = roles

    dense_roles = {"intermediate": output_value, "output": "output"}
    ingress["C"] = np.ascontiguousarray(inputs["C"], dtype=np.uint16)
    role_sizes["C"] = ingress["C"].nbytes
    dense_shape = tuple(vector_region.compute_analysis.output.shape)
    zero_specs[output_value] = (dense_shape, np.dtype(np.uint16))
    zero_specs["output"] = (dense_shape, np.dtype(np.uint16))
    role_sizes[output_value] = math.prod(dense_shape) * np.dtype(np.uint16).itemsize
    role_sizes["output"] = role_sizes[output_value]

    input_roles = tuple(sorted(ingress))
    zero_roles = tuple(sorted(set(zero_specs) - {"output"}))
    output_roles = ("output",)
    roles = tuple(dict.fromkeys(input_roles + zero_roles + output_roles))
    host = _emit_host(
        input_roles, zero_roles, output_roles, role_sizes, ((0, 4), (1, 1))
    )
    device = _emit_device(physical, scalar_phase, shards, role_by_shard, dense_roles)
    project = _generate_project(
        dst,
        device,
        ingress,
        {"output": zero_specs["output"]},
        lab_name,
        host_source=host,
        struct_source=_emit_struct(roles),
    )
    return APUv1HybridProject(
        Path(project),
        ingress,
        {"output": zero_specs["output"]},
        {"output": "output"},
        device,
        host,
        ((0, 4), (1, 1)),
    )


def _prepare_vector_region(region, physical_phase, inputs, ingress, zero_specs, sizes):
    shards = realize_apu_v1_region_shards(region, physical_phase.executable.partition)
    roles_by_apuc = {}
    output_name = region.compute_analysis.output.value
    for item in shards:
        arrays = {}
        for value in item.artifact.values:
            if value.intent in {"out", "inout"}:
                arrays[value.name] = np.zeros(value.shape, dtype=value.dtype)
            elif value.name in inputs:
                source = np.asarray(inputs[value.name])[item.value_slice(value.name)]
                arrays[value.name] = np.asarray(source, dtype=value.dtype)
            else:
                arrays[value.name] = np.zeros(value.shape, dtype=value.dtype)
        routed = item.artifact.abi.transfer_input_images(arrays)
        roles = {}
        for name, image in routed.items():
            role = f"{region.id}_apuc{item.shard.apuc}_{name}"
            image = np.ascontiguousarray(image)
            if name in inputs:
                ingress[role] = image
            else:
                zero_specs[role] = (image.shape, image.dtype)
            sizes[role] = image.nbytes
            roles[name] = role
        packed = item.artifact.abi.pack(arrays, values=(output_name,))[output_name]
        raw_output = np.ascontiguousarray(packed.data[:, 0, :])
        output_role = f"{region.id}_apuc{item.shard.apuc}_{output_name}"
        zero_specs[output_role] = (raw_output.shape, raw_output.dtype)
        sizes[output_role] = raw_output.nbytes
        roles[output_name] = output_role
        output_value = next(v for v in item.artifact.values if v.name == output_name)
        mapping = np.empty(math.prod(output_value.shape), dtype=np.uint32)
        for linear, index in enumerate(np.ndindex(output_value.shape)):
            coordinates = item.artifact.abi._coordinates(output_value, index)
            if len(coordinates) != 1 or coordinates[0][1] != 0:
                raise NotImplementedError("hybrid gather needs one APUC-local owner")
            batch, _apuc, lane = coordinates[0]
            mapping[linear] = batch * VR_LANES + lane
        map_role = f"{region.id}_apuc{item.shard.apuc}_gather_map"
        ingress[map_role] = mapping
        sizes[map_role] = mapping.nbytes
        roles["__map__"] = map_role
        roles_by_apuc[item.shard.apuc] = roles
    return shards, roles_by_apuc


def _transition_table(source_item, destination_item, value_name):
    """Map every valid destination replica to its packed source coordinate."""

    destination_value = next(
        value for value in destination_item.artifact.values if value.name == value_name
    )
    count = math.prod(destination_value.shape)
    if count >= np.iinfo(np.uint16).max:
        raise NotImplementedError("transition tag carrier exceeds uint16")
    tag_bits = np.arange(1, count + 1, dtype=np.uint16).reshape(destination_value.shape)
    arrays = {
        value.name: (
            tag_bits.astype(value.dtype, copy=False)
            if value.name == value_name
            else np.zeros(value.shape, dtype=value.dtype)
        )
        for value in destination_item.artifact.values
    }
    destination = destination_item.artifact.abi.transfer_input_images(arrays)[
        value_name
    ].reshape(-1)
    source_value = next(
        value for value in source_item.artifact.values if value.name == value_name
    )
    entries = []
    for destination_offset in np.flatnonzero(destination):
        logical_linear = int(destination[destination_offset]) - 1
        logical_index = np.unravel_index(logical_linear, destination_value.shape)
        coordinates = source_item.artifact.abi._coordinates(source_value, logical_index)
        if len(coordinates) != 1 or coordinates[0][1] != 0:
            raise NotImplementedError("transition source needs one APUC-local owner")
        batch, _apuc, lane = coordinates[0]
        entries.append(
            (source_item.shard.apuc, batch * VR_LANES + lane, destination_offset)
        )
    return np.asarray(entries, dtype=np.uint32)


def _vector_device_fragments(region, shards, roles_by_apuc):
    fragments, calls = [], []
    for item in shards:
        artifact = item.artifact
        old = re.sub(r"\W", "_", artifact.plan.name) + "_vector"
        new = f"{region.id}_{old}_apuc{item.shard.apuc}"
        fragments.append(_rename_vector_function(artifact.device_source(), old, new))
        arguments = []
        for name, level in _pointer_values(artifact):
            if level != "L4":
                raise NotImplementedError(
                    "hybrid runner currently requires shard-local L4 vector pointers"
                )
            role = roles_by_apuc[item.shard.apuc][name]
            arguments.append(
                f"(uint16_t *)gal_mem_handle_to_apu_ptr(data->{_field(role)})"
            )
        calls.append(
            f"        case {item.shard.apuc}: return {new}({', '.join(arguments)});"
        )
    function = (
        f"static int run_{region.id}(struct program_data *data) {{\n"
        "    switch (data->core_id) {\n"
        + "\n".join(calls)
        + "\n        default: return -1;\n    }\n}\n"
    )
    return fragments, function


def _emit_two_mm_device(
    physical,
    regions,
    scalar_phase,
    shard_sets,
    roles_by_region,
    transition_roles,
):
    fragments, vector_functions = [], []
    for region in regions:
        source, function = _vector_device_fragments(
            region, shard_sets[region.id], roles_by_region[region.id]
        )
        fragments.extend(source)
        vector_functions.append(function)

    source_region, destination_region = regions
    repack_cases = []
    for apuc in range(4):
        transition = transition_roles[apuc]
        source_role = roles_by_region[source_region.id][apuc][
            source_region.compute_analysis.output.value
        ]
        destination_role = roles_by_region[destination_region.id][apuc][
            source_region.compute_analysis.output.value
        ]
        repack_cases.append(
            f"""        case {apuc}: {{
            uint16_t *source = (uint16_t *)gal_mem_handle_to_apu_ptr(
                data->{_field(source_role)});
            uint16_t *destination = (uint16_t *)gal_mem_handle_to_apu_ptr(
                data->{_field(destination_role)});
            uint32_t (*map)[3] = (uint32_t (*)[3])gal_mem_handle_to_apu_ptr(
                data->{_field(transition['role'])});
            for (uint32_t index = 0; index < {transition['count']}; ++index)
                destination[map[index][2]] = source[map[index][1]];
            return 0;
        }}"""
        )

    final_region = destination_region
    final_shards = shard_sets[final_region.id]
    final_roles = roles_by_region[final_region.id]
    dense_name = final_region.compute_analysis.output.value
    gather = []
    for item in final_shards:
        roles = final_roles[item.shard.apuc]
        rows, columns = item.analysis.output.shape
        global_offset = item.shard.start * columns
        gather.append(
            f"""    {{
        uint16_t *packed = (uint16_t *)gal_mem_handle_to_apu_ptr(
            data->{_field(roles[dense_name])});
        uint32_t *map = (uint32_t *)gal_mem_handle_to_apu_ptr(
            data->{_field(roles['__map__'])});
        for (uint32_t index = 0; index < {rows * columns}; ++index)
            dense[{global_offset} + index] = packed[map[index]];
    }}"""
        )
    scalar_logical = next(
        region
        for phase in physical.logical.phases
        for region in phase.regions
        if region.id == scalar_phase.name
    )
    bindings = dict(scalar_phase.bindings)
    scalar_arguments = [
        _pointer_cast(
            operand.storage_dtype,
            operand.shape,
            f"gal_mem_handle_to_apu_ptr(data->{_field(bindings[operand.value])})",
        )
        for operand in scalar_logical.operands
    ]
    scalar_source = scalar_phase.executable.device_source
    scalar_function = scalar_phase.executable.device_function
    return f"""#include <stdint.h>
#include <stddef.h>
#include <gsi/libsys.h>
#include <gsi/libgal.h>
#include <gsi_device_profiling.h>
#include "struct.h"
{chr(10).join(fragments)}
{scalar_source}
{chr(10).join(vector_functions)}
static int run_repack(struct program_data *data) {{
    switch (data->core_id) {{
{chr(10).join(repack_cases)}
        default: return -1;
    }}
}}
static int run_final(struct program_data *data) {{
    if (data->core_id != 0) return 0;
    uint16_t *dense = (uint16_t *)gal_mem_handle_to_apu_ptr(
        data->{_field(dense_name)});
{chr(10).join(gather)}
    {scalar_function}({', '.join(scalar_arguments)});
    return 0;
}}
PROF_VAR(total);
GAL_TASK_ENTRY_POINT(apu_kernel_task, in, out) {{
    struct program_cmd *command = (struct program_cmd *)in;
    arc_counters_init(); PROF_INIT(total); PROF_START(total);
    int ret;
    switch (command->data.phase_id) {{
        case 0: ret = run_{source_region.id}(&command->data); break;
        case 1: ret = run_repack(&command->data); break;
        case 2: ret = run_{destination_region.id}(&command->data); break;
        case 3: ret = run_final(&command->data); break;
        default: ret = -1;
    }}
    PROF_END(total); PROF_PRINT(total); return ret;
}}
"""


def emit_apu_v1_hybrid_two_mm_project(
    compiled, inputs, dst, *, lab_name="tenon-hybrid-2mm"
):
    """Emit two vector batches with an in-L4 packed transition and epilogue."""

    program = getattr(compiled, "compiled", compiled)
    hybrid = (
        compiled
        if hasattr(compiled, "physical_manifest")
        else getattr(program, "hybrid_callable", None)
    )
    if hybrid is None:
        raise ValueError("compiled program has no physical hybrid callable")
    physical = hybrid.physical_manifest
    regions = [region for phase in physical.logical.phases for region in phase.regions]
    vector_regions = [region for region in regions if region.kind == "vector"]
    scalar_regions = [region for region in regions if region.kind == "scalar"]
    if len(vector_regions) != 2 or len(scalar_regions) != 1:
        raise NotImplementedError("2mm runner requires vector/vector/scalar topology")
    physical_by_name = {phase.name: phase for phase in physical.phases}
    ingress, zero_specs, sizes = {}, {}, {}
    shard_sets, roles_by_region = {}, {}
    for region in vector_regions:
        shards, roles = _prepare_vector_region(
            region,
            physical_by_name[region.id],
            inputs,
            ingress,
            zero_specs,
            sizes,
        )
        shard_sets[region.id] = shards
        roles_by_region[region.id] = roles

    source_region, destination_region = vector_regions
    transition_value = source_region.compute_analysis.output.value
    transition_roles = {}
    for apuc, (source_item, destination_item) in enumerate(
        zip(shard_sets[source_region.id], shard_sets[destination_region.id])
    ):
        if source_item.shard != destination_item.shard:
            raise NotImplementedError("2mm transition requires identical row shards")
        table = _transition_table(source_item, destination_item, transition_value)
        role = f"{source_region.id}_to_{destination_region.id}_apuc{apuc}_map"
        ingress[role] = table
        sizes[role] = table.nbytes
        transition_roles[apuc] = {"role": role, "count": len(table)}

    final_region = destination_region
    dense_name = final_region.compute_analysis.output.value
    dense_shape = tuple(final_region.compute_analysis.output.shape)
    zero_specs[dense_name] = (dense_shape, np.dtype(np.uint16))
    sizes[dense_name] = math.prod(dense_shape) * np.dtype(np.uint16).itemsize
    for name in ("D",):
        ingress[name] = np.ascontiguousarray(inputs[name], dtype=np.uint16)
        sizes[name] = ingress[name].nbytes
    zero_specs["output"] = (dense_shape, np.dtype(np.uint16))
    sizes["output"] = sizes[dense_name]

    input_roles = tuple(sorted(ingress))
    zero_roles = tuple(sorted(set(zero_specs) - {"output"}))
    output_roles = ("output",)
    all_roles = tuple(dict.fromkeys(input_roles + zero_roles + output_roles))
    launch_plan = ((0, 4), (1, 4), (2, 4), (3, 1))
    host = _emit_host(input_roles, zero_roles, output_roles, sizes, launch_plan)
    scalar_phase = physical_by_name[scalar_regions[0].id]
    device = _emit_two_mm_device(
        physical,
        vector_regions,
        scalar_phase,
        shard_sets,
        roles_by_region,
        transition_roles,
    )
    project = _generate_project(
        dst,
        device,
        ingress,
        {"output": zero_specs["output"]},
        lab_name,
        host_source=host,
        struct_source=_emit_struct(all_roles),
    )
    return APUv1HybridProject(
        Path(project),
        ingress,
        {"output": zero_specs["output"]},
        {"output": "output"},
        device,
        host,
        launch_plan,
    )


def _global_transition_table(source_items, destination_item, value_name):
    """Map a replicated destination image to all producer row shards."""

    destination_value = next(
        value for value in destination_item.artifact.values if value.name == value_name
    )
    count = math.prod(destination_value.shape)
    if count >= np.iinfo(np.uint16).max:
        raise NotImplementedError("transition tag carrier exceeds uint16")
    tag_bits = np.arange(1, count + 1, dtype=np.uint16).reshape(destination_value.shape)
    arrays = {
        value.name: (
            tag_bits.astype(value.dtype, copy=False)
            if value.name == value_name
            else np.zeros(value.shape, dtype=value.dtype)
        )
        for value in destination_item.artifact.values
    }
    destination = destination_item.artifact.abi.transfer_input_images(arrays)[
        value_name
    ].reshape(-1)
    entries = []
    for destination_offset in np.flatnonzero(destination):
        logical_linear = int(destination[destination_offset]) - 1
        global_index = np.unravel_index(logical_linear, destination_value.shape)
        source_item = next(
            item
            for item in source_items
            if item.shard.start <= global_index[0] < item.shard.stop
        )
        local_index = (global_index[0] - source_item.shard.start,) + global_index[1:]
        source_value = next(
            value for value in source_item.artifact.values if value.name == value_name
        )
        coordinates = source_item.artifact.abi._coordinates(source_value, local_index)
        if len(coordinates) != 1 or coordinates[0][1] != 0:
            raise NotImplementedError("transition source needs one APUC-local owner")
        batch, _apuc, lane = coordinates[0]
        entries.append(
            (
                source_item.shard.apuc,
                batch * VR_LANES + lane,
                destination_offset,
            )
        )
    return np.asarray(entries, dtype=np.uint32)


def _emit_three_mm_device(
    regions, shard_sets, roles_by_region, transition_roles, output_role
):
    fragments, vector_functions = [], []
    for region in regions:
        source, function = _vector_device_fragments(
            region, shard_sets[region.id], roles_by_region[region.id]
        )
        fragments.extend(source)
        vector_functions.append(function)

    consumer = regions[2]
    repack_cases = []
    for apuc in range(4):
        edge_blocks = []
        for edge_index, producer in enumerate(regions[:2]):
            value_name = producer.compute_analysis.output.value
            transition = transition_roles[(edge_index, apuc)]
            destination_role = roles_by_region[consumer.id][apuc][value_name]
            source_roles = roles_by_region[producer.id]
            source_switch = " ".join(
                f"case {source_apuc}: value = ((uint16_t *)"
                f"gal_mem_handle_to_apu_ptr(data->"
                f"{_field(source_roles[source_apuc][value_name])}))[map[index][1]]; break;"
                for source_apuc in range(4)
            )
            edge_blocks.append(
                f"""        {{
            uint16_t *destination = (uint16_t *)gal_mem_handle_to_apu_ptr(
                data->{_field(destination_role)});
            uint32_t (*map)[3] = (uint32_t (*)[3])gal_mem_handle_to_apu_ptr(
                data->{_field(transition['role'])});
            for (uint32_t index = 0; index < {transition['count']}; ++index) {{
                uint16_t value = 0;
                switch (map[index][0]) {{ {source_switch} default: return -1; }}
                destination[map[index][2]] = value;
            }}
        }}"""
            )
        repack_cases.append(
            f"        case {apuc}: {{\n"
            + "\n".join(edge_blocks)
            + "\n            return 0;\n        }"
        )

    output_name = consumer.compute_analysis.output.value
    gather = []
    for item in shard_sets[consumer.id]:
        roles = roles_by_region[consumer.id][item.shard.apuc]
        rows, columns = item.analysis.output.shape
        global_offset = item.shard.start * columns
        gather.append(
            f"""    {{
        uint16_t *packed = (uint16_t *)gal_mem_handle_to_apu_ptr(
            data->{_field(roles[output_name])});
        uint32_t *map = (uint32_t *)gal_mem_handle_to_apu_ptr(
            data->{_field(roles['__map__'])});
        for (uint32_t index = 0; index < {rows * columns}; ++index)
            output[{global_offset} + index] = packed[map[index]];
    }}"""
        )
    return f"""#include <stdint.h>
#include <stddef.h>
#include <gsi/libsys.h>
#include <gsi/libgal.h>
#include <gsi_device_profiling.h>
#include "struct.h"
{chr(10).join(fragments)}
{chr(10).join(vector_functions)}
static int run_repack_join(struct program_data *data) {{
    switch (data->core_id) {{
{chr(10).join(repack_cases)}
        default: return -1;
    }}
}}
static int run_final_gather(struct program_data *data) {{
    if (data->core_id != 0) return 0;
    uint16_t *output = (uint16_t *)gal_mem_handle_to_apu_ptr(
        data->{_field(output_role)});
{chr(10).join(gather)}
    return 0;
}}
PROF_VAR(total);
GAL_TASK_ENTRY_POINT(apu_kernel_task, in, out) {{
    struct program_cmd *command = (struct program_cmd *)in;
    arc_counters_init(); PROF_INIT(total); PROF_START(total);
    int ret;
    switch (command->data.phase_id) {{
        case 0: ret = run_{regions[0].id}(&command->data); break;
        case 1: ret = run_{regions[1].id}(&command->data); break;
        case 2: ret = run_repack_join(&command->data); break;
        case 3: ret = run_{regions[2].id}(&command->data); break;
        case 4: ret = run_final_gather(&command->data); break;
        default: ret = -1;
    }}
    PROF_END(total); PROF_PRINT(total); return ret;
}}
"""


def emit_apu_v1_hybrid_three_mm_project(
    compiled, inputs, dst, *, lab_name="tenon-hybrid-3mm"
):
    """Emit three vector batches, an in-L4 join repack, and final gather."""

    program = getattr(compiled, "compiled", compiled)
    hybrid = (
        compiled
        if hasattr(compiled, "physical_manifest")
        else getattr(program, "hybrid_callable", None)
    )
    if hybrid is None:
        raise ValueError("compiled program has no physical hybrid callable")
    physical = hybrid.physical_manifest
    regions = [region for phase in physical.logical.phases for region in phase.regions]
    vector_regions = [region for region in regions if region.kind == "vector"]
    if len(vector_regions) != 3 or any(region.kind == "scalar" for region in regions):
        raise NotImplementedError("3mm runner requires three vector regions")
    physical_by_name = {phase.name: phase for phase in physical.phases}
    ingress, zero_specs, sizes = {}, {}, {}
    shard_sets, roles_by_region = {}, {}
    for region in vector_regions:
        shards, roles = _prepare_vector_region(
            region,
            physical_by_name[region.id],
            inputs,
            ingress,
            zero_specs,
            sizes,
        )
        shard_sets[region.id] = shards
        roles_by_region[region.id] = roles

    consumer = vector_regions[2]
    transition_roles = {}
    for edge_index, producer in enumerate(vector_regions[:2]):
        value_name = producer.compute_analysis.output.value
        for apuc, destination_item in enumerate(shard_sets[consumer.id]):
            if value_name == consumer.compute_analysis.lhs.value:
                table = _transition_table(
                    shard_sets[producer.id][apuc], destination_item, value_name
                )
            else:
                table = _global_transition_table(
                    shard_sets[producer.id], destination_item, value_name
                )
            role = f"{producer.id}_to_{consumer.id}_apuc{apuc}_map"
            ingress[role] = table
            sizes[role] = table.nbytes
            transition_roles[(edge_index, apuc)] = {"role": role, "count": len(table)}

    output_shape = tuple(consumer.compute_analysis.output.shape)
    zero_specs["output"] = (output_shape, np.dtype(np.uint16))
    sizes["output"] = math.prod(output_shape) * np.dtype(np.uint16).itemsize
    input_roles = tuple(sorted(ingress))
    zero_roles = tuple(sorted(set(zero_specs) - {"output"}))
    output_roles = ("output",)
    all_roles = tuple(dict.fromkeys(input_roles + zero_roles + output_roles))
    launch_plan = ((0, 4), (1, 4), (2, 4), (3, 4), (4, 1))
    host = _emit_host(input_roles, zero_roles, output_roles, sizes, launch_plan)
    device = _emit_three_mm_device(
        vector_regions,
        shard_sets,
        roles_by_region,
        transition_roles,
        "output",
    )
    project = _generate_project(
        dst,
        device,
        ingress,
        {"output": zero_specs["output"]},
        lab_name,
        host_source=host,
        struct_source=_emit_struct(all_roles),
    )
    return APUv1HybridProject(
        Path(project),
        ingress,
        {"output": zero_specs["output"]},
        {"output": "output"},
        device,
        host,
        launch_plan,
    )


def run_apu_v1_hybrid(compiled, inputs, *, lab_name=None):
    """Build and run a supported physical hybrid without host intermediates."""

    from ..spmw_codegen import _apu_v1_unavailable_reason

    reason = _apu_v1_unavailable_reason()
    if reason:
        return RunResult(None, reason, "apu_v1")
    root = tempfile.mkdtemp(prefix="tenon-apu-v1-hybrid-")
    try:
        physical = compiled.physical_manifest
        vector_count = sum(
            region.kind == "vector"
            for phase in physical.logical.phases
            for region in phase.regions
        )
        if vector_count == 1:
            lab_name = lab_name or "tenon-hybrid-gemm"
            emitted = emit_apu_v1_hybrid_gemm_project(
                compiled, inputs, Path(root) / "project", lab_name=lab_name
            )
        elif vector_count == 2:
            lab_name = lab_name or "tenon-hybrid-2mm"
            emitted = emit_apu_v1_hybrid_two_mm_project(
                compiled, inputs, Path(root) / "project", lab_name=lab_name
            )
        elif vector_count == 3:
            lab_name = lab_name or "tenon-hybrid-3mm"
            emitted = emit_apu_v1_hybrid_three_mm_project(
                compiled, inputs, Path(root) / "project", lab_name=lab_name
            )
        else:
            raise NotImplementedError(
                "hybrid device runner supports GEMM, 2mm, and 3mm topologies"
            )
        paths = {}
        for role, value in emitted.inputs.items():
            path = Path(root) / f"in_{role}.bin"
            value.tofile(path)
            paths[role] = path
        output_path = Path(root) / "out_output.bin"
        build = subprocess.run(
            ["make"], cwd=emitted.path, capture_output=True, timeout=600, check=False
        )
        if build.returncode:
            raise RuntimeError(
                "APU v1 hybrid build failed:\n"
                + build.stderr.decode(errors="replace")[-10000:]
            )
        binary = emitted.path / "build" / "debug" / lab_name
        argv = [str(binary)]
        argv.extend(str(paths[name]) for name in sorted(emitted.inputs))
        argv.append(str(output_path))
        process = None
        text = ""
        for attempt in range(3):
            process = subprocess.run(
                argv, cwd=emitted.path, capture_output=True, timeout=900, check=False
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
                f"APU v1 hybrid device run failed ({code}):\n{text[-8000:]}"
            )
        shape, dtype = emitted.output_specs["output"]
        output = np.fromfile(output_path, dtype=dtype).reshape(shape)
        if "output" in inputs:
            np.copyto(inputs["output"], output, casting="same_kind")
        device_log = _ledag_log()
        cycles, phase_cycles = _parse_hybrid_profile(device_log, emitted.launch_plan)
        has_scalar_region = any(
            region.kind == "scalar"
            for phase in physical.logical.phases
            for region in phase.regions
        )
        return RunResult(
            cycles,
            text + "\n" + device_log,
            "apu_v1",
            extra={
                "outputs": {"output": output},
                "host_intermediate_round_trips": 0,
                "persistent_l4": True,
                "vector_apucs": (0, 1, 2, 3),
                "scalar_apucs": (0,) if has_scalar_region else (),
                "final_gather_apucs": (0,),
                "phase_cycles": phase_cycles,
                "device_source": emitted.device_source,
                "host_source": emitted.host_source,
            },
        )
    finally:
        if os.environ.get("TENON_APU_V1_KEEP_TMP") != "1":
            shutil.rmtree(root, ignore_errors=True)


__all__ = [
    "APUv1HybridProject",
    "emit_apu_v1_hybrid_gemm_project",
    "emit_apu_v1_hybrid_two_mm_project",
    "emit_apu_v1_hybrid_three_mm_project",
    "run_apu_v1_hybrid",
]
