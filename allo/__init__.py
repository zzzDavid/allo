# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin

from . import frontend, backend, ir, passes, library, _mlir
from .customize import customize, Partition
from .backend.llvm import invoke_mlir_parser, LLVMModule
from .backend.hls import HLSModule
from .backend.ip import IPModule
from .dsl import *
from .template import *
from .verify import verify
from .memory import Memory, Layout
from .dataflow import kernel as work, get_pid as get_wid
from .spmw_target import (
    target,
    unit,
    device,
    reg,
    get_uid,
    move,
    op,
    any_,
    or_,
)
from .spmw_target import memory as mem

# Host-transfer dispatch surface (spec 001). NOTE: the proxy is `allo.host_xfer`,
# NOT `allo.backend` — `allo.backend` is already the codegen-backend submodule
# (imported on line 5; llvm/hls/ip/aie). Re-exporting the proxy as `backend`
# would shadow it and break the FPGA/AIE path. See backend-host-transfer-dispatch.md
# open question.
from .spmw_target import (
    broadcast,
    scatter,
    gather,
    move_only,
    host_xfer,
    record_host_moves,
    BackendHandle,
    HandleToken,
    HostMoveRecord,
    host_program,
    launch,
    BufferToken,
    LaunchRecord,
    HostProgram,
)
# HostXcel collective surface removed (2026-06-30): host<->device transfers
# are now explicit `allo.move`s declared on an @allo.unit(mode="host") scope,
# naming device memory (e.g. hbm_pim.banks) as an endpoint. See spmw_target.py
# (@allo.device / DeviceScope) and allo/pim/targets.py (the Samsung host scope).
# SPMW/PIM exports below load on first attribute access (PEP 562), so
# `import allo` does not import the PIM compiler stack.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "HostSchedule": ("allo.spmw_host_program", "HostSchedule"),
    "HostGroup": ("allo.spmw_host_program", "HostGroup"),
    "HostLaunch": ("allo.spmw_host_program", "HostLaunch"),
    "analyze_host_program": ("allo.spmw_host_program", "analyze"),
    "resolve_host_program_shapes": ("allo.spmw_host_program", "resolve_shapes"),
    "schedule_residency": ("allo.spmw_host_program", "schedule_residency"),
    "schedule_cost": ("allo.spmw_host_program", "schedule_cost"),
    "CostSpec": ("allo.perf", "CostSpec"),
    "BoundCostSpec": ("allo.perf", "BoundCostSpec"),
    "cost": ("allo.perf", "cost"),
    "rule": ("allo.perf", "rule"),
    "compile_op_pattern": ("allo.spmw_match_engine", "compile_op_pattern"),
    "compile_target_patterns": ("allo.spmw_match_engine", "compile_target_patterns"),
    "match_workload": ("allo.spmw_match_engine", "match_workload"),
    "Placement": ("allo.spmw_autoschedule", "Placement"),
    "autoschedule": ("allo.spmw_autoschedule", "autoschedule"),
    "compile_for_target": ("allo.spmw_codegen", "compile_for_target"),
    "Compiled": ("allo.spmw_codegen", "Compiled"),
    "PIMCmd": ("allo.spmw_codegen", "PIMCmd"),
    "SamsungCtx": ("allo.spmw_samsung", "SamsungCtx"),
    "ResolvedHostMove": ("allo.spmw_codegen", "ResolvedHostMove"),
    "LinearLayout": ("allo.spmw_linear_layout", "LinearLayout"),
    "materialise_handle": ("allo.spmw_linear_layout", "materialise_handle"),
    "compile": ("allo.compiler", "compile"),
    "CompiledCallable": ("allo.compiler", "CompiledCallable"),
    "APUv1VectorCallable": ("allo.pim.apu_v1_vector_program", "APUv1VectorCallable"),
    "APUv1Phase": ("allo.pim.apu_v1_program", "APUv1Phase"),
    "APUv1PrecisionPolicy": ("allo.pim.apu_v1_program", "APUv1PrecisionPolicy"),
    "APUv1Program": ("allo.pim.apu_v1_program", "APUv1Program"),
    "APUG2ComposedContractionCallable": ("allo.pim.apu_g2_composed_program", "APUG2ComposedContractionCallable"),
    "APUG2ComposedContractionProgram": ("allo.pim.apu_g2_composed_program", "APUG2ComposedContractionProgram"),
    "APUG2DotEpilogue": ("allo.pim.apu_g2_composed_program", "APUG2DotEpilogue"),
    "APUG2DotEpilogueMode": ("allo.pim.apu_g2_composed_program", "APUG2DotEpilogueMode"),
    "APUG2ComposedHostFiles": ("allo.pim.apu_g2_composed_layout", "APUG2ComposedHostFiles"),
    "build_apu_g2_composed_host_command": ("allo.pim.apu_g2_composed_layout", "build_apu_g2_composed_host_command"),
    "decode_apu_g2_u16_bitpatterns": ("allo.pim.apu_g2_composed_layout", "decode_apu_g2_u16_bitpatterns"),
    "deinterleave_apu_g2_streams": ("allo.pim.apu_g2_composed_layout", "deinterleave_apu_g2_streams"),
    "encode_apu_g2_u16_bitpatterns": ("allo.pim.apu_g2_composed_layout", "encode_apu_g2_u16_bitpatterns"),
    "gather_apu_g2_composed_output": ("allo.pim.apu_g2_composed_layout", "gather_apu_g2_composed_output"),
    "interleave_apu_g2_dot_streams": ("allo.pim.apu_g2_composed_layout", "interleave_apu_g2_dot_streams"),
    "pack_apu_g2_batched_gemm_dots": ("allo.pim.apu_g2_composed_layout", "pack_apu_g2_batched_gemm_dots"),
    "pack_apu_g2_composed_auxiliary": ("allo.pim.apu_g2_composed_layout", "pack_apu_g2_composed_auxiliary"),
    "pack_apu_g2_composed_operand": ("allo.pim.apu_g2_composed_layout", "pack_apu_g2_composed_operand"),
    "pack_apu_g2_matrix_vector_dots": ("allo.pim.apu_g2_composed_layout", "pack_apu_g2_matrix_vector_dots"),
    "APUG2ScalarType": ("allo.pim.apu_g2_typed_program", "APUG2ScalarType"),
    "APUG2TypedCallable": ("allo.pim.apu_g2_typed_program", "APUG2TypedCallable"),
    "APUG2TypedOperation": ("allo.pim.apu_g2_typed_program", "APUG2TypedOperation"),
    "APUG2TypedProgram": ("allo.pim.apu_g2_typed_program", "APUG2TypedProgram"),
    "pack_apu_g2_gemm_as_independent_dots": ("allo.pim.apu_g2_typed_program", "pack_apu_g2_gemm_as_independent_dots"),
    "unpack_apu_g2_gemm_dots": ("allo.pim.apu_g2_typed_program", "unpack_apu_g2_gemm_dots"),
    "CompiledWorkload": ("allo.compiler", "CompiledWorkload"),
}


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'allo' has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
