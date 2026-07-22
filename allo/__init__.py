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
from .spmw_host_program import (
    HostSchedule,
    HostGroup,
    HostLaunch,
    analyze as analyze_host_program,
    resolve_shapes as resolve_host_program_shapes,
    schedule_residency,
    schedule_cost,
)

# HostXcel collective surface removed (2026-06-30): host<->device transfers
# are now explicit `allo.move`s declared on an @allo.unit(mode="host") scope,
# naming device memory (e.g. hbm_pim.banks) as an endpoint. See spmw_target.py
# (@allo.device / DeviceScope) and allo/pim/targets.py (the Samsung host scope).
from .perf import CostSpec, BoundCostSpec, cost, rule
from .spmw_match_engine import (
    compile_op_pattern,
    compile_target_patterns,
    match_workload,
)
from .spmw_autoschedule import Placement, autoschedule
from .spmw_codegen import (
    compile_for_target,
    Compiled,
    PIMCmd,
    SamsungCtx,
    ResolvedHostMove,
)
from .spmw_linear_layout import LinearLayout, materialise_handle
from .compiler import compile, CompiledCallable
from .pim.schedule_promotion import PromotionEvidence
from .pim.aim_program import (
    AimOp,
    AimContraction,
    AimElementwise,
    AimActivation,
    AimHostTransfer,
    AimDistributedHostTransfer,
    AimBankCopy,
    AimAllBankWrite,
    AimSync,
    AimProgram,
    AimProgramCallable,
)
from .pim.apu_v1_vector_program import APUv1VectorCallable
from .pim.upmem_program import (
    UPMEMArray,
    UPMEMDenseTile,
    UPMEMDotTile,
    UPMEMRank1Tile,
    UPMEMPhase,
    UPMEMProgram,
)
from .pim.apu_v1_program import APUv1Phase, APUv1PrecisionPolicy, APUv1Program
from .pim.apu_g2_program import APUG2Callable, APUG2Operation, APUG2Program
from .pim.apu_g2_vector_program import (
    APUG2AtaxCallable,
    APUG2GemvCallable,
    APUG2IndependentContractionsCallable,
    APUG2RankNContractionCallable,
)
