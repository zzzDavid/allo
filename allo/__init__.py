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
from .spmw_target import target, unit, reg, const, get_uid, move, op, any_, or_
from .spmw_target import memory as mem
from .spmw_host import HostXcel, host_xcel, primitive, NotSupported, host_cpu
from .spmw_host import (
    broadcast,
    scatter,
    gather,
    reduce,
    all_reduce,
    all_gather,
    reduce_scatter,
    StageRequest,
    staging_scope,
    resolve_staging,
    weight_resident_from_staging,
    resolve_collective_axis,
)
from .spmw_cost import cost, get_cost
from .spmw_cost_model import (
    CostModel,
    OpCost,
    MoveCost,
    OpCostCtx,
    MoveCostCtx,
    ComposeCtx,
    CostResult,
    register_cost_model,
    get_cost_model,
)
from .spmw_match_engine import (
    compile_op_pattern,
    compile_target_patterns,
    match_workload,
)
from .spmw_autoschedule import Placement, autoschedule
from .spmw_codegen import compile_for_target, Compiled, PIMCmd, SamsungCtx
from .spmw_linear_layout import LinearLayout, materialise_handle
from .spmw_regalloc import allocate
from . import spmw_cost_models  # noqa: F401  — registers @cost factories
