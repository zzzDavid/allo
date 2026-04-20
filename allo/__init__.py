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

# SPMW target-description decorator surface (see allo/unit.py). Sits
# next to the workload-side @allo.work decorator so users write
# @allo.unit for targets and @allo.work for workloads.
#
# Note: the ``memory`` decorator exported here rebinds the ``allo.memory``
# top-level attribute from the submodule to the decorator function. This
# only affects attribute-access (``allo.memory``); regular
# ``from allo.memory import Memory, Layout`` goes through ``sys.modules``
# and keeps working. The pre-existing ``allo.Memory`` class (from the
# dataflow memory module) is re-exported at line 13 above and is
# untouched.
from .unit import target, unit, memory, op, stream, cost  # noqa: F811

# SPMW workload-side decorator + MVP compile entry point.
from .work import work, Work  # noqa: F811
from .compile import compile  # noqa: F811,A001

# PIM subpackage: target descriptions, backends, and runtime drivers.
# Surface ``allo.pim`` as a top-level attribute so ``from allo import pim``
# and attribute access work without a separate explicit import.
from . import pim  # noqa: F401
