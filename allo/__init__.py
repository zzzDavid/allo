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
# @allo.unit for targets and @allo.work for workloads. Imported lazily
# inside a try/except because allo.unit depends on `pim_dsl`, which
# isn't on every install's import path yet; a missing `pimdsl` should
# not break `import allo` for users who aren't doing PIM target-
# description work.
#
# Note: the ``memory`` decorator exported here rebinds the ``allo.memory``
# top-level attribute from the submodule to the decorator function. This
# only affects attribute-access (``allo.memory``); regular
# ``from allo.memory import Memory, Layout`` goes through ``sys.modules``
# and keeps working. The pre-existing ``allo.Memory`` class (from the
# dataflow memory module) is re-exported at line 13 above and is
# untouched.
try:
    from .unit import target, unit, memory, op, stream, cost  # noqa: F811
except ImportError:
    pass
