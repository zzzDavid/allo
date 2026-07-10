# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete canonical heat_3d MLIR workload for the UPMEM reference backend."""

from lib.upmem_polybench import build_upmem_program, get_case

CASE = get_case("heat_3d")


def build():
    return build_upmem_program(CASE)
