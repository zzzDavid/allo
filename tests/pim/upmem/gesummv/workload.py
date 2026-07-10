# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete canonical gesummv MLIR workload for the UPMEM reference backend."""

from lib.upmem_polybench import build_upmem_program, get_case

CASE = get_case("gesummv")


def build():
    return build_upmem_program(CASE)
