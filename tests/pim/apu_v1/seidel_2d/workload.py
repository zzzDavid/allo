# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical seidel_2d MLIR program for the APU v1 scalar ARC fallback."""

from lib import apu_v1


def build():
    return apu_v1.build_program("seidel_2d")
