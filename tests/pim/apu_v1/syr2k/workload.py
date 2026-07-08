# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical syr2k MLIR program for the APU v1 scalar ARC fallback."""

from lib import apu_v1


def build():
    return apu_v1.build_program("syr2k")
