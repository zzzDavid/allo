# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical 3mm MLIR program with structurally selected hybrid regions."""

from lib import apu_v1


def build():
    return apu_v1.build_hybrid_program("3mm")
