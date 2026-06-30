# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PIM device descriptions for the SPMW/Tenon backends."""

from __future__ import annotations

from .targets import (
    build_samsung_target,
    build_aim_target,
    build_upmem_target,
    build_apu_v1_target,
    build_apu_v2_target,
)

__all__ = [
    "build_samsung_target",
    "build_aim_target",
    "build_upmem_target",
    "build_apu_v1_target",
    "build_apu_v2_target",
]
from .performance import default_profile, virtual_target

__all__ = ["default_profile", "virtual_target"]
