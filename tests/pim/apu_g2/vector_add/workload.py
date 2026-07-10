# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""First hardware-only APUg2 workload: one coalesced uint16 VL64 add."""

from allo.pim import APUG2Program


def build():
    return APUG2Program(repetitions=256)
