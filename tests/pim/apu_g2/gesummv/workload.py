# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct-VL64 uint16 specialization of PolyBench GESUMMV."""

from allo.pim import APUG2Program
from lib.shapes import shape


N = shape("gesummv")["N"]

# Smallest positive integer pair preserving PolyBench's 1.5:1.2 ratio.
ALPHA = 5
BETA = 4


def build():
    return APUG2Program(
        operation="gesummv_u16",
        shape=(N, N),
        alpha=ALPHA,
        beta=BETA,
        repetitions=8,
    )


STAGES = [(N, N), (N, N)]
