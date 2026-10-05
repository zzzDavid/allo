# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Software-only tests for calibrated compiler-owned UPMEM physical plans."""

import pytest

from allo.pim.upmem_physical import (
    UPMEMElementwisePlan,
    UPMEMGEMMPlan,
    UPMEMMatrixVectorPlan,
    UPMEMSumReductionPlan,
)


@pytest.mark.parametrize(
    "constructor, message",
    [
        (lambda: UPMEMElementwisePlan(128, dma_bytes=12), "8-byte aligned"),
        (lambda: UPMEMElementwisePlan(128, dma_bytes=2048), "fused operand DMA"),
        (lambda: UPMEMSumReductionPlan(128, dma_bytes=4), "between 8 and 2048"),
        (
            lambda: UPMEMMatrixVectorPlan(
                12, 64, dma_bytes=128, vector_mode="shared_wram"
            ),
            "fused_replicated",
        ),
        (lambda: UPMEMGEMMPlan(12, 64, 128, column_tile=32), "between 8 and 2048"),
    ],
)
def test_illegal_dma_or_unsupported_physical_modes_fail_closed(constructor, message):
    with pytest.raises(ValueError, match=message):
        constructor()
