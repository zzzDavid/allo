# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-simulator tests for physical UPMEM schedule search and calibration."""

from __future__ import annotations

import pytest

from allo.pim.upmem_physical_search import (
    UPMEMDataLayout,
    UPMEMNoLegalPhysicalPlan,
    UPMEMOperandResidency,
    UPMEMOwnership,
    UPMEMPhysicalDecision,
    UPMEMPhysicalProblem,
    rank_upmem_physical_candidates,
    select_upmem_physical_plan,
)


def test_unsupported_ownership_and_wram_illegal_plan_are_rejected_fail_closed():
    cyclic = UPMEMPhysicalDecision(
        UPMEMDataLayout.SEPARATE,
        128,
        32,
        ownership=UPMEMOwnership.CYCLIC,
    )
    cyclic_result = rank_upmem_physical_candidates(
        UPMEMPhysicalProblem.pointwise(12_288), (cyclic,)
    )
    assert not cyclic_result.ranked
    assert "only proven contiguous" in cyclic_result.rejected[0].reason
    with pytest.raises(UPMEMNoLegalPhysicalPlan):
        select_upmem_physical_plan(UPMEMPhysicalProblem.pointwise(12_288), (cyclic,))

    shared_full_rhs = UPMEMPhysicalDecision(
        UPMEMDataLayout.SEPARATE,
        2048,
        256,
        rhs_residency=UPMEMOperandResidency.SHARED_WRAM,
        nc=256,
        kc=256,
    )
    wram_result = rank_upmem_physical_candidates(
        UPMEMPhysicalProblem.gemm(12, 256, 256), (shared_full_rhs,)
    )
    assert not wram_result.ranked
    assert "WRAM bytes" in wram_result.rejected[0].reason
