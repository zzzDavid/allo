# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-simulator tests for irregular UPMEM physical plans."""

import pytest

from allo.pim.upmem_irregular import (
    UPMEMFeatureGradientPlan,
    UPMEMHistogramPlan,
    UPMEMKMeansDistancesPlan,
    UPMEMKMeansPlan,
    UPMEMStableSelectionPlan,
)


def test_legality_checks_fail_before_emitting_an_invalid_translation_unit():
    with pytest.raises(ValueError, match="exactly 12 tasklets"):
        UPMEMHistogramPlan(128, 8, 3, num_tasklets=8)
    with pytest.raises(ValueError, match="8-byte-aligned DMA"):
        UPMEMHistogramPlan(128, 8, 3, dma_elements=3)
    with pytest.raises(ValueError, match="64 MiB MRAM"):
        UPMEMHistogramPlan(20_000_000, 8, 3)
    with pytest.raises(ValueError, match="64 KiB WRAM"):
        UPMEMStableSelectionPlan(20_000)
    with pytest.raises(ValueError, match="predicate='odd'"):
        UPMEMStableSelectionPlan(128, predicate="positive")
    with pytest.raises(ValueError, match="exactly one k-means iteration"):
        UPMEMKMeansPlan(120, 8, 4, iterations=2)
    with pytest.raises(ValueError, match="dimension must be even"):
        UPMEMKMeansPlan(120, 7, 4)
    with pytest.raises(ValueError, match="divide evenly across 12 tasklets"):
        UPMEMKMeansDistancesPlan(5, 2, 2)
    with pytest.raises(ValueError, match="fused packet DMA"):
        UPMEMKMeansDistancesPlan(12, 257, 1, max_abs_value=0)
    with pytest.raises(ValueError, match="grouped output DMA"):
        UPMEMKMeansDistancesPlan(12 * 257, 1, 1, max_abs_value=0)
    with pytest.raises(ValueError, match="does not fit signed int32"):
        UPMEMKMeansDistancesPlan(12, 256, 1, max_abs_value=2_000)
    with pytest.raises(ValueError, match="exceeds the declared max_abs_value"):
        UPMEMKMeansDistancesPlan(6, 2, 2, max_abs_value=2).reference_distances(
            (3,) * 48
        )
    with pytest.raises(ValueError, match="2048-byte DMA"):
        UPMEMFeatureGradientPlan(120, 512, "linear")
    with pytest.raises(ValueError, match="gradient formula"):
        UPMEMFeatureGradientPlan(120, 8, "softmax")
    with pytest.raises(ValueError, match="overflow_shift"):
        UPMEMFeatureGradientPlan(120, 8, "linear", overflow_shift=63)
