# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-neutral retained-MLIR contraction analysis interface.

The implementation originated with the APUv1 vector backend, but its parser
and proof objects contain no target decisions.  This module is the shared
compiler boundary used by newer targets while preserving the established
APUv1 import surface.
"""

from .apu_v1_vectorize import (
    ContractionAnalysis,
    ContractionAnalysisError,
    IllegalContractionError,
    LogicalAxis,
    NoContractionError,
    ValueAccess,
    analyze_apu_v1_contraction,
    analyze_apu_v1_contractions,
)


analyze_contraction = analyze_apu_v1_contraction
analyze_contractions = analyze_apu_v1_contractions


__all__ = [
    "ContractionAnalysis",
    "ContractionAnalysisError",
    "IllegalContractionError",
    "LogicalAxis",
    "NoContractionError",
    "ValueAccess",
    "analyze_contraction",
    "analyze_contractions",
]
