# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry for complete non-contraction APUv1 vector lowerings.

Contractions continue through :mod:`apu_v1_vectorize`.  This small registry is
for complete structural recipes such as distance/argmin and frequency
reductions whose control flow cannot be represented as a contraction region.
Every recognizer is fail-closed and is considered only when a phase explicitly
requires vectorization.
"""

from __future__ import annotations

from .apu_v1_distance_argmin import (
    APUv1SquaredL2ArgminLowering,
    analyze_apu_v1_squared_l2_argmin,
)
from .apu_v1_histogram import (
    APUv1DenseHistogramLowering,
    analyze_apu_v1_dense_histogram,
)
from .apu_v1_moments import (
    APUv1BivariateMomentsLowering,
    analyze_apu_v1_bivariate_moments,
)
from .apu_v1_record_frequency import (
    APUv1RecordFrequencyLowering,
    analyze_apu_v1_record_frequency,
)


def _discover_squared_l2_argmin(
    source_mlir,
    arguments,
    *,
    function,
    argument_bounds=(),
):
    analysis = analyze_apu_v1_squared_l2_argmin(
        source_mlir,
        arguments,
        function=function,
        argument_bounds=argument_bounds,
    )
    return None if analysis is None else APUv1SquaredL2ArgminLowering(analysis)


def _discover_dense_histogram(
    source_mlir,
    arguments,
    *,
    function,
    argument_bounds=(),
):
    del argument_bounds
    analysis = analyze_apu_v1_dense_histogram(
        source_mlir,
        arguments,
        function=function,
    )
    return None if analysis is None else APUv1DenseHistogramLowering(analysis)


def _discover_record_frequency(
    source_mlir,
    arguments,
    *,
    function,
    argument_bounds=(),
):
    del argument_bounds
    analysis = analyze_apu_v1_record_frequency(
        source_mlir,
        arguments,
        function=function,
    )
    return None if analysis is None else APUv1RecordFrequencyLowering(analysis)


def _discover_bivariate_moments(
    source_mlir,
    arguments,
    *,
    function,
    argument_bounds=(),
):
    del argument_bounds
    analysis = analyze_apu_v1_bivariate_moments(
        source_mlir,
        arguments,
        function=function,
    )
    return None if analysis is None else APUv1BivariateMomentsLowering(analysis)


# Ordered deliberately: a future recipe that accepts a more general skeleton
# belongs after a more specific semantic proof.
APU_V1_NATIVE_VECTOR_DISCOVERERS = (
    _discover_squared_l2_argmin,
    _discover_dense_histogram,
    _discover_record_frequency,
    _discover_bivariate_moments,
)


def discover_apu_v1_native_vector_lowering(phase, artifact, arguments, function):
    """Return the unique required-vector recipe that proves this phase."""

    if getattr(phase, "vectorize", False) != "required":
        return None
    matches = tuple(
        lowering
        for discover in APU_V1_NATIVE_VECTOR_DISCOVERERS
        for lowering in (
            discover(
                artifact.source_mlir,
                arguments,
                function=function,
                argument_bounds=getattr(phase, "argument_bounds", ()),
            ),
        )
        if lowering is not None
    )
    if len(matches) > 1:
        routes = ", ".join(item.route for item in matches)
        raise ValueError(f"ambiguous native APUv1 vector lowering: {routes}")
    return matches[0] if matches else None


__all__ = [
    "APU_V1_NATIVE_VECTOR_DISCOVERERS",
    "discover_apu_v1_native_vector_lowering",
]
