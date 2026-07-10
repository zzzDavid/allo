# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Gemini-II milestone gate against NumPy uint16 arithmetic."""

import numpy as np
import pytest

import allo
from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


@pytest.mark.apu_g2_device
def test_vector_add_apu_g2_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "vector_add")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))

    rng = np.random.default_rng(20260708)
    lhs = rng.integers(0, 65536, size=(4, 65536), dtype=np.uint16)
    rhs = rng.integers(0, 65536, size=(4, 65536), dtype=np.uint16)
    # Explicit overflow cases prove modulo-2^16 rather than widened arithmetic.
    lhs[0, :8] = np.array(
        [0, 1, 65535, 65535, 32768, 60000, 65530, 42], dtype=np.uint16
    )
    rhs[0, :8] = np.array([0, 65535, 1, 65535, 32768, 6000, 10, 65535], dtype=np.uint16)
    expected = np.add(lhs, rhs, dtype=np.uint16)
    output = np.empty_like(lhs)

    import conftest

    with conftest.apu_g2_board_lock():
        result = compiled(lhs, rhs, output)

    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(result.extra["outputs"]["out"], expected)
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["repetitions"] == 256
    assert result.extra["total_ticks"] > 0
    assert result.extra["final_pipeline_ticks"] > 0
    assert result.extra["per_call_ticks"]["pipeline"] > 0
    # The executable cost is calibrated to repeated full-pipeline throughput.
    # Real-card micro-timing shifts by roughly 8-10% when the shared ARC task
    # library is relinked with additional entry points; exact NumPy equality is
    # the hardware acceptance criterion, and this remains a cost smoke check.
    predicted = compiled.estimate().cycles
    assert abs(result.cycles - predicted) / predicted <= 0.10

    device_source = result.extra["sources"]["device/apu_g2_u16_add.cc"]
    assert "gsi::g2_64vl" in device_source
    assert "MmbVectors_seg0" in device_source
    assert "MmbVectors_seg1" in device_source
    assert "add(v.lhs_mmb, v.rhs_mmb, v.out_mmb);" in device_source
    assert "seu_barrier();" in device_source
