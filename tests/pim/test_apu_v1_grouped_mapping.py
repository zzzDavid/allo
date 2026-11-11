# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""V1-20: a dataflow region with more work items than APUCs on APU v1 is
refused at enumeration with the shared work-grid policy's reason, instead of
being accepted by the grouped enumerator and failing in costing with a unit
coordinate error; the four-work-item form (the realized grouped program: one
SPMD body on the four APUCs, the output rows split over the cores) compiles."""

import pytest

import allo
from allo.ir.types import float16
from allo.pim import apu_v1_cost, build_apu_v1_target

K = 8


def _gemv_region(items: int):
    @allo.dataflow.region()
    def grouped_gemv(a: float16[items, K], b: float16[K], c: float16[items]):
        @allo.work(mapping=[items], args=[a, b, c])
        def reduce_row(mat: float16[items, K], vec: float16[K], dst: float16[items]):
            row = allo.get_wid()
            acc: float16 = 0
            for k in allo.reduction(K):
                acc += mat[row, k] * vec[k]
            dst[row] = acc

    return grouped_gemv


def _compile(region):
    return allo.compile(
        region,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
        reduction_order="reassociate",
    )


def test_four_work_items_compile_through_the_grouped_route():
    compiled = _compile(_gemv_region(4))
    (kernel,) = compiled.placement_ranking.kernels
    assert kernel.scores[kernel.chosen_position][2] == "grouped_f16"
    assert compiled.estimate().cycles > 0


@pytest.mark.parametrize("items", [8, 16])
def test_more_work_items_than_apucs_are_refused_before_costing(items):
    with pytest.raises(ValueError) as excinfo:
        _compile(_gemv_region(items))
    message = str(excinfo.value)
    assert message.startswith("work grid does not partition target 'apu_v1'")
    assert f"declares the work grid [{items}] ({items} items)" in message
    assert "one SPMD body on the 4 APUCs" in message and "mapping=[4]" in message
    assert "V1-20" in message
    # the former failure mode (a unit coordinate error from the cost evaluator)
    assert "work coordinate" not in message and "outside 'apuc' extent" not in message
