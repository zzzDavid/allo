# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""APU v1 vector plans under the reduction-order policy (release issue V1-03).

Compile-only and functional: no board access.  Under the default
``reduction_order="strict"`` an FP16 contraction ranks only source-order
(temporal) plans; ``"reassociate"`` admits the subgroup-tree plans, and the
functional backend then reproduces the selected plan's order instead of the
source order.  Integer contractions keep every plan.
"""

from __future__ import annotations

import numpy as np
import pytest

import allo
from allo.ir.types import float16, uint16
from allo.pim.apu_v1_vector_program import apu_v1_plan_regroups
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target
from allo.spmw_reduction_policy import ReductionOrderError

M, K = 1024, 256


def fp16_gemv(W: float16[M, K], x: float16[K], y: float16[M]):
    for i in allo.grid(M):
        for k in allo.reduction(K):
            y[i] += W[i, k] * x[k]


def u16_gemv(W: uint16[M, K], x: uint16[K], y: uint16[M]):
    for i in allo.grid(M):
        for k in allo.reduction(K):
            y[i] += W[i, k] * x[k]


def _compile(workload, **kwargs):
    return allo.compile(
        workload, build_apu_v1_target(), apu_v1_cost, backend="functional", **kwargs
    )


@pytest.fixture(scope="module")
def operands():
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    source = np.zeros(M, np.float16)
    for k in range(K):
        source = (source + (W[:, k] * x[k]).astype(np.float16)).astype(np.float16)
    return W, x, source


def test_strict_ranks_only_source_order_plans():
    compiled = _compile(fp16_gemv)
    assert compiled.reduction_order == "strict"
    assert compiled.ranked_plans
    assert not any(apu_v1_plan_regroups(plan) for plan in compiled.ranked_plans)
    assert not apu_v1_plan_regroups(compiled.selected_plan)
    pruned = dict(compiled.reduction_policy_pruned)
    assert {"baseline_spatial_reduction", "spatial_gemv_group_reduction"} <= set(pruned)
    assert all("reduction_order='strict' on a f16" in r for r in pruned.values())
    assert "gvml_add_subgrps" not in compiled.device_source()

    with pytest.raises(ReductionOrderError, match="spatial_gemv_group_reduction"):
        _compile(fp16_gemv, layout="spatial_gemv_group_reduction")


def test_reassociate_admits_tree_plans():
    compiled = _compile(fp16_gemv, reduction_order="reassociate")
    assert compiled.reduction_policy_pruned == ()
    assert any(apu_v1_plan_regroups(plan) for plan in compiled.ranked_plans)
    assert compiled.selected_plan.name == "spatial_gemv_group_reduction"
    assert "gvml_add_subgrps" in compiled.device_source()


def test_functional_numerics_follow_the_selected_plan(operands):
    W, x, source = operands
    exact = W.astype(np.float32) @ x.astype(np.float32)

    strict = _compile(fp16_gemv)
    y = np.zeros(M, np.float16)
    result = strict(W, x, y)
    assert result.extra["reduction_order"] == "strict"
    assert result.extra["plan_regroups_reduction"] is False
    assert result.extra["numerics"] == "source order"
    np.testing.assert_array_equal(y.view(np.uint16), source.view(np.uint16))

    relaxed = _compile(fp16_gemv, reduction_order="reassociate")
    y = np.zeros(M, np.float16)
    result = relaxed(W, x, y)
    assert result.extra["plan_regroups_reduction"] is True
    assert result.extra["numerics"].startswith("selected plan order")
    # Halving FP16 tree over the 256-element group, then one add into y.
    products = (W * x).astype(np.float16)
    width = K
    while width > 1:
        width //= 2
        products = (products[:, :width] + products[:, width : 2 * width]).astype(
            np.float16
        )
    np.testing.assert_array_equal(y, (np.float16(0) + products[:, 0]).astype(np.float16))
    assert np.any(y.view(np.uint16) != source.view(np.uint16))
    np.testing.assert_allclose(y.astype(np.float32), exact, rtol=1e-2, atol=1e-2)


def test_integer_contraction_keeps_every_plan():
    strict = _compile(u16_gemv)
    relaxed = _compile(u16_gemv, reduction_order="reassociate")
    assert strict.reduction_policy_pruned == ()
    assert [c.name for c in strict.candidates] == [c.name for c in relaxed.candidates]
    assert strict.selected_plan.name == relaxed.selected_plan.name
