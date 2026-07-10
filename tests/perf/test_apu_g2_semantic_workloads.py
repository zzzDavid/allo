# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained-semantics gates for specialized APUg2 incumbent workloads."""

from pathlib import Path
import re
import sys
from types import SimpleNamespace

import allo
import numpy as np
import pytest

from allo.pim.apu_g2_vector_program import (
    APUG2ColumnBatchedGemmCallable,
    APUG2ContractionChainCallable,
    APUG2CorrelationCallable,
    APUG2CovarianceCallable,
    APUG2GemverCallable,
    APUG2SymmCallable,
    APUG2TrmmCallable,
    compile_apu_g2_vector_workload,
)
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.targets import build_apu_g2_target


_PIM_TEST_ROOT = Path(__file__).resolve().parents[1] / "pim"
if str(_PIM_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIM_TEST_ROOT))

from lib import cell  # noqa: E402


def _workload(name):
    return cell.load_workload(_PIM_TEST_ROOT / "apu_g2" / name, name)


def _alpha_rename(text):
    text = re.sub(r"@[-\w.$]+", "@renamed_function", text)
    names = {}

    def replace_ssa(match):
        source = match.group(0)
        return names.setdefault(source, f"%renamed_{len(names)}")

    return re.sub(r"%[-\w.$]+", replace_ssa, text)


def _renamed_gemver(q0, q1, q2, q3, q4, q5, q6, q7, q8):
    pass


def _renamed_symm(q0, q1, q2):
    pass


def _renamed_trmm(q0, q1):
    pass


def _renamed_covariance(q0, q1, q2):
    pass


def _renamed_correlation(q0, q1, q2, q3):
    pass


def _virtual_inputs(name):
    if name == "gemver":
        matrix = np.zeros((120, 120), dtype=np.uint16)
        vectors = [np.zeros(120, dtype=np.uint16) for _ in range(8)]
        return [matrix, *vectors]
    if name == "symm":
        return [
            np.zeros((60, 60), dtype=np.uint16),
            np.zeros((60, 80), dtype=np.uint16),
            np.zeros((60, 80), dtype=np.uint16),
        ]
    if name == "trmm":
        return [
            np.zeros((60, 60), dtype=np.uint16),
            np.zeros((60, 80), dtype=np.uint16),
        ]
    if name == "covariance":
        return [
            np.zeros((100, 80), dtype=np.uint16),
            np.zeros(80, dtype=np.uint16),
            np.zeros((80, 80), dtype=np.uint16),
        ]
    return [
        np.zeros((100, 80), dtype=np.uint16),
        np.zeros((100, 80), dtype=np.uint16),
        np.zeros((100, 80), dtype=np.uint16),
        np.zeros((80, 80), dtype=np.uint16),
    ]


@pytest.mark.parametrize(
    "name,callable_type,program,hardware_tasks",
    [
        ("symm", APUG2SymmCallable, "column_batched_symm_u16", 3),
        (
            "covariance",
            APUG2CovarianceCallable,
            "transport_aware_covariance_u16",
            9,
        ),
        (
            "correlation",
            APUG2CorrelationCallable,
            "transport_aware_correlation_u16",
            17,
        ),
    ],
)
def test_explicit_semantic_bodies_preserve_incumbent_materialization(
    name, callable_type, program, hardware_tasks
):
    workload = _workload(name)
    target = build_apu_g2_target()
    compiled = allo.compile(
        workload.build(), target, apu_g2_cost, backend="virtual"
    )

    assert isinstance(compiled, callable_type)
    assert compiled.execution_graph.metadata["program"] == program
    assert compiled.execution_graph.metadata["hardware_tasks"] == hardware_tasks
    assert compiled.execution_graph.metadata["transport_schedule"][
        "batch_columns"
    ] == 31


@pytest.mark.parametrize(
    "name,callable_type,epilogues,hardware_tasks",
    [
        ("2mm", APUG2ContractionChainCallable, ((1, 1), (5, 1)), 5),
        ("syrk", APUG2ColumnBatchedGemmCallable, (5, 4), 3),
        ("syr2k", APUG2ContractionChainCallable, ((5, 4), (5, 1)), 6),
    ],
)
def test_scaled_contractions_recover_incumbent_epilogues_from_retained_semantics(
    name, callable_type, epilogues, hardware_tasks
):
    target = build_apu_g2_target()
    compiled = allo.compile(
        _workload(name).build(), target, apu_g2_cost, backend="virtual"
    )

    assert isinstance(compiled, callable_type)
    actual_epilogues = (
        compiled.stage_epilogues
        if hasattr(compiled, "stage_epilogues")
        else compiled.epilogue
    )
    assert actual_epilogues == epilogues
    assert compiled.execution_graph.metadata["hardware_tasks"] == hardware_tasks


def test_covariance_body_retains_centering_gram_and_unsigned_scaling():
    module = allo.customize(_workload("covariance").build(), enable_tensor=False).module
    mlir = str(module)

    assert mlir.count("arith.divui") == 2
    assert "arith.subi" in mlir
    assert "reduction" in mlir
    assert mlir.count('to = "data"') >= 1
    assert mlir.count('to = "cov"') >= 2


def test_correlation_body_retains_normalization_sqrt_gram_and_diagonal_select():
    module = allo.customize(_workload("correlation").build(), enable_tensor=False).module
    mlir = str(module)

    assert mlir.count("arith.divui") == 3
    assert mlir.count("math.sqrt") == 1
    assert "arith.cmpi eq" in mlir
    assert "reduction" in mlir
    assert mlir.count('to = "corr"') >= 3


def test_symm_body_retains_epilogue_coefficients_and_symmetric_predicate():
    module = allo.customize(_workload("symm").build(), enable_tensor=False).module
    mlir = str(module)

    assert "arith.constant 4" in mlir
    assert "arith.constant 5" in mlir
    assert "arith.cmpi sle" in mlir
    assert mlir.count('to = "C"') == 3


@pytest.mark.parametrize(
    "name,callable_type,renamed_workload",
    [
        ("gemver", APUG2GemverCallable, _renamed_gemver),
        ("symm", APUG2SymmCallable, _renamed_symm),
        ("trmm", APUG2TrmmCallable, _renamed_trmm),
        ("covariance", APUG2CovarianceCallable, _renamed_covariance),
        ("correlation", APUG2CorrelationCallable, _renamed_correlation),
    ],
)
def test_alpha_renamed_semantics_preserve_dispatch_graph_and_operand_roles(
    name, callable_type, renamed_workload
):
    workload = _workload(name).build()
    schedule = allo.customize(workload, enable_tensor=False)
    target = build_apu_g2_target()
    cost = apu_g2_cost.bind(target)
    baseline = compile_apu_g2_vector_workload(
        workload, target, schedule, cost=cost, backend="virtual"
    )
    renamed_schedule = SimpleNamespace(module=_alpha_rename(str(schedule.module)))
    renamed = compile_apu_g2_vector_workload(
        renamed_workload,
        target,
        renamed_schedule,
        cost=cost,
        backend="virtual",
    )

    assert isinstance(renamed, callable_type)
    assert renamed.plan == baseline.plan
    assert renamed.execution_graph.name == baseline.execution_graph.name
    assert renamed.execution_graph.metadata == baseline.execution_graph.metadata
    assert renamed.execution_graph.activities == baseline.execution_graph.activities
    assert renamed.estimate().cycles == baseline.estimate().cycles
    assert renamed(*_virtual_inputs(name)).backend == "virtual"


def test_special_workloads_and_compiler_have_no_identity_tags():
    tokens = ("APUG2_" + "WORKLOAD_KIND", "APUG2_" + "PROGRAM_KIND")
    paths = [
        Path("allo/pim/apu_g2_vector_program.py"),
        *(
            _PIM_TEST_ROOT / "apu_g2" / name / "workload.py"
            for name in ("gemver", "symm", "trmm", "covariance", "correlation")
        ),
    ]

    for path in paths:
        source = path.read_text()
        assert all(token not in source for token in tokens)
