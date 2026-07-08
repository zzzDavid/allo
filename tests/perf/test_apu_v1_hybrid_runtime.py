# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused physical orchestration tests for explicit APU v1 hybrid manifests."""

from types import SimpleNamespace

import numpy as np
import pytest

from allo.pim.apu_v1_hybrid import (
    APUv1BarrierManifest,
    APUv1APUCShard,
    APUv1ConversionManifest,
    APUv1HybridManifest,
    APUv1PhaseManifest,
    APUv1PrecisionPolicy,
    APUv1RegionExecutable,
    APUv1RegionPartition,
    APUv1RegionManifest,
    APUv1RegionOperand,
    APUv1ValueManifest,
    compile_apu_v1_hybrid,
    lower_apu_v1_hybrid_manifest,
)
from allo.pim.targets import build_apu_v1_target
from allo.pim.apu_v1_hybrid_runtime import _parse_hybrid_profile


def _operand(position, role, value, shape, access, compute="f16"):
    return APUv1RegionOperand(position, role, value, shape, "f32", compute, access)


def _hybrid_manifest():
    values = (
        APUv1ValueManifest("A", (4, 3), "f32", program_input=True),
        APUv1ValueManifest("B", (3, 4), "f32", program_input=True),
        APUv1ValueManifest("C", (4, 2), "f32", program_input=True),
        APUv1ValueManifest(
            "T", (4, 4), "f32", intermediate=True, zero_initialized=True
        ),
        APUv1ValueManifest(
            "D", (4, 2), "f32", intermediate=True, zero_initialized=True
        ),
        APUv1ValueManifest(
            "Y", (4, 2), "f32", program_output=True, zero_initialized=True
        ),
    )
    vector_facts = {
        "compute_analysis": SimpleNamespace(numeric_type="f16"),
        "selected_plan": SimpleNamespace(name="broadcast_friendly"),
    }
    mm1 = APUv1RegionManifest(
        "region0",
        "mm1",
        0,
        "vector",
        (
            _operand(0, "lhs", "A", (4, 3), "read"),
            _operand(1, "rhs", "B", (3, 4), "read"),
            _operand(2, "output", "T", (4, 4), "write"),
        ),
        ("A", "B"),
        ("T",),
        ("T",),
        (),
        (),
        **vector_facts,
    )
    mm2 = APUv1RegionManifest(
        "region1",
        "mm2",
        1,
        "vector",
        (
            _operand(0, "lhs", "T", (4, 4), "read"),
            _operand(1, "rhs", "C", (4, 2), "read"),
            _operand(2, "output", "D", (4, 2), "write"),
        ),
        ("T", "C"),
        ("D",),
        ("D",),
        ("T",),
        ("region0",),
        barrier="barrier0",
        **vector_facts,
    )
    epilogue = APUv1RegionManifest(
        "region2",
        "epilogue",
        2,
        "scalar",
        (
            _operand(0, "input0", "D", (4, 2), "read", compute="f32"),
            _operand(1, "output", "Y", (4, 2), "write", compute="f32"),
        ),
        ("D",),
        ("Y",),
        (),
        ("D",),
        ("region1",),
        barrier="barrier1",
    )
    barriers = (
        APUv1BarrierManifest(
            "barrier0", ("region0",), ("region1",), ("T",), "data_dependency"
        ),
        APUv1BarrierManifest(
            "barrier1", ("region1",), ("region2",), ("D",), "engine_transition"
        ),
    )
    conversions = (
        APUv1ConversionManifest(
            "convert_A", "A", (4, 3), "f32", "f16", before_region="region0"
        ),
        APUv1ConversionManifest(
            "convert_B", "B", (3, 4), "f32", "f16", before_region="region0"
        ),
        APUv1ConversionManifest(
            "convert_C", "C", (4, 2), "f32", "f16", before_region="region1"
        ),
        APUv1ConversionManifest(
            "convert_D", "D", (4, 2), "f16", "f32", after_region="region1"
        ),
    )
    phase = APUv1PhaseManifest(
        "two_mm",
        "two_mm",
        (mm1, mm2, epilogue),
        barriers,
        conversions,
        ("A", "B", "C"),
        ("Y",),
        ("T", "D"),
    )
    return APUv1HybridManifest(
        "two_mm_hybrid", values, (phase,), APUv1PrecisionPolicy.f32_to_f16()
    )


def _executables(observed):
    shards = tuple(APUv1APUCShard(index, index, index + 1) for index in range(4))
    partition0 = APUv1RegionPartition("region0", "m", 4, shards, ("A", "T"), ("B",))
    partition1 = APUv1RegionPartition("region1", "m", 4, shards, ("T", "D"), ("C",))

    def mm1(buffers, _phase):
        observed.append(("mm1", {name: value.dtype for name, value in buffers.items()}))
        buffers["T"][:] = buffers["A"] @ buffers["B"]

    def mm2(buffers, _phase):
        observed.append(("mm2", {name: value.dtype for name, value in buffers.items()}))
        buffers["D"][:] = buffers["T"] @ buffers["C"]

    def epilogue(buffers, _phase):
        observed.append(
            ("epilogue", {name: value.dtype for name, value in buffers.items()})
        )
        buffers["Y"][:] = buffers["D"] + np.float32(1.0)

    return {
        "region0": APUv1RegionExecutable(
            "region0", functional=mm1, cycles=10, partition=partition0
        ),
        "region1": APUv1RegionExecutable(
            "region1", functional=mm2, cycles=20, partition=partition1
        ),
        "region2": APUv1RegionExecutable("region2", functional=epilogue, cycles=3),
    }


def test_physical_manifest_owns_persistent_l4_and_explicit_launch_barriers():
    physical = lower_apu_v1_hybrid_manifest(_hybrid_manifest(), _executables([]))
    kinds = [phase.kind for phase in physical.phases]

    assert kinds == [
        "convert",
        "convert",
        "vector",
        "barrier",
        "convert",
        "vector",
        "convert",
        "barrier",
        "arc_scalar",
    ]
    assert all(
        phase.apucs == (0, 1, 2, 3)
        for phase in physical.phases
        if phase.kind in {"vector", "convert"}
    )
    assert next(
        phase for phase in physical.phases if phase.kind == "arc_scalar"
    ).apucs == (0,)
    allocation_names = {item.name for item in physical.allocations}
    assert "T" not in allocation_names  # vector-only f32 image is never materialized
    assert "T__f16__resident" in allocation_names
    assert "D__f16__resident" in allocation_names
    assert physical.manifest()["host_intermediate_round_trips"] == 0


def test_functional_hybrid_executes_same_conversion_and_region_sequence():
    observed = []
    compiled = compile_apu_v1_hybrid(
        _hybrid_manifest(),
        _executables(observed),
        build_apu_v1_target(),
        backend="functional",
    )
    a = np.arange(12, dtype=np.float32).reshape(4, 3) / 10
    b = np.arange(12, dtype=np.float32).reshape(3, 4) / 20
    c = np.arange(8, dtype=np.float32).reshape(4, 2) / 30
    run = compiled(a, b, c)
    expected = (
        (a.astype(np.float16) @ b.astype(np.float16)).astype(np.float16)
        @ c.astype(np.float16)
    ).astype(np.float32) + 1

    np.testing.assert_allclose(run.extra["outputs"]["Y"], expected, atol=2e-2)
    assert run.cycles == 33
    assert [name for name, _dtypes in observed] == ["mm1", "mm2", "epilogue"]
    assert all(dtype == np.dtype(np.float16) for dtype in observed[0][1].values())
    assert all(dtype == np.dtype(np.float16) for dtype in observed[1][1].values())
    assert all(dtype == np.dtype(np.float32) for dtype in observed[2][1].values())
    assert run.extra["host_intermediate_round_trips"] == 0


def test_host_and_device_sources_expose_four_apuc_then_scalar_barriers():
    compiled = compile_apu_v1_hybrid(
        _hybrid_manifest(), _executables([]), build_apu_v1_target()
    )
    host = compiled.host_source()
    device = compiled.device_source()

    assert "persistent_l4_bytes=" in host
    assert "host_to_l4_once: A" in host
    assert "l4_to_host_once: Y" in host
    assert "region0:vector apucs=(0, 1, 2, 3)" in host
    assert "region2:arc_scalar apucs=(0,)" in host
    assert host.count("blocking batch = barrier") == 7
    assert "explicit host launch barrier; L4 retained" in host
    assert "f32->f16" in device and "f16->f32" in device
    assert "if (core_id != 0) return 0;" in device


def test_hybrid_lowering_fails_closed_without_region_artifact():
    with pytest.raises(ValueError, match="region1"):
        lower_apu_v1_hybrid_manifest(
            _hybrid_manifest(), {"region0": _executables([])["region0"]}
        )


def test_hybrid_profile_uses_parallel_vector_critical_path_plus_scalar():
    log = "\n".join(
        [
            "ARCT[0]: *** total - hits:1 seu:1 crun:100 iall:2",
            "ARCT[2]: *** total - hits:1 seu:1 crun:130 iall:2",
            "ARCT[1]: *** total - hits:1 seu:1 crun:120 iall:2",
            "ARCT[3]: *** total - hits:1 seu:1 crun:110 iall:2",
            "ARCT[0]: *** total - hits:1 seu:0 crun:40 iall:2",
        ]
    )
    cycles, phases = _parse_hybrid_profile(log)
    assert cycles == 170
    assert phases == {
        "vector_per_apuc": {0: 100, 2: 130, 1: 120, 3: 110},
        "vector_critical": 130,
        "scalar_core0": 40,
    }
