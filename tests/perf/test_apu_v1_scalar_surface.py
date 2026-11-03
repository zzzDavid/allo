# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""APU v1 scalar-route sqrt (release V1-12) and public surface (V1-11).

Compile-only.  The device helper is compiled for the host with the system C
compiler and compared with the C library's correctly rounded ``sqrt`` at both
ends of the double and float domains.
"""

from __future__ import annotations

import ctypes
import math
import shutil
import subprocess

import numpy as np
import pytest

import allo
from allo.ir.types import float32, float64
from allo.pim import apu_v1_program
from allo.pim.apu_v1_program import APU_V1_SQRT_HELPERS, _replace_local_arrays
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target


def root32(x: float32[4]) -> float32[4]:
    y: float32[4] = 0.0
    for i in range(4):
        y[i] = allo.sqrt(x[i])
    return y


def root64(x: float64[4]) -> float64[4]:
    y: float64[4] = 0.0
    for i in range(4):
        y[i] = allo.sqrt(x[i])
    return y


@pytest.fixture(scope="module")
def helper(tmp_path_factory):
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("no host C compiler")
    root = tmp_path_factory.mktemp("apu_v1_sqrt")
    source = root / "helper.c"
    source.write_text(
        "#include <stdint.h>\n"
        + APU_V1_SQRT_HELPERS
        + "double t_sqrt(double v) { return allo_sqrt(v); }\n"
        + "float t_sqrtf(float v) { return allo_sqrtf(v); }\n"
    )
    library = root / "helper.so"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-fno-builtin",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(library))
    lib.t_sqrt.restype = ctypes.c_double
    lib.t_sqrt.argtypes = [ctypes.c_double]
    lib.t_sqrtf.restype = ctypes.c_float
    lib.t_sqrtf.argtypes = [ctypes.c_float]
    return lib


def _ulps(lhs, rhs):
    return abs(
        int(np.float64(lhs).view(np.int64)) - int(np.float64(rhs).view(np.int64))
    )


def test_sqrt_is_exact_at_the_audit_failure_points(helper):
    # The removed 16-step Newton helper was off by 0.74 at 1e10 and 14.3 at 1e12.
    for value, root in ((1e10, 1e5), (1e12, 1e6), (1e30, 1e15), (4.0, 2.0)):
        assert helper.t_sqrt(value) == root
        assert helper.t_sqrtf(value) == np.float32(root)


@pytest.mark.parametrize(
    "value",
    [
        5e-324,  # smallest subnormal
        2.2250738585072014e-308,  # smallest normal
        1e-300,
        0.5,
        2.0,
        1e300,
        1.7976931348623157e308,  # largest finite
    ],
)
def test_sqrt_double_domain_ends_within_one_ulp(helper, value):
    assert _ulps(helper.t_sqrt(value), math.sqrt(value)) <= 1


def test_sqrt_special_values(helper):
    assert helper.t_sqrt(0.0) == 0.0
    assert math.copysign(1.0, helper.t_sqrt(-0.0)) == -1.0
    assert helper.t_sqrt(math.inf) == math.inf
    assert math.isnan(helper.t_sqrt(math.nan))
    assert math.isnan(helper.t_sqrt(-1.0))
    assert math.isnan(helper.t_sqrt(-math.inf))
    assert math.isnan(helper.t_sqrtf(-1.0))
    assert helper.t_sqrtf(np.float32(3.4028235e38)) == np.sqrt(np.float32(3.4028235e38))
    assert helper.t_sqrtf(1e-45) == np.sqrt(np.float32(1e-45))


def test_sqrt_random_sample_across_both_ranges(helper):
    rng = np.random.default_rng(20261102)
    doubles = rng.integers(1, 0x7FF0000000000000, size=20000, dtype=np.uint64).view(
        np.float64
    )
    assert max(_ulps(helper.t_sqrt(float(v)), math.sqrt(v)) for v in doubles) <= 1
    floats = rng.integers(1, 0x7F800000, size=20000, dtype=np.uint32).view(np.float32)
    mismatches = [
        v for v in floats if np.float32(helper.t_sqrtf(float(v))) != np.sqrt(v)
    ]
    assert not mismatches


def test_scalar_route_rewrites_sqrt_calls_to_the_helpers():
    source = "void f(double a, float b, double *o, float *p) {\n  *o = sqrt(a);\n  *p = sqrtf(b);\n}\n"
    transformed, _ = _replace_local_arrays(source)
    assert "*o = allo_sqrt(a);" in transformed
    assert "*p = allo_sqrtf(b);" in transformed
    assert transformed.count("static double allo_sqrt(double value)") == 1
    assert "allo_allo_sqrt" not in transformed


@pytest.mark.parametrize("kernel", [root32, root64])
def test_scalar_program_uses_the_helper_and_matches_the_functional_route(kernel):
    program = allo.APUv1Program((allo.APUv1Phase(kernel, result_names=("y",)),))
    target = build_apu_v1_target()
    virtual = allo.compile(program, target, apu_v1_cost, backend="virtual")
    assert "= allo_sqrt(" in virtual.compiled.device_source
    assert "static double allo_sqrt(double value)" in virtual.compiled.device_source
    validity = virtual.execution_graph.metadata["cost_validity"]
    assert validity["cost_validity"] == "extrapolated"

    functional = allo.compile(program, target, apu_v1_cost, backend="functional")
    dtype = np.float32 if kernel is root32 else np.float64
    x = np.array([0.0, 1e10, 1e12, 2.0], dtype=dtype)
    y = functional(x).extra["outputs"]["y"]
    np.testing.assert_array_equal(y, np.sqrt(x))


def test_dead_apu_v1_surface_is_removed():
    assert "APUv1PrecisionPolicy" not in allo._LAZY_EXPORTS
    assert not hasattr(allo.pim, "APUv1PrecisionPolicy")
    assert "APUv1PrecisionPolicy" not in apu_v1_program.__all__
    for field in (
        "precision_policy",
        "vector_layout",
        "produces",
        "bindings",
        "dependencies",
        "zero_initialize",
    ):
        with pytest.raises(TypeError):
            allo.APUv1Phase(root32, **{field: ()})
    phase = allo.APUv1Phase(root32, result_names=("y",))
    other = allo.APUv1Phase(root64, result_names=("y",))
    with pytest.raises(ValueError, match="exactly one phase"):
        allo.APUv1Program((phase, other))
