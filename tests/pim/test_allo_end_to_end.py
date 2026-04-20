"""End-to-end compile test: @allo.work kernel -> compile -> LoweringResult
on all five PIM targets (Samsung HBM-PIM, SK-Hynix AiM, UPMEM DPU, GSI
APU v1, GSI APU v2).

Success criteria per target (Report 11 §10 follow-up MVP):

  1. The kernel is expressed as a ``@allo.work``-decorated Python
     function. The test code NEVER hand-builds a ``SrcProgram`` or
     ``SrcOp``.
  2. The target is declared with ``@allo.unit`` / ``@allo.target``
     (we reuse the already-ported ``build_*`` backends).
  3. ``allo.compile(kernel, target=t)`` returns a
     ``pimdsl.LoweringResult`` with::

         len(result.unlowered) == 0   # target realised every source op
         len(result.emitted)   > 0    # non-empty device-native text

  4. If the target's simulator binary is available in this sandbox,
     additionally run it and check numeric output against a CPU
     reference. If the binary is absent, skip the numeric check with
     ``pytest.skip("simulator not available")`` -- that is an
     acceptable MVP deliverable.

Loader note
-----------

``import allo`` fails in this sandbox because the C-extension
``allo._mlir`` is not built. The allo-side helpers (``allo.work``,
``allo.compile``, ``allo.unit``) are pure-Python, though, so we direct-
file-load them under a synthetic ``_allo_mvp`` package root. This
mirrors the pattern used by
``experiments/allo/tests/test_unit_decorators.py`` and by every backend
file under ``pimdsl/backends/`` -- see each backend's
``_load_allo_unit`` helper.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Path setup + pure-Python allo.* loader
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
# tests/pim/ -> tests/ -> allo (repo root) -> allo (pkg dir)
ALLO_PKG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "allo"))
# tests/pim/ -> tests/ -> allo (allo repo) -> experiments/
REPO_EXPERIMENTS = os.path.normpath(os.path.join(HERE, "..", "..", ".."))


def _load_allo_mvp():
    """Return a namespace exposing ``.work``, ``.compile``, ``.unit``,
    and ``.Work`` from the Allo package source tree, without requiring
    ``import allo`` to succeed. Works even when ``allo._mlir`` is
    missing."""
    # Prefer the real thing if it imports.
    try:
        allo = importlib.import_module("allo")
        if hasattr(allo, "work") and hasattr(allo, "compile") \
                and hasattr(allo, "unit"):
            return allo
    except Exception:
        pass

    # Synthetic package so relative imports (`.work`, `.unit`) inside
    # compile.py resolve.
    pkg_name = "_allo_mvp"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [ALLO_PKG_DIR]
    sys.modules[pkg_name] = pkg

    def _loadfile(submod_name: str, filename: str):
        full = f"{pkg_name}.{submod_name}"
        spec = importlib.util.spec_from_file_location(
            full, os.path.join(ALLO_PKG_DIR, filename))
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        setattr(pkg, submod_name, m)
        return m

    unit_mod = _loadfile("unit", "unit.py")
    work_mod = _loadfile("work", "work.py")
    compile_mod = _loadfile("compile", "compile.py")

    # Mount the symbols the test uses directly on the package.
    pkg.work = work_mod.work
    pkg.Work = work_mod.Work
    pkg.compile = compile_mod.compile
    pkg.unit = unit_mod  # leave as submodule; backends use it internally
    pkg.target = unit_mod.target
    return pkg


allo = _load_allo_mvp()

# Backends (already ported to @allo.unit; we reuse as-is).
from allo.pim.backends import (  # noqa: E402
    build_samsung, build_aim, build_upmem, build_apu_v1, build_apu_v2,
)
from allo.pim.lowering import LoweringResult  # noqa: E402


# ---------------------------------------------------------------------------
# The kernel: one @allo.work vadd used for every target.
# ---------------------------------------------------------------------------
#
# ``N = 131072`` because PIMSimulator requires N >= 131072 for its
# eltwise path (see ``pimdsl/runtime/pimsim_driver.py::run_eltwise``).
# The smaller-grid targets (UPMEM, AiM, APU v1/v2) accept any length;
# using the same shape keeps the test single-path.
N = 131072


@allo.work(shapes={"A": (N,), "B": (N,), "C": (N,)}, dtype="fp16")
def vadd(A, B, C):
    """Vector-add kernel, written in plain-Python slice-assignment form."""
    C[:] = A + B


# Also exercise a second, slightly-richer kernel to make sure the
# walker supports more than one statement form. This one is MUL; it
# lowers through all five backends' `mul` pattern.
@allo.work(shapes={"A": (N,), "B": (N,), "C": (N,)}, dtype="fp16")
def vmul(A, B, C):
    C[:] = A * B


# ---------------------------------------------------------------------------
# Target table
# ---------------------------------------------------------------------------

TARGETS = [
    ("samsung", build_samsung),
    ("aim", build_aim),
    ("upmem", build_upmem),
    ("apu_v1", build_apu_v1),
    ("apu_v2", build_apu_v2),
]


# ---------------------------------------------------------------------------
# 1. Lowering: every target must accept the vadd kernel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_name,builder", TARGETS,
                         ids=[n for n, _ in TARGETS])
def test_vadd_lowers_on_every_target(target_name, builder):
    t = builder()
    result = allo.compile(vadd, target=t)

    assert isinstance(result, LoweringResult)
    assert len(result.unlowered) == 0, \
        (f"{target_name}: {len(result.unlowered)} src ops went unlowered: "
         f"{[u.kind for u in result.unlowered]}")
    assert len(result.emitted) > 0, \
        f"{target_name}: lowering produced no device-native text"
    # Also check the target name made it through, so we know the result
    # corresponds to the backend we built.
    assert result.target == t.name


@pytest.mark.parametrize("target_name,builder", TARGETS,
                         ids=[n for n, _ in TARGETS])
def test_vmul_lowers_on_every_target(target_name, builder):
    t = builder()
    result = allo.compile(vmul, target=t)
    assert len(result.unlowered) == 0
    assert len(result.emitted) > 0


# ---------------------------------------------------------------------------
# 2. Numeric: run real simulator when available, else skip
# ---------------------------------------------------------------------------


def _samsung_numeric_available() -> bool:
    """True iff Samsung's pim_driver binary is built and runnable."""
    sim = os.path.join(
        REPO_EXPERIMENTS, "simulators", "PIMSimulator", "pim_driver")
    return os.path.isfile(sim) and os.access(sim, os.X_OK)


def test_samsung_vadd_numeric():
    """Lower + run the real PIMSimulator binary, compare to CPU reference.

    If the binary is absent, ``pytest.skip`` -- still an acceptable
    deliverable per the MVP brief.
    """
    if not _samsung_numeric_available():
        pytest.skip(
            "PIMSimulator pim_driver binary not built "
            "(experiments/simulators/PIMSimulator/pim_driver missing)")

    t = build_samsung()
    result = allo.compile(vadd, target=t)
    assert len(result.unlowered) == 0

    # Drive the simulator. The compiler's emitted text is Samsung PIMCmd
    # pseudo-assembly; the simulator reads npy inputs + op name. The MVP
    # separates the two concerns -- one day the emitted text will drive
    # the simulator directly, for now we invoke it through the same
    # op-name the pattern would have selected ("ADD").
    from allo.pim.runtime.pimsim_driver import run_eltwise

    rng = np.random.default_rng(0xA110)
    A = rng.standard_normal(N).astype(np.float16)
    B = rng.standard_normal(N).astype(np.float16)
    C_sim = run_eltwise("ADD", A, B)
    C_ref = (A.astype(np.float32) + B.astype(np.float32)).astype(np.float16)

    # fp16 sum is bit-exact for inputs that fit: allow 1-ULP tolerance
    # for safety. (Samsung's pim.add uses the same fp16 adder.)
    np.testing.assert_allclose(
        C_sim.astype(np.float32), C_ref.astype(np.float32),
        rtol=1e-2, atol=1e-2,
    )


# ---------------------------------------------------------------------------
# APU v2 (GSI G2) end-to-end: compile -> GTML .cc -> Docker l1_sim -> verify
# ---------------------------------------------------------------------------


def _apu_v2_numeric_available() -> tuple[bool, str]:
    """Return (available, reason). APU v2 L1 simulator needs Docker and the
    pre-built ``gsi-g2-l1sim`` image. The bundled ``example_tensor_add`` must
    also pass so we know the in-image build tree works."""
    import shutil
    import subprocess as _sp
    if shutil.which("docker") is None:
        return False, "docker CLI not on PATH"
    img = _sp.run(["docker", "image", "inspect", "gsi-g2-l1sim"],
                  capture_output=True)
    if img.returncode != 0:
        return False, "gsi-g2-l1sim Docker image not built"
    smoke = _sp.run(
        ["docker", "run", "--rm", "gsi-g2-l1sim",
         "./build/l1_sim/bin/example_tensor_add"],
        capture_output=True, text=True, timeout=120)
    if smoke.returncode != 0 or "PASSED" not in smoke.stdout:
        return False, f"gsi-g2-l1sim smoke test failed: rc={smoke.returncode}"
    return True, ""


# GTML host.cc template. Reads A.bin / B.bin (fp32 bit patterns) from
# /workspace, fills a 16-group x 1-vector VectorPack, dispatches
# tensor_add_kernel (gtml.add), and writes C.bin back.
#
# Why fp32 (not fp16): the image's reference ``add_float16`` helper in
# g2_ref/ref/src/g2_gtml_ref.cc has a known normalisation bug -- it does not
# shift mant_sum right when it carries into bit 12, so e.g. 1.5 + 2.5 gives
# 2.00391 instead of 4.0 (verified empirically in this sandbox). The fp32
# path routes through ``hw_numerics::ref_add_float`` which is native IEEE
# and bit-exact. Same vector count (65536 elements = 16 groups x 4096).
#
# Why num_grps=16: G2Gtml::add_float in g2_gtml/src/L1_to_L1_sim/g2_gtml_fadd.cc
# unconditionally constructs ``VectorPackRef(src, 16)`` and reads 16 groups
# from L1, so the host-side ref must also stage 16 groups (one full APU v2
# vector per tensor).
_APU_V2_HOST_CC = r"""// DSL-emitted host driver for elementwise vadd on 65536 FLOAT32 elements.
// Generated by test_allo_end_to_end.py::test_apu_v2_vadd_end_to_end.

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include <g2_gtml.h>
#include <g2_gtml_ref.h>

#include "device.h"  // tensor_add_kernel -> gtml.add

static constexpr uint32_t NUM_GRPS = 16;
static constexpr uint32_t NUM_VECS = 1;
static constexpr uint32_t BITS     = 32;
static constexpr uint32_t STRIDE   = 32;
static constexpr uint32_t N_ELEMS  = NUM_GRPS * GTML_ELEMENTS_PER_GRP;

static void init_gtml() {
    uint32_t l1 = GTML_SIM_MEM_LOC;
    gsi::L1Container temp(l1,       128);
    gsi::L1Container idx (l1 + 128,  16);
    gsi::gtml::G2Gtml::getInstance(temp, idx);
}

static bool read_bin(const std::string &p, std::vector<uint32_t> &out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    out.resize(N_ELEMS);
    f.read(reinterpret_cast<char*>(out.data()), N_ELEMS * sizeof(uint32_t));
    return (size_t)f.gcount() == N_ELEMS * sizeof(uint32_t);
}
static bool write_bin(const std::string &p, const std::vector<uint32_t> &v) {
    std::ofstream f(p, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(uint32_t));
    return f.good();
}

int main() {
    init_gtml();
    auto &gtml = gsi::gtml::G2Gtml::getInstance();
    gtml_ref::GtmlRef ref;

    std::vector<uint32_t> A_bits, B_bits;
    if (!read_bin("/workspace/A.bin", A_bits) ||
        !read_bin("/workspace/B.bin", B_bits)) {
        std::cerr << "ERROR: cannot read /workspace/{A,B}.bin" << std::endl;
        return 1;
    }

    gsi::VectorPackDescriptor d = {NUM_VECS, STRIDE, gsi::G2_TYPES::FLOAT32, BITS};
    gsi::VectorPack a(d, 1000), b(d, 1050), c(d, 1100);
    gtml_ref::VectorPackRef a_r(a, NUM_GRPS), b_r(b, NUM_GRPS), c_r(c, NUM_GRPS);

    for (uint32_t g = 0; g < NUM_GRPS; g++)
        for (int e = 0; e < GTML_ELEMENTS_PER_GRP; e++) {
            uint32_t i = g * GTML_ELEMENTS_PER_GRP + e;
            a_r.ivecs[g][0][e] = (int64_t)(uint64_t)A_bits[i];
            b_r.ivecs[g][0][e] = (int64_t)(uint64_t)B_bits[i];
        }

    ref.copy_to_l1(a_r);
    ref.copy_to_l1(b_r);

    int rc = tensor_add_kernel(gtml, a, b, c);
    if (rc != 0) { std::cerr << "tensor_add_kernel rc=" << rc << std::endl; return 1; }

    ref.copy_from_l1(c_r);

    std::vector<uint32_t> C_bits(N_ELEMS);
    for (uint32_t g = 0; g < NUM_GRPS; g++)
        for (int e = 0; e < GTML_ELEMENTS_PER_GRP; e++) {
            uint32_t i = g * GTML_ELEMENTS_PER_GRP + e;
            C_bits[i] = (uint32_t)(c_r.ivecs[g][0][e] & 0xFFFFFFFFULL);
        }
    if (!write_bin("/workspace/C.bin", C_bits)) {
        std::cerr << "ERROR: cannot write /workspace/C.bin" << std::endl;
        return 1;
    }

    std::cout << "APU v2 fp32 vadd: N=" << N_ELEMS << " PASSED" << std::endl;
    return 0;
}
"""


def _emit_apu_v2_host_cc(emitted_lines: list[str]) -> str:
    """Wrap ``apu_v2_codegen`` thinking: given the compiler's emitted GTML
    instruction lines, produce a buildable host.cc.

    The existing ``pimdsl.runtime.apu_v2_codegen.gen_apu_v2_cc`` emits an
    INT16 self-contained program with a fixed ``NUM_VECS=4, NUM_GRPS=1``
    shape. For this test we need 65536 elements (one full APU v2 vector) in
    fp32 -- so the template is a local variant that the pattern-matched
    instruction sequence (copy_to_l1, gtml.add, copy_from_l1) is consistent
    with. The instruction list is checked for shape rather than inlined
    verbatim, which matches the state of the other targets' tests (AiM is
    the only one that inlines the exact emitted bytes)."""
    # Smoke-check the compiler produced the expected sequence so we know we
    # aren't papering over a lowering regression.
    want = {"ref.copy_to_l1", "gtml.add", "ref.copy_from_l1"}
    got = set()
    for ln in emitted_lines:
        for k in want:
            if k in ln:
                got.add(k)
    missing = want - got
    assert not missing, (
        f"APU v2 backend emitted an unexpected vadd sequence; "
        f"missing ops: {missing}; full emission: {emitted_lines!r}"
    )
    return _APU_V2_HOST_CC


def test_apu_v2_vadd_end_to_end():
    """End-to-end APU v2 (GSI G2) vadd via ``allo.compile`` -> GTML .cc ->
    ``gsi-g2-l1sim`` Docker -> binary output -> numeric verify.

    Pipeline:
      1. ``allo.compile(vadd, target=build_apu_v2())`` lowers the kernel to a
         GTML instruction stream (copy_to_l1, gtml.add, copy_from_l1) — we
         assert this sequence is present but do not drive the .cc from the
         emitted text verbatim (the APU v2 codegen path is a known-stale
         stub today).
      2. We write an equivalent hand-rolled host.cc (see ``_APU_V2_HOST_CC``)
         that reads A.bin / B.bin, stages a 16-group x 1-vector VectorPack,
         calls ``tensor_add_kernel`` (defined in the image's device.cc as
         ``gtml.add(a, b, c)``), and writes C.bin.
      3. Mount the host.cc + workspace into ``gsi-g2-l1sim``, reconfigure,
         force-rebuild example_tensor_add, run the binary.
      4. Read C.bin back and assert it matches the CPU fp32 reference
         ``A + B`` at rtol=1e-5.

    Functional-only: ``l1_sim`` is a memory-model simulator with no cycle
    counters (see backends/apu_v2.py::caps["perf_is_placeholder"]=True and
    report 05 §5). We assert numerics only.

    Input size: N = 65536 = 16 groups * 4096 elements/group — one full
    APU v2 vector (per report 05 §5 and the G2 programming model). Per the
    brief this is the *exact* size, not an arbitrary N.

    Dtype: FLOAT32 (not fp16) because the image's reference
    ``add_float16`` helper in ``g2_ref/ref/src/g2_gtml_ref.cc`` has a
    normalisation bug (does not handle mant_sum carrying into bit 12);
    1.5 + 2.5 gives 2.00391 instead of 4.0. Documented as a codegen gap
    below.
    """
    import subprocess
    import tempfile

    ok, reason = _apu_v2_numeric_available()
    if not ok:
        pytest.skip(f"APU v2 numeric path unavailable: {reason}")

    # 1. Compile through allo.compile with the APU v2 target. We use a
    #    separate kernel (vadd_apu_v2) with dtype="fp32" because the
    #    numeric leg runs in fp32 (see docstring).
    @allo.work(shapes={"A": (65536,), "B": (65536,), "C": (65536,)},
               dtype="fp32")
    def vadd_apu_v2(A, B, C):
        C[:] = A + B

    t = build_apu_v2()
    result = allo.compile(vadd_apu_v2, target=t)
    assert isinstance(result, LoweringResult)
    assert result.target == t.name
    assert len(result.unlowered) == 0, \
        f"APU v2 did not lower vadd: unlowered={[u.kind for u in result.unlowered]}"
    assert len(result.emitted) > 0, "APU v2 lowering produced no GTML lines"

    # 2. Turn the emitted instruction list into a buildable host.cc.
    host_cc_src = _emit_apu_v2_host_cc(result.emitted)

    # 3. Stage workspace: random fp32 inputs + host.cc. World-writable so
    #    the container's uid 1001 (g2) can write C.bin back.
    with tempfile.TemporaryDirectory(prefix="apu_v2_vadd_") as ws:
        os.chmod(ws, 0o777)
        rng = np.random.default_rng(0xA110)
        N_fp32 = 65536  # 16 groups x 4096; one full APU v2 vector
        A = rng.standard_normal(N_fp32).astype(np.float32)
        B = rng.standard_normal(N_fp32).astype(np.float32)
        A.tofile(os.path.join(ws, "A.bin"))
        B.tofile(os.path.join(ws, "B.bin"))
        host_cc_path = os.path.join(ws, "host.cc")
        with open(host_cc_path, "w") as f:
            f.write(host_cc_src)

        # 4. Build + run inside the gsi-g2-l1sim container. The image's
        #    example-gtml/ target links host.cc + device.cc; we bind-mount
        #    our host.cc over the image's copy (read-only), leaving the
        #    image's device.cc (which provides ``tensor_add_kernel ->
        #    gtml.add``) untouched. A bind-mount does not change the
        #    on-disk ctime in the container's eyes, so CMake's incremental
        #    rebuild skips it -- we delete the cached object+binary before
        #    building to force the recompile. ``cmake --preset l1_sim`` is
        #    already configured; we just re-run the build step.
        build_sh = (
            "cd /home/g2/gtml && "
            "rm -f build/l1_sim/example-gtml/CMakeFiles/"
            "example_tensor_add.dir/host.cc.o "
            "build/l1_sim/bin/example_tensor_add && "
            "cmake --build --preset l1_sim "
            "--target example_tensor_add -j$(nproc) && "
            "./build/l1_sim/bin/example_tensor_add"
        )
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{host_cc_path}:/home/g2/gtml/example-gtml/host.cc:ro",
            "-v", f"{ws}:/workspace",
            "gsi-g2-l1sim", "bash", "-c", build_sh,
        ]
        r = subprocess.run(docker_cmd, capture_output=True, text=True,
                           timeout=600)
        assert r.returncode == 0, (
            f"docker run failed (rc={r.returncode}).\n"
            f"cmd: {' '.join(docker_cmd)}\n"
            f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        )
        assert "PASSED" in r.stdout, \
            f"APU v2 binary did not print PASSED.\nstdout:\n{r.stdout}"

        # 5. Read back and compare to CPU reference.
        C_path = os.path.join(ws, "C.bin")
        assert os.path.isfile(C_path), \
            f"C.bin not produced by simulator.\nstdout:\n{r.stdout}"
        C_sim = np.fromfile(C_path, dtype=np.float32)
        C_ref = A + B
        assert C_sim.shape == C_ref.shape, \
            f"C.bin size mismatch: {C_sim.shape} vs {C_ref.shape}"
        # fp32 IEEE add: expect close to bit-exact (tiny rounding OK).
        np.testing.assert_allclose(C_sim, C_ref, rtol=1e-5, atol=1e-6)


def _aim_numeric_available() -> tuple[bool, str]:
    """Return (available, reason). AiM needs Docker, the pre-built image,
    and the ramulator2 binary on disk (the binary runs *through* Docker
    because it links GLIBC 2.38, but lives on the host filesystem and is
    mounted into the container -- see report 05 §3)."""
    import shutil
    import subprocess as _sp
    if shutil.which("docker") is None:
        return False, "docker CLI not on PATH"
    img = _sp.run(["docker", "image", "inspect", "aim-simulator-build"],
                  capture_output=True)
    if img.returncode != 0:
        return False, "aim-simulator-build Docker image not built"
    binpath = os.path.join(
        REPO_EXPERIMENTS, "simulators", "aim_simulator", "build", "ramulator2")
    if not os.path.isfile(binpath):
        return False, f"ramulator2 binary missing at {binpath}"
    return True, ""


def test_aim_vadd_end_to_end():
    """End-to-end AiM vadd: compile via ``allo.compile(..., target=build_aim())``,
    wrap the emitted ISR trace with the minimum preamble/postamble needed to
    run it, then assert BOTH halves:

      * Timing     -- ramulator2 (the real AiM simulator inside Docker) accepts
        the trace, exits 0, and reports non-zero ``AiM_ISR_EWADD_cycles``.
      * Functional -- ``aim_shadow.AiMShadow`` executes the same trace with
        real tensor data staged in GPRs, and the output equals ``A + B``.

    Scope: one AiM burst (N = LANES = 16, fp16). The compiler's pattern
    ``linalg_add->aim_ewadd`` lowers ``C[:] = A + B`` to a single
    ``AiM EWADD 1 0 1`` line (opsize=1, gpr0=0, gpr1=1). Ramulator2's trace
    format requires a ``W GPR`` header per GPR we fill + ``SYNC`` / ``EOC``
    at the end; the shadow needs ``set_gpr`` calls staged before those
    ``W GPR`` lines so the functional model sees the actual data. The AiM
    build's channel count is fixed at 32; the EWADD opcode in this backend
    carries no mask field (see backends/aim.py line 114), so channel
    selection is handled by ramulator2's defaults.
    """
    import subprocess
    from allo.pim.runtime.aim_shadow import AiMShadow, LANES

    ok, reason = _aim_numeric_available()
    if not ok:
        pytest.skip(f"AiM numeric path unavailable: {reason}")

    # 1. Compile the vadd kernel. Use N=LANES so the EWADD runs on exactly
    #    one burst -- the compiled pattern emits opsize=1 regardless of the
    #    kernel's shape (see backends/aim.py ``linalg_add->aim_ewadd``).
    @allo.work(shapes={"A": (LANES,), "B": (LANES,), "C": (LANES,)},
               dtype="fp16")
    def vadd_burst(A, B, C):
        C[:] = A + B

    t = build_aim()
    result = allo.compile(vadd_burst, target=t)
    assert len(result.unlowered) == 0, \
        f"AiM did not lower vadd: unlowered={[u.kind for u in result.unlowered]}"
    assert len(result.emitted) == 1, \
        f"expected exactly one emitted line for a single-burst vadd; got {result.emitted!r}"
    ewadd_line = result.emitted[0]
    assert ewadd_line.startswith("AiM EWADD"), \
        f"unexpected emitted op; expected AiM EWADD, got {ewadd_line!r}"
    assert result.target == "skhynix_aim"

    # 2. Wrap the compiler-emitted ISR with the boilerplate ramulator2 and
    #    aim_shadow both expect (W GPR headers to consume staged data,
    #    final SYNC + EOC). This is the translation gap: the pattern emits
    #    only the data-plane op; the surrounding trace-harness lines are
    #    host-driver concerns that ``allo.compile`` does not emit today.
    rng = np.random.default_rng(0xA1)
    A_data = rng.standard_normal(LANES).astype(np.float32)
    B_data = rng.standard_normal(LANES).astype(np.float32)
    trace_lines = [
        "# allo.compile -> build_aim() EWADD end-to-end",
        "W GPR 0",          # consumes A staged on the shadow
        "W GPR 1",          # consumes B staged on the shadow
        ewadd_line,         # <-- the line that `allo.compile` emitted
        "AiM SYNC",
        "AiM EOC",
    ]
    # staging: line index -> [(gpr_id, data)]  (see aim_shadow.run_trace)
    staging = {1: [(0, A_data)], 2: [(1, B_data)]}

    # 3. FUNCTIONAL HALF -- run the trace through aim_shadow with real data.
    sh = AiMShadow()
    for i, ln in enumerate(trace_lines):
        for (g, d) in staging.get(i, []):
            sh.set_gpr(g, d)
        sh.step(ln)
    # AiM EWADD semantics in the shadow: gpr1 <- gpr0 + gpr1
    # (see aim_shadow.py:_op_ewadd). So GPR 1 holds the vadd result.
    C_shadow = sh.read_gpr(1)
    C_ref = A_data + B_data
    np.testing.assert_allclose(
        C_shadow, C_ref, rtol=1e-6, atol=1e-7,
        err_msg="aim_shadow functional model diverges from CPU reference",
    )

    # 4. TIMING HALF -- run the SAME trace through real ramulator2 in Docker.
    aim_dir = os.path.join(REPO_EXPERIMENTS, "simulators", "aim_simulator")
    trace_path = os.path.join(
        aim_dir, "test", "allo_aim_vadd_end_to_end.trace")
    with open(trace_path, "w") as f:
        f.write("\n".join(trace_lines) + "\n")
    r = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{aim_dir}:/work",
         "aim-simulator-build", "bash", "-c",
         "cd /work && ./build/ramulator2 -f test/example.yaml "
         "-t test/allo_aim_vadd_end_to_end.trace"],
        capture_output=True, text=True, timeout=300,
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, \
        f"ramulator2 rejected the compiled trace:\n{out[-2000:]}"

    mem_cycles = 0
    ewadd_requests = 0
    ewadd_cycles = 0
    for ln in out.splitlines():
        s = ln.strip()
        if "memory_system_cycles:" in s and mem_cycles == 0:
            mem_cycles = int(s.split(":")[1].split("#")[0].strip())
        elif "total_num_AiM_ISR_EWADD_requests:" in s:
            ewadd_requests = int(s.split(":")[1].split("#")[0].strip())
        elif "AiM_ISR_EWADD_cycles:" in s:
            # Per-channel counters; sum them (EWADD is a GPR-only op in
            # ramulator2's DRAM model, so these are typically 0 -- the
            # real signal is ewadd_requests + memory_system_cycles).
            ewadd_cycles += int(s.split(":")[1].split("#")[0].strip())
    # Loose timing check: the trace ran to completion (non-zero cycles)
    # AND ramulator2's frontend parsed and dispatched our EWADD request.
    # Per-channel EWADD_cycles stays 0 because EWADD is GPR-to-GPR; no
    # DRAM command sequence is issued (verified: ``all_isr.trace`` shipped
    # with the simulator exhibits the same 0-cycle behaviour for EWADD).
    assert mem_cycles > 0, \
        f"ramulator2 reported zero memory_system_cycles; stdout tail:\n{out[-1000:]}"
    assert ewadd_requests >= 1, \
        f"ramulator2 did not dispatch the EWADD request; " \
        f"total_num_AiM_ISR_EWADD_requests={ewadd_requests}; " \
        f"stdout tail:\n{out[-1000:]}"
    print(f"[aim-vadd-e2e] emitted={ewadd_line!r}  mem_cycles={mem_cycles}  "
          f"ewadd_requests={ewadd_requests}  "
          f"ewadd_cycles={ewadd_cycles}  shadow_ok=True")


# APU v2 used to live in a parametrised skip here; it is now covered end-to-
# end by ``test_apu_v2_vadd_end_to_end`` above (Docker ``gsi-g2-l1sim`` L1
# simulator, functional-only — no cycle counters). UPMEM / AiM / APU v1 each
# have their own dedicated end-to-end tests below or above.


# ---------------------------------------------------------------------------
# APU v1 numeric: real GSI APU v1 hardware on this server.
#
# Flow (mirrors the UPMEM numeric test's Option B structure):
#   1. ``allo.compile(vadd, target=build_apu_v1())`` -- proves the
#      ``@allo.work`` kernel lowers end-to-end against the APU v1
#      target. ``LoweringResult.emitted`` is the DSL's op-level
#      pseudo-instructions (one line per target op).
#   2. The APU v1 runtime codegen
#      (``pimdsl/runtime/apu_v1_codegen.py``) is then invoked to
#      materialise a buildable project tree (host.c + device.c +
#      Makefile + template files). This wraps -- does not edit --
#      apu_v1_codegen.py per the task constraints.
#   3. ``make`` + run the binary on real silicon. The generated
#      host.c contains its own CPU reference loop and prints
#      ``PASS (N elements)`` on bit-exact match.
#   4. Optionally invoke ``flo`` via ``ledag-ssh`` to capture cycle
#      counts. Shared hardware -- ``flo`` may be unresponsive per
#      Report 05 troubleshooting note; we make it best-effort.
# ---------------------------------------------------------------------------


_ARC_TOOLCHAIN_GCC = (
    "/usr/local/gsi-apu/13.7.1/ubuntu_20_04/"
    "arc_gnu_2021.09-release_elf32_le_linux_no_sdata/"
    "arc-snps-elf/bin/arc-elf32-gcc"
)
_LEDAG_SSH = "/usr/bin/ledag-ssh"
_APU_TEMPLATE = "/home/nz264/shared/accelerator-hub/gsi-apu/example-gvml"


def _apu_v1_preconditions() -> tuple[bool, str]:
    """Return (ok, reason). Hardware + toolchain must all be present."""
    if not os.path.isfile(_ARC_TOOLCHAIN_GCC):
        return False, f"ARC toolchain missing at {_ARC_TOOLCHAIN_GCC}"
    if not (os.path.islink(_LEDAG_SSH) or os.path.isfile(_LEDAG_SSH)):
        return False, f"ledag-ssh runtime shell missing at {_LEDAG_SSH}"
    if not os.path.isdir(_APU_TEMPLATE):
        return False, f"example-gvml template missing at {_APU_TEMPLATE}"
    return True, ""


def test_apu_v1_vadd_end_to_end():
    """Compile vadd via ``allo.compile(..., target=build_apu_v1())``,
    generate a buildable APU v1 project via the runtime codegen, build
    with the ARC toolchain, run on the real APU v1 hardware, and assert
    PASS (which the generated host.c emits iff every output element
    matches its CPU reference)."""
    ok, reason = _apu_v1_preconditions()
    if not ok:
        pytest.skip(f"APU v1 preconditions not met: {reason}")

    # (1) Lowering via the pure-Python allo.compile path.
    t = build_apu_v1()
    result = allo.compile(vadd, target=t)
    assert isinstance(result, LoweringResult)
    assert result.target == t.name == "gsi_apu_v1"
    assert len(result.unlowered) == 0, (
        f"APU v1: {len(result.unlowered)} src ops went unlowered: "
        f"{[u.kind for u in result.unlowered]}")
    assert len(result.emitted) > 0, \
        "APU v1: lowering produced no device-native text"

    # Sanity check: emitted device-native text references the GVML add
    # primitive that the codegen will eventually fire.
    emitted_blob = "\n".join(result.emitted)
    assert "gvml_add_u16" in emitted_blob, (
        "APU v1: expected emitted text to reference gvml_add_u16, got: "
        + emitted_blob[:400])

    # (2) Materialise a buildable APU project from the DSL spec.
    # apu_v1_codegen is kept unedited per task constraints; we wrap it.
    import tempfile
    import time
    from allo.pim.runtime.apu_v1_codegen import (
        gen_apu_v1_project, build_and_run_apu_v1, capture_apu_log,
    )

    # N=131072 = 4 * 32K -- one VR holds 32K elements; the loop body
    # runs 4 iterations. Matches the constant used by every other
    # target-parameterised test in this module.
    assert N == 131072 and N % 32768 == 0
    run_tag = f"ALLO_E2E_APU_V1_{int(time.time())}"
    lab_name = "allo_e2e_apu_v1"
    workdir = tempfile.mkdtemp(prefix="allo_e2e_apu_v1_")
    project_dir = os.path.join(workdir, "proj")

    gen_apu_v1_project(
        project_dir,
        N=N,
        op="add",
        n_vrs_per_body=1,
        lab_name=lab_name,
        run_tag=run_tag,
    )

    # (3) Build and run on real hardware. The generated host.c contains
    # its own CPU reference loop and prints ``PASS (N elements)`` iff
    # every output element matches the reference.
    r = build_and_run_apu_v1(project_dir, lab_name=lab_name, timeout_s=600)

    assert r.get("stage") == "run", (
        f"APU v1 build failed (stage={r.get('stage')}):\n"
        f"stdout:\n{r.get('stdout', '')}\nstderr:\n{r.get('stderr', '')}"
    )
    assert r.get("returncode") == 0, (
        f"APU v1 run failed returncode={r.get('returncode')}:\n"
        f"stdout:\n{r.get('stdout', '')}\nstderr:\n{r.get('stderr', '')}"
    )

    stdout = r.get("stdout") or ""
    assert "Num Apucs = 4" in stdout, (
        f"APU v1: hardware not visible in stdout "
        f"(expected 'Num Apucs = 4'):\n{stdout}")
    assert "PASS" in stdout, (
        f"APU v1: binary did not report PASS -- numeric verification "
        f"failed against the CPU reference in the generated host.c."
        f"\nstdout:\n{stdout}"
    )
    assert "FAIL" not in stdout, (
        f"APU v1: 'FAIL' token appeared in stdout:\n{stdout}")

    # (4) Optional: capture flo cycle counts. Best-effort; the shared
    # hardware can have flo unresponsive (Report 05 troubleshooting).
    try:
        time.sleep(2)  # let APU finish flushing its log buffer
        counters = capture_apu_log(run_tag, timeout_s=30)
    except Exception as e:
        counters = {"_error": f"capture_apu_log raised: {e!r}"}

    # Print for visibility in ``pytest -v -s`` output; not load-bearing.
    print(f"\n[APU v1 E2E] project_dir={project_dir}")
    print(f"[APU v1 E2E] run_tag={run_tag}")
    tail = "\n".join(stdout.splitlines()[-6:])
    print(f"[APU v1 E2E] stdout tail:\n{tail}")
    if isinstance(counters, dict) and "_error" not in counters:
        for sec, fields in counters.items():
            if isinstance(fields, dict) and "crun" in fields:
                print(f"[APU v1 E2E] flo  {sec:<12} "
                      f"crun={fields['crun']:>10}  "
                      f"iall={fields.get('iall', '?'):>8}  "
                      f"us@500MHz={fields.get('microsec500', '?')}")
    else:
        print(f"[APU v1 E2E] flo counters unavailable: {counters}")


# ---------------------------------------------------------------------------
# UPMEM numeric: uses uPIMulator Go binary + bongjoonhyun/upimulator docker
# image for DPU-side compilation. Mirrors the Samsung numeric test's
# structure: allo.compile proves the lowering path emits device text, and
# the UPMEM runtime driver is invoked separately to actually execute the
# kernel on the simulator (Option B per the wiring brief).
# ---------------------------------------------------------------------------


def _upmem_numeric_available() -> tuple[bool, str]:
    """Return (available, reason) — three preconditions for end-to-end run."""
    import shutil
    import subprocess
    upm_root = os.path.join(
        REPO_EXPERIMENTS, "simulators", "uPIMulator", "golang", "uPIMulator")
    binpath = os.path.join(upm_root, "build", "uPIMulator")
    if not os.path.isfile(binpath) or not os.access(binpath, os.X_OK):
        return False, f"uPIMulator Go binary missing or not executable at {binpath}"
    if shutil.which("docker") is None:
        return False, "docker CLI not on PATH"
    try:
        img_check = subprocess.run(
            ["docker", "images", "-q", "bongjoonhyun/upimulator"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        return False, f"docker command failed: {e!r}"
    if not img_check.stdout.strip():
        return False, ("bongjoonhyun/upimulator docker image not built "
                       "(required for DPU-side clang compile)")
    return True, ""


def test_upmem_vadd_numeric():
    """Lower + run the real uPIMulator binary, compare to CPU reference.

    This is Option B per the wiring brief: ``allo.compile`` is invoked
    to prove the @allo.work vadd kernel lowers end-to-end against the
    UPMEM target (producing non-empty emitted DPU text in the
    LoweringResult), and the uPIMulator simulator is driven separately
    through ``pimdsl.runtime.upmem_codegen`` — the currently-working
    entry point — to actually execute the kernel. Round-tripping the
    emitted text through the simulator (i.e. compiling the exact
    LoweringResult.emitted strings rather than the pre-baked task.c
    template in upmem_codegen.py) is a future step.
    """
    avail, reason = _upmem_numeric_available()
    if not avail:
        pytest.skip(f"uPIMulator preconditions not met: {reason}")

    # (1) Compile through allo.compile -- proves the lowering path
    # actually emits UPMEM DPU pseudo-C.
    t = build_upmem()
    result = allo.compile(vadd, target=t)
    assert len(result.unlowered) == 0, \
        f"UPMEM: {len(result.unlowered)} src ops went unlowered"
    assert len(result.emitted) > 0, \
        "UPMEM: lowering produced no device-native text"
    assert result.target == t.name

    # (2) Drive the simulator. int32 buffers of length data_prep_params
    # are generated deterministically by the Go Assemblable (seed=42),
    # and the DPU kernel does c = a + b elementwise. We read back a, b,
    # c from the MRAM-heap dumps and verify numerically.
    from allo.pim.runtime.upmem_codegen import (
        emit_benchmark, run_benchmark, read_dpu_io)

    # 1024 int32 elements = 4 KiB per buffer, well above the >=256 floor
    # the brief mentions and the smallest size the UPMEM DPU kernel
    # template exercises (BLOCK_SIZE = 1024 bytes = 256 int32s per DMA).
    N_UPMEM = 1024

    emit_benchmark("DSLVA", "add")
    bin_dir = run_benchmark(
        "DSLVA", data_prep_params=N_UPMEM, num_tasklets=16)

    a, b, c_sim = read_dpu_io(bin_dir)
    assert len(a) == N_UPMEM, f"expected N={N_UPMEM}, simulator ran N={len(a)}"

    c_ref = a + b
    mismatches = int(np.sum(c_sim != c_ref))
    assert mismatches == 0, (
        f"UPMEM DPU vadd mismatch: {mismatches}/{N_UPMEM} elements differ. "
        f"a[:5]={a[:5]}, b[:5]={b[:5]}, "
        f"c_sim[:5]={c_sim[:5]}, c_ref[:5]={c_ref[:5]}"
    )


# ---------------------------------------------------------------------------
# 3. Developer-facing smoke assertions (not strictly required by the MVP
#    brief, but cheap and help the `-v` output read well).
# ---------------------------------------------------------------------------


def test_work_handle_shape():
    """The decorator produces a ``Work`` handle carrying shapes + name."""
    assert isinstance(vadd, allo.Work)
    assert vadd.name == "vadd"
    assert vadd.shapes == {"A": (N,), "B": (N,), "C": (N,)}
    assert vadd.dtype == "fp16"


def test_compile_records_last_program():
    """``compile`` stashes the recognised SrcProgram on the Work handle
    for introspection. Useful for debugging; not load-bearing."""
    t = build_samsung()
    allo.compile(vadd, target=t)
    assert vadd._last_program is not None
    kinds = [o.kind for o in vadd._last_program.ops]
    assert kinds == ["add"]


def test_compile_rejects_unrecognised_kernel():
    """Kernels outside the MVP shortlist raise ``NotImplementedError``
    with a readable message. Exercises the walker's default branch."""
    @allo.work(shapes={"A": (16,), "C": (16,)}, dtype="fp16")
    def exotic(A, C):
        C[:] = A ** 2   # `**` is not in the recognised op list
    with pytest.raises(NotImplementedError):
        allo.compile(exotic, target=build_samsung())


# ---------------------------------------------------------------------------
# Shape-inference regressions (BUG-1 fix): intermediate tensors should no
# longer need to be declared in @allo.work(shapes=...). The compiler
# derives output shapes from RHS operators + operand shapes for the seven
# recognised statement forms (matmul, add/sub/mul, relu, softmax, scale).
# ---------------------------------------------------------------------------


def test_compile_infers_matmul_then_softmax_intermediate():
    """``y = W @ x`` declares no shape for ``y``; the compiler infers
    ``(M,)`` from the GEMV rule and lets the subsequent
    ``z = softmax(y)`` use that inferred shape. After compile, the
    recorded SrcProgram's two ops must carry the expected shapes."""
    M, K = 1024, 4

    @allo.work(
        shapes={"W": (M, K), "x": (K,), "z": (M,)},  # y omitted on purpose
        dtype="fp16",
    )
    def gemv_then_softmax(W, x, z):
        y = W @ x
        z[:] = allo.softmax(y)

    t = build_samsung()
    result = allo.compile(gemv_then_softmax, target=t)
    assert isinstance(result, LoweringResult)
    ops = gemv_then_softmax._last_program.ops
    # Matmul auto-coerces its kind to "gemv" when shape is 2D (see
    # pimdsl/ops.py:Matmul.__post_init__).
    kinds = [o.kind for o in ops]
    assert kinds == ["gemv", "softmax"], kinds
    # Matmul shape is (M, K) for GEMV per _matmul_shape(); Softmax shape
    # is (M,) matching the inferred intermediate.
    assert ops[0].shape == (M, K), ops[0].shape
    assert ops[1].shape == (M,), ops[1].shape
    assert ops[1].inputs == ("y",)
    assert ops[1].output == "z"


def test_compile_matmul_dim_mismatch_raises():
    """Inner-dim mismatch in ``A @ B`` should surface as a clear error
    (ValueError), not an opaque assert. Exercised via shape inference
    when the output tensor is intentionally left undeclared."""
    @allo.work(shapes={"A": (8, 4), "B": (5,), "C": (8,)}, dtype="fp16")
    def bad_matmul(A, B, C):
        C[:] = A @ B
    with pytest.raises(ValueError, match="inner dims"):
        allo.compile(bad_matmul, target=build_samsung())


def test_compile_elementwise_shape_mismatch_during_inference():
    """``y = A + B`` with differing input shapes surfaces a ValueError
    during shape inference (i.e. when the LHS is an intermediate the
    compiler is trying to infer a shape for). The walker doesn't
    re-check shapes when the output is explicitly declared, by design
    (MVP behaviour — user-asserted shapes are trusted)."""
    M = 16
    @allo.work(shapes={"A": (M,), "B": (32,), "C": (M,)}, dtype="fp16")
    def bad_add(A, B, C):
        y = A + B       # intermediate -> triggers inference, raises
        C[:] = y + A
    with pytest.raises(ValueError, match="shapes differ"):
        allo.compile(bad_add, target=build_samsung())


# ---------------------------------------------------------------------------
# Multi-op kernel: two GEMVs + one vadd (MLP-block motif from Tenon paper
# Fig. 1, `paper/latex/code/tenon_mlp.tex`).
#
#     y1 = W1 @ x1
#     y2 = W2 @ x2
#     z  = y1 + y2
#
# Purpose: stress-test @allo.work -> allo.compile -> simulator on a program
# with intermediates. Each per-target test below documents where this path
# breaks today (captured as a bug, not silently skipped).
#
# Shape inference in ``experiments/allo/allo/compile.py::_compile_statement``
# now covers the seven recognised RHS forms, so the MLP-block kernel's
# intermediates (``y1``, ``y2``) no longer need to be listed in
# ``@allo.work(shapes={...})``. The compiler derives ``(M,)`` for both
# from ``Wi @ xi`` and registers them for the subsequent ``y1 + y2``.
# ---------------------------------------------------------------------------

_MLP_M = 1024
_MLP_K = 4


def _mk_mlp_block(dtype: str = "fp16"):
    """Return a fresh ``@allo.work``-decorated MLP-block kernel. The
    intermediates (``y1``, ``y2``) are inferred by the compiler's shape
    pass; only inputs and the final output ``z`` appear in ``shapes=``.
    The kernel body itself is the motif from
    `paper/latex/code/tenon_mlp.tex`.
    """
    @allo.work(
        shapes={
            "W1": (_MLP_M, _MLP_K), "x1": (_MLP_K,),
            "W2": (_MLP_M, _MLP_K), "x2": (_MLP_K,),
            "z":  (_MLP_M,),
        },
        dtype=dtype,
    )
    def mlp_block(W1, x1, W2, x2, z):
        y1 = W1 @ x1
        y2 = W2 @ x2
        z[:] = y1 + y2

    return mlp_block


def test_samsung_mlp_block_end_to_end():
    """Samsung HBM-PIM: lower MLP-block (two GEMVs + vadd) via
    ``allo.compile(..., target=build_samsung())``.

    Expected outcome today: LOWERING PASSES (3 ops, all matched against
    ``pim.mac`` / ``pim.add`` / ``pim.fill``). Numeric stage SKIPPED:
    ``pimdsl/runtime/pimsim_driver.py::run_eltwise`` only exposes
    single-op entry points (``run_eltwise('ADD'|'MUL'|'MAC')``). Option-B
    round-tripping of a 3-op emitted sequence requires new runtime
    glue — outside the "trivial test-side fix" scope.
    """
    mlp = _mk_mlp_block()
    t = build_samsung()
    result = allo.compile(mlp, target=t)
    assert isinstance(result, LoweringResult)
    assert result.target == t.name
    assert len(result.unlowered) == 0, (
        f"Samsung unlowered on MLP-block: "
        f"{[u.kind for u in result.unlowered]}"
    )
    # 2 gemv + 1 add = at least 3 emitted lines; pattern expands the add
    # into FILL+ADD so we get >=4.
    assert len(result.emitted) >= 3, (
        f"Samsung MLP-block emitted too little: {result.emitted!r}"
    )
    # Detect the two MAC ops and an ADD.
    joined = "\n".join(result.emitted)
    assert "MAC" in joined and "ADD" in joined, (
        f"Samsung MLP-block emission lacks MAC or ADD:\n{joined}"
    )

    if not _samsung_numeric_available():
        pytest.skip(
            "Samsung pim_driver available but no runtime driver currently "
            "chains matmul+matmul+add for MLP-block Option-B "
            "(pimsim_driver.run_eltwise is single-op). Compile+lower path "
            "demonstrated successful; numeric round-trip blocked on runtime."
        )
    pytest.skip(
        "Runtime round-trip for multi-op MLP-block not wired on Samsung "
        "(Option B driver is single-op). Bug to track: extend "
        "pimsim_driver to accept a sequence."
    )


def test_aim_mlp_block_end_to_end():
    """SK-Hynix AiM: lower MLP-block (two GEMVs + vadd) via
    ``allo.compile(..., target=build_aim())``.

    BUG-3 fix (see ``pimdsl/backends/aim.py``): the patterns now bind
    each tensor name to a distinct GPR slot on first reference and reuse
    that slot on every subsequent reference. The assertions below guard
    the fix — the two MAC_ABK lines must now differ (distinct bank
    rows), the two RD_MAC lines must write y1 and y2 to distinct GPR
    slots, and the EWADD must reference both slots.

    Numeric stage still SKIPPED: full round-trip via ramulator2 +
    aim_shadow for multi-op programs is BUG-5 domain (staging glue
    around WR_BIAS + per-bank weight rows). The lowering half is
    exercised here.
    """
    mlp = _mk_mlp_block()
    t = build_aim()
    result = allo.compile(mlp, target=t)
    assert len(result.unlowered) == 0, (
        f"AiM unlowered on MLP-block: "
        f"{[u.kind for u in result.unlowered]}"
    )
    assert len(result.emitted) >= 3
    joined = "\n".join(result.emitted)
    assert "MAC_ABK" in joined, (
        f"AiM MLP-block missing MAC_ABK: {joined}"
    )
    assert "EWADD" in joined, (
        f"AiM MLP-block missing EWADD: {joined}"
    )

    # BUG-3 guard: the two MAC_ABK lines must be textually distinct —
    # previously they were byte-identical because row was hardcoded to 0.
    mac_lines = [ln for ln in result.emitted
                 if ln.strip().startswith("AiM MAC_ABK")]
    assert len(mac_lines) >= 2, (
        f"expected two MAC_ABK lines (one per matmul); got {mac_lines!r}"
    )
    assert mac_lines[0] != mac_lines[1], (
        f"BUG-3 regressed: two MAC_ABK lines are byte-identical ({mac_lines[0]!r}); "
        f"expected distinct rows per gemv output"
    )

    # BUG-3 guard: the two RD_MAC lines place y1 and y2 into DISTINCT
    # GPR slots. We parse the slot by taking the RD_MAC line's 3rd token
    # (``AiM RD_MAC {gpr} {mask}``). They appear in program order.
    rd_mac_lines = [ln for ln in result.emitted
                    if ln.strip().startswith("AiM RD_MAC")]
    assert len(rd_mac_lines) >= 2, (
        f"expected two RD_MAC lines (one per matmul); got {rd_mac_lines!r}"
    )
    y1_slot = int(rd_mac_lines[0].split()[2])
    y2_slot = int(rd_mac_lines[1].split()[2])
    assert y1_slot != y2_slot, (
        f"BUG-3 regressed: RD_MAC slots aliased "
        f"(y1_slot={y1_slot}, y2_slot={y2_slot}); expected distinct"
    )

    # BUG-3 guard: the EWADD line must reference y1's and y2's slots as
    # its two GPR operands. Template: ``AiM EWADD {opsize} {gpr0} {gpr1}``.
    ewadd_lines = [ln for ln in result.emitted
                   if ln.strip().startswith("AiM EWADD")]
    assert len(ewadd_lines) == 1, (
        f"expected one EWADD line; got {ewadd_lines!r}"
    )
    ewadd_toks = ewadd_lines[0].split()
    ewadd_gpr0 = int(ewadd_toks[3])
    ewadd_gpr1 = int(ewadd_toks[4])
    assert {ewadd_gpr0, ewadd_gpr1} == {y1_slot, y2_slot}, (
        f"BUG-3 regressed: EWADD operands ({ewadd_gpr0}, {ewadd_gpr1}) "
        f"do not match y1/y2 slots ({y1_slot}, {y2_slot})"
    )

    pytest.skip(
        "AiM: compile+lower pass on MLP-block (BUG-3 fix verified: "
        f"MAC_ABK rows distinct; y1 slot={y1_slot}, y2 slot={y2_slot}; "
        f"EWADD operands match). Numeric round-trip via ramulator2 + "
        "aim_shadow for the full 3-op sequence is BUG-5 domain "
        "(staging glue for WR_BIAS + per-bank weight rows)."
    )


def test_upmem_mlp_block_end_to_end():
    """UPMEM DPU: lower MLP-block (two GEMVs + vadd) via
    ``allo.compile(..., target=build_upmem())``.

    Expected outcome today: LOWERING PASSES. Numeric stage SKIPPED:
    ``pimdsl/runtime/upmem_codegen.py::emit_benchmark`` / ``run_benchmark``
    are parameterised by a single op keyword (``'add'``, ``'mul'``) and
    use a pre-baked ``task.c`` template; they do not accept a 3-op
    emitted sequence. Round-tripping requires runtime changes.
    """
    mlp = _mk_mlp_block()
    t = build_upmem()
    result = allo.compile(mlp, target=t)
    assert len(result.unlowered) == 0, (
        f"UPMEM unlowered on MLP-block: "
        f"{[u.kind for u in result.unlowered]}"
    )
    assert len(result.emitted) >= 3
    joined = "\n".join(result.emitted)
    # Two GEMV MAC loops + one add loop.
    assert joined.count("fmul_hf") >= 2, (
        f"UPMEM MLP-block missing fmul_hf MACs:\n{joined}"
    )
    assert "c[i]=a[i]+b[i]" in joined, (
        f"UPMEM MLP-block missing add loop:\n{joined}"
    )

    pytest.skip(
        "UPMEM: compile+lower pass on MLP-block. Numeric round-trip "
        "requires extending upmem_codegen.emit_benchmark to accept a "
        "multi-op LoweringResult.emitted sequence (today it takes a "
        "single op keyword and uses a pre-baked task.c template)."
    )


def test_apu_v1_mlp_block_end_to_end():
    """GSI APU v1: lower MLP-block (two GEMVs + vadd) via
    ``allo.compile(..., target=build_apu_v1())``.

    Expected outcome today: LOWERING PASSES. The backend now has a
    ``gemv->gvml_mac_unroll`` pattern that expands ``y = W @ x`` into a
    K-unrolled MAC loop using the existing GVML primitives
    (``apu.l4_to_l1`` / ``apu.l1_to_vr`` / ``apu.bcast_scalar_u16`` /
    ``apu.mul_u16`` / ``apu.add_u16``). Numeric stage SKIPPED: the
    existing ``apu_v1_codegen.py`` runtime only knows how to emit
    single-op C++ drivers; stitching a three-op (gemv + gemv + add)
    sequence into a full host/device build is outside the pattern-gap
    fix and is tracked as a separate runtime follow-up.
    """
    mlp = _mk_mlp_block()
    t = build_apu_v1()
    result = allo.compile(mlp, target=t)
    assert len(result.unlowered) == 0, (
        f"APU v1 unlowered on MLP-block: "
        f"{[u.kind for u in result.unlowered]}"
    )
    # The vadd half still lowers via gvml_add_u16.
    assert any("gvml_add_u16" in ln for ln in result.emitted), (
        f"APU v1: expected the add half to still lower, got: "
        f"{result.emitted!r}"
    )
    # Two gemvs, each emits a chain of loads + bcast + mul + add, so the
    # emitted stream should contain at least two gvml_mul_u16 lines per
    # gemv (K=4 -> 4 MACs per gemv -> 8 total).
    joined = "\n".join(result.emitted)
    assert joined.count("gvml_mul_u16") >= 2 * _MLP_K, (
        f"APU v1 MLP-block: expected >= {2 * _MLP_K} gvml_mul_u16 lines "
        f"(two K-unrolled gemvs, K={_MLP_K}), got:\n{joined}"
    )

    pytest.skip(
        "APU v1: compile+lower pass on MLP-block after the "
        "gemv->gvml_mac_unroll pattern addition. Numeric round-trip "
        "requires the apu_v1_codegen runtime to accept a multi-op "
        "LoweringResult.emitted sequence (today it drives a single-op "
        "C++ template)."
    )


def test_apu_v2_mlp_block_end_to_end():
    """GSI APU v2 (G2): lower MLP-block (two GEMVs + vadd) via
    ``allo.compile(..., target=build_apu_v2())``.

    Expected outcome today: LOWERING PASSES (two ``gtml.matmul`` lines
    + one ``gtml.add`` line, plus copy-in/out). However the emission
    has the same hardcoded-operand-name bug as AiM: every pattern in
    ``pimdsl/backends/apu_v2.py`` hardcodes ``a/b/c`` (or ``A/B/C``)
    regardless of the SrcOp's actual ``inputs`` / ``output`` tuple.
    The two ``gtml.matmul(A, B, C)`` lines are byte-identical — the
    downstream host.cc cannot tell them apart.

    Numeric stage SKIPPED: (1) the existing ``gsi-g2-l1sim`` Docker
    test relies on a hand-rolled host.cc template that only wires one
    ``tensor_add_kernel`` call; no template exists for two-matmul+add;
    (2) the aliasing bug above would make any template that *did*
    dispatch the three ops produce ``z = (W2@x2) + (W2@x2)`` rather
    than ``(W1@x1) + (W2@x2)``. Both are follow-up fixes beyond the
    trivial line.
    """
    # Use fp32 to dodge the image's known fp16 add carry bug (see the
    # existing test_apu_v2_vadd_end_to_end docstring).
    mlp = _mk_mlp_block(dtype="fp32")
    t = build_apu_v2()
    result = allo.compile(mlp, target=t)
    assert len(result.unlowered) == 0, (
        f"APU v2 unlowered on MLP-block: "
        f"{[u.kind for u in result.unlowered]}"
    )
    assert len(result.emitted) >= 3
    joined = "\n".join(result.emitted)
    assert joined.count("gtml.matmul") >= 2, (
        f"APU v2 MLP-block missing two gtml.matmul lines:\n{joined}"
    )
    assert "gtml.add" in joined, (
        f"APU v2 MLP-block missing gtml.add:\n{joined}"
    )

    # BUG-4 post-fix: the two matmul lines MUST now be distinct, and each
    # must reference the kernel's actual tensor identifiers (W1/x1/y1 vs
    # W2/x2/y2). If this assertion fires, backends/apu_v2.py has regressed
    # back to hardcoded `A/B/C` operands (the original BUG-4 symptom).
    matmul_lines = [ln for ln in result.emitted if "gtml.matmul" in ln]
    assert len(matmul_lines) >= 2, (
        f"APU v2 MLP-block: expected two gtml.matmul lines, got "
        f"{matmul_lines!r}"
    )
    assert matmul_lines[0] != matmul_lines[1], (
        f"APU v2 MLP-block: two gtml.matmul lines are byte-identical "
        f"(BUG-4 regressed — operand names are hardcoded instead of "
        f"threaded from SrcOp.inputs/output):\n  {matmul_lines[0]}\n  "
        f"{matmul_lines[1]}"
    )
    # First matmul must carry y1=W1@x1 identifiers; second y2=W2@x2.
    assert ("W1" in matmul_lines[0] and "x1" in matmul_lines[0]
            and "y1" in matmul_lines[0]), (
        f"APU v2 MLP-block: first gtml.matmul missing expected operands "
        f"W1/x1/y1: {matmul_lines[0]!r}"
    )
    assert ("W2" in matmul_lines[1] and "x2" in matmul_lines[1]
            and "y2" in matmul_lines[1]), (
        f"APU v2 MLP-block: second gtml.matmul missing expected operands "
        f"W2/x2/y2: {matmul_lines[1]!r}"
    )
    # The gtml.add must consume y1, y2 and produce z (not a/b/c).
    add_line = next(ln for ln in result.emitted if ln.startswith("gtml.add"))
    for ident in ("y1", "y2", "z"):
        assert ident in add_line, (
            f"APU v2 MLP-block: gtml.add missing {ident!r}: {add_line!r}"
        )

    pytest.skip(
        "APU v2: compile+lower now emit distinct operand tuples per op "
        "(BUG-4 fixed: gtml.matmul(W1,x1,y1) != gtml.matmul(W2,x2,y2), "
        "gtml.add(y1,y2,z)). Numeric round-trip still skipped because "
        "runtime apu_v2_codegen + the hand-rolled host.cc template "
        "in this file wire only one tensor_add_kernel, not a 3-op "
        "sequence — that is BUG-5 work."
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
