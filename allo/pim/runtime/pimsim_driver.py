"""Python wrapper around the compiled Samsung PIMSimulator `pim_driver` binary.

End-to-end: emit .npy inputs -> run pim_driver -> read fp16 output -> verify.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# runtime/ -> pim/ -> allo/ -> allo/ -> experiments/ -> repo root
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
SIM_DIR = os.path.join(REPO, "experiments", "simulators", "PIMSimulator")
BIN = os.path.join(SIM_DIR, "pim_driver")


def _ensure_binary():
    if not os.path.isfile(BIN) or not os.access(BIN, os.X_OK):
        raise RuntimeError(
            f"pim_driver not built. Run: cd {SIM_DIR} && scons -j32 pim_driver"
        )


def _read_fp16_blob(path: str, n_elems: int) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read()
    assert len(raw) == 2 * n_elems, f"size {len(raw)} != {2*n_elems}"
    return np.frombuffer(raw, dtype=np.float16).copy()


def _env():
    env = os.environ.copy()
    # libgtest is from conda
    conda = env.get("CONDA_PREFIX")
    if conda:
        env["LD_LIBRARY_PATH"] = conda + "/lib:" + env.get("LD_LIBRARY_PATH", "")
    return env


def run_eltwise(op: str, a: np.ndarray, b: np.ndarray = None) -> np.ndarray:
    """Run ADD/MUL/RELU on real PIMSimulator. `a`, `b` must be fp16."""
    _ensure_binary()
    assert op in ("ADD", "MUL", "RELU")
    assert a.dtype == np.float16
    n = a.shape[0]
    assert n % 16 == 0 and n >= 131072, \
        f"PIMSimulator eltwise needs N >= 131072, got {n}"
    if op != "RELU":
        assert b is not None and b.dtype == np.float16 and b.shape == a.shape

    with tempfile.TemporaryDirectory() as td:
        in0 = os.path.join(td, "a.npy")
        np.save(in0, a)
        cmd = [BIN, "--op", op, "--in0", in0, "--n", str(n)]
        if b is not None:
            in1 = os.path.join(td, "b.npy")
            np.save(in1, b)
            cmd += ["--in1", in1]
        out = os.path.join(td, "out.bin")
        cmd += ["--out", out]
        r = subprocess.run(cmd, cwd=SIM_DIR, env=_env(),
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"pim_driver failed:\n{r.stdout}\n{r.stderr}")
        return _read_fp16_blob(out, n)


def run_gemv(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Run a GEMV. W: (M, K) fp16, x: (K,) fp16. Returns raw burst results;
    caller tree-reduces the 16 partial sums per output element."""
    _ensure_binary()
    assert W.dtype == np.float16 and x.dtype == np.float16
    assert W.ndim == 2 and x.ndim == 1 and W.shape[1] == x.shape[0]
    M, K = W.shape
    assert K % 16 == 0

    with tempfile.TemporaryDirectory() as td:
        wp = os.path.join(td, "w.npy"); np.save(wp, W)
        xp = os.path.join(td, "x.npy"); np.save(xp, x)
        op = os.path.join(td, "out.bin")
        cmd = [BIN, "--op", "GEMV",
               "--weight", wp, "--in", xp, "--out", op,
               "--output-dim", str(M), "--input-dim", str(K)]
        r = subprocess.run(cmd, cwd=SIM_DIR, env=_env(),
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"pim_driver failed:\n{r.stdout}\n{r.stderr}")
        # one burst (16 fp16) per output element -> we sum within each burst
        bursts = _read_fp16_blob(op, M * 16).reshape(M, 16)
        return bursts.astype(np.float32).sum(axis=1).astype(np.float16)
