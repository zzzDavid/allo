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


# ----------------------------------------------------------------------------- #
# Multi-op driver (BUG-5 closure).
#
# `LoweringResult.schedule` carries a typed trace of SrcOps in program order
# (with .kind / .inputs / .output / .shape). A multi-op program runs by
# dispatching each schedule entry to the single-op helper appropriate for its
# kind, threading intermediate tensors through a host-side `state` dict keyed
# by tensor name. pim_driver itself has no "program" mode — it's one invocation
# per op — so we stitch at the Python level.
# ----------------------------------------------------------------------------- #

# Samsung pim_driver has hardware-imposed minimum sizes:
#   - eltwise (ADD/MUL/RELU): N >= 131072
#   - GEMV: any M, K % 16 == 0
# Tensors coming from a high-level @allo.work kernel may be smaller (e.g. the
# MLP-block test uses M=1024, vvadd of 1024). For the eltwise path we therefore
# tile-and-repeat the small tensor up to 131072 and trim on the way out. A
# future compile pass will pad at the IR level; for now this is a host-side
# fixup documented in the docstring.
_PIM_ELTWISE_MIN = 131072


def _eltwise_padded(op: str, a: np.ndarray, b: np.ndarray = None) -> np.ndarray:
    """Dispatch an eltwise op at any N >= 16 (multiple of 16). If N below the
    simulator's hard floor of 131072, tile + trim. Result has the original N.
    """
    n = a.shape[0]
    if n >= _PIM_ELTWISE_MIN:
        return run_eltwise(op, a, b)
    # tile to the minimum, then trim
    reps = (_PIM_ELTWISE_MIN + n - 1) // n
    # round reps up so reps*n % 16 == 0 is automatic (n already mult of 16)
    pad_len = reps * n
    a_pad = np.tile(a, reps).astype(np.float16)[:pad_len]
    b_pad = None if b is None else np.tile(b, reps).astype(np.float16)[:pad_len]
    # pad to exact minimum (pad_len may be slightly above; trim inputs first
    # so we have a clean tile of length pad_len then trim to 131072 if needed)
    if pad_len > _PIM_ELTWISE_MIN:
        # Use the first 131072 elements (still a full tile of the original)
        a_pad = a_pad[:_PIM_ELTWISE_MIN].copy()
        if b_pad is not None:
            b_pad = b_pad[:_PIM_ELTWISE_MIN].copy()
    elif pad_len < _PIM_ELTWISE_MIN:
        # Shouldn't happen since reps was ceil, but guard anyway
        extra = _PIM_ELTWISE_MIN - pad_len
        a_pad = np.concatenate([a_pad, np.zeros(extra, dtype=np.float16)])
        if b_pad is not None:
            b_pad = np.concatenate([b_pad, np.zeros(extra, dtype=np.float16)])
    out = run_eltwise(op, a_pad, b_pad)
    return out[:n].copy()


def run_program(schedule, tensors: dict) -> dict:
    """Execute a multi-op Samsung program described by ``schedule``.

    ``schedule`` is the ``LoweringResult.schedule`` list — each entry is
    ``{"where": ..., "src": SrcOp, "pattern": ...}`` in program order.
    ``tensors`` is a ``{name: np.ndarray[fp16]}`` host-side state dict holding
    inputs (e.g. W1, x1, W2, x2) and slots for outputs (may be None).

    Returns the mutated ``tensors`` dict with every produced tensor present.
    Recognized kinds: ``gemv`` / ``matmul`` / ``mac`` (-> pim_driver GEMV),
    ``add`` / ``mul`` / ``relu`` (-> pim_driver eltwise). Other kinds raise.
    """
    state = dict(tensors)
    for step in schedule:
        src = step["src"]
        kind = src.kind
        if kind in ("gemv", "matmul", "mac"):
            wname, xname = src.inputs[0], src.inputs[1]
            W = state[wname]
            x = state[xname]
            assert W.dtype == np.float16 and x.dtype == np.float16, \
                f"Samsung run_program: {wname}, {xname} must be fp16"
            y = run_gemv(W, x)
            state[src.output] = y
        elif kind == "add":
            a, b = src.inputs
            state[src.output] = _eltwise_padded(
                "ADD", state[a].astype(np.float16), state[b].astype(np.float16))
        elif kind == "mul":
            a, b = src.inputs
            state[src.output] = _eltwise_padded(
                "MUL", state[a].astype(np.float16), state[b].astype(np.float16))
        elif kind == "relu":
            a = src.inputs[0]
            state[src.output] = _eltwise_padded(
                "RELU", state[a].astype(np.float16))
        else:
            raise NotImplementedError(
                f"Samsung run_program: kind={kind!r} not implemented "
                f"(supported: gemv/matmul/mac, add, mul, relu)")
    return state
