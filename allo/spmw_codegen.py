# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW codegen — Step E.Q3 of report 16.

Lowers a `(target, MatchTrace)` pair to a runnable backend artifact.
Q3's scope is exactly one workload (GEMV) on exactly one backend
(Samsung HBM-PIM via PIMSimulator). Anything outside that envelope
raises `NotImplementedError`.

Codegen consumes only the trace + the target spec — never the source
MLIR module. The trace already names every memref the workload binds
its op operands to (`OperandBinding.memref_name`), so we can route the
host's numpy inputs into the right roles without re-walking the IR.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

from .spmw_match import MatchTrace, MatchedOp


# --------------------------------------------------------------------- #
# Samsung PIMSimulator binary location
# --------------------------------------------------------------------- #


_HERE = os.path.dirname(os.path.abspath(__file__))
# allo/spmw_codegen.py -> allo/ -> experiments/allo/ -> experiments/ -> repo
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_SIM_DIR = os.path.join(_REPO, "experiments", "simulators", "PIMSimulator")
_BIN = os.path.join(_SIM_DIR, "pim_driver")


def _ensure_binary() -> None:
    if not os.path.isfile(_BIN) or not os.access(_BIN, os.X_OK):
        raise RuntimeError(
            f"pim_driver not built. Run: cd {_SIM_DIR} && scons -j32 pim_driver"
        )


def _sim_env() -> dict[str, str]:
    env = os.environ.copy()
    conda = env.get("CONDA_PREFIX")
    if conda:
        env["LD_LIBRARY_PATH"] = conda + "/lib:" + env.get("LD_LIBRARY_PATH", "")
    return env


def _read_fp16_blob(path: str, n_elems: int) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) != 2 * n_elems:
        raise RuntimeError(
            f"unexpected output blob size: {len(raw)} bytes, want {2 * n_elems}"
        )
    return np.frombuffer(raw, dtype=np.float16).copy()


_CYC_RE = re.compile(
    r"PIM_CYCLES\s+total=(\d+)\s+preload=(\d+)\s+exec=(\d+)\s+readback=(\d+)"
)


def _parse_cycles(stdout: str) -> dict[str, int]:
    m = _CYC_RE.search(stdout)
    if not m:
        return {"cycles": 0, "preload": 0, "exec": 0, "readback": 0}
    total, pre, exe, rb = (int(m.group(i)) for i in (1, 2, 3, 4))
    return {"cycles": total, "preload": pre, "exec": exe, "readback": rb}


# --------------------------------------------------------------------- #
# GEMV-shape detection
# --------------------------------------------------------------------- #


_AFFINE_CONST_RE = re.compile(r"\(\s*\)\s*->\s*\(\s*(-?\d+)\s*\)")


def _affine_const(text: str) -> int | None:
    """Parse an MLIR affine_map constant like `affine_map<() -> (32)>`.

    Returns the integer if it's a constant map, else None.
    """
    if text is None:
        return None
    m = _AFFINE_CONST_RE.search(str(text))
    return int(m.group(1)) if m else None


@dataclass
class _GemvShape:
    """Detected GEMV-loop shape, derived from a MatchTrace."""

    rows_per_tile: int  # outer-loop trip count per work-item
    K: int  # inner-loop trip count (reduction dim)
    num_work_items: int  # number of distinct func_name's bearing MAC matches
    M_total: int  # rows_per_tile * num_work_items
    x_memref: str  # memref name bound to op-role "x"
    y_memref: str  # memref name bound to op-role "y"
    acc_memref: str  # memref name bound to op-role "acc"


def _detect_gemv_shape(trace: MatchTrace) -> _GemvShape:
    macs = trace.by_target_op("MAC")
    if not macs:
        raise NotImplementedError(
            "compile_for_target: trace contains no MAC matches; "
            "Q3 only supports the GEMV (MAC-reduction) shape."
        )

    # All MAC matches must share the same operand-role / memref binding.
    proto = macs[0]
    proto_roles = [b.role for b in proto.operands]
    if proto_roles != ["x", "y", "acc"]:
        raise NotImplementedError(
            f"compile_for_target: GEMV expects MAC with roles "
            f"['x','y','acc'], got {proto_roles!r}"
        )
    proto_memrefs = tuple((b.role, b.memref_name) for b in proto.operands)
    proto_loop_carried = [b.is_loop_carried for b in proto.operands]
    if proto_loop_carried != [False, False, True]:
        raise NotImplementedError(
            "compile_for_target: GEMV expects only the `acc` operand "
            f"to be loop-carried, got {proto_loop_carried!r}"
        )
    if proto.result_memref_name != proto.operands[2].memref_name:
        raise NotImplementedError(
            "compile_for_target: GEMV expects MAC to reduce into the "
            f"`acc` memref, got result_memref_name={proto.result_memref_name!r} "
            f"vs acc memref={proto.operands[2].memref_name!r}"
        )

    for m in macs[1:]:
        memrefs = tuple((b.role, b.memref_name) for b in m.operands)
        if memrefs != proto_memrefs:
            raise NotImplementedError(
                "compile_for_target: GEMV expects every MAC match to bind "
                f"the same operand memrefs; {m.func_name} differs "
                f"({memrefs!r} vs {proto_memrefs!r})"
            )

    # Two enclosing loops, both with constant bounds: outer = rows_per_tile,
    # inner = K. The matcher reports them outer-first.
    if len(proto.enclosing_loops) < 2:
        raise NotImplementedError(
            "compile_for_target: GEMV expects MAC nested in >=2 affine.for "
            f"loops, got {len(proto.enclosing_loops)}"
        )
    outer_lb = _affine_const(proto.enclosing_loops[-2][1])
    outer_ub = _affine_const(proto.enclosing_loops[-2][2])
    inner_lb = _affine_const(proto.enclosing_loops[-1][1])
    inner_ub = _affine_const(proto.enclosing_loops[-1][2])
    if None in (outer_lb, outer_ub, inner_lb, inner_ub):
        raise NotImplementedError(
            "compile_for_target: GEMV expects constant affine bounds, got "
            f"loops={proto.enclosing_loops!r}"
        )
    rows_per_tile = outer_ub - outer_lb
    K = inner_ub - inner_lb
    if K <= 0 or rows_per_tile <= 0:
        raise NotImplementedError(
            f"compile_for_target: degenerate GEMV bounds "
            f"rows_per_tile={rows_per_tile}, K={K}"
        )

    num_work_items = len({m.func_name for m in macs})
    M_total = rows_per_tile * num_work_items

    return _GemvShape(
        rows_per_tile=rows_per_tile,
        K=K,
        num_work_items=num_work_items,
        M_total=M_total,
        x_memref=proto.operands[0].memref_name,
        y_memref=proto.operands[1].memref_name,
        acc_memref=proto.operands[2].memref_name,
    )


# --------------------------------------------------------------------- #
# Compiled artifact
# --------------------------------------------------------------------- #


_LOCAL_PREFIX = "local_"


def _strip_local(name: str) -> str:
    """Drop the `local_` prefix that allo's frontend adds to work-arg
    memrefs. The user-facing tensor for memref `local_W` is `W`.
    """
    if name and name.startswith(_LOCAL_PREFIX):
        return name[len(_LOCAL_PREFIX):]
    return name


class _SamsungGemvCompiled:
    """Compiled artifact: spawns PIMSimulator on .run(...).

    The trace tells us:
      - which kwarg name is the weight matrix (op role `x` -> `local_W` -> `W`)
      - which kwarg name is the input vector  (op role `y` -> `local_x` -> `x`)
      - which kwarg name is the output vector (`local_y` if present in the
        workload signature; we don't actually need it from the trace because
        the result comes back as a fresh array, but we surface it for the
        caller as the named output if they pass an empty placeholder).
    """

    def __init__(self, target, trace: MatchTrace, shape: _GemvShape):
        self._target = target
        self._trace = trace
        self._shape = shape
        # Tensor names for the runtime-input dict, derived from memrefs.
        self.weight_name = _strip_local(shape.x_memref)  # "W"
        self.vector_name = _strip_local(shape.y_memref)  # "x"
        # Output memref is `acc` (per the trace) — but the workload's external
        # output tensor is `local_y` -> `y`. The trace doesn't explicitly bind
        # `y` (the host-visible output) since the MAC's `dst` is the reg
        # `grf_b`/acc. We still expose the symbolic output name `y` so the
        # returned dict matches what the caller authored.
        self.output_name = "y"

    def run(self, **inputs: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        sh = self._shape
        if self.weight_name not in inputs:
            raise KeyError(
                f"missing required input {self.weight_name!r} "
                f"(expected the weight matrix)"
            )
        if self.vector_name not in inputs:
            raise KeyError(
                f"missing required input {self.vector_name!r} "
                f"(expected the input vector)"
            )
        W = inputs[self.weight_name]
        x = inputs[self.vector_name]

        # Shape & dtype validation
        if W.dtype != np.float16:
            raise TypeError(
                f"input {self.weight_name!r} must be fp16, got {W.dtype}"
            )
        if x.dtype != np.float16:
            raise TypeError(
                f"input {self.vector_name!r} must be fp16, got {x.dtype}"
            )
        if W.ndim != 2 or W.shape != (sh.M_total, sh.K):
            raise ValueError(
                f"input {self.weight_name!r} must have shape "
                f"({sh.M_total}, {sh.K}), got {W.shape}"
            )
        if x.ndim != 1 or x.shape != (sh.K,):
            raise ValueError(
                f"input {self.vector_name!r} must have shape "
                f"({sh.K},), got {x.shape}"
            )
        if sh.K % 16 != 0:
            raise NotImplementedError(
                f"PIMSimulator GEMV requires K % 16 == 0, got K={sh.K}"
            )
        if sh.M_total < 16:
            raise NotImplementedError(
                f"PIMSimulator GEMV requires M >= 16, got M={sh.M_total}"
            )

        _ensure_binary()
        with tempfile.TemporaryDirectory() as td:
            wp = os.path.join(td, "w.npy")
            xp = os.path.join(td, "x.npy")
            outp = os.path.join(td, "out.bin")
            np.save(wp, W)
            np.save(xp, x)
            cmd = [
                _BIN,
                "--op", "GEMV",
                "--weight", wp,
                "--in", xp,
                "--out", outp,
                "--output-dim", str(sh.M_total),
                "--input-dim", str(sh.K),
            ]
            r = subprocess.run(
                cmd,
                cwd=_SIM_DIR,
                env=_sim_env(),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    "pim_driver failed:\n"
                    f"  cmd: {' '.join(cmd)}\n"
                    f"  stdout:\n{r.stdout}\n"
                    f"  stderr:\n{r.stderr}"
                )
            # GEMV emits 16-burst raw fp16 per output row; reduce in fp32.
            bursts = _read_fp16_blob(outp, sh.M_total * 16).reshape(sh.M_total, 16)
            y = bursts.astype(np.float32).sum(axis=1).astype(np.float16)
            stats = _parse_cycles(r.stdout)
            stats["stdout"] = r.stdout
            stats["cmd"] = " ".join(cmd)

        outputs = {self.output_name: y}
        return outputs, stats


# Public type alias
Compiled = _SamsungGemvCompiled


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #


def compile_for_target(target: Any, trace: MatchTrace) -> Compiled:
    """Lower a `(target, trace)` pair to a runnable Samsung HBM-PIM artifact.

    The only target/workload combination supported in Q3 is
    `samsung_hbm_pim` × GEMV (per report 16). Everything else raises
    `NotImplementedError`.
    """
    target_name = getattr(target, "name", None)
    if target_name != "samsung_hbm_pim":
        raise NotImplementedError(
            f"compile_for_target: only target 'samsung_hbm_pim' is "
            f"supported in Q3, got {target_name!r}"
        )
    if trace.target_name != "samsung_hbm_pim":
        raise NotImplementedError(
            f"compile_for_target: trace target {trace.target_name!r} "
            f"does not match target spec 'samsung_hbm_pim'"
        )

    shape = _detect_gemv_shape(trace)
    return _SamsungGemvCompiled(target, trace, shape)
