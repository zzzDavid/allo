# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GSI APU v2 codegen context and l1_sim runtime (moved from spmw_codegen)."""

from __future__ import annotations

from .spmw_target import MemoryRef
from .spmw_codegen import (
    CodegenContext,
    RunResult,
    SimulatorUnavailable,
    Compiled,
)
from .spmw_simenv import (
    docker_image_unavailable_reason as _docker_image_unavailable_reason,
)


class APUv2Ctx(CodegenContext):
    """Codegen ctx for GSI APU v2.

    Output is a list of C++ source lines on `self.cmds`. Each entry is
    one GTML call (e.g. `g.add(out, lhs, rhs);`). `get_program_src()`
    wraps them with the standard GTML singleton boilerplate to produce
    a compilable `.cc` file.
    """

    def __init__(self, target):
        super().__init__(target)
        # Override the parent's PIMCmd list -- APU v2 emits C++ text.
        self.cmds: list[str] = []
        self._handle_names: dict[int, str] = {}
        self._tmp_counter = 0

    def bind_handle(self, handle, cpp_name: str) -> None:
        """Teach the ctx a handle's C++ container variable name.

        The autoscheduler populates this before walk_and_emit.
        """
        self._handle_names[id(handle)] = cpp_name

    def _name(self, handle) -> str:
        cached = self._handle_names.get(id(handle))
        if cached is not None:
            return cached
        if isinstance(handle, MemoryRef):
            mem = handle.memory
            return f"{mem.name}_ref_{handle.idx!r}"
        return f"opd_{id(handle)}"

    def cmd(self, name: str, dst=None, src0=None, src1=None, **kwargs):
        """Emit one GTML call as a C++ line.

        `name` is the fully-qualified function name (e.g. `"g.matmul"`
        or `"copy_to_l1"`). Operand order: dst first (out-parameter
        convention in GTML), then src0, src1.
        """
        parts = []
        if dst is not None:
            parts.append(self._name(dst))
        if src0 is not None:
            parts.append(self._name(src0))
        if src1 is not None:
            parts.append(self._name(src1))
        for k, v in kwargs.items():
            parts.append(f"/* {k}= */ {v!r}")
        self.cmds.append(f"{name}({', '.join(parts)});")

    def emit_mac_matmul(self, acc, x, y) -> None:
        """Expand MAC into the GTML matmul + add pattern.

        For a length-K dot product per work-item, this is one
        `g.matmul` (rank-1 x rank-1 -> scalar) followed by an add into
        the running acc. The canonical declarative form emits the
        per-iter expansion; a fully-vectorised matmul would replace
        this with a single outer-granularity `g.matmul`.
        """
        acc_n = self._name(acc)
        x_n = self._name(x)
        y_n = self._name(y)
        tmp = f"mac_tmp_{self._tmp_counter}"
        self._tmp_counter += 1
        self.cmds.append(f"L1Container {tmp} = g.alloc_vector();")
        self.cmds.append(f"g.matmul({tmp}, {x_n}, {y_n});")
        self.cmds.append(f"g.add({acc_n}, {acc_n}, {tmp});")

    def append(self, line: str) -> None:
        """Low-level escape hatch -- append a raw C++ source line."""
        self.cmds.append(line)

    def resolve_moves(self, role, src_handle=None, dst_handle=None):
        # l1_sim ignores moves at functional level; the cost model
        # already prices COPY_TO_L1 / COPY_FROM_L1 for autoschedule.
        return (None, None)

    def get_program_src(self) -> str:
        """Return the assembled G2 program source.

        Wraps `self.cmds` with the standard GTML singleton accessor
        plus `main()`. The coder may replace this wrapper later.
        """
        header = (
            '#include "gtml.h"\n\n'
            "int main() {\n"
            "    G2Gtml &g = G2Gtml::instance();\n"
        )
        body = "\n".join("    " + line for line in self.cmds)
        footer = "\n    return 0;\n}\n"
        return header + body + footer


def _run_apu_v2(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled APU v2 artifact through the `gsi-g2-l1sim` image.

    APU v2 / l1_sim is functional-only (no cycle counts; see report 13
    §1, `caps["perf_is_placeholder"] = True`). We pass the emitted
    C++ source through to the container so the user can inspect it,
    but always return `cycles=None`.
    """
    cpp_src = "\n".join(str(c) for c in compiled.cmds if isinstance(c, str))
    reason = _docker_image_unavailable_reason("gsi-g2-l1sim")
    if reason is not None:
        raise SimulatorUnavailable("apu_v2", reason)
    # l1_sim does not report cycles; we don't bother compiling here,
    # since the GTML build pipeline needs a real project tree. The
    # functional-correctness build flow is spec 014's territory.
    return RunResult(
        cycles=None,
        stdout="apu_v2 l1_sim is functional-only; cycles=None by design",
        backend="apu_v2",
        extra={"program_src": cpp_src},
    )
