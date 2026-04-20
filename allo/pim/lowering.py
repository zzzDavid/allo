"""Lower a SrcProgram against a Target, and (optionally) interpret it.

Outputs:
  - emitted:  list of target-native text lines (PIMCmd / AiM ISR / DPU C).
  - perf_ir:  per-source-op cost dicts + program-level rollup.
  - schedule: list of {"where": "pim"|"host", "src": SrcOp, "pattern": name}.
  - unlowered: source ops the target cannot realize *even with host fallback*.

`execute(result, prog, target, state)` runs each pattern's `compute` closure in
order (falling back to `src.compute(state)` for typed ops from allo.pim.ops).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .ops import SrcOp, SrcProgram
from .target import Target


@dataclass
class LoweringResult:
    target: str
    emitted: List[str] = field(default_factory=list)
    perf_ir: List[Dict] = field(default_factory=list)
    schedule: List[Dict] = field(default_factory=list)
    unlowered: List[SrcOp] = field(default_factory=list)

    @property
    def total_cycles(self) -> int:
        return sum(p.get("cycles", 0) for p in self.perf_ir)

    @property
    def total_energy_pJ(self) -> float:
        return sum(p.get("energy_pJ", 0.0) for p in self.perf_ir)

    @property
    def host_ops(self) -> int:
        return sum(1 for s in self.schedule if s["where"] == "host")

    def summary(self) -> str:
        lines = [f"# target={self.target}  cycles={self.total_cycles}  "
                 f"energy_pJ={self.total_energy_pJ:.1f}  "
                 f"host_ops={self.host_ops}"]
        if self.unlowered:
            lines.append("# UNLOWERED: "
                         + ", ".join(f"{u.kind}{u.shape}" for u in self.unlowered))
        return "\n".join(lines)


def lower(prog: SrcProgram, target: Target) -> LoweringResult:
    res = LoweringResult(target=target.name)
    for src in prog.ops:
        pat = target.find_pattern(src)
        if pat is None:
            res.unlowered.append(src)
            res.perf_ir.append({"src": src, "cycles": 0, "per_op": [],
                                "note": "no-pattern"})
            res.schedule.append({"where": "none", "src": src, "pattern": None})
            continue

        instrs = pat.lower(src, target)
        where = "pim"
        for (opname, args) in instrs:
            op = target.ops[opname]
            res.emitted.append(op.render(args))
            if opname.startswith("host."):
                where = "host"
        cost = target.analyze_cost(instrs, src)
        cost["src"] = src
        cost["pattern"] = pat.name
        res.perf_ir.append(cost)
        res.schedule.append({"where": where, "src": src, "pattern": pat.name})
    return res


def execute(result: LoweringResult, prog: SrcProgram, target: Target, state: Dict):
    """Run each source op's compute in order, mutating `state`.

    Priority: pattern-provided `compute` wins; otherwise fall back to
    `src.compute(state)` from the typed op class (allo.pim.ops).
    """
    for src in prog.ops:
        pat = target.find_pattern(src)
        if pat is None:
            raise RuntimeError(
                f"cannot execute: no pattern for {src.kind}{src.shape} on {target.name}"
            )
        if pat.compute is not None:
            pat.compute(src, state)
        else:
            # use the op's built-in semantics (typed subclass)
            try:
                src.compute(state)
            except NotImplementedError as e:
                raise RuntimeError(
                    f"cannot execute {src.kind}{src.shape} on {target.name}: "
                    f"pattern has no compute closure and the op has no built-in "
                    f"semantics. Either use a typed op from allo.pim.ops (Add, "
                    f"Matmul, Softmax, …) or pass `compute=` on the pattern."
                ) from e
