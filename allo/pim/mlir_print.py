"""Serialize a Target + LoweringResult as MLIR-like textual IR.

This is a *textual sketch* of the `pim_target` dialect proposed in report 07
section 8. It is not parsed by a real MLIR context; the point is to show the
exact form the C++/TableGen dialect would take, and to let downstream passes
prototype against a stable text format.

Three IR levels:
  1. Target description  — `pim_target.describe / memory / op / pattern`.
  2. Source program with attached cost attributes — each source op is rendered
     with a `pim_target.cost` attribute carrying (cycles, energy_pJ, where).
  3. A rolled-up cost for the whole program as a `pim_target.program_cost` op.
"""
from __future__ import annotations

from typing import Dict, List

from .ops import SrcProgram
from .lowering import LoweringResult
from .target import Target


def target_to_mlir(t: Target) -> str:
    lines = []
    caps = ", ".join(f"{k}={str(v).lower()}" for k, v in t.caps.items())
    lines.append(f'pim_target.describe @{t.name} {{parallel_units = {t.parallel_units}, '
                 f'caps = #pim_target.caps<{caps}>}}')
    for name, m in t.memories.items():
        lines.append(f'pim_target.memory @{name} {{ parent = @{t.name}, '
                     f'capacity_bytes = {m.capacity_bytes} : i64, '
                     f'lanes = {m.lanes} : i32, scope = "{m.scope}", '
                     f'parallel_units = {m.parallel_units} : i32 }}')
    for name, o in t.ops.items():
        host = "true" if name.startswith("host.") else "false"
        lines.append(f'pim_target.op @"{name}" {{ parent = @{t.name}, '
                     f'lanes = {o.lanes} : i32, latency = {o.latency} : i32, '
                     f'throughput = {o.throughput} : f32, '
                     f'energy_pJ = {o.energy_pJ} : f32, is_host = {host} }}')
    for p in t.patterns:
        lines.append(f'pim_target.pattern @"{p.name}" {{ target = @{t.name} }}')
    return "\n".join(lines)


def program_to_mlir(prog: SrcProgram, res: LoweringResult) -> str:
    lines = [f'// Source program lowered for @{res.target}',
             f'func.func @kernel() {{']
    for src, perf, sch in zip(prog.ops, res.perf_ir, res.schedule):
        where = sch["where"]
        steps = [e["op"] for e in perf.get("per_op", [])]
        cost_attr = (f'#pim_target.cost<cycles = {perf.get("cycles",0)}, '
                     f'energy_pJ = {perf.get("energy_pJ",0.0)}, '
                     f'where = "{where}", pattern = "{sch["pattern"]}", '
                     f'steps = [{", ".join(repr(s) for s in steps)}]>')
        ishape = 'x'.join(str(d) for d in src.shape)
        lines.append(
            f'  %{src.output or src.name} = "src.{src.kind}"({", ".join("%" + i for i in src.inputs)}) '
            f'{{shape = [{ishape}], dtype = "{src.dtype}"}} '
            f': () -> tensor<{ishape}x{src.dtype}>  '
            f'{{pim_target.cost = {cost_attr}}}'
        )
    lines.append(f'  "pim_target.program_cost"() {{'
                 f'total_cycles = {res.total_cycles}, '
                 f'total_energy_pJ = {res.total_energy_pJ:.2f}, '
                 f'host_ops = {res.host_ops}}} : () -> ()')
    lines.append('  return')
    lines.append('}')
    return "\n".join(lines)
