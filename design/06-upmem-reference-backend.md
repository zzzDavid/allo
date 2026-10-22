# UPMEM as a Tenon reference backend

UPMEM is represented with the same three independent programming abstractions
as Samsung HBM-PIM and SK hynix AiM:

1. `build_upmem_target()` describes structure and legal operations.
2. `upmem_cost` is executable Python that maps those operations to cycles and
   structural occupancy.
3. Each PolyBench leaf contains a complete MLIR-driven workload plus a
   declarative launch/partition plan and invokes
   `allo.compile(workload, target, cost)`.

No cycle formula is stored in the target.

## Sources and modeled structure

The hardware structure follows Gomez-Luna et al., *Benchmarking a New
Paradigm: An Experimental Analysis of a Real Processing-in-Memory
Architecture* (arXiv:2105.03814):

- one modeled rank has 64 independent DPUs;
- each DPU has 64 MiB MRAM, 64 KiB WRAM, 24 KiB IRAM, 256 B atomic memory,
  and 24 32-bit general-purpose registers per tasklet;
- a DPU supports at most 24 tasklets;
- tasklets share the DPU pipeline and DMA engine;
- DPUs cannot communicate directly, so cross-DPU combination is a host step.

The initial cycle constants follow the local uPIMulator checkout at commit
`870d916334e9ff0b190f555f951a9ec3c4257781` and its HPCA 2024 paper
(*Pathfinding Future PIM Architectures by Demystifying a Commercial PIM
Technology*): 14 pipeline stages, an 11-cycle revolver issue interval, 1 KiB
MRAM wordlines, 8-byte minimum DMA granularity, and the simulator's default
DRAM timing values. The physical 64/24/24 KiB capacities come from the device
paper; uPIMulator's doubled WRAM/IRAM address apertures are simulator
implementation details and are not exposed as extra physical capacity.

## MLIR compilation, ABI, and parallel execution

`UPMEMProgram` bypasses the contraction matcher. Every phase is first
customized into Allo MLIR, passed through the shared MLIR lowering pipeline,
and emitted as portable C. The retained source MLIR is also summarized into
general integer/float arithmetic, control-flow, and WRAM events for the cost
program. This keeps functional lowering and performance analysis tied to the
same compiler IR.

`ProgramABI`/`LaunchABI` describe BLOCK or BROADCAST placement, arbitrary
partition axes, halos, scalar metadata, aligned MRAM slots, and INPUT/OUTPUT/
INOUT readback. Packing always creates 64 equal-layout DPU images and gathering
reconstructs the exact logical NumPy arrays. Host-visible launch boundaries are
global barriers; the per-kernel execution manifest records required gather,
reduce, broadcast, redistribution, halo, pivot, temporal, and wavefront steps.
Each tensor also carries an F2 `LinearLayout` from its logical partition and
inner indices to physical `(dpu, tasklet, local)` coordinates. Shard ownership
and analytical DPU/tasklet fanout are derived from this map; see
`design/09-aim-upmem-linear-layout.md`.

The functional virtual runtime executes MLIR-derived C on the host. Canonical
loop-site identities and conservative dependence checks are derived from the
retained MLIR; stores must carry an exact unit-coefficient induction coordinate,
so modulo, floor-divide, scaled, and mixed-symbol indices fail closed. Function
names, phase labels, benchmark names, and emitted-C labels do not select
parallel regions. Proven frontiers receive an OpenMP static partition with up
to 64 workers, while recurrence and wavefront loops remain ordered.
Non-power-of-two tasklet requests map the largest contained F2 subgroup and
leave the remaining tasklets to runtime or DMA work.

Autoschedule candidates are ranked only when every phase owns a complete,
candidate-specific DPU translation unit. The current complete lowering is the
structurally proven rank-one pointwise subset. Its frozen source contains the
DPU `main`, ABI descriptor loads, MRAM transfers, barriers, exact
`NR_TASKLETS` build flag, and F2-active tasklet fanout. Each candidate retains
an immutable source/ABI manifest and distinct executable. Other generic MLIR
programs keep the exact portable-C incumbent but fail before candidate scoring
or promotion.

## Cost source

For an aggregate instruction stream `I` running on `T` tasklets, the initial
pipeline estimate is:

```text
issue_cycles = ceil(I * 11 / min(T, 11))
cycles       = issue_cycles + 13
```

An integer ADD contributes one instruction. A general 32-bit integer multiply
uses the paper's worst-case 32 `mul_step` instructions, so a source-level MAC
contributes 33 instructions. MRAM DMA rules use the simulator's row and burst
timings. Explicit host scatter, broadcast, and gather rules occupy the rank
link. These are calibration starting points, not target facts: an agent should
edit `allo/pim/costs/upmem.py` after running microprofiles on a real DIMM.

The operation constants are intentionally visible calibration assumptions.
There is no workload-specific correction factor: an agent should compile and
microprofile generated kernels, then edit this ordinary Python cost program.
Cycle estimates are never presented as simulator measurements.

## PolyBench scope and correctness

All 30 PolyBench leaves compile the repository's complete canonical Allo
function through MLIR. They cover returned arrays, mutations, reverse loops,
triangular bounds, branches/selects, division, square root, temporal stencils,
pivots, and wavefront plans. Each leaf compares every observable fp32 result
against that module's own NumPy reference and records `PASS` only after the
64-DPU ABI pack/gather round trip succeeds. Known repository semantics that
differ from upstream PolyBench (Durbin, Gram-Schmidt, and fused Heat-3D) are
recorded explicitly in the corresponding artifacts.

This is currently a functional **virtual target**, not a claim that all 30
programs ran in uPIMulator or on a DIMM. The checked-in simulator cannot expose
general floating-point array results, and this environment has no UPMEM SDK.
Unsupported generated device output remains fail-closed as a DPU
translation-unit fragment; its manifest lists the missing capabilities and
cannot provide a promotion fingerprint. Supported rank-one pointwise phases
emit a complete frozen unit, but no result is presented as hardware execution
without separate platform-bound evidence. `results.json` keeps portable-C
correctness separate from analytical cost cycles. The exact kernel-by-kernel
physical launch plan is in `design/08-upmem-polybench-execution-matrix.md`.
