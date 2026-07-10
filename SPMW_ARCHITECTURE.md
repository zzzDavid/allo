# Tenon architecture in Allo

This is the living map of the aggressively refactored implementation. Historical
design proposals under `design/` are useful rationale, but this file describes
the supported interfaces and paths in the current tree.

## Three programming abstractions

Tenon deliberately separates three concerns:

1. A **target specification** describes units, memories, registers, legal
   moves, operations, capacities, mappings, and spatial axes. It contains no
   latency formula.
2. A **cost specification** is ordinary executable Python. It maps retained
   operation/move events and their shape/layout metrics to cycle estimates and
   resource occupancy. An agent calibrates a backend by editing this code after
   microprofiling a device.
3. A **workload specification** is an Allo/MLIR program plus the partition and
   launch information required by its target.

The public composition point is:

```python
compiled = allo.compile(workload, target, cost)
result = compiled(*numpy_inputs)
estimate = compiled.estimate()
```

The target and cost specifications together form the virtual target used by
the autoscheduler and by simulator-free performance evaluation.

## Module map

| Module | Responsibility |
|---|---|
| `allo/compiler.py` | Public `allo.compile` facade and NumPy-callable wrapper. |
| `allo/spmw_target.py` | Structure-only target data model and decorators. |
| `allo/spmw_match.py`, `allo/spmw_match_engine.py` | MLIR matching for operation-pattern backends. |
| `allo/spmw_autoschedule.py` | Matcher-path placement enumeration for Samsung, AiM, and APU. |
| `allo/spmw_codegen.py` | Matcher-path code generation and Samsung/AiM/APU physical runners. |
| `allo/spmw_plan.py` | Target-neutral execution graph, cost events, dependencies, and resources. |
| `allo/spmw_linear_layout.py` | F2 `LinearLayout` algebra used by AiM and UPMEM placement. |
| `allo/perf/` | `CostSpec`, binding, event rules, and execution-graph evaluation. |
| `allo/pim/targets.py` | Reference Samsung, AiM, UPMEM, and APU v1 target specifications. |
| `allo/pim/costs/` | Executable reference-backend cost specifications. |
| `allo/pim/upmem_program.py` | Multi-phase MLIR compilation, launch orchestration, retained graph, and callable host-C runtime. |
| `allo/pim/upmem_abi.py` | LinearLayout-derived 64-DPU MRAM pack/gather ABI. |
| `allo/pim/upmem_analysis.py` | MLIR instruction/control-flow summary used to emit UPMEM cost events. |
| `allo/backend/c.py` | MLIR-to-portable-C backend shared by UPMEM and the APU v1 scalar path. |
| `allo/pim/apu_v1_program.py` | Complete MLIR-to-ARC-C APU v1 programs and NumPy/L4 ABI. |
| `allo/pim/apu_v1_layout.py` | Immutable APU iteration/value/transfer/reduction plans over `LinearLayout`. |
| `allo/pim/apu_v1_vectorize.py` | Retained-MLIR contraction analysis and four MICRO-inspired plans. |
| `allo/pim/apu_v1_vector_codegen.py` | Plan-driven GVML lowering, VR allocation, and NumPy packing ABI. |
| `allo/pim/apu_v1_vector_cost.py` | Executable plan-to-cost-graph bridge and candidate ranking. |
| `allo/pim/apu_v1_vector_program.py` | Public `allo.compile` callable for ordinary APU contractions. |
| `allo/pim/apu_v1_vector_runtime.py` | Real-board GDL harness for a realized APU vector plan. |
| `allo/pim/apu_v1_hybrid.py` | Structural region discovery, persistent-L4 physical manifests, hybrid execution graph, and functional orchestration. |
| `allo/pim/apu_v1_hybrid_runtime.py` | Four-APUC shard packing, persistent-L4 GDL project emission, packed-result gather, and hybrid device execution. |

## Backend paths

### Samsung HBM-PIM and SK hynix AiM

A plain Allo workload is customized into MLIR, matched against target
operations, autoscheduled, lowered to backend commands, and compiled into a
callable. AiM MAC bank fanout is derived from a `LinearLayout`, which selects
single-bank versus all-bank execution and drives both code generation and cost.

### UPMEM

UPMEM supports arbitrary multi-phase MLIR workloads through `UPMEMProgram`.
Every launch declares tensor directions, BLOCK/BROADCAST placement, partition
axes, halos, scalars, and orchestration barriers or collectives. Each tensor has
a logical-index map to physical `(dpu, tasklet, local)` coordinates. The same
map drives:

- MRAM shard ownership and pack/gather behavior;
- DPU and tasklet parallelism in the cost graph;
- the retained physical launch manifest.

The old contraction-only `UPMEMCtx`/uPIMulator matcher path has been removed.
Both `allo.compile` and `compile_for_target` fail closed if a plain callable or
matcher trace attempts to target UPMEM. This leaves one supported UPMEM path:

```python
allo.compile(allo.UPMEMProgram(...), build_upmem_target(), upmem_cost)
```

The host functional runtime executes MLIR-derived portable C and validates the
64-DPU ABI. Generated physical DPU C remains a fail-closed translation-unit
fragment until the SDK wrapper supplies DPU `main`, MRAM descriptor binding,
tasklet-strided entry, and build/run integration. Analytical cycles are never
reported as simulator or device measurements.

### GSI APU v1

The fixed spatial hierarchy is four APUCs, each with a named 32K-lane VR
axis. A GVML group is a workload-selected power-of-two partition of that lane
axis, not a fixed target unit. Scalar `mapping=N` means exactly `N` groups per
VR, so `group_size=32768/N`. The compiler coalesces the source work replicas
and builds a `LinearLayout` from logical `(group, lane_in_group)` to physical
`vr_lane`; all groups execute in one GVML operation.

Grouped FP16 contractions lower to `gvml_mul_f16` followed by
`gvml_add_subgrps_f16_grp`. The callable ABI packs one dot product per group,
partitions output rows over all four APUCs, streams multiple VR batches when
needed, and gathers the result at each group head. The host launches four GDL
tasks as one batch with per-APUC L4 slices.

Ordinary non-dataflow contractions use retained-MLIR access analysis to create
four immutable APU plans. `TransferLayout` and `TransferRouteStep` relate
compact L4/L3 representations, resident VR banks, and expanded compute
layouts; `OutputBatching` relates compute work tiles to dense physical output
VRs. The same relations drive NumPy packing, GVML lookup/subgroup lowering,
VR liveness, and executable costs. The full 1024x1024x64 reference test runs
all four candidates on silicon and preserves the measured ranking.

Fixed-width bitwise complement, `allo.xnor`, and `allo.popcount` are
target-neutral frontend operations. Popcount remains `math.ctpop` in retained
MLIR and lowers through both LLVM and portable C. The APU analyzer recognizes
the structural XNOR/popcount dataflow and emits faithful GVML
XOR/NOT/POPCOUNT/ADD operations; it does not implicitly apply the distinct
bipolar `2*popcount-word_bits` transform. The 32x1024xK8 reference leaf is
bit-exact on hardware. Contraction discovery now accepts one or more output
axes, while plan construction remains fail-closed outside the proven rank-2
layout; rank-1 ATAX/BiCG/MVT reductions are the next native-vector subset.

Complete non-vectorized programs use an explicit `APUv1Program` containing one
`APUv1Phase`. Allo MLIR lowers to portable C, the GSI ARC toolchain builds it,
and NumPy arguments and results use L4-backed buffers. This correctness path
currently runs on APUC 0 because arbitrary programs do not yet declare the
partition and barrier semantics needed for a faithful four-APUC launch.

An `APUv1Phase` with `vectorize=True` uses the same retained MLIR to discover a
hybrid region graph. Dense contractions become vector regions only after
illegal plans have been filtered by dtype, reduction, VR-pressure, zero-input,
and realization constraints; unmatched functions remain explicit ARC scalar
regions. Produced/consumed values derive dependencies and barriers, and an
explicit precision policy inserts conversion phases. No benchmark name is used
for region selection or code generation.

The physical lowering assigns persistent-L4 allocations and emits ordered
conversion, four-APUC vector, barrier, and APUC-0 scalar phases. The same
logical graph builds the analytical cost graph: independent regions may
overlap subject to concrete APUC/ARC/SEU/DMA/L4 occupancy, while data edges and
barriers serialize required transitions. The functional backend executes that
physical phase sequence without host intermediate round trips.

Canonical GEMM, 2mm, and 3mm leaf tests now call `allo.compile` directly and
assert the discovered region manifests. Their staged-FP16 functional references
match exactly and are recorded separately from the existing scalar real-device
CRUN evidence. A vector region has an explicit balanced four-APUC partition;
each shard is replanned with a local zero-based output axis, physical
power-of-two padding, exact validity masks, sliced LHS/output ownership, and a
replicated RHS.

Canonical GEMM now has a physical unified runner. Initial ABI packing creates
four shard-local ingress images, one stitched 67 MB L4 arena remains live across
a four-task GVML batch and a retained-MLIR scalar `ele_add` task on APUC 0, and
the host gathers only the final output. The measured critical path is 7,470,927
cycles: 4,456,146 for the slowest vector shard plus 3,014,781 for packed-result
gather and the scalar epilogue. The result passes both staged-FP16 and canonical
PolyBench tolerance. The public `backend=None` path installs this runner rather
than substituting the scalar baseline.

The same persistent-arena runner covers the milestone's multi-vector graphs.
For 2mm, an ARC repack batch maps packed `mm1` output coordinates directly to
`mm2` ingress before the second vector batch and scalar epilogue. It measures
18,566,489 cycles and passes both staged and canonical tolerance. For 3mm, the
join batch repacks row-sharded `out_AB` and all-gathers differently sharded
`out_CD` into every `mm3` ingress image. The complete graph measures 31,828,554
cycles, but compounded FP16 produces 104 staged and 129 canonical values outside
the repository tolerance. A sequential GVML-style reference that rounds every
multiply and add still has 75 values outside tolerance. It is recorded as
executed hardware evidence, not a correctness pass. In both cases,
intermediate data stays in L4.

`allo/pim/costs/apu_v1.py` contains the independently measured GVML/DMA
constants, explicit grouped-kernel startup/batch-control calibration, and a
coarse scalar ARC rule. The target contains only resources and legal
operations. See
`design/10-apu-v1-reference-backend.md`.

## Cost and parallel execution

A cost program emits events into an execution graph. Dependencies serialize
events; resource coordinates determine contention; disjoint spatial
coordinates may overlap. Operand shapes, layout image sizes, byte counts,
iteration counts, and candidate properties are event metrics rather than target
timing fields.

Reference costs are in:

- `allo/pim/costs/samsung.py`
- `allo/pim/costs/aim.py`
- `allo/pim/costs/upmem.py`
- `allo/pim/costs/apu_v1.py`

The UPMEM cost uses the DPU/tasklet fanout derived from tensor layouts. It does
not divide all phases by an unconditional rank size. Its constants are initial
assumptions sourced from the reference paper and simulator implementation and
are intended to be edited directly after real-device microprofiling.

## PolyBench coverage

The Samsung and AiM suites use their operation-pattern workloads. The UPMEM and
APU v1 suites contain all 30 repository PolyBench programs as complete MLIR
workloads. Each leaf calls `allo.compile` and validates every observable result
against the repository NumPy reference. UPMEM records analytical cost beside a
portable-C functional run; APU v1 records measured CRUN from the real scalar
ARC execution. APU v1 GEMM also tests the optimized four-APUC grouped-GVML path.
GEMM, 2mm, and 3mm additionally retain distinct hybrid functional and physical
evidence. GEMM and 2mm pass their physical precision contracts; 3mm is explicitly
marked out-of-tolerance rather than promoted to a false PASS.

See:

- `design/06-upmem-reference-backend.md`
- `design/08-upmem-polybench-execution-matrix.md`
- `design/09-aim-upmem-linear-layout.md`
- `design/10-apu-v1-reference-backend.md`

## Compatibility boundary

Tenon does not change upstream Allo's legacy `allo.memory.Layout` or FPGA/AIE
build paths. The PIM interfaces are additive at the public package boundary.
Backend-specific code must continue to keep cost formulas out of target
specifications and must not silently fall back from an unsupported physical
runner to an analytical estimate.
