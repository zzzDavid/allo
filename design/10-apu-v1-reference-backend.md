# GSI APU v1 reference backend

This document describes the APU v1 implementation in the current tree. It is
the reference for the target, cost, and workload abstractions; superseded
fixed-tiling and binary-LUT designs are intentionally omitted.

## Sources

- GSI manuals, headers, examples, profiling tutorial, and multicore tutorial in
  `/home/nz264/shared/accelerator-hub/gsi-apu`.
- [APU Performance Modeling](https://github.com/liya-mei/APU-Performance-Modeling),
  inspected at commit `20161ca64985b234b35b0028b633cbea21c922f4`.
- [GVML semantics](https://github.com/zzzDavid/gvml-semantics), inspected at
  commit `dd39afdc548c46a1262fc05081cb0414a1f2b86d`.
- A Gemini board running GSI 13.7.1, profiled on 2026-07-01 and
  2026-07-07.

## Target specification

`build_apu_v1_target()` describes structure and legality only. The device has
four APUCs. Each APUC contains an ARC controller, an asynchronous SEU, a DMA
engine, fifteen writable 32K x 16-bit vector registers, the read-only index
VR, forty-eight VMRs, one VIOR, and eight marker vectors. L4 is shared and has
four DMA ports so independent APUC transfers can overlap.

APU v1 vector code targets GVML. VL64 is a Gemini-II/APU-G2 interface and is
not part of this backend's lowering, runtime, or cost specification.

The target exposes two fixed spatial axes:

- `apuc`, with extent 4;
- `lane`, with extent 32,768 on every VR-like register.

A GVML group is not a hardware unit and is therefore not a child in the target
tree. It is a workload-selected partition of the `lane` axis. No latency or
throughput number appears in the target specification.

## Scalar mapping means group count

The minimal user interface is a scalar mapping:

```python
@allo.work(mapping=8, args=[A, B, C])
def gemm_k(A, B, C):
    ...
```

For APU v1, `mapping=N` means that every 32K VR is divided into exactly `N`
groups:

```text
group_count = N
group_size  = 32768 / N
vr_lane     = group * group_size + lane_in_group
```

`N` must be a power-of-two divisor of 32,768. The compiler coalesces the `N`
source work replicas into one GVML operation and constructs a `LinearLayout`
from logical `(group, lane_in_group)` to physical `vr_lane`. Thus `mapping=8`
selects eight 4K groups and emits one `GVML_P2_4K` operation, not eight serial
operations. A grouped reduction is legal only when its reduction extent fits
within `group_size`.

The four APUCs are an implicit physical launch dimension. Users do not include
them in `mapping`; the grouped callable partitions output rows across all four
cores.

## Three MLIR compilation paths

All supported paths use the same public composition point:

```python
compiled = allo.compile(workload, target, cost)
result = compiled(*numpy_arrays)
estimate = compiled.estimate()
```

### Group-parallel GVML

A plain dataflow workload with scalar mapping is customized to MLIR, matched,
and autoscheduled. The grouped compatibility path supports dense FP16
contractions. It binds the two inputs and accumulator to distinct VRs and
emits:

```c
gvml_mul_f16(mac_tmp_vr, vr0, vr1);
gvml_add_subgrps_f16_grp(
    vr2, mac_tmp_vr, GROUP_SIZE, GVML_P2_1, 0, GVML_VM_3, reduce_tmp_vr);
```

The NumPy ABI packs one dot product into each group, streams as many complete
VR batches as required, and gathers each result from its group head. The host
launches four GDL tasks as one batch using per-APUC L4 slices.

The `mapping=8` GEMM integration test emits `GVML_P2_4K`, uses all four APUCs,
and matches the NumPy result. Its calibration run measured 14,947,831 CRUN
against a 14,906,715-cycle analytical estimate.

### Ordinary contraction vector planning

An ordinary, non-dataflow Allo contraction now takes a separate retained-MLIR
path. `apu_v1_vectorize.py` proves the two parallel output axes and one
reduction axis from SSA accesses, then creates four immutable plans inspired by
the MICRO'25 mapping progression:

1. spatial group reduction;
2. temporal/SVP reduction;
3. temporal reduction with 32K DMA coalescing;
4. a broadcast-friendly temporal layout.

The plans use the same F2 `LinearLayout` algebra as AiM and UPMEM. Low tile
bits map to `vr_lane`; high output/reduction/temporal bits map injectively to
`vr_batch`. `ValueLayout` records operand axes and replica axes separately, so
the NumPy ABI, VR allocation, code generator, and cost program consume the
same physical meaning rather than each reconstructing a GEMM packing rule.

`allo.compile` ranks these plans with `apu_v1_vector_cost.py`, accepts a plan
name or plan object through `layout=`, and exposes `candidates`,
`selected_plan`, `realization`, `device_source()`, and `estimate()` on the
returned callable. Unknown arithmetic, unsafe reductions, unsupported dtypes,
and VR pressure above VR16_0..14 fail closed.

The full real-device milestone is
`tests/pim/apu_v1/vector_gemm/test_vector_gemm_apu_v1.py`: one ordinary Allo
1024x1024x64 FP16 contraction generates all four candidates. Every candidate
compiles through MLIR to GVML and matches all 1,048,576 NumPy outputs on the
Gemini board.

The same plan abstraction now realizes canonical `uint16` contractions with
`gvml_mul_u16`, `gvml_add_u16`, and `gvml_add_subgrps_u16_grp`. Unsignedness is
recovered from Allo's function ABI and load attributes even though MLIR stores
the element type as signless `i16`.

Transfers relate explicit `TransferLayout` values through ordered
`TransferRouteStep` objects. A `ReuseWindow` states where a representation is
created and across which logical tiles it remains resident. The optimized plan
therefore represents and realizes the MICRO mapping directly:

- one compact 128 KiB LHS image is copied to L3;
- 2,048 32-entry lookup tables expand to 32K compute layouts;
- eight RHS VRs each retain eight 1K rows, repeated across four 8K groups;
- those RHS VRs are loaded once and reused across 32 output batches;
- 2,048 subgroup operations select `k/8`'s resident VR and subgroup `k%8`;
- `OutputBatching` distinguishes 2,048 compute batches from 32 physical C
  batches and records every work-tile-to-output-lane placement.

Measured CRUN is 7,349,987 for the broadcast-friendly plan, 93,449,834 for
coalesced DMA, 6,254,825,903 for spatial baseline, and 6,258,832,854 for
temporal SVP. The executable model predicts the same ordering. The optimized
layout is 12.71x faster than coalesced DMA. Full evidence is retained in
`tests/pim/apu_v1/vector_gemm/{results.json,RESULTS.md}`.

### Target-neutral XNOR/popcount

Packed binary arithmetic is expressed in ordinary Allo rather than selected
by a backend-specific helper:

```python
result[row, column] += allo.popcount(
    ~(left[row, depth] ^ right[depth, column])
)
```

The frontend retains the popcount as `math.ctpop`. The contraction analyzer
then proves the SSA dataflow `xori(xori(lhs, rhs), all_ones) -> ctpop` and
records `allo.xnor_popcount` plus the packed operand width. The APU realization
lowers that neutral operation to GVML XOR, NOT, POPCOUNT, and signed ADD. Raw
popcount semantics remain distinct from the MICRO binary-dot expression
`2*popcount-word_bits`, whose shift/subtract sequence is only emitted when
that bipolar operation is requested explicitly.

The LLVM execution path lowers `math.ctpop` directly. The structured HLS/C
emitter also handles it with a width-preserving unsigned cast before the
compiler builtin, so negative narrow integers count their declared
two's-complement width rather than the width introduced by C integer
promotion. This keeps the APU scalar baseline and UPMEM portable-C path
semantically aligned with the vector path.

The real-device leaf
`tests/pim/apu_v1/vector_binary/test_vector_binary_apu_v1.py` compiles a
32x1024xK8 `int16` contraction through retained MLIR. It measured 248,424 CRUN
on GSI 13.7.1 and matched all 32,768 NumPy outputs bit-for-bit. The workload,
run command, environment, and evidence are retained in
`tests/pim/apu_v1/vector_binary/{results.json,RESULTS.md}`.

### Complete scalar ARC program

Control-heavy and non-vectorized programs are represented explicitly as an
`APUv1Program` containing one `APUv1Phase`:

```python
program = allo.APUv1Program((allo.APUv1Phase(kernel),))
compiled = allo.compile(program, target, cost)
```

The phase is customized to Allo MLIR and lowered by the portable C backend.
The real GSI ARC toolchain then compiles and links the result. NumPy argument
and result arrays are L4-backed; emitted local arrays are moved from the small
ARC stack to aligned L4 scratch. In-place results are copied to explicit result
buffers for host readback. A local Newton implementation supplies scalar square
root for kernels such as Cholesky and correlation.

This path currently executes on APUC 0. It is an explicit,
correctness-complete scalar baseline, not a performance claim: using four APUCs
faithfully requires phase boundaries, partition ownership, and cross-APUC
barrier semantics that arbitrary scalar programs do not yet declare.

### Structurally discovered hybrid programs

An `APUv1Phase` may request structural vectorization while retaining the
canonical Allo program as its only source of semantics:

```python
phase = allo.APUv1Phase(
    kernel,
    vectorize=True,
    vector_layout="temporal_dma_coalescing",
)
compiled = allo.compile(
    allo.APUv1Program((phase,)), target, cost, backend="functional"
)
```

The compiler walks retained MLIR, discovers dense contraction regions from SSA
accesses, and leaves all other functions as scalar ARC regions. It derives
dependencies from produced and consumed values, inserts barriers only at
cross-region boundaries. Canonical APU v1 PolyBench storage and vector compute
are both `uint16`, so these programs require no precision-conversion phases.
Selection is structural; neither discovery nor code generation switches on a
benchmark name.

The first canonical hybrid milestone covers GEMM, 2mm, and 3mm:

| program | discovered regions | resident boundary |
|---|---|---|
| GEMM | `mm1: vector -> ele_add: scalar` | `out_AB` |
| 2mm | `mm1: vector -> mm2: vector -> ele_add: scalar` | `out_AB`, `out_ABC` |
| 3mm | independent `mm1`/`mm2: vector -> mm3: vector` | join on `out_AB`, `out_CD` |

The functional executor uses the same region graph and modular arithmetic. It
keeps intermediates in its program arena, so all three programs report zero
host intermediate round trips. Each leaf compares bit-for-bit with a staged
`uint16` matrix-product reference; unlike the former FP16 policy, this is the
canonical program contract rather than an approximate precision mode.

Lowering also materializes a physical manifest with persistent-L4 values,
region artifacts, conversions, barriers, and inspectable host/device source.
Every vector region derives a balanced four-APUC row partition. The compiler
then replans each shard with a local zero-based row axis; LHS and output slices
shrink to the declared half-open interval, RHS values remain replicated, and
the ordinary `APUVectorABI` owns padding, ingress packing, and output
coordinates.

Canonical GEMM has a buildable unified GDL realization. Host ingress packing
creates four shard-local A/B images. One stitched 67 MB L4 arena holds those
images, four packed outputs, gather maps, dense `out_AB`, `C`, and the final
output for the complete run. A blocking four-task GVML batch covers row shards
`[0,15)`, `[15,30)`, `[30,45)`, and `[45,60)`; a second task on APUC 0 gathers
the packed uint16 values and calls the retained-MLIR `ele_add`. Only the final
output returns to the host.

The 2026-07-08 uint16 real-board critical path is 5,110,640 cycles. Per-APUC
vector CRUN is 4,398,074, 4,400,581, 4,419,068, and 4,415,788 cycles, followed
by 691,572 cycles for gather plus scalar epilogue. Output is bit-exact against
the modular reference. This is separate from the scalar-baseline record.

The same mechanism realizes the multi-vector programs without a host
intermediate. 2mm inserts a four-core ARC repack from `mm1` packed output to
`mm2` ingress, then runs the second vector batch and retained scalar epilogue.
Its measured uint16 critical path is 16,788,652 cycles and it is bit-exact.
3mm inserts a four-core join: `out_AB` is repacked between
matching row shards, while the differently partitioned `out_CD` is all-gathered
from its four producers into each `mm3` RHS image. It then runs `mm3` and a
core-0 final gather for 31,577,401 cycles. It is now bit-exact: the old
compounded-FP16 out-of-tolerance result is obsolete. The join/all-gather remains
the dominant phase at 20,873,332 cycles, exposing the next layout/residency
optimization target.

## Executable cost specification

`allo/pim/costs/apu_v1.py` is ordinary Python and is independent of the target
specification. It is the calibration surface an agent edits after running
microprofiles. All values are cycles. CRUN is end-to-end latency and SEU is
retained as asynchronous resource occupancy; no clock conversion is used.

Measured primitive values include:

| operation | SEU | CRUN |
|---|---:|---:|
| `gvml_add_f16` | 204 | 264 |
| `gvml_sub_f16` | 209 | 290 |
| `gvml_mul_f16` | 196 | 254 |
| group-reduce FP16, group 64 | 1,724 | 4,931 |
| group-reduce FP16, group 128 | 1,962 | 5,308 |
| group-reduce FP16, group 256 | 2,213 | 5,652 |
| `gvml_add_u16` | — | 12 |
| `gvml_sub_u16` | — | 13 |
| `gvml_mul_u16` | — | 114 |
| uint16 reduction, group 128 | — | 527 |
| direct L4-to-L1 32K DMA | 81 | 22,731 |
| direct L1-to-L4 32K DMA | 81 | 21,886 |

Unmeasured legal group sizes use a linear fit in `log2(group_size)`. Complete
grouped-kernel measurements separate a 72,555-cycle startup term from 14,651
cycles of ARC control per VR batch. The scalar ARC rule currently uses 52 CRUN
per retained MLIR operation plus startup, calibrated from canonical SMALL GEMM;
it should be replaced by an operation-mix regression as more profiles are
collected.

Cost events occupy named APUC, ARC, SEU, DMA, L4, operation, and move resources.
This lets the evaluator serialize contention and overlap independent APUCs or
DMA paths without encoding performance inside the target.

## PolyBench organization and coverage

`tests/pim/apu_v1` contains all 30 canonical PolyBench kernels. Every kernel
has the same leaf structure used by the other reference backends:

```text
<kernel>/
  workload.py
  test_<kernel>_apu_v1.py
  results.json
  RESULTS.md
```

Each leaf loads its local workload, constructs the target and cost, and calls
`allo.compile` visibly. A shared APU workload adapter specializes all 30 kernels
to `uint16`, quantizes deterministic inputs and coefficients, and preserves
modulo-2^16 stores. The canonical complete program runs through MLIR and the
scalar ARC path on the real board. Twenty-five kernels retain their repository
NumPy oracle. ADI, correlation, covariance, Durbin, and FDTD-2D mix widened
integers, modular arrays, and floating temporaries; for those, retained MLIR
compiled to host C is the exact scalar oracle used to differentially validate
the independent ARC/GVML lowering. All 30 SMALL cases compile and match their
oracle exactly. `results.json` stores measured CRUN and `RESULTS.md` mirrors it;
`tests/pim/COVERAGE.tsv` is regenerated from those records.

GEMM, 2mm, and 3mm additionally exercise structural hybrid discovery through
the same visible `allo.compile` call in each leaf. Their functional and
real-device evidence records exact uint16 results, explicit region/barrier
graphs, persistent-L4 execution, and zero host intermediate round trips.

GEMM additionally contains `group_workload.py` and a physical grouped-GVML
test for `mapping=8`. This keeps correctness coverage of arbitrary programs
separate from the optimized vector path.

The retained-MLIR analyzer now discovers dense contractions with one or more
parallel output axes, while APU vector-plan generation remains deliberately
restricted to the proven two-output-axis layout:

| PolyBench vector pattern | analysis | native APU vector plan | missing abstraction |
|---|---|---|---|
| GEMM and GEMM stages in 2mm/3mm | supported | supported | — |
| ATAX, BiCG, and MVT matrix-vector reductions | supported | fail closed | singleton second output/layout axis |
| Doitgen batched contraction | supported | fail closed | batched output-layout axis |
| SYRK/SYR2K triangular updates | not yet legal | fail closed | triangular-domain validity and symmetric ownership |
| Gemver multi-stage reductions/updates | individual forms only | fail closed | phase/fusion and intermediate residency |

The next enabled native-vector subset is ATAX/BiCG/MVT. Their one-dimensional
matrix-vector reductions already have proven load/reduction structure; they
need a singleton output axis carried consistently through `ValueLayout`,
`OutputBatching`, the NumPy ABI, and GVML egress. Doitgen is the following
layout extension because its extra output axis requires genuine batching,
not a singleton. Triangular and fused programs remain separate legality work.

## Current boundaries

- Native group-parallel lowering currently recognizes dense FP16
  contractions; canonical uint16 contractions use the ordinary retained-MLIR
  planner.
- Ordinary vector planning realizes rank-2 FP16 and uint16 contractions plus
  packed XNOR/popcount contractions. Rank-1 and batched contractions are analyzed but
  fail plan generation until their output-layout extensions are implemented.
  The spatial group-reduction realization requires a zero-initialized `C` and
  rejects nonzero accumulators until group-head input placement is modeled.
- The complete scalar realization currently accepts one phase and runs it on
  APUC 0.
- Hybrid functional execution supports persistent intermediates across vector,
  conversion, barrier, and scalar regions. Physical four-APUC execution is
  bit-exact for canonical uint16 GEMM, 2mm, and 3mm.
- The scalar cost is a coarse retained-operation model, not yet a per-op ARC
  regression.
- Device-context acquisition is an external GDL resource and may require a
  retry when another process has just released the board.
