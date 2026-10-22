# UPMEM PolyBench execution matrix

This document is an implementation inventory for faithful execution of the 30
PolyBench programs in `examples/polybench`.  It describes the semantics of the
actual Allo programs in this repository.  In a few places those programs differ
from canonical PolyBench (noted below).

## Fidelity contract

A UPMEM leaf is faithful only when all of the following are true:

1. The complete workload is lowered through Allo's MLIR pipeline to DPU C.  A
   dense contraction substituted for a larger algorithm is not the workload.
2. Every input, output, in/out buffer, scalar, and alias is represented by the
   host/DPU ABI.  NumPy is used only to construct inputs and validate outputs;
   it cannot perform a missing scale, mask, recurrence, reduction, or postpass.
3. Every DPU-local array access has a physical MRAM/WRAM allocation and every
   cross-DPU dependency appears as a host launch boundary plus an explicit
   gather, reduce, broadcast, redistribution, or halo exchange.  UPMEM DPUs do
   not communicate directly.
4. Work is distributed over the 64-DPU rank by flattened output elements,
   tiles, independent lines, or frontier elements.  Every generated kernel has
   a tail guard (`global_id < logical_extent`).  All 64 DPUs participate when
   the legal parallel frontier has at least 64 elements; a recurrence with a
   smaller frontier necessarily leaves some DPUs idle.
5. DPU execution uses tasklet-strided loops and barriers where needed.  Float
   division, square root, comparison, and conversion must remain real MLIR
   operations lowered to the UPMEM compiler/runtime implementation, not be
   silently converted to integer arithmetic or omitted.

The common MLIR lowering surface required by the suite is `func`, `scf` and/or
`affine` loops, calls, `memref` load/store/copy, `arith` integer/index and
floating add/subtract/multiply/divide/compare/select, and `math.sqrt`.  Reverse
loops, dynamic triangular bounds, and nested `scf.if` are required.  A launch
graph above individual DPU modules is needed for cross-DPU phases.

## Kernel-by-kernel matrix

`SMALL` values come from `examples/polybench/psize.json`.  “Partition” always
means across the 64 DPUs; each DPU then divides its local range over tasklets.

| # | Kernel and SMALL shape | MLIR-visible buffers | Exact dependencies and UPMEM partition/launch plan | Extra scalar operations |
|---:|---|---|---|---|
| 1 | `2mm`: P=40,Q=70,R=50,S=80 | in A[P,Q], B[Q,R], C[R,S], D[P,S]; return output[P,S] | Launch AB=A@B over flattened AB elements; gather/redistribute AB; launch ABC=AB@C; final elementwise `beta*ABC + alpha*D` in the second launch or a third launch; gather output. | float mul/add |
| 2 | `3mm`: P=40,Q=60,R=50,S=80,T=70 | in A[P,Q], B[Q,R], C[R,S], D[S,T]; return G[P,T] | AB=A@B and CD=C@D are independent launches (or one launch graph level); gather/redistribute both; launch G=AB@CD over flattened G elements; gather G. | mul/add |
| 3 | `adi`: TSTEPS=40,N=60 | in/out u,v,p,q [N,N] | Per timestep: partition the N-2 independent column solves for v; each line has a serial forward p/q recurrence and reverse substitution. Barrier and redistribute orientation. Partition N-2 independent row solves for u with the same forward/reverse structure. Barrier before next timestep. At most 58 independent lines exist at SMALL. | div, add/sub, mul, negation, reverse index |
| 4 | `atax`: M=116,N=124 | in A[M,N], x[N]; out y[N] (tmp[M] internal) | Launch tmp=A*x by output rows; gather/broadcast tmp. Launch y=A^T*tmp by output columns, with A repacked by columns or broadcast. Gather y. | mul/add |
| 5 | `bicg`: M=116,N=124 (source A[N,M]) | in A[N,M] (plus source-level copy), p[M], r[N]; in/out zero q[N],s[M] | q=A*p and s=A^T*r are independent. Use row-packed A for q and column-packed/rebroadcast A for s, or compute per-DPU partial s followed by host sum. Gather both. Preserve accumulation semantics. | mul/add |
| 6 | `cholesky`: N=120 | in/out A[N,N] | Outer row i is a strict frontier. For each j<i, parallelize the k<j dot into 64 partial sums, host-reduce, update/divide A[i,j], then continue j. Parallel-reduce the diagonal dot, update A[i,i], and apply sqrt. A blocked implementation may widen the frontier but must preserve the same dependencies. | div, sqrt, sub, mul |
| 7 | `correlation`: M=80,N=100 | source ABI has three read copies of data[N,M] and out corr[M,M]; mean/stddev/centered data are internal | Phase 1 partition columns for means. Reduce each column locally because a whole column is assigned to one DPU. Phase 2 compute variance/stddev and epsilon clamp by column. Broadcast mean/stddev. Phase 3 center/scale flattened data. Phase 4 distribute upper-triangle (i,j) correlations, then mirror. Host barriers separate phases. | div, sqrt, compare<=, branch/select, sub, mul/add |
| 8 | `covariance`: M=80,N=100 | in data[N,M]; out mean[M], cov[M,M] | Compute means by columns; broadcast means. Distribute all covariance pairs (or one triangle and mirror), reduce over N locally, divide by N-1, gather mean/cov. The Allo kernel does not mutate data. | div, sub, mul/add |
| 9 | `deriche`: W=192,H=128 | in imgIn[W,H]; out/in-out imgOut,y1,y2 [W,H] | Partition W independent rows for horizontal forward and reverse recurrences; combine y1/y2. Barrier, transpose/repack, then partition H independent columns for vertical forward and reverse recurrences; final combine. Each individual line is serial. Coefficients are host-computed scalars; exp/pow are not DPU-loop operations. | mul/add, reverse index |
| 10 | `doitgen`: R=25,Q=20,P=S=30 | in/out A[R,Q,S], in x[P,S], out scratch sum[P] | Flatten (r,q) into 500 independent vector-matrix products and distribute them. Each DPU computes complete output p values, writes A in place, and the ABI returns the final scratch semantics if exposed. Gather A. | mul/add |
| 11 | `durbin`: N=120 | in r[N]; out y[N] (`z` internal) | k=1..N-1 is serial. Per k, split the length-k dot product over DPUs and host-reduce; broadcast alpha. Partition z and y prefix updates over DPUs; barrier before k+1. SMALL uses all DPUs once k>=64. Repository semantics intentionally omit canonical `alpha /= beta`; fidelity means matching this source unless it is fixed globally. | mul/add/sub, negation; no active div in repository source |
| 12 | `fdtd_2d`: Tmax=40,Nx=60,Ny=80 | in/out ex,ey,hz[Nx,Ny]; in fict[Tmax] | Per t: set ey boundary; update ey from hz and ex from hz over flattened cells (these two fields can share a launch); exchange the required row/column halos or gather/redistribute; update hz from the new ex/ey; barrier before next t. 2-D tiles/flattened cell ranges allow all 64 DPUs despite Nx<64. | sub, mul/add |
| 13 | `floyd_warshall`: N=180 | in/out path[N,N] | k is serial. At each k broadcast pivot row path[k,:]; row-partition path so path[i,k] is local (or broadcast the pivot column too); update all (i,j) in parallel; barrier before k+1. All 64 DPUs participate. | add, compare>=, min/select |
| 14 | `gemm`: P=60,Q=80,R=70 | in A[P,Q],B[Q,R],C[P,R]; out output[P,R] | Distribute flattened output elements (not only 60 rows) so all 64 DPUs run; each computes dot(A row,B column), then `output=dot+beta*C`; gather output. Read-only operands may be packed by tile or broadcast at this SMALL size. | mul/add |
| 15 | `gemver`: N=120 | in/out A[N,N],x[N],w[N]; in u1,u2,v1,v2,y,z[N] | Phase 1 row-partition rank-2 update of A. Barrier. Phase 2 compute A^T*y: column-pack A or host-reduce row-partitioned partial x, apply beta and add the original x and z. Broadcast resulting x. Phase 3 row-partition w += alpha*A*x. Gather A,x,w. | mul/add |
| 16 | `gesummv`: N=90 | in A,B[N,N],x[N]; out y[N] | Row-partition outputs. In one launch compute tmp=A*x and bx=B*x, then y=alpha*tmp+beta*bx. Gather y. | mul/add |
| 17 | `gramschmidt`: M=60,N=80 | in/out A[M,N]; out Q[M,N],R[N,N] | k is serial. Reduce column norm over rows; broadcast R[k,k]; partition Q[:,k] normalization. For all j>k, compute distributed dot partials, host-reduce R[k,j], broadcast the R row, then partition updates to A[:,j]. Barrier before k+1. Repository semantics deliberately store squared norm rather than sqrt; match this source unless fixed globally. | div, mul/add/sub; no active sqrt in repository source |
| 18 | `heat_3d`: TSTEPS=40,N=20 | in/out A,B[N,N,N] | The repository fuses B-cell and A-cell updates inside a lexicographic i,j,k loop; its NumPy reference does too. Therefore this is not a conventional two-barrier Jacobi stencil. Preserve its read-after-write order with 3-D wavefronts and halo exchange at each wavefront, twice per cell, or first fix both global references to canonical two-phase PolyBench. Flattened 3-D frontiers use all 64 DPUs when wide enough. | add/sub, mul |
| 19 | `jacobi_1d`: TSTEPS=40,N=120 | in/out A,B[N] | Per timestep launch B from A over N-2 points, exchange one-element halos/barrier, launch A from B, exchange/barrier. All 64 DPUs can participate across 118 interior points. | add, mul |
| 20 | `jacobi_2d`: TSTEPS=40,N=90 | in/out A,B[N,N] | Per timestep launch B from A over flattened interior, exchange one-cell tile halos/barrier, launch A from B, exchange/barrier. | add, mul |
| 21 | `lu`: N=120 | in/out A[N,N] | i is a serial row frontier. Lower entries j<i are serial because A[i,j] consumes the current-row prefix; each k dot may use DPU partial reductions. Once the lower prefix is complete, distribute upper entries j>=i across DPUs. Barrier before i+1. | div, sub, mul |
| 22 | `ludcmp`: N=120 | in/out A[N,N]; in b[N]; out x,y[N] | First perform the same phased LU frontier as `lu`. Forward substitution has serial i with a distributed j<i dot reduction; backward substitution has reverse serial i with distributed j>i dot reduction and division. Gather A,x,y. | div, sub, mul, reverse index |
| 23 | `mvt`: N=120 | in A and source-level A_copy[N,N], y1,y2,x1,x2[N]; out x1_out,x2_out[N] | x1_out=x1+A*y1 by rows. x2_out=x2+A^T*y2 by column-packed A or host-reduced partials. The two launches are independent and must retain the input x vectors, not overwrite them with bare products. | mul/add |
| 24 | `nussinov`: N=180 | in seq[N]; in/out zero table[N,N] | Execute increasing interval length. All (i,j) intervals of one diagonal are independent after shorter diagonals complete, so distribute them across DPUs; reduce max over split k values within a DPU/tasklets or via host partial maxima. Barrier per diagonal. Early/late diagonals have fewer than 64 intervals. | compare, max/select, equality, add, nested branches |
| 25 | `seidel_2d`: TSTEPS=40,N=120 | in/out A[N,N] | This is an in-place lexicographic 9-point sweep, not Jacobi. Per timestep execute skewed wavefront `2*i+j`: the coefficient 2 is necessary because `(i,j)` reads the already-updated north-east value `(i-1,j+1)`. Exchange boundaries and impose a barrier per wavefront. Wide frontiers use all 64 DPUs. | add, div by 9 |
| 26 | `symm`: M=60,N=80 | source ABI duplicates read-only A[M,M],B[M,N]; in/out C[M,N] | Partition the 80 independent output columns across DPUs. Within a column execute the exact i/k triangular update order, including updates to C[k,j], diagonal term, alpha and beta. Broadcast A and column-pack B/C. This avoids cross-DPU C races. | compare/bounded loop, mul/add |
| 27 | `syr2k`: M=60,N=80 | duplicated A,B[N,M], Cin[N,N]; out Cout[N,N] | Partition flattened lower-triangle (i,j). Compute `beta*C + alpha*A_i dot B_j + alpha*B_i dot A_j`; copy upper triangle unchanged exactly as source. Gather Cout. | compare<=, mul/add |
| 28 | `syrk`: M=60,N=80 | duplicated A[N,M], Cin[N,N]; out Cout[N,N] | Partition flattened lower-triangle. Compute `beta*C + alpha*A_i dot A_j`; copy upper triangle unchanged exactly as source. Gather Cout. | compare<=, mul/add |
| 29 | `trisolv`: N=120 | in L[N,N],b[N]; out x[N] | i is serial. Split j<i dot product into DPU partial sums, host-reduce, divide by L[i,i], broadcast x[i], then advance. Uses all DPUs once i>=64. | div, sub, mul |
| 30 | `trmm`: M=60,N=80 | in A[M,M]; in/out B[M,N] | Partition 80 independent B columns across 64 DPUs. Within each column execute i in increasing order and k>i using still-unmodified later B rows, then multiply by alpha. Broadcast A and column-pack B; gather B. | compare/bounded loop, mul/add |

## Implementation status

All 30 leaves now compile the complete repository workload through MLIR, use
the general 64-DPU ABI and launch graph, and compare gathered results against
the repository NumPy reference. The source-loop inventory below remains the
partitioning rationale for those launch specifications.

## Source loop names and scalar bindings

The table below is intended to drive static DPU-partition annotations.  A
“safe” source loop means iterations can own disjoint outputs at the stated
phase boundary.  Reduction loops can be tasklet-local or deliberately outlined
as cross-DPU partial reductions, but must not simply be marked parallel.  A
“derived frontier” requires an MLIR transform because no rectangular source
loop has the needed independence.

| Kernel | Safe source loop(s), by phase | Serial/reduction loop(s) | Ambient scalar binding before `customize` |
|---|---|---|---|
| `2mm` | `mm1`: `(i0,j0)`; `mm2`: `(i1,j1)`; `ele_add`: `(i2,j2)` | reductions `k0`,`k1` | wrapper-local `alpha=0.1`, `beta=0.5` are free names in `ele_add`; make them explicit |
| `3mm` | `(i0,j0)` in `mm1`, `(i1,j1)` in `mm2`, `(i2,j2)` in `mm3` | `k0`,`k1`,`k2` reductions | none |
| `adi` | first directional solve outer `i`; second directional solve outer `i` (different orientation) | `t`, inner `j`, `j_rev` | globals `a,b,c,d,e,f` computed from N/TSTEPS in `adi()` |
| `atax` | `m` in `stage_M`; `n` in `stage_N` | `r`,`k` reductions | none |
| `bicg` | `stageQ.i1`; output index `j0` in an outlined/transposed `stageS` | `stageQ.j1`; source `stageS.i0` is the reduction contributing to every `s[j0]` | none |
| `cholesky` | second-region `j` (j>=i) only after the lower prefix; reduction fragments of `k` | outer `i`; first-region `j`; `k` reductions | none |
| `correlation` | mean `x`; stddev `x`; center `(x,y)`; correlation output pair `(i,j)` after phase extraction | inner `k`/`m` reductions | globals `N_float`, `epsilon` assigned by `correlation()` |
| `covariance` | mean `x`; covariance output pair `(i,j)` | `k`/`p` reductions | none; divisors derive from N |
| `deriche` | horizontal sweep outer `i`; vertical sweep outer `j`; combine `(i,j)` | forward/reverse line indices `j`,`j_inv`,`i`,`i_inv` | globals `a1..a8,b1,b2,c1,c2` assigned by `deriche()` |
| `doitgen` | `(r,q)` after privatizing `sum_`; output `p` within one `(r,q)` | inner `s` reduction | none |
| `durbin` | each prefix-update `i` after the sum/alpha barrier | outer `k`; first `i` is a reduction | none; canonical alpha/beta division is commented out |
| `fdtd_2d` | boundary `j`; ey cell `(i,j)`; ex cell `(i,j)`; hz cell `(i,j)`, each at its phase | timestep `m` | constants 0.5 and 0.7 are inline |
| `floyd_warshall` | `(i,j)` for a fixed extracted `k` phase | `k` | none |
| `gemm` | `mm1.(i0,j0)`; `ele_add.(i2,j2)` | `k0` reduction | wrapper parameter/default `beta=0.1` is a free name in `ele_add`; make explicit |
| `gemver` | rank update `(i,j)`; transpose product output `i`; z update `i`; final product output `i` | `j` reductions in product phases | wrapper defaults `alpha=beta=0.1` are free names; make explicit |
| `gesummv` | output `i` (`compute_tmp.tmp` after reduction); `store.i1`; `compute_y.load.i0` | `compute_tmp.j` reduction | wrapper defaults `alpha=beta=0.1` are free names; make explicit |
| `gramschmidt` | normalize `i`; projection column `j` (with private reduction), then update `i` for each j | outer `k`; norm `i` and projection `i` reductions | none; sqrt is commented out in both kernel/reference |
| `heat_3d` | derived 3-D frontier `i+j+k` only; execute B then A for each cell | timestep `m`; direct rectangular `i`,`j`,`k` are not parallel-safe in the fused repository program | inline 0.125 and 2.0 |
| `jacobi_1d` | first phase `i0`; second phase `i1` | timestep `m` | inline 0.33333 |
| `jacobi_2d` | `compute_A.(i0,j0)` and `compute_B.(i1,j1)` | timestep `m` | `TSTEPS` is a free name captured when `jacobi_2d()` customizes; make explicit |
| `lu` | upper-region `j` after lower prefix; outlined partial `k` reductions | outer `i`; lower-region `j` | none |
| `ludcmp` | LU upper-region `j`; outlined dot fragments in solve phases | LU `i` and lower `j`; solve `i`/`i_inv`; dot `j` reductions | none |
| `mvt` | `stageA.i0`, `stageB.i1` | `j0`,`j1` reductions | none |
| `nussinov` | derived interval-length diagonal `(i,j=i+length)` | source `i_inv`,`j`; `k` is max reduction | none |
| `seidel_2d` | derived skewed wavefront `2*i+j` | timestep `t`; direct `i`,`j` and plain `i+j` diagonals are not parallel-safe because `(i,j)` reads updated `(i-1,j+1)` | divisor 9 inline |
| `symm` | output column `j` after outlining the whole algorithm per column; `compute_sum.(i1,j1)` is also safe with private reduction | outer `i` order within each column; `k`/`k1` reductions | wrapper defaults `alpha=1.5`, `beta=1.2` are free names; make explicit |
| `syr2k` | `update.(i0,j0)`; output pair `(i1,j1)` after treating `k1` as reduction; `store.(i2,j2)` | `k1` | wrapper defaults `alpha=1.5`, `beta=1.2` are free names; make explicit |
| `syrk` | `update.(i0,j0)`; output pair `(i1,j1)` after treating `k1` as reduction; `store.(i2,j2)` | `k1` | wrapper defaults `alpha=1.5`, `beta=1.2` are free names; make explicit |
| `trisolv` | outlined fragments of `j` dot reduction only | outer `i`; `j` as a reduction | none |
| `trmm` | output column `j1` for the complete triangular update; S1 `(i0,j0)` after S0 barrier | S0 `i1` order within a column; `k1` reduction | wrapper default `alpha=1.5` is a free name in S1; make explicit |

Free Python/module names should not survive as hidden backend state.  The new
workload specifications should either expose them as scalar arguments in the
MLIR function ABI or materialize them as explicit compile-time attributes.
