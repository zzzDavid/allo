# AiM and UPMEM LinearLayout integration

This note records where the F2 `LinearLayout` is now load-bearing for the AiM
and UPMEM reference backends. It does not route non-power-of-two host fanout
through F2: host collectives remain ordinary integer fan degrees. The layouts
below describe device-local power-of-two topology only.

## AiM

An AiM MAC placement carries inputs `work_bank`, `lane_bank`, and `k`, with a
physical `bank` output of size 16.

- `work_bank -> bank` is identity for both candidates.
- Single-bank maps every `lane_bank` basis vector to zero. Its bank image has
  size 1 and 15 nonzero lane collisions.
- All-bank maps `lane_bank -> bank` identically. Its image has size 16 and zero
  collisions.

The image size selects `MAC_SBK` or `MAC_ABK`, becomes the cost model's bank
fanout, determines the reduction column width, and controls the emitted
`opsize`. The single-bank memory reference is materialized from the same map.
Duplicated `Placement.extra` fields cannot override these derived properties.

The native trace lowering retains the matched operand memref types as well as
the enclosing loop bounds.  From those two programming-abstraction facts it
recovers `(output extent, reduction extent, independent batch extent)` for a
GEMV, transposed GEMV, or matrix multiplication expressed as batched GEMV.
`MAC_ABK` is already an all-channel/all-bank command, so the 32 unrolled SPMW
work-item functions are one physical dispatch, not 32 serial copies.  One
native ABK segment contains `ceil(output/16)` row groups with
`ceil(reduction/16)` BF16 columns.  An independent matrix-output column is an
identical host dispatch; it is retained compactly as `repeat=B`.

The Ramulator2 runner profiles each distinct native segment directly and sums
`measured_cycles * repeat`.  This is the same summed-independent-GEMV protocol
used by the PolyBench comparison corpus.  It avoids both the former one-row
under-count and millions of redundant trace lines for LARGE GEMM; the repeat
factor is derived from MLIR loops, not supplied by a benchmark table.

## UPMEM MLIR programs

Every tensor launch carries a logical-index map to `(dpu, tasklet, local)`:

- Halo-free BLOCK: `dpu_lane -> dpu` is identity, so the first 64 partition
  elements occupy distinct DPUs before any DPU receives a second element.
- Halo-bearing BLOCK: `dpu_block -> dpu` is identity and preserves contiguous
  neighborhoods. `local_partition` and flattened `inner` bits then stripe over
  tasklets and local offsets.
- BROADCAST: `replica -> dpu` is identity, while the complete tensor payload
  maps over the tasklet/local outputs independently on every DPU.
- Arbitrary logical extents are padded to power-of-two local spans. Padding is
  not gathered back and empty DPU shards remain explicit.

The MRAM ABI derives contiguous or strided shard ownership from this map and
exposes exact layout coordinates in its manifest. The cost graph obtains DPU parallelism
from the number of non-empty BLOCK owners and tasklet parallelism from the
layout image. It no longer divides every annotated phase by an unconditional
64.

The current physical DPU-C output remains a fail-closed translation-unit
fragment; the layout is nevertheless shared by the functional ABI and virtual
cycle model, so removing or changing it changes observable placement and cost.
