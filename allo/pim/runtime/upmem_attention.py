"""Multi-stage UPMEM codegen for self-attention.

Two chained benchmarks:
    DSLATTN_QKT     scores[s] = sum_d Q[d] * K[s*D+d]          (int32, int32 acc)
    DSLATTN_AV      out[d]    = (sum_s weights[s] * V[s*D+d]) >> Q_FRAC  (int32 rescale)

Between them the host runs softmax in float and quantizes probs to Q.15.

Inputs flow Python -> Go: both benchmarks read a file at
    $UPM/dsl_data/<bench>_input.bin
which the Assemblable parses into MRAM. Python writes this file before each
`./build/uPIMulator --benchmark <bench>` invocation.

MRAM layouts
------------
Stage 1 (DSLATTN_QKT):
    offset 0              : Q [D int32]
    offset D*4            : K [S,D int32]
    offset (D + S*D)*4    : scores [S int32]   (written by DPU)

Stage 2 (DSLATTN_AV):
    offset 0              : weights [S int32, Q.15 fixed-point]
    offset S*4            : V [S,D int32]
    offset (S + S*D)*4    : out [D int32]       (written by DPU)

Kernel parallelism: S queries (stage 1) / D output lanes (stage 2) are split
across NR_TASKLETS. Each tasklet loads one Q-row (or weights) plus one K/V
row into WRAM and does a dot product.
"""
from __future__ import annotations

import os
import struct
import subprocess
import textwrap
from typing import Tuple

import numpy as np

from .upmem_codegen import (
    UPM, _write_file, _patch_once, _patch_assembler_go,
    _conda_env, rebuild_upimulator,
)

# fixed-point scale for host-computed softmax weights
Q_FRAC = 15
SCALE = 1 << Q_FRAC

DSL_DATA = os.path.join(UPM, "dsl_data")


# ----------------------------------------------------------------------------- #
# DPU source (task.c / common.h) per stage
# ----------------------------------------------------------------------------- #

_COMMON_H_ATTN = """\
#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
    uint32_t S;
    uint32_t D;
    uint32_t q_frac;      // used by stage 2 to rescale
    enum kernels {
        kernel1 = 0,
        nr_kernels = 1,
    } kernel;
} dpu_arguments_t;

#define T int32_t
#ifndef MAX_S
#define MAX_S 4096
#endif
#ifndef MAX_D
#define MAX_D 4096
#endif

#define align_up8(n) (((n) + 7) & ~7)
#define PRINT 0
#endif
"""


_TASK_C_QKT = r"""
/* Stage 1 kernel: scores[s] = sum_d Q[d] * K[s*D + d]. int32 inputs/outputs. */
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
BARRIER_INIT(my_barrier, NR_TASKLETS);

/* File-scope arrays live in WRAM and are shared across tasklets. */
__dma_aligned int32_t shared_Q[MAX_D];
__dma_aligned int32_t shared_scores[MAX_S];

extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) { return kernels[DPU_INPUT_ARGUMENTS.kernel](); }

int main_kernel1() {
    unsigned int tasklet_id = me();
    if (tasklet_id == 0) mem_reset();
    barrier_wait(&my_barrier);

    uint32_t S = DPU_INPUT_ARGUMENTS.S;
    uint32_t D = DPU_INPUT_ARGUMENTS.D;

    uint32_t Q_addr      = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t K_addr      = Q_addr + align_up8(D * 4);
    uint32_t scores_addr = K_addr + align_up8(S * D * 4);

    /* tasklet 0 loads Q into shared WRAM */
    if (tasklet_id == 0) {
        mram_read((__mram_ptr void const*)Q_addr, shared_Q, align_up8(D * 4));
    }
    barrier_wait(&my_barrier);

    /* per-tasklet scratch for one K row */
    int32_t *cache_K = (int32_t *) mem_alloc(align_up8(D * 4));

    for (uint32_t s = tasklet_id; s < S; s += NR_TASKLETS) {
        mram_read((__mram_ptr void const*)(K_addr + s * align_up8(D * 4)),
                  cache_K, align_up8(D * 4));
        int32_t acc = 0;
        for (uint32_t d = 0; d < D; d++) {
            acc += shared_Q[d] * cache_K[d];
        }
        shared_scores[s] = acc;
    }
    barrier_wait(&my_barrier);

    if (tasklet_id == 0) {
        mram_write(shared_scores, (__mram_ptr void*)scores_addr,
                   align_up8(S * 4));
    }
    return 0;
}
"""


_TASK_C_AV = r"""
/* Stage 2 kernel: out[d] = (sum_s weights[s] * V[s*D + d]) >> Q_FRAC.
   Q_FRAC is a compile-time constant so the shift is an immediate op. */
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include "../support/common.h"

#ifndef Q_FRAC
#define Q_FRAC 15
#endif

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
BARRIER_INIT(my_barrier, NR_TASKLETS);

__dma_aligned int32_t shared_W[MAX_S];
__dma_aligned int32_t shared_out[MAX_D];

extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) { return kernels[DPU_INPUT_ARGUMENTS.kernel](); }

int main_kernel1() {
    unsigned int tasklet_id = me();
    if (tasklet_id == 0) mem_reset();
    barrier_wait(&my_barrier);

    uint32_t S = DPU_INPUT_ARGUMENTS.S;
    uint32_t D = DPU_INPUT_ARGUMENTS.D;

    uint32_t W_addr   = (uint32_t) DPU_MRAM_HEAP_POINTER;
    uint32_t V_addr   = W_addr + align_up8(S * 4);
    uint32_t out_addr = V_addr + align_up8(S * D * 4);

    if (tasklet_id == 0) {
        mram_read((__mram_ptr void const*)W_addr, shared_W, align_up8(S * 4));
        for (uint32_t d = 0; d < D; d++) shared_out[d] = 0;
    }
    barrier_wait(&my_barrier);

    int32_t *cache_V = (int32_t *) mem_alloc(align_up8(D * 4));

    /* For each s: load V[s,:]; tasklets split the d range; accumulate
     *   shared_out[d] += (W[s] * V[s,d]) >> Q_FRAC
     */
    for (uint32_t s = 0; s < S; s++) {
        mram_read((__mram_ptr void const*)(V_addr + s * align_up8(D * 4)),
                  cache_V, align_up8(D * 4));
        int32_t w = shared_W[s];
        for (uint32_t d = tasklet_id; d < D; d += NR_TASKLETS) {
            /* inputs are small (w < 2^15, V[s,d] < 2^10) so int32 suffices.
             * Keep the shift immediate to use DPU's SR.S opcode, not SErri. */
            int32_t prod = w * cache_V[d];
            shared_out[d] += prod >> Q_FRAC;
        }
        barrier_wait(&my_barrier);
    }

    if (tasklet_id == 0) {
        mram_write(shared_out, (__mram_ptr void*)out_addr, align_up8(D * 4));
    }
    return 0;
}
"""


_DPU_CMAKE_ATTN = """\
SET(BL 10)
SET(TYPE INT32)
set(CMAKE_C_COMPILER "dpu-upmem-dpurte-clang")
set(CMAKE_C_FLAGS "-w -I${{CMAKE_CURRENT_SOURCE_DIR}}/../support -O2 -S -DNR_TASKLETS=${{NR_TASKLETS}} -DBL=${{BL}} -D${{TYPE}} -DMAX_S={MAX_S} -DMAX_D={MAX_D} -DQ_FRAC={Q_FRAC}")
file(GLOB_RECURSE SRCS *.c)
add_executable({target} ${{SRCS}})
"""

_BENCH_CMAKE = "#add_subdirectory(host)\nadd_subdirectory(dpu)\n"


# ----------------------------------------------------------------------------- #
# Go Assemblable — reads input from a Python-written file
# ----------------------------------------------------------------------------- #

def _qkt_go(struct_name: str, bench_name: str) -> str:
    return textwrap.dedent(f"""\
    /* DSL-generated attention stage 1 (Q @ K^T).  Reads the input tensor
       layout [Q(D) || K(S,D)] (int32 little-endian) from
       $UPM/dsl_data/{bench_name}_input.bin, computes expected scores as
       reference, and plumbs dpu_arguments_t (S, D, q_frac=0) for the kernel. */
    package prim

    import (
        "errors"
        "encoding/binary"
        "io/ioutil"
        "os"
        "path/filepath"
        "uPIMulator/src/abi/encoding"
        "uPIMulator/src/abi/word"
        "uPIMulator/src/misc"
    )

    type {struct_name} struct {{
        num_dpus       int
        num_tasklets   int
        num_executions int
        S, D           int64
        Q              []int32
        K              []int32
        scores         []int32
    }}

    func alignUp8(n int64) int64 {{ return (n + 7) &^ 7 }}

    func (this *{struct_name}) Init(p *misc.CommandLineParser) {{
        num_channels := int(p.IntParameter("num_channels"))
        num_ranks_per_channel := int(p.IntParameter("num_ranks_per_channel"))
        num_dpus_per_rank := int(p.IntParameter("num_dpus_per_rank"))
        this.num_dpus = num_channels * num_ranks_per_channel * num_dpus_per_rank
        this.num_tasklets = int(p.IntParameter("num_tasklets"))
        this.num_executions = 1

        params := p.DataPrepParams()
        this.S = int64(params[0])
        this.D = int64(params[1])

        root := p.StringParameter("root_dirpath")
        in_path := filepath.Join(root, "dsl_data", "{bench_name}_input.bin")
        f, err := os.Open(in_path)
        if err != nil {{ panic(err) }}
        defer f.Close()
        raw, err := ioutil.ReadAll(f)
        if err != nil {{ panic(err) }}
        want := int64(this.D + this.S*this.D) * 4
        if int64(len(raw)) != want {{
            panic(errors.New("input file size mismatch"))
        }}
        this.Q = make([]int32, this.D)
        for i := int64(0); i < this.D; i++ {{
            this.Q[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
        }}
        this.K = make([]int32, this.S*this.D)
        for i := int64(0); i < this.S*this.D; i++ {{
            this.K[i] = int32(binary.LittleEndian.Uint32(raw[(this.D+i)*4:]))
        }}
        this.scores = make([]int32, this.S)
        for s := int64(0); s < this.S; s++ {{
            var acc int64 = 0
            for d := int64(0); d < this.D; d++ {{
                acc += int64(this.Q[d]) * int64(this.K[s*this.D+d])
            }}
            this.scores[s] = int32(acc)
        }}
    }}

    func (this *{struct_name}) InputDpuHost(e, id int) map[string]*encoding.ByteStream {{
        s := new(encoding.ByteStream); s.Init()
        for _, v := range []int64{{this.S, this.D, 0, 0}} {{   // S, D, q_frac=0, kernel=0
            w := new(word.Word); w.Init(32); w.SetValue(v); s.Merge(w.ToByteStream())
        }}
        return map[string]*encoding.ByteStream{{"DPU_INPUT_ARGUMENTS": s}}
    }}

    func (this *{struct_name}) OutputDpuHost(e, id int) map[string]*encoding.ByteStream {{
        return make(map[string]*encoding.ByteStream, 0)
    }}

    func writeAlignedI32Block(bs *encoding.ByteStream, values []int32) {{
        for _, v := range values {{
            w := new(word.Word); w.Init(32); w.SetValue(int64(v)); bs.Merge(w.ToByteStream())
        }}
        /* pad to 8-byte alignment */
        for bs.Size()%8 != 0 {{
            w := new(word.Word); w.Init(8); w.SetValue(0); bs.Merge(w.ToByteStream())
        }}
    }}

    func (this *{struct_name}) InputDpuMramHeapPointerName(e, id int) (int64, *encoding.ByteStream) {{
        bs := new(encoding.ByteStream); bs.Init()
        writeAlignedI32Block(bs, this.Q)
        writeAlignedI32Block(bs, this.K)
        return 0, bs
    }}

    func (this *{struct_name}) OutputDpuMramHeapPointerName(e, id int) (int64, *encoding.ByteStream) {{
        /* offset = align_up8(D*4) + align_up8(S*D*4) */
        offset := alignUp8(this.D*4) + alignUp8(this.S*this.D*4)
        bs := new(encoding.ByteStream); bs.Init()
        writeAlignedI32Block(bs, this.scores)
        return offset, bs
    }}

    func (this *{struct_name}) NumExecutions() int {{ return this.num_executions }}
    """)


def _av_go(struct_name: str, bench_name: str) -> str:
    return textwrap.dedent(f"""\
    /* DSL-generated attention stage 2 (weights @ V).  Reads [weights(S) ||
       V(S,D)] from $UPM/dsl_data/{bench_name}_input.bin; q_frac is the
       fixed-point scale of weights (typically 15). */
    package prim

    import (
        "errors"
        "encoding/binary"
        "io/ioutil"
        "os"
        "path/filepath"
        "strconv"
        "uPIMulator/src/abi/encoding"
        "uPIMulator/src/abi/word"
        "uPIMulator/src/misc"
    )

    type {struct_name} struct {{
        num_dpus       int
        num_tasklets   int
        num_executions int
        S, D, qFrac    int64
        W              []int32
        V              []int32
        out            []int32
    }}

    func alignUp8v2(n int64) int64 {{ return (n + 7) &^ 7 }}

    func (this *{struct_name}) Init(p *misc.CommandLineParser) {{
        num_channels := int(p.IntParameter("num_channels"))
        num_ranks_per_channel := int(p.IntParameter("num_ranks_per_channel"))
        num_dpus_per_rank := int(p.IntParameter("num_dpus_per_rank"))
        this.num_dpus = num_channels * num_ranks_per_channel * num_dpus_per_rank
        this.num_tasklets = int(p.IntParameter("num_tasklets"))
        this.num_executions = 1

        params := p.DataPrepParams()
        this.S = int64(params[0])
        this.D = int64(params[1])
        if len(params) >= 3 {{
            this.qFrac = int64(params[2])
        }} else {{
            this.qFrac = 15
        }}
        _ = strconv.Itoa  /* keep import */

        root := p.StringParameter("root_dirpath")
        in_path := filepath.Join(root, "dsl_data", "{bench_name}_input.bin")
        f, err := os.Open(in_path)
        if err != nil {{ panic(err) }}
        defer f.Close()
        raw, err := ioutil.ReadAll(f)
        if err != nil {{ panic(err) }}
        want := int64(this.S + this.S*this.D) * 4
        if int64(len(raw)) != want {{
            panic(errors.New("input file size mismatch"))
        }}
        this.W = make([]int32, this.S)
        for i := int64(0); i < this.S; i++ {{
            this.W[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
        }}
        this.V = make([]int32, this.S*this.D)
        for i := int64(0); i < this.S*this.D; i++ {{
            this.V[i] = int32(binary.LittleEndian.Uint32(raw[(this.S+i)*4:]))
        }}
        this.out = make([]int32, this.D)
        for d := int64(0); d < this.D; d++ {{
            var acc int32 = 0
            for s := int64(0); s < this.S; s++ {{
                /* match DPU kernel: int32 product, immediate shift */
                prod := this.W[s] * this.V[s*this.D+d]
                acc += prod >> uint(this.qFrac)
            }}
            this.out[d] = acc
        }}
    }}

    func (this *{struct_name}) InputDpuHost(e, id int) map[string]*encoding.ByteStream {{
        s := new(encoding.ByteStream); s.Init()
        for _, v := range []int64{{this.S, this.D, this.qFrac, 0}} {{
            w := new(word.Word); w.Init(32); w.SetValue(v); s.Merge(w.ToByteStream())
        }}
        return map[string]*encoding.ByteStream{{"DPU_INPUT_ARGUMENTS": s}}
    }}

    func (this *{struct_name}) OutputDpuHost(e, id int) map[string]*encoding.ByteStream {{
        return make(map[string]*encoding.ByteStream, 0)
    }}

    func writeAlignedI32BlockAV(bs *encoding.ByteStream, values []int32) {{
        for _, v := range values {{
            w := new(word.Word); w.Init(32); w.SetValue(int64(v)); bs.Merge(w.ToByteStream())
        }}
        for bs.Size()%8 != 0 {{
            w := new(word.Word); w.Init(8); w.SetValue(0); bs.Merge(w.ToByteStream())
        }}
    }}

    func (this *{struct_name}) InputDpuMramHeapPointerName(e, id int) (int64, *encoding.ByteStream) {{
        bs := new(encoding.ByteStream); bs.Init()
        writeAlignedI32BlockAV(bs, this.W)
        writeAlignedI32BlockAV(bs, this.V)
        return 0, bs
    }}

    func (this *{struct_name}) OutputDpuMramHeapPointerName(e, id int) (int64, *encoding.ByteStream) {{
        offset := alignUp8v2(this.S*4) + alignUp8v2(this.S*this.D*4)
        bs := new(encoding.ByteStream); bs.Init()
        writeAlignedI32BlockAV(bs, this.out)
        return offset, bs
    }}

    func (this *{struct_name}) NumExecutions() int {{ return this.num_executions }}
    """)


# ----------------------------------------------------------------------------- #
# Emission
# ----------------------------------------------------------------------------- #

def _emit_stage(bench_name: str, struct_name: str, task_c: str, go_src: str,
                max_s: int, max_d: int):
    bench_dir = os.path.join(UPM, "benchmark", bench_name)
    _write_file(os.path.join(bench_dir, "support", "common.h"), _COMMON_H_ATTN)
    _write_file(os.path.join(bench_dir, "CMakeLists.txt"), _BENCH_CMAKE)
    _write_file(os.path.join(bench_dir, "dpu", "CMakeLists.txt"),
                _DPU_CMAKE_ATTN.format(target=f"{bench_name}_device",
                                       MAX_S=max_s, MAX_D=max_d,
                                       Q_FRAC=Q_FRAC))
    _write_file(os.path.join(bench_dir, "dpu", "task.c"), task_c)
    _write_file(os.path.join(UPM, "src", "assembler", "prim",
                             bench_name.lower() + ".go"),
                go_src)
    _patch_once(os.path.join(UPM, "benchmark", "CMakeLists.txt"),
                f"add_subdirectory({bench_name})",
                f"add_subdirectory({bench_name})")
    _patch_assembler_go(bench_name, struct_name)


def emit_attention_benchmarks(max_s: int = 256, max_d: int = 128):
    os.makedirs(DSL_DATA, exist_ok=True)
    _emit_stage("DSLATTN_QKT", "DslattnQkt", _TASK_C_QKT,
                _qkt_go("DslattnQkt", "DSLATTN_QKT"), max_s, max_d)
    _emit_stage("DSLATTN_AV", "DslattnAv", _TASK_C_AV,
                _av_go("DslattnAv", "DSLATTN_AV"), max_s, max_d)


# ----------------------------------------------------------------------------- #
# Orchestrator
# ----------------------------------------------------------------------------- #

def _run_stage(bench: str, S: int, D: int, q_frac: int, num_tasklets: int = 16):
    """Run `./build/uPIMulator --benchmark <bench> ...` after we've written
    the input file. Returns bin_dir."""
    bin_dir = os.path.join(UPM, "bin")
    # fresh bin/
    if os.path.isdir(bin_dir):
        for f in os.listdir(bin_dir):
            try:
                os.remove(os.path.join(bin_dir, f))
            except IsADirectoryError:
                pass
    os.makedirs(bin_dir, exist_ok=True)
    params = f"{S},{D},{q_frac}" if q_frac else f"{S},{D}"
    r = subprocess.run(
        [os.path.join(UPM, "build", "uPIMulator"),
         "--root_dirpath", UPM, "--bin_dirpath", bin_dir,
         "--benchmark", bench,
         "--num_channels", "1", "--num_ranks_per_channel", "1",
         "--num_dpus_per_rank", "1", "--num_tasklets", str(num_tasklets),
         "--data_prep_params", params],
        cwd=UPM, capture_output=True, text=True,
        env=_conda_env(), timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"uPIMulator {bench} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return bin_dir


def _read_output(bin_dir: str) -> np.ndarray:
    out_files = [f for f in os.listdir(bin_dir)
                 if f.startswith("output_dpu_mram_heap_pointer_name_")]
    assert len(out_files) == 1, out_files
    path = os.path.join(bin_dir, out_files[0])
    with open(path) as f:
        raw = bytes(int(x) for x in f.read().split() if x.strip())
    return np.frombuffer(raw, dtype="<i4").copy()


def _write_ints_le(path: str, ints: np.ndarray):
    assert ints.dtype == np.int32, ints.dtype
    with open(path, "wb") as f:
        f.write(ints.tobytes())


def run_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                  num_tasklets: int = 16
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full attention on real uPIMulator + host softmax between stages.

    Q:(D,) int32, K:(S,D) int32, V:(S,D) int32
    Returns (out_dpu, scores_dpu, probs_q) — all int32 arrays.
    The reference scale is: probs_q = round(softmax(scores/sqrt(D)) * 2^Q_FRAC),
    and out is computed on DPU as sum_s (probs_q[s] * V[s,d]) >> Q_FRAC.
    """
    D = Q.shape[0]
    S = K.shape[0]
    assert K.shape == (S, D) and V.shape == (S, D)
    assert Q.dtype == np.int32 and K.dtype == np.int32 and V.dtype == np.int32
    os.makedirs(DSL_DATA, exist_ok=True)

    # -- stage 1 --
    in1 = os.path.join(DSL_DATA, "DSLATTN_QKT_input.bin")
    _write_ints_le(in1, np.concatenate([Q, K.reshape(-1)]).astype(np.int32))
    bin_dir = _run_stage("DSLATTN_QKT", S, D, 0, num_tasklets)
    scores = _read_output(bin_dir)[:S].copy()

    # -- host softmax + quantization --
    scores_f = scores.astype(np.float64) / float(np.sqrt(D) * 1.0)
    shifted = scores_f - scores_f.max()
    ex = np.exp(shifted)
    probs = ex / ex.sum()
    probs_q = np.round(probs * SCALE).astype(np.int32)

    # -- stage 2 --
    in2 = os.path.join(DSL_DATA, "DSLATTN_AV_input.bin")
    _write_ints_le(in2, np.concatenate([probs_q, V.reshape(-1)]).astype(np.int32))
    bin_dir = _run_stage("DSLATTN_AV", S, D, Q_FRAC, num_tasklets)
    out = _read_output(bin_dir)[:D].copy()

    return out, scores, probs_q


def numpy_reference(Q, K, V) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute what the DPU kernels *should* produce, using the exact same
    quantization and integer arithmetic as the kernels. This is the oracle
    we compare against."""
    D = Q.shape[0]; S = K.shape[0]
    # stage 1: scores[s] = sum_d Q[d]*K[s,d]  (int32, no rescale)
    scores = (Q.astype(np.int64)[None, :] * K.astype(np.int64)).sum(axis=1).astype(np.int32)
    # host softmax + quantize
    scores_f = scores.astype(np.float64) / float(np.sqrt(D) * 1.0)
    shifted = scores_f - scores_f.max()
    ex = np.exp(shifted)
    probs = ex / ex.sum()
    probs_q = np.round(probs * SCALE).astype(np.int32)
    # stage 2: per-product rescale (matches the DPU kernel to avoid int64 shift)
    #   out[d] = sum_s ((W[s]*V[s,d]) as int32) >> Q_FRAC
    prods_i32 = (probs_q[:, None].astype(np.int32) *
                 V.astype(np.int32))           # relies on inputs fitting int32
    out = (prods_i32 >> Q_FRAC).sum(axis=0).astype(np.int32)
    return out, scores, probs_q
