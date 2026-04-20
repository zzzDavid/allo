"""UPMEM uPIMulator code generation.

Given an IR-level description (currently: vector-add and vector-mul on int32),
this module emits every file needed to stand up a new uPIMulator benchmark:

  benchmark/<NAME>/CMakeLists.txt
  benchmark/<NAME>/dpu/CMakeLists.txt
  benchmark/<NAME>/dpu/task.c
  benchmark/<NAME>/support/common.h
  src/assembler/prim/<name>.go    (Go Assemblable — identical pattern to VA)

plus in-place idempotent patches to:
  benchmark/CMakeLists.txt        (add_subdirectory(<NAME>))
  src/assembler/assembler.go      (assemblables["<NAME>"] = new(prim.<Name>))

The Python side then invokes `python script/build.py` to rebuild uPIMulator and
runs `./build/uPIMulator --benchmark <NAME> ...` to execute. Outputs land in
`bin/output_dpu_mram_heap_pointer_name_*.bin` as ASCII-decimal-per-byte; the
driver here reads them back as int32 arrays.
"""
from __future__ import annotations

import os
import struct
import subprocess
import textwrap
from typing import List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# runtime/ -> pim/ -> allo/ -> allo/ -> experiments/ -> repo root
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
UPM = os.path.join(
    REPO, "experiments", "simulators", "uPIMulator", "golang", "uPIMulator")


# ----------------------------------------------------------------------------- #
# DPU-side code gen
# ----------------------------------------------------------------------------- #

_COMMON_H = """\
#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
    uint32_t size;
    uint32_t transfer_size;
    enum kernels {
        kernel1 = 0,
        nr_kernels = 1,
    } kernel;
} dpu_arguments_t;

#ifdef BL
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#else
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#endif

#ifdef UINT32
#define T uint32_t
#define DIV 2
#elif INT32
#define T int32_t
#define DIV 2
#else
#define T int32_t
#define DIV 2
#endif

#define PRINT 0
#define divceil(n, m) (((n)-1) / (m) + 1)
#define roundup(n, m) ((n / m) * m + m)
#endif
"""


def _task_c(op: str) -> str:
    if op == "add":
        compute = "bufferB[i] += bufferA[i];"
    elif op == "mul":
        compute = "bufferB[i] *= bufferA[i];"
    elif op == "relu":
        compute = "bufferB[i] = (bufferA[i] > 0) ? bufferA[i] : 0;"
    else:
        raise ValueError(f"unknown op {op!r}")

    return textwrap.dedent(f"""\
    /*
     * DSL-generated {op} kernel
     */
    #include <stdint.h>
    #include <stdio.h>
    #include <defs.h>
    #include <mram.h>
    #include <alloc.h>
    #include <perfcounter.h>
    #include <barrier.h>

    #include "../support/common.h"

    __host dpu_arguments_t DPU_INPUT_ARGUMENTS;

    void __attribute__ ((noinline))
    dsl_kernel(T *bufferB, T *bufferA, unsigned int l_size) {{
        for (unsigned int i = 0; i < l_size; i++) {{
            {compute}
        }}
    }}

    BARRIER_INIT(my_barrier, NR_TASKLETS);

    extern int main_kernel1(void);
    int (*kernels[nr_kernels])(void) = {{main_kernel1}};

    int main(void) {{
        return kernels[DPU_INPUT_ARGUMENTS.kernel]();
    }}

    int main_kernel1() {{
        unsigned int tasklet_id = me();
        if (tasklet_id == 0) {{ mem_reset(); }}
        barrier_wait(&my_barrier);

        uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size;
        uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size;

        uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;
        uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
        uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes_transfer);

        T *cache_A = (T *) mem_alloc(BLOCK_SIZE);
        T *cache_B = (T *) mem_alloc(BLOCK_SIZE);

        for (unsigned int byte_index = base_tasklet;
             byte_index < input_size_dpu_bytes;
             byte_index += BLOCK_SIZE * NR_TASKLETS) {{
            uint32_t l_size_bytes = (byte_index + BLOCK_SIZE >= input_size_dpu_bytes)
                ? (input_size_dpu_bytes - byte_index) : BLOCK_SIZE;

            mram_read((__mram_ptr void const*)(mram_base_addr_A + byte_index),
                      cache_A, l_size_bytes);
            mram_read((__mram_ptr void const*)(mram_base_addr_B + byte_index),
                      cache_B, l_size_bytes);

            dsl_kernel(cache_B, cache_A, l_size_bytes >> DIV);

            mram_write(cache_B,
                       (__mram_ptr void*)(mram_base_addr_B + byte_index),
                       l_size_bytes);
        }}
        return 0;
    }}
    """)


_DPU_CMAKE = """\
SET(BL 10)
SET(TYPE INT32)

set(CMAKE_C_COMPILER "dpu-upmem-dpurte-clang")
set(CMAKE_C_FLAGS "-w -I${{CMAKE_CURRENT_SOURCE_DIR}}/../support -O2 -S -DNR_TASKLETS=${{NR_TASKLETS}} -DBL=${{BL}} -D${{TYPE}}")

file(GLOB_RECURSE SRCS *.c)

add_executable({target} ${{SRCS}})
"""

_BENCH_CMAKE = "#add_subdirectory(host)\nadd_subdirectory(dpu)\n"


# ----------------------------------------------------------------------------- #
# Go-side Assemblable (mirrors VA's pattern; parametric op semantics baked in)
# ----------------------------------------------------------------------------- #

def _dslva_go(struct_name: str, benchmark_name: str, op: str) -> str:
    if op == "add":
        compute = "c := a + b"
    elif op == "mul":
        compute = "c := a * b"
    elif op == "relu":
        compute = "c := int64(0)\n\t\tif a > 0 { c = a }"
    else:
        raise ValueError(op)

    return textwrap.dedent(f"""\
    // DSL-generated assemblable for benchmark {benchmark_name} ({op})
    package prim

    import (
        "errors"
        "math"
        "math/rand"
        "uPIMulator/src/abi/encoding"
        "uPIMulator/src/abi/word"
        "uPIMulator/src/misc"
    )

    type {struct_name} struct {{
        num_dpus       int
        num_tasklets   int
        num_executions int

        input_size_dpu_8bytes int64
        buffer_a              []int64
        buffer_b              []int64
        buffer_c              []int64
        sizes                 []int64
        transfer_sizes        []int64
        kernels               []int64
    }}

    func (this *{struct_name}) Init(command_line_parser *misc.CommandLineParser) {{
        num_channels := int(command_line_parser.IntParameter("num_channels"))
        num_ranks_per_channel := int(command_line_parser.IntParameter("num_ranks_per_channel"))
        num_dpus_per_rank := int(command_line_parser.IntParameter("num_dpus_per_rank"))

        this.num_dpus = num_channels * num_ranks_per_channel * num_dpus_per_rank
        this.num_tasklets = int(command_line_parser.IntParameter("num_tasklets"))
        this.num_executions = 1

        buffer_size := int64(command_line_parser.DataPrepParams()[0])
        elem_size := int64(4)

        input_size := buffer_size

        var input_size_8bytes int64
        if (input_size*elem_size)%8 == 0 {{
            input_size_8bytes = input_size
        }} else {{
            input_size_8bytes = int64(math.Ceil(float64(input_size)/float64(8)) * 8)
        }}

        input_size_dpu := (input_size-1)/int64(this.num_dpus) + 1
        if (input_size_dpu*elem_size)%8 == 0 {{
            this.input_size_dpu_8bytes = input_size_dpu
        }} else {{
            this.input_size_dpu_8bytes = int64(math.Ceil(float64(input_size_dpu)/float64(8)) * 8)
        }}

        // deterministic seed so Python can re-derive a,b from seed + params
        rand.Seed(42)

        this.buffer_a = make([]int64, 0)
        this.buffer_b = make([]int64, 0)
        this.buffer_c = make([]int64, 0)
        for i := int64(0); i < this.input_size_dpu_8bytes*int64(this.num_dpus); i++ {{
            a := int64(rand.Intn(256))
            b := int64(rand.Intn(256))
            {compute}
            this.buffer_a = append(this.buffer_a, a)
            this.buffer_b = append(this.buffer_b, b)
            this.buffer_c = append(this.buffer_c, c)
        }}

        this.sizes = make([]int64, 0)
        for i := 0; i < this.num_dpus-1; i++ {{
            this.sizes = append(this.sizes, this.input_size_dpu_8bytes*elem_size)
        }}
        size := (input_size_8bytes - this.input_size_dpu_8bytes*int64(this.num_dpus-1)) * elem_size
        this.sizes = append(this.sizes, size)

        this.transfer_sizes = make([]int64, 0)
        for i := 0; i < this.num_dpus; i++ {{
            this.transfer_sizes = append(this.transfer_sizes, this.input_size_dpu_8bytes*elem_size)
        }}

        this.kernels = make([]int64, 0)
        for i := 0; i < this.num_dpus; i++ {{
            this.kernels = append(this.kernels, 0)
        }}
    }}

    func (this *{struct_name}) InputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {{
        if execution >= this.num_executions || dpu_id >= this.num_dpus {{
            panic(errors.New("out-of-range"))
        }}
        s := new(encoding.ByteStream); s.Init()
        for _, v := range []int64{{this.sizes[dpu_id], this.transfer_sizes[dpu_id], this.kernels[dpu_id]}} {{
            w := new(word.Word); w.Init(32); w.SetValue(v)
            s.Merge(w.ToByteStream())
        }}
        return map[string]*encoding.ByteStream{{"DPU_INPUT_ARGUMENTS": s}}
    }}

    func (this *{struct_name}) OutputDpuHost(execution int, dpu_id int) map[string]*encoding.ByteStream {{
        if execution >= this.num_executions || dpu_id >= this.num_dpus {{
            panic(errors.New("out-of-range"))
        }}
        return make(map[string]*encoding.ByteStream, 0)
    }}

    func (this *{struct_name}) InputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {{
        if execution >= this.num_executions || dpu_id >= this.num_dpus {{
            panic(errors.New("out-of-range"))
        }}
        bs := new(encoding.ByteStream); bs.Init()
        start := this.input_size_dpu_8bytes * int64(dpu_id)
        for i := int64(0); i < this.input_size_dpu_8bytes; i++ {{
            w := new(word.Word); w.Init(32); w.SetValue(this.buffer_a[start+i]); bs.Merge(w.ToByteStream())
        }}
        for i := int64(0); i < this.input_size_dpu_8bytes; i++ {{
            w := new(word.Word); w.Init(32); w.SetValue(this.buffer_b[start+i]); bs.Merge(w.ToByteStream())
        }}
        return 0, bs
    }}

    func (this *{struct_name}) OutputDpuMramHeapPointerName(execution int, dpu_id int) (int64, *encoding.ByteStream) {{
        if execution >= this.num_executions || dpu_id >= this.num_dpus {{
            panic(errors.New("out-of-range"))
        }}
        bs := new(encoding.ByteStream); bs.Init()
        start := this.input_size_dpu_8bytes * int64(dpu_id)
        for i := int64(0); i < this.input_size_dpu_8bytes; i++ {{
            w := new(word.Word); w.Init(32); w.SetValue(this.buffer_c[start+i]); bs.Merge(w.ToByteStream())
        }}
        return this.input_size_dpu_8bytes * 4, bs
    }}

    func (this *{struct_name}) NumExecutions() int {{ return this.num_executions }}
    """)


# ----------------------------------------------------------------------------- #
# File I/O helpers
# ----------------------------------------------------------------------------- #

def _write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _patch_once(path, marker, new_line):
    """Append `new_line` to `path` only if `marker` not already present."""
    with open(path) as f:
        content = f.read()
    if marker in content:
        return False
    if not content.endswith("\n"):
        content += "\n"
    content += new_line.rstrip() + "\n"
    with open(path, "w") as f:
        f.write(content)
    return True


def _patch_assembler_go(bench_name: str, struct_name: str):
    path = os.path.join(UPM, "src", "assembler", "assembler.go")
    marker = f'"{bench_name}"'
    line = f'\tthis.assemblables["{bench_name}"] = new(prim.{struct_name})'
    with open(path) as f:
        content = f.read()
    if marker in content:
        return False
    # insert just before the closing brace of Init(); simplest: add after the
    # existing last `assemblables[...] = new(prim....)` line.
    anchor = '\tthis.assemblables["VA"] = new(prim.Va)'
    if anchor not in content:
        raise RuntimeError("couldn't find anchor line in assembler.go")
    content = content.replace(anchor, anchor + "\n" + line)
    with open(path, "w") as f:
        f.write(content)
    return True


# ----------------------------------------------------------------------------- #
# Public entry points
# ----------------------------------------------------------------------------- #

def emit_benchmark(bench_name: str, op: str) -> dict:
    """Write all DSL-generated files for benchmark `bench_name` doing `op`.

    Returns a dict of paths written (for test diagnostics)."""
    struct_name = bench_name.title().replace("-", "")    # DSLVA -> Dslva
    bench_dir = os.path.join(UPM, "benchmark", bench_name)
    support_dir = os.path.join(bench_dir, "support")

    out = {}
    out["common_h"]   = os.path.join(support_dir, "common.h")
    out["bench_cmake"] = os.path.join(bench_dir, "CMakeLists.txt")
    out["dpu_cmake"]   = os.path.join(bench_dir, "dpu", "CMakeLists.txt")
    out["task_c"]      = os.path.join(bench_dir, "dpu", "task.c")
    out["assemblable"] = os.path.join(UPM, "src", "assembler", "prim",
                                       bench_name.lower() + ".go")

    _write_file(out["common_h"], _COMMON_H)
    _write_file(out["bench_cmake"], _BENCH_CMAKE)
    _write_file(out["dpu_cmake"], _DPU_CMAKE.format(
        target=f"{bench_name}_device"))
    _write_file(out["task_c"], _task_c(op))
    _write_file(out["assemblable"], _dslva_go(struct_name, bench_name, op))

    # register in benchmark/CMakeLists.txt and assembler.go (idempotent)
    _patch_once(
        os.path.join(UPM, "benchmark", "CMakeLists.txt"),
        f"add_subdirectory({bench_name})",
        f"add_subdirectory({bench_name})",
    )
    _patch_assembler_go(bench_name, struct_name)
    return out


def _conda_env():
    """Return an env where `go` and `cmake` from pim-dev are on PATH."""
    env = os.environ.copy()
    conda = env.get("CONDA_PREFIX")
    # If we weren't run under conda activate, point at pim-dev explicitly.
    if conda is None or "pim-dev" not in conda:
        guess = os.path.expanduser("~/anaconda3/envs/pim-dev")
        if os.path.isdir(guess):
            env["CONDA_PREFIX"] = guess
            env["PATH"] = guess + "/bin:" + env.get("PATH", "")
    return env


def rebuild_upimulator():
    """Rebuild the Go binary after assembler.go patch."""
    r = subprocess.run(["python", "script/build.py"],
                       cwd=UPM, capture_output=True, text=True,
                       env=_conda_env(), timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"uPIMulator build failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def run_benchmark(bench_name: str, data_prep_params: int = 1024,
                  num_tasklets: int = 16) -> str:
    """Run `--benchmark bench_name ...`. Returns the bin dir path."""
    bin_dir = os.path.join(UPM, "bin")
    # fresh bin/
    if os.path.isdir(bin_dir):
        for f in os.listdir(bin_dir):
            try:
                os.remove(os.path.join(bin_dir, f))
            except IsADirectoryError:
                pass
    os.makedirs(bin_dir, exist_ok=True)

    r = subprocess.run(
        [os.path.join(UPM, "build", "uPIMulator"),
         "--root_dirpath", UPM, "--bin_dirpath", bin_dir,
         "--benchmark", bench_name,
         "--num_channels", "1", "--num_ranks_per_channel", "1",
         "--num_dpus_per_rank", "1", "--num_tasklets", str(num_tasklets),
         "--data_prep_params", str(data_prep_params)],
        cwd=UPM, capture_output=True, text=True,
        env=_conda_env(), timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError(f"uPIMulator run failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return bin_dir


# ----------------------------------------------------------------------------- #
# Output recovery
# ----------------------------------------------------------------------------- #

def _decode_bytestream_file(path) -> bytes:
    """Each line of the file is a decimal byte 0..255."""
    with open(path) as f:
        vals = [int(x) for x in f.read().split() if x.strip()]
    return bytes(vals)


def read_dpu_io(bin_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (a, b, c_from_sim) int32 arrays recovered from the run.

    a, b come from the input MRAM heap file (concatenated then split).
    c_from_sim comes from the output MRAM heap file.
    """
    inp_files = [f for f in os.listdir(bin_dir)
                 if f.startswith("input_dpu_mram_heap_pointer_name_")]
    out_files = [f for f in os.listdir(bin_dir)
                 if f.startswith("output_dpu_mram_heap_pointer_name_")]
    assert len(inp_files) == 1, inp_files
    assert len(out_files) == 1, out_files

    raw_in = _decode_bytestream_file(os.path.join(bin_dir, inp_files[0]))
    raw_out = _decode_bytestream_file(os.path.join(bin_dir, out_files[0]))
    n_in = len(raw_in) // 4
    n_out = len(raw_out) // 4
    assert n_in % 2 == 0, f"input not an even number of int32s: {n_in}"
    N = n_in // 2

    ab = np.frombuffer(raw_in, dtype="<i4")
    c  = np.frombuffer(raw_out, dtype="<i4")
    assert n_out == N, f"output has {n_out} int32s but input has N={N} each buffer"
    return ab[:N].copy(), ab[N:].copy(), c.copy()
