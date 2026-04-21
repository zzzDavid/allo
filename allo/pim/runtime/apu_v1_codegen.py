"""GSI APU v1 codegen — emits a buildable APU project from a DSL spec.

Strategy: copy the `example-gvml/` template (Common/, Makefile, gsi_dma.{c,h},
gsi_device_profiling.h) verbatim into a fresh project directory, then
overwrite host.c, device.c, struct.h with DSL-emitted versions.

Public API:

    gen_apu_v1_project(dst_dir, N, op="add", n_vrs_per_body=1,
                       lab_name="apu_dsl")
        -> writes a full project tree ready to `make`.

    build_and_run_apu_v1(project_dir, timeout_s=120)
        -> runs `make` then the resulting binary; returns captured stdout
           (host PASS/FAIL + ARCT flo-style counter dump).

Layout A/B knob: `n_vrs_per_body` picks how many VR triples (a, b, c=a+b) are
packed into one task body. N must be a multiple of 32768 * n_vrs_per_body.
With n_vrs_per_body=1, the task loops `N / 32K` times reusing VR0/VR1/VR2.
With n_vrs_per_body=k, the task unrolls k chunks in straight-line code using
VR0..VR_{3k-1}.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

APU_TEMPLATE_DIR = Path(
    "/home/nz264/shared/accelerator-hub/gsi-apu/example-gvml"
)
_32K = 32768


# ----------------------------------------------------------------------------
# codegen
# ----------------------------------------------------------------------------

_GVML_BINOP = {
    "add": "gvml_add_u16",
    "mul": "gvml_mul_u16",
    "sub": "gvml_sub_u16",
}


def _device_c(op: str, n_chunks: int, n_vrs_per_body: int,
              run_tag: str = "DSL_RUN") -> str:
    """Emit device.c for `n_chunks` 32K-element chunks.

    If n_vrs_per_body==1, a loop handles all chunks with VR0/VR1/VR2.
    Otherwise the body is unrolled across VR triples; we may still need an
    outer loop if n_chunks > n_vrs_per_body.
    """
    assert n_chunks % n_vrs_per_body == 0, \
        f"n_chunks={n_chunks} not divisible by n_vrs_per_body={n_vrs_per_body}"
    outer_iters = n_chunks // n_vrs_per_body
    binop = _GVML_BINOP[op]

    header = '''\
#include <gsi/libsys/assert.h>
#include <gsi/libsys.h>
#include <gsi/libgal.h>
#include <gsi/gal-fast-funcs.h>
#include <gsi/libgvml_memory.h>
#include <gsi/libgvml_element_wise.h>
#include <gsi/libgvml_debug.h>

#include "struct.h"
#include "gsi_dma.h"
#include <gsi_device_profiling.h>

PROF_VAR(total);
PROF_VAR(l4_to_l1);
PROF_VAR(calc);
PROF_VAR(l1_to_l4);
PROF_VAR(host_to_l4);

static void prof_init(void)
{
    arc_counters_init();
    PROF_INIT(total);
    PROF_INIT(l4_to_l1);
    PROF_INIT(calc);
    PROF_INIT(l1_to_l4);
    PROF_INIT(host_to_l4);
}

static void prof_print(void)
{
    PROF_PRINT(total);
    PROF_PRINT(l4_to_l1);
    PROF_PRINT(calc);
    PROF_PRINT(l1_to_l4);
    PROF_PRINT(host_to_l4);
}
'''

    body_lines = ["", "static int my_kernel(struct program_data *data) {", "",
                  "    prof_init();", "    PROF_START(total);", ""]
    body_lines.append("    uint16_t *A = (uint16_t *)gal_mem_handle_to_apu_ptr(data->mem_hndl_input1);")
    body_lines.append("    uint16_t *B = (uint16_t *)gal_mem_handle_to_apu_ptr(data->mem_hndl_input2);")
    body_lines.append("    uint16_t *C = (uint16_t *)gal_mem_handle_to_apu_ptr(data->mem_hndl_output1);")
    body_lines.append("    gal_set_l2dma_dma_mode(GAL_L2DMA_MODE_DIRECT);")
    body_lines.append("")

    if n_vrs_per_body == 1:
        # single-VR loop: reuse VR0/VR1/VR2 across all chunks
        body_lines.append(f"    for (uint32_t t = 0; t < {outer_iters}; ++t) {{")
        body_lines.append("        PROF_START(l4_to_l1);")
        body_lines.append("        direct_dma_l4_to_l1_32k(GVML_VM_0, A + t * 32768);")
        body_lines.append("        gvml_load_16(GVML_VR16_0, GVML_VM_0);")
        body_lines.append("        direct_dma_l4_to_l1_32k(GVML_VM_0, B + t * 32768);")
        body_lines.append("        gvml_load_16(GVML_VR16_1, GVML_VM_0);")
        body_lines.append("        PROF_END(l4_to_l1);")
        body_lines.append("        PROF_START(calc);")
        body_lines.append(f"        {binop}(GVML_VR16_2, GVML_VR16_0, GVML_VR16_1);")
        body_lines.append("        PROF_END(calc);")
        body_lines.append("        PROF_START(l1_to_l4);")
        body_lines.append("        gvml_store_16(GVML_VM_0, GVML_VR16_2);")
        body_lines.append("        direct_dma_l1_to_l4_32k(C + t * 32768, GVML_VM_0);")
        body_lines.append("        PROF_END(l1_to_l4);")
        body_lines.append("    }")
    else:
        # unrolled body: VR{3j}, VR{3j+1}, VR{3j+2} for chunk j in [0, n_vrs_per_body)
        # each outer iter processes `n_vrs_per_body` chunks at chunk offset t*n_vrs + j
        body_lines.append(f"    for (uint32_t t = 0; t < {outer_iters}; ++t) {{")
        body_lines.append("        PROF_START(l4_to_l1);")
        for j in range(n_vrs_per_body):
            body_lines.append(f"        direct_dma_l4_to_l1_32k(GVML_VM_0, A + (t * {n_vrs_per_body} + {j}) * 32768);")
            body_lines.append(f"        gvml_load_16(GVML_VR16_{3*j}, GVML_VM_0);")
            body_lines.append(f"        direct_dma_l4_to_l1_32k(GVML_VM_0, B + (t * {n_vrs_per_body} + {j}) * 32768);")
            body_lines.append(f"        gvml_load_16(GVML_VR16_{3*j+1}, GVML_VM_0);")
        body_lines.append("        PROF_END(l4_to_l1);")
        body_lines.append("        PROF_START(calc);")
        for j in range(n_vrs_per_body):
            body_lines.append(f"        {binop}(GVML_VR16_{3*j+2}, GVML_VR16_{3*j}, GVML_VR16_{3*j+1});")
        body_lines.append("        PROF_END(calc);")
        body_lines.append("        PROF_START(l1_to_l4);")
        for j in range(n_vrs_per_body):
            body_lines.append(f"        gvml_store_16(GVML_VM_0, GVML_VR16_{3*j+2});")
            body_lines.append(f"        direct_dma_l1_to_l4_32k(C + (t * {n_vrs_per_body} + {j}) * 32768, GVML_VM_0);")
        body_lines.append("        PROF_END(l1_to_l4);")
        body_lines.append("    }")

    body_lines.append("")
    body_lines.append("    PROF_END(total);")
    body_lines.append("    prof_print();")
    body_lines.append("    return 0;")
    body_lines.append("}")

    footer = (
        "\nGAL_TASK_ENTRY_POINT(apu_kernel_task, in, out)\n"
        "{\n"
        "    struct program_cmd *cmd = (struct program_cmd *)in;\n"
        "    int ret;\n"
        "\n"
        f'    gsi_info("\\n{run_tag}\\n");\n'
        "    gvml_init_once();\n"
        "    ret = my_kernel(&cmd->data);\n"
        "    return ret;\n"
        "}\n"
    )
    return header + "\n".join(body_lines) + "\n" + footer


def _host_c(op: str, N: int) -> str:
    """Emit host.c that drives the DSL-emitted kernel with N-element vectors."""
    ref_op = {"add": "A[i] + B[i]", "mul": "A[i] * B[i]", "sub": "A[i] - B[i]"}[op]
    return f'''\
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <gsi/libgdl.h>
#include <gsi/libsys.h>
#include <gsi/gsi_sim_config.h>

GDL_TASK_DECLARE(apu_kernel_task);

#include "struct.h"
#include "gsi_dma.h"

#define N {N}

static int run_cmd(gdl_context_handle_t ctx_id)
{{
    int ret;
    uint16_t *A = (uint16_t *)malloc(sizeof(uint16_t) * N);
    uint16_t *B = (uint16_t *)malloc(sizeof(uint16_t) * N);
    uint16_t *out = (uint16_t *)malloc(sizeof(uint16_t) * N);

    for (uint32_t i = 0; i < N; ++i) {{ A[i] = rand() & 0x7fff; B[i] = rand() & 0x7fff; }}

    const uint64_t bytes = sizeof(uint16_t) * N;
    const uint64_t total = bytes * 4;  /* A, B, C, debug (debug unused, kept for struct compat) */

    gdl_mem_handle_t dev_buf = gdl_mem_alloc_aligned(
        ctx_id, total, GDL_CONST_MAPPED_POOL, GDL_ALIGN_32);
    if (gdl_mem_handle_is_null(dev_buf)) {{
        gsi_error("gdl_mem_alloc failed");
        ret = gsi_status(ENOMEM);
        goto CLEAN_UP;
    }}

    ret = gdl_mem_cpy_to_dev(dev_buf, A, bytes); if (ret) goto CLEAN_UP;
    struct program_cmd cmd = {{ .data.mem_hndl_input1 = dev_buf }};
    ret = gdl_add_to_mem_handle(&cmd.data.mem_hndl_input2, cmd.data.mem_hndl_input1, bytes); if (ret) goto CLEAN_UP;
    ret = gdl_mem_cpy_to_dev(cmd.data.mem_hndl_input2, B, bytes); if (ret) goto CLEAN_UP;
    ret = gdl_add_to_mem_handle(&cmd.data.mem_hndl_debug,   cmd.data.mem_hndl_input2, bytes); if (ret) goto CLEAN_UP;
    ret = gdl_add_to_mem_handle(&cmd.data.mem_hndl_output1, cmd.data.mem_hndl_debug,  bytes); if (ret) goto CLEAN_UP;

    gdl_mem_handle_t dev_cmd = gdl_mem_alloc_aligned(
        ctx_id, sizeof(cmd), GDL_CONST_MAPPED_POOL, GDL_ALIGN_32);
    ret = gdl_mem_cpy_to_dev(dev_cmd, &cmd, sizeof(cmd)); if (ret) goto CLEAN_UP;

    ret = gdl_run_task_timeout(
        ctx_id, GDL_TASK(apu_kernel_task), dev_cmd, GDL_MEM_HANDLE_NULL,
        GDL_TEMPORARY_DEFAULT_MEM_BUF, GDL_TEMPORARY_DEFAULT_MEM_BUF_SIZE,
        GDL_TEMPORARY_DEFAULT_CORE_INDEX, NULL, 0, GDL_USER_MAPPING);
    if (ret) {{ gsi_error("gdl_run_task failed"); goto CLEAN_UP; }}

    ret = gdl_mem_cpy_from_dev(out, cmd.data.mem_hndl_output1, bytes); if (ret) goto CLEAN_UP;

    int errors = 0;
    for (uint32_t i = 0; i < N; ++i) {{
        uint16_t ref = (uint16_t)({ref_op});
        if (ref != out[i]) {{
            if (errors < 5) printf("mismatch at %u: got %u expected %u\\n", i, out[i], ref);
            errors++;
        }}
    }}
    if (errors == 0) printf("\\033[0;32mPASS (%u elements)\\033[0m\\n", N);
    else             printf("\\033[0;31mFAIL: %d errors\\033[0m\\n", errors);

CLEAN_UP:
    free(A); free(B); free(out);
    return ret;
}}

enum {{ NUM_CTXS = 1 }};
static struct gsi_sim_contexts g_ctxs[NUM_CTXS] = {{
    {{ .apu_count = 1, .apucs_per_apu = 4, .mem_size = 0x40000000 }}
}};

int main(int GSI_UNUSED(argc), char *argv[])
{{
    uint32_t num_ctxs;
    struct gdl_context_desc contexts_desc[GDL_MAX_NUM_CONTEXTS];

    int ret = gsi_libsys_init("dsl-apu", true);
    if (ret) gsi_fatal("gsi_libsys_init: %s", gsi_status_errorstr(ret));
    gsi_sim_create_simulator(NUM_CTXS, g_ctxs);
    ret = gdl_init();
    if (ret) gsi_fatal("gdl_init: %s", gsi_status_errorstr(ret));
    ret = gdl_context_count_get(&num_ctxs);
    if (ret) gsi_fatal("ctx count: %s", gsi_status_errorstr(ret));
    ret = gdl_context_desc_get(contexts_desc, num_ctxs);
    if (ret) gsi_fatal("ctx desc: %s", gsi_status_errorstr(ret));
    printf("Num Contexts = %u\\n", num_ctxs);

    gdl_context_handle_t valid_ctx_id = 0;
    uint32_t ctx;
    for (ctx = 0; ctx < num_ctxs; ++ctx) {{
        if (contexts_desc[ctx].status == GDL_CONTEXT_READY) {{
            valid_ctx_id = contexts_desc[ctx].ctx_id;
            printf("Memory Size = %0.1fG\\n", (float)contexts_desc[ctx].mem_size/1024/1024/1024);
            printf("Num Apucs = %d\\n", contexts_desc[ctx].num_apucs);
            break;
        }}
    }}
    if (ctx == num_ctxs) gsi_fatal("no valid ctx");

    long long unsigned int req = 3LL<<30, got1, got2;
    ret = gdl_context_alloc(valid_ctx_id, req, &got1, &got2);
    if (ret) gsi_fatal("ctx alloc: %s", gsi_status_errorstr(ret));
    ret = run_cmd(valid_ctx_id);
    gdl_context_free(valid_ctx_id);
    gdl_exit();
    gsi_libsys_exit();
    if (ret != 0) printf("\\nFailure\\n");
    else          printf("\\nSuccess\\n");
    return ret;
}}
'''


# ----------------------------------------------------------------------------
# project assembly
# ----------------------------------------------------------------------------

def gen_apu_v1_project(dst_dir: str, N: int, op: str = "add",
                       n_vrs_per_body: int = 1,
                       lab_name: str = "apu_dsl",
                       run_tag: str = "DSL_RUN") -> str:
    """Materialize a complete APU v1 project in `dst_dir`.

    Copies Makefile, Common/, gsi_dma.{c,h}, gsi_device_profiling.h, struct.h
    from the example-gvml template, then writes DSL-generated host.c and
    device.c tuned for (N, op, n_vrs_per_body).
    """
    assert N % 32768 == 0, f"N={N} must be a multiple of 32768"
    n_chunks = N // 32768
    assert n_chunks % n_vrs_per_body == 0, \
        f"n_chunks={n_chunks} not divisible by n_vrs_per_body={n_vrs_per_body}"

    dst = Path(dst_dir)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # copy template (everything except host.c/device.c/build)
    for name in ("Common", "gsi_dma.c", "gsi_dma.h", "gsi_device_profiling.h",
                 "struct.h"):
        src = APU_TEMPLATE_DIR / name
        dst_path = dst / name
        if src.is_dir():
            shutil.copytree(src, dst_path)
        else:
            shutil.copyfile(src, dst_path)

    # Makefile with the requested lab_name
    makefile = (
        "GNU_TOOLCHAIN_FOR_ARC_BASE := "
        "/usr/local/gsi-apu/13.7.1/ubuntu_20_04/"
        "arc_gnu_2021.09-release_elf32_le_linux_no_sdata/arc-snps-elf/\n"
        "export PATH:=${GNU_TOOLCHAIN_FOR_ARC_BASE}/bin:${PATH}\n\n"
        f"lab_name := {lab_name}\n"
        "TOP_DIR  := $(shell pwd)\n"
        "include $(TOP_DIR)/Common/common.mk\n"
    )
    (dst / "Makefile").write_text(makefile)

    # generated sources
    (dst / "host.c").write_text(_host_c(op, N))
    (dst / "device.c").write_text(
        _device_c(op, n_chunks, n_vrs_per_body, run_tag=run_tag))

    return str(dst)


# ----------------------------------------------------------------------------
# APU device log capture via ledag-ssh
# ----------------------------------------------------------------------------

def capture_apu_log(banner: str, timeout_s: int = 30) -> dict:
    """Pipe `connect localhost; flo; quit` into ledag-ssh and parse the log
    from the last occurrence of `banner` (the gsi_info line emitted by our
    generated kernel). Returns {section: {crun, iall, icm, dcm, seu,
    microsec500}} for each PROF_PRINT section found.
    """
    script = "connect localhost\nflo\nquit\n"
    r = subprocess.run(
        ["ledag-ssh", "-o", "localhost"],
        input=script, capture_output=True, text=True, timeout=timeout_s)
    text = r.stdout + "\n" + r.stderr

    # find the LAST occurrence of the banner — the most recent run
    idx = text.rfind(banner)
    if idx < 0:
        return {"_error": f"banner {banner!r} not found in flo output",
                "_raw_tail": text[-2000:]}
    tail = text[idx:]

    out: dict = {}
    for line in tail.splitlines():
        if "crun:" not in line or "iall:" not in line:
            continue
        # format: "ARCT[0]:  *** <section> - hits:... crun:... iall:... ..."
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        # section name is the last token before " -"
        name = left.strip().split()[-1]
        fields = {}
        for tok in right.split():
            if ":" in tok:
                k, v = tok.split(":", 1)
                # strip trailing '@500Mhz' style
                if k == "microsec@500Mhz":
                    k = "microsec500"
                try:
                    fields[k] = int(v)
                except ValueError:
                    fields[k] = v
        out[name] = fields
    return out


# ============================================================================
# Two-matvec chain codegen (case study for report 08 §4.4)
# ============================================================================
#
# Workload (MLP-block pattern):
#     y1 = W1 @ x1       W1: [M=32768, K] uint16, x1: [K] uint16
#     y2 = W2 @ x2       W2: [M=32768, K] uint16, x2: [K] uint16
#     z  = y1 + y2       residual/skip combine (elementwise add)
#
# Both GEMVs are lowered in SVP (scalar-vector product) form, the paper's
# recommended "communication-aware reduction mapping" (micro25 §4.2, Fig 8b).
# That mapping is held constant between the two layouts — the A/B contrast is
# specifically about intermediate-value persistence across the stage boundary.
#
# Layout A — "no vr-persistence axis"  (paper §5.1 "No Opt"-style baseline):
#   - Load W1 into VR0..K-1, compute y1 into VR(K), store y1 to L4.
#   - Load W2 over the same VR0..K-1 (clobbers W1), compute y2 into VR(K).
#   - Re-DMA y1 from L4 into a fresh VR, add VR_y1 + VR_y2 into VR_z, store z.
#
# Layout B — "linear-layout-aware"  (paper §4.2 + §4.3 in miniature):
#   - Upfront DMA: W1 into VR0..K-1, W2 into VR(K+1)..VR(2K) — disjoint VRs.
#   - Compute y1 into VR_y1, leave resident.
#   - Compute y2 into VR_y2.
#   - Add VR_y1 + VR_y2 into VR_z (no reload), store z.
#
# In linear-layout algebra:
#   A: bases = {k: [(1,), (2,), ...] (log2(K) bits)}           out_dim = vr
#      intermediate `stage` axis is collapsed into time — weight VRs are reused
#   B: bases = {k: [(1,), (2,), ...], stage: [(K,)]}           out_dim = vr
#      `stage` is lifted to a 1-bit spatial VR input, so both matrices are
#      co-resident and the intermediate y1 keeps its VR across stages.
#
# Cycle savings expected: elimination of the y1 VR->L1->L4 store and the
# subsequent L4->L1->VR reload — one full DMA round-trip for a 32K-element
# vector, i.e. roughly (l4_to_l1 + l1_to_l4) / 4 from §4.4's chunk-cost
# measurements, ~110K cycles.


def _matmul_chain_device_c(K: int, layout: str, run_tag: str,
                           x1: list, x2: list) -> str:
    """Emit device.c for the two-matvec chain under the chosen layout.

    `x1` and `x2` are host-supplied K-element uint16 scalars; they are
    baked into the device code as `uint16_t` constants. K must be <= 7
    (layout B reserves VR0..K-1 for W1, VR K..2K-1 for W2, VR 2K for y1,
    VR 2K+1 for y2, VR 2K+2 for scratch, VR 2K+3 for z — 2K+4 <= 15).
    """
    assert layout in ("A", "B")
    assert 1 <= K <= 7
    # VR assignments
    if layout == "B":
        vr_w1 = [f"GVML_VR16_{i}"     for i in range(K)]
        vr_w2 = [f"GVML_VR16_{i + K}" for i in range(K)]
        vr_y1      = f"GVML_VR16_{2 * K}"
        vr_y2      = f"GVML_VR16_{2 * K + 1}"
        vr_scratch = f"GVML_VR16_{2 * K + 2}"
        vr_z       = f"GVML_VR16_{2 * K + 3}"
    else:  # layout A — reuse VR0..K-1 for both weight matrices
        vr_w   = [f"GVML_VR16_{i}" for i in range(K)]
        vr_w1  = vr_w
        vr_w2  = vr_w
        vr_y1  = f"GVML_VR16_{K}"
        vr_y2  = f"GVML_VR16_{K + 1}"
        vr_scratch = f"GVML_VR16_{K + 2}"
        vr_z   = f"GVML_VR16_{K + 3}"

    x1_lit = ", ".join(f"0x{v & 0xffff:04x}" for v in x1)
    x2_lit = ", ".join(f"0x{v & 0xffff:04x}" for v in x2)

    # --- code gen per layout ---
    lines: list = []
    lines.append('#include <gsi/libsys/assert.h>')
    lines.append('#include <gsi/libsys.h>')
    lines.append('#include <gsi/libgal.h>')
    lines.append('#include <gsi/gal-fast-funcs.h>')
    lines.append('#include <gsi/libgvml_memory.h>')
    lines.append('#include <gsi/libgvml_element_wise.h>')
    lines.append('#include <gsi/libgvml_debug.h>')
    lines.append('')
    lines.append('#include "struct.h"')
    lines.append('#include "gsi_dma.h"')
    lines.append('#include <gsi_device_profiling.h>')
    lines.append('')
    lines.append('PROF_VAR(total);')
    lines.append('PROF_VAR(load_W);')
    lines.append('PROF_VAR(stage1);')
    lines.append('PROF_VAR(stage1_spill);')
    lines.append('PROF_VAR(load_W2);')
    lines.append('PROF_VAR(stage2);')
    lines.append('PROF_VAR(combine);')
    lines.append('PROF_VAR(store_z);')
    lines.append('')
    lines.append('static void prof_init(void) {')
    lines.append('    arc_counters_init();')
    lines.append('    PROF_INIT(total);')
    lines.append('    PROF_INIT(load_W);')
    lines.append('    PROF_INIT(stage1);')
    lines.append('    PROF_INIT(stage1_spill);')
    lines.append('    PROF_INIT(load_W2);')
    lines.append('    PROF_INIT(stage2);')
    lines.append('    PROF_INIT(combine);')
    lines.append('    PROF_INIT(store_z);')
    lines.append('}')
    lines.append('')
    lines.append('static void prof_print(void) {')
    lines.append('    PROF_PRINT(total);')
    lines.append('    PROF_PRINT(load_W);')
    lines.append('    PROF_PRINT(stage1);')
    lines.append('    PROF_PRINT(stage1_spill);')
    lines.append('    PROF_PRINT(load_W2);')
    lines.append('    PROF_PRINT(stage2);')
    lines.append('    PROF_PRINT(combine);')
    lines.append('    PROF_PRINT(store_z);')
    lines.append('}')
    lines.append('')
    lines.append(f'static const uint16_t g_x1[{K}] = {{ {x1_lit} }};')
    lines.append(f'static const uint16_t g_x2[{K}] = {{ {x2_lit} }};')
    lines.append('')
    lines.append('static int my_kernel(struct program_data *data) {')
    lines.append('    prof_init();')
    lines.append('    PROF_START(total);')
    lines.append('')
    # note: mem_hndl_input1 = W1 base, input2 = W2 base, output1 = z, debug = y1 spill
    lines.append('    uint16_t *W1 = (uint16_t *)gal_mem_handle_to_apu_ptr(data->mem_hndl_input1);')
    lines.append('    uint16_t *W2 = (uint16_t *)gal_mem_handle_to_apu_ptr(data->mem_hndl_input2);')
    lines.append('    uint16_t *Z  = (uint16_t *)gal_mem_handle_to_apu_ptr(data->mem_hndl_output1);')
    lines.append('    uint16_t *Y1_SPILL = (uint16_t *)gal_mem_handle_to_apu_ptr(data->mem_hndl_debug);')
    lines.append('    gal_set_l2dma_dma_mode(GAL_L2DMA_MODE_DIRECT);')
    lines.append('')

    if layout == "B":
        # ---- Layout B: upfront DMA of W1 + W2 into disjoint VRs ----
        lines.append('    // Layout B: upfront load W1 and W2 into disjoint VRs (no re-DMA).')
        lines.append('    PROF_START(load_W);')
        for k in range(K):
            lines.append(f'    direct_dma_l4_to_l1_32k(GVML_VM_0, W1 + {k} * 32768);')
            lines.append(f'    gvml_load_16({vr_w1[k]}, GVML_VM_0);')
        for k in range(K):
            lines.append(f'    direct_dma_l4_to_l1_32k(GVML_VM_0, W2 + {k} * 32768);')
            lines.append(f'    gvml_load_16({vr_w2[k]}, GVML_VM_0);')
        lines.append('    PROF_END(load_W);')
        lines.append('')
        lines.append('    // Stage 1: y1 = W1 @ x1 via SVP (temporal reduction over k)')
        lines.append('    PROF_START(stage1);')
        lines.append(f'    gvml_cpy_imm_16({vr_y1}, 0);')
        for k in range(K):
            lines.append(f'    gvml_cpy_imm_16({vr_scratch}, g_x1[{k}]);')
            lines.append(f'    gvml_mul_u16({vr_scratch}, {vr_w1[k]}, {vr_scratch});')
            lines.append(f'    gvml_add_u16({vr_y1}, {vr_y1}, {vr_scratch});')
        lines.append('    PROF_END(stage1);')
        lines.append('')
        lines.append('    // Stage 2: y2 = W2 @ x2 (y1 stays in VR across stages)')
        lines.append('    PROF_START(stage2);')
        lines.append(f'    gvml_cpy_imm_16({vr_y2}, 0);')
        for k in range(K):
            lines.append(f'    gvml_cpy_imm_16({vr_scratch}, g_x2[{k}]);')
            lines.append(f'    gvml_mul_u16({vr_scratch}, {vr_w2[k]}, {vr_scratch});')
            lines.append(f'    gvml_add_u16({vr_y2}, {vr_y2}, {vr_scratch});')
        lines.append('    PROF_END(stage2);')
        lines.append('')
        lines.append('    // Combine: z = y1 + y2 — no reload, both in VR')
        lines.append('    PROF_START(combine);')
        lines.append(f'    gvml_add_u16({vr_z}, {vr_y1}, {vr_y2});')
        lines.append('    PROF_END(combine);')
        lines.append('')
        lines.append('    // Store z')
        lines.append('    PROF_START(store_z);')
        lines.append(f'    gvml_store_16(GVML_VM_0, {vr_z});')
        lines.append('    direct_dma_l1_to_l4_32k(Z, GVML_VM_0);')
        lines.append('    PROF_END(store_z);')
    else:
        # ---- Layout A: naive sequential, re-DMA W2, spill y1 to L4, reload ----
        lines.append('    // Layout A: naive — load W1, compute y1, spill y1 to L4,')
        lines.append('    //   re-load W2 over W1 VRs, compute y2, reload y1, combine.')
        lines.append('    PROF_START(load_W);')
        for k in range(K):
            lines.append(f'    direct_dma_l4_to_l1_32k(GVML_VM_0, W1 + {k} * 32768);')
            lines.append(f'    gvml_load_16({vr_w1[k]}, GVML_VM_0);')
        lines.append('    PROF_END(load_W);')
        lines.append('')
        lines.append('    // Stage 1')
        lines.append('    PROF_START(stage1);')
        lines.append(f'    gvml_cpy_imm_16({vr_y1}, 0);')
        for k in range(K):
            lines.append(f'    gvml_cpy_imm_16({vr_scratch}, g_x1[{k}]);')
            lines.append(f'    gvml_mul_u16({vr_scratch}, {vr_w1[k]}, {vr_scratch});')
            lines.append(f'    gvml_add_u16({vr_y1}, {vr_y1}, {vr_scratch});')
        lines.append('    PROF_END(stage1);')
        lines.append('')
        lines.append('    // Spill y1 → L4 (will be reloaded after stage 2)')
        lines.append('    PROF_START(stage1_spill);')
        lines.append(f'    gvml_store_16(GVML_VM_0, {vr_y1});')
        lines.append('    direct_dma_l1_to_l4_32k(Y1_SPILL, GVML_VM_0);')
        lines.append('    PROF_END(stage1_spill);')
        lines.append('')
        lines.append('    // Re-load W2 into the SAME VR0..K-1, clobbering W1')
        lines.append('    PROF_START(load_W2);')
        for k in range(K):
            lines.append(f'    direct_dma_l4_to_l1_32k(GVML_VM_0, W2 + {k} * 32768);')
            lines.append(f'    gvml_load_16({vr_w2[k]}, GVML_VM_0);')
        lines.append('    PROF_END(load_W2);')
        lines.append('')
        lines.append('    // Stage 2')
        lines.append('    PROF_START(stage2);')
        lines.append(f'    gvml_cpy_imm_16({vr_y2}, 0);')
        for k in range(K):
            lines.append(f'    gvml_cpy_imm_16({vr_scratch}, g_x2[{k}]);')
            lines.append(f'    gvml_mul_u16({vr_scratch}, {vr_w2[k]}, {vr_scratch});')
            lines.append(f'    gvml_add_u16({vr_y2}, {vr_y2}, {vr_scratch});')
        lines.append('    PROF_END(stage2);')
        lines.append('')
        lines.append('    // Reload y1 from L4 (the layout-A penalty) + combine')
        lines.append('    PROF_START(combine);')
        lines.append('    direct_dma_l4_to_l1_32k(GVML_VM_0, Y1_SPILL);')
        lines.append(f'    gvml_load_16({vr_y1}, GVML_VM_0);')
        lines.append(f'    gvml_add_u16({vr_z}, {vr_y1}, {vr_y2});')
        lines.append('    PROF_END(combine);')
        lines.append('')
        lines.append('    PROF_START(store_z);')
        lines.append(f'    gvml_store_16(GVML_VM_0, {vr_z});')
        lines.append('    direct_dma_l1_to_l4_32k(Z, GVML_VM_0);')
        lines.append('    PROF_END(store_z);')

    lines.append('')
    lines.append('    PROF_END(total);')
    lines.append('    prof_print();')
    lines.append('    return 0;')
    lines.append('}')
    lines.append('')
    lines.append('GAL_TASK_ENTRY_POINT(apu_kernel_task, in, out) {')
    lines.append('    struct program_cmd *cmd = (struct program_cmd *)in;')
    lines.append(f'    gsi_info("\\n{run_tag}\\n");')
    lines.append('    gvml_init_once();')
    lines.append('    return my_kernel(&cmd->data);')
    lines.append('}')
    return "\n".join(lines) + "\n"


def _matmul_chain_host_c(K: int, x1: list, x2: list) -> str:
    """Host side: allocate W1, W2 in L4; the reference computes expected z."""
    x1_lit = ", ".join(str(v & 0xffff) for v in x1)
    x2_lit = ", ".join(str(v & 0xffff) for v in x2)
    return f'''\
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <gsi/libgdl.h>
#include <gsi/libsys.h>
#include <gsi/gsi_sim_config.h>

GDL_TASK_DECLARE(apu_kernel_task);

#include "struct.h"
#include "gsi_dma.h"

#define M 32768u
#define K {K}u
static const uint16_t X1[K] = {{ {x1_lit} }};
static const uint16_t X2[K] = {{ {x2_lit} }};

static int run_cmd(gdl_context_handle_t ctx_id) {{
    int ret;
    uint16_t *W1 = (uint16_t *)malloc(M * K * sizeof(uint16_t));
    uint16_t *W2 = (uint16_t *)malloc(M * K * sizeof(uint16_t));
    uint16_t *Z  = (uint16_t *)malloc(M * sizeof(uint16_t));
    uint16_t *Y1_spill = (uint16_t *)malloc(M * sizeof(uint16_t));

    srand(0x12345);
    for (uint32_t i = 0; i < M * K; ++i) W1[i] = rand() & 0x7fff;
    for (uint32_t i = 0; i < M * K; ++i) W2[i] = rand() & 0x7fff;

    const uint64_t w_bytes = sizeof(uint16_t) * M * K;
    const uint64_t y_bytes = sizeof(uint16_t) * M;
    const uint64_t total = 2 * w_bytes + 2 * y_bytes;  /* W1 W2 Z Y1_spill */

    gdl_mem_handle_t dev_buf = gdl_mem_alloc_aligned(
        ctx_id, total, GDL_CONST_MAPPED_POOL, GDL_ALIGN_32);
    if (gdl_mem_handle_is_null(dev_buf)) {{
        gsi_error("gdl_mem_alloc failed"); ret = gsi_status(ENOMEM); goto CLEAN_UP;
    }}

    ret = gdl_mem_cpy_to_dev(dev_buf, W1, w_bytes); if (ret) goto CLEAN_UP;
    struct program_cmd cmd = {{ .data.mem_hndl_input1 = dev_buf }};
    ret = gdl_add_to_mem_handle(&cmd.data.mem_hndl_input2, cmd.data.mem_hndl_input1, w_bytes); if (ret) goto CLEAN_UP;
    ret = gdl_mem_cpy_to_dev(cmd.data.mem_hndl_input2, W2, w_bytes); if (ret) goto CLEAN_UP;
    ret = gdl_add_to_mem_handle(&cmd.data.mem_hndl_debug,   cmd.data.mem_hndl_input2, w_bytes); if (ret) goto CLEAN_UP;
    ret = gdl_add_to_mem_handle(&cmd.data.mem_hndl_output1, cmd.data.mem_hndl_debug,  y_bytes); if (ret) goto CLEAN_UP;

    gdl_mem_handle_t dev_cmd = gdl_mem_alloc_aligned(
        ctx_id, sizeof(cmd), GDL_CONST_MAPPED_POOL, GDL_ALIGN_32);
    ret = gdl_mem_cpy_to_dev(dev_cmd, &cmd, sizeof(cmd)); if (ret) goto CLEAN_UP;

    ret = gdl_run_task_timeout(
        ctx_id, GDL_TASK(apu_kernel_task), dev_cmd, GDL_MEM_HANDLE_NULL,
        GDL_TEMPORARY_DEFAULT_MEM_BUF, GDL_TEMPORARY_DEFAULT_MEM_BUF_SIZE,
        GDL_TEMPORARY_DEFAULT_CORE_INDEX, NULL, 0, GDL_USER_MAPPING);
    if (ret) {{ gsi_error("gdl_run_task failed"); goto CLEAN_UP; }}

    ret = gdl_mem_cpy_from_dev(Z, cmd.data.mem_hndl_output1, y_bytes); if (ret) goto CLEAN_UP;

    /* reference: z[m] = sum_k W1[k,m] * X1[k] + sum_k W2[k,m] * X2[k] (u16 wrap) */
    int errors = 0;
    for (uint32_t m = 0; m < M; ++m) {{
        uint32_t s1 = 0, s2 = 0;
        for (uint32_t k = 0; k < K; ++k) {{
            s1 += (uint32_t)W1[k * M + m] * (uint32_t)X1[k];
            s2 += (uint32_t)W2[k * M + m] * (uint32_t)X2[k];
        }}
        uint16_t ref = (uint16_t)((s1 + s2) & 0xffff);
        if (ref != Z[m]) {{
            if (errors < 5) printf("mismatch at %u: got %u expected %u\\n", m, Z[m], ref);
            errors++;
        }}
    }}
    if (errors == 0) printf("\\033[0;32mPASS (M=%u, K=%u)\\033[0m\\n", M, K);
    else             printf("\\033[0;31mFAIL: %d errors\\033[0m\\n", errors);

CLEAN_UP:
    free(W1); free(W2); free(Z); free(Y1_spill);
    return ret;
}}

enum {{ NUM_CTXS = 1 }};
static struct gsi_sim_contexts g_ctxs[NUM_CTXS] = {{
    {{ .apu_count = 1, .apucs_per_apu = 4, .mem_size = 0x40000000 }}
}};

int main(int GSI_UNUSED(argc), char *argv[]) {{
    uint32_t num_ctxs;
    struct gdl_context_desc contexts_desc[GDL_MAX_NUM_CONTEXTS];

    int ret = gsi_libsys_init("matmul-chain", true);
    if (ret) gsi_fatal("gsi_libsys_init: %s", gsi_status_errorstr(ret));
    gsi_sim_create_simulator(NUM_CTXS, g_ctxs);
    ret = gdl_init();
    if (ret) gsi_fatal("gdl_init: %s", gsi_status_errorstr(ret));
    ret = gdl_context_count_get(&num_ctxs);
    if (ret) gsi_fatal("ctx count: %s", gsi_status_errorstr(ret));
    ret = gdl_context_desc_get(contexts_desc, num_ctxs);
    if (ret) gsi_fatal("ctx desc: %s", gsi_status_errorstr(ret));
    printf("Num Contexts = %u\\n", num_ctxs);

    gdl_context_handle_t valid_ctx_id = 0;
    uint32_t ctx;
    for (ctx = 0; ctx < num_ctxs; ++ctx) {{
        if (contexts_desc[ctx].status == GDL_CONTEXT_READY) {{
            valid_ctx_id = contexts_desc[ctx].ctx_id;
            printf("Memory Size = %0.1fG\\n", (float)contexts_desc[ctx].mem_size/1024/1024/1024);
            printf("Num Apucs = %d\\n", contexts_desc[ctx].num_apucs);
            break;
        }}
    }}
    if (ctx == num_ctxs) gsi_fatal("no valid ctx");

    long long unsigned int req = 3LL<<30, got1, got2;
    ret = gdl_context_alloc(valid_ctx_id, req, &got1, &got2);
    if (ret) gsi_fatal("ctx alloc: %s", gsi_status_errorstr(ret));
    ret = run_cmd(valid_ctx_id);
    gdl_context_free(valid_ctx_id);
    gdl_exit();
    gsi_libsys_exit();
    printf(ret == 0 ? "\\nSuccess\\n" : "\\nFailure\\n");
    return ret;
}}
'''


def gen_apu_v1_multiop_project(dst_dir: str,
                               schedule,
                               tensors: dict = None,
                               layout: str = "A",
                               lab_name: str = "apu_multiop",
                               run_tag: str = "APU_MULTIOP") -> str:
    """Materialize a multi-op APU v1 project from a ``LoweringResult.schedule``.

    Today we support the MLP-block motif (two gemvs + one add) directly via
    the ``_matmul_chain_device_c`` template. For other schedule shapes this
    raises; extend as new motifs become important.

    The schedule must contain exactly three ops in order: ``matmul|gemv``,
    ``matmul|gemv``, ``add``. The two matmuls must share ``K`` (the inner
    dim of ``W``). ``tensors`` optionally supplies the ``x1``, ``x2``
    vector values; if missing we use the same deterministic defaults as
    ``gen_apu_v1_matmul_chain_project``.
    """
    # Validate the schedule shape.
    kinds = [s["src"].kind for s in schedule]
    if kinds != ["gemv", "gemv", "add"] and kinds != ["matmul", "matmul", "add"]:
        raise NotImplementedError(
            f"APU v1 multi-op codegen only supports MLP-block (gemv/gemv/add) "
            f"today; got kinds={kinds}. Extend "
            f"apu_v1_codegen.gen_apu_v1_multiop_project for other motifs.")
    mm1, mm2, addop = schedule[0]["src"], schedule[1]["src"], schedule[2]["src"]
    # Both matmuls: shape = (M, K) carrying inner dim.
    M1, K1 = mm1.shape[0], mm1.shape[-1]
    M2, K2 = mm2.shape[0], mm2.shape[-1]
    if K1 != K2 or M1 != M2:
        raise NotImplementedError(
            f"APU v1 multi-op codegen requires matching matmul shapes; "
            f"got {mm1.shape} vs {mm2.shape}")
    K = K1
    if not (1 <= K <= 7):
        raise NotImplementedError(
            f"APU v1 _matmul_chain_device_c template needs 1 <= K <= 7; got {K}")

    x1 = None
    x2 = None
    if tensors:
        # The schedule tells us which names hold x1 / x2.
        if mm1.inputs[1] in tensors:
            x1 = [int(v) & 0xffff for v in tensors[mm1.inputs[1]]]
        if mm2.inputs[1] in tensors:
            x2 = [int(v) & 0xffff for v in tensors[mm2.inputs[1]]]

    return gen_apu_v1_matmul_chain_project(
        dst_dir, K=K, layout=layout, lab_name=lab_name, run_tag=run_tag,
        x1=x1, x2=x2,
    )


def gen_apu_v1_matmul_chain_project(dst_dir: str, K: int = 4,
                                    layout: str = "A",
                                    lab_name: str = "apu_chain",
                                    run_tag: str = "APU_CHAIN",
                                    x1: list = None,
                                    x2: list = None) -> str:
    """Materialize a two-matvec chain project (see module docstring §Two-matvec)."""
    assert layout in ("A", "B")
    assert 1 <= K <= 7

    if x1 is None:
        x1 = [(7 * k + 3) & 0x7fff for k in range(K)]   # deterministic, small
    if x2 is None:
        x2 = [(11 * k + 5) & 0x7fff for k in range(K)]

    dst = Path(dst_dir)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    for name in ("Common", "gsi_dma.c", "gsi_dma.h", "gsi_device_profiling.h",
                 "struct.h"):
        src = APU_TEMPLATE_DIR / name
        dst_path = dst / name
        if src.is_dir():
            shutil.copytree(src, dst_path)
        else:
            shutil.copyfile(src, dst_path)

    makefile = (
        "GNU_TOOLCHAIN_FOR_ARC_BASE := "
        "/usr/local/gsi-apu/13.7.1/ubuntu_20_04/"
        "arc_gnu_2021.09-release_elf32_le_linux_no_sdata/arc-snps-elf/\n"
        "export PATH:=${GNU_TOOLCHAIN_FOR_ARC_BASE}/bin:${PATH}\n\n"
        f"lab_name := {lab_name}\n"
        "TOP_DIR  := $(shell pwd)\n"
        "include $(TOP_DIR)/Common/common.mk\n"
    )
    (dst / "Makefile").write_text(makefile)
    (dst / "host.c").write_text(_matmul_chain_host_c(K, x1, x2))
    (dst / "device.c").write_text(
        _matmul_chain_device_c(K, layout, run_tag, x1, x2))
    return str(dst)


# ----------------------------------------------------------------------------
# build + run
# ----------------------------------------------------------------------------

def build_and_run_apu_v1(project_dir: str, timeout_s: int = 300,
                         lab_name: str = "apu_dsl") -> dict:
    """make + run; return {'stdout': ..., 'returncode': ..., 'counters': {...}}."""
    r_make = subprocess.run(
        ["make"], cwd=project_dir,
        capture_output=True, text=True, timeout=timeout_s)
    if r_make.returncode != 0:
        return {"stage": "build", "returncode": r_make.returncode,
                "stdout": r_make.stdout, "stderr": r_make.stderr}

    binary = os.path.join(project_dir, "build", "debug", lab_name)
    r_run = subprocess.run(
        [binary], cwd=project_dir,
        capture_output=True, text=True, timeout=timeout_s)

    counters = _parse_flo_counters(r_run.stdout + "\n" + r_run.stderr)
    return {"stage": "run", "returncode": r_run.returncode,
            "stdout": r_run.stdout, "stderr": r_run.stderr,
            "counters": counters}


def _parse_flo_counters(text: str) -> dict:
    """Extract PROF_PRINT lines into a {section_name: {crun, iall, ...}} dict.

    Looking for lines like:
        ARCT[0]: ***        l4_to_l1 - hits:1  seu:... crun:70960 iall:16597 ...
    """
    out: dict = {}
    for line in text.splitlines():
        if "crun:" not in line or "iall:" not in line:
            continue
        # section name is before the " - hits:"
        dash = line.split(" - ")
        if len(dash) < 2:
            continue
        name = dash[0].strip().split()[-1]
        fields = {}
        for tok in dash[1].split():
            if ":" in tok:
                k, v = tok.split(":", 1)
                try:
                    fields[k] = int(v)
                except ValueError:
                    fields[k] = v
        out[name] = fields
    return out
