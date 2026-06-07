# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Buildable ARC project emitter for the APU v1 (Gemini 1) backend.

Translates a `Compiled` artifact (with `cmds: list[str]` of GVML C-source
calls) into a project tree that can be built with the GSI ARC GNU
toolchain and run on real Gemini silicon. See
`experiments/allo/.orchestrator/specs/017-apu-v1-run-spec.md` for the
contract.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np  # noqa: F401

    from .spmw_codegen import Compiled


_DEFAULT_TEMPLATE_DIR = "/home/nz264/shared/accelerator-hub/gsi-apu/example-gvml"
_DEFAULT_TOOLCHAIN_BASE = (
    "/usr/local/gsi-apu/13.7.1/ubuntu_20_04/"
    "arc_gnu_2021.09-release_elf32_le_linux_no_sdata/arc-snps-elf/"
)

# Default GVML SDK include root. GSI's stock `Common/common.mk` already
# adds `-I $(GSI_USR_LOCAL_INCLUDE)` (= `/usr/local/include`) on every
# dev_modules compile, so we do not pass an extra `-I` into the emitted
# Makefile -- this constant is used only by the Python-side probe.
_DEFAULT_GVML_INCLUDE_ROOT = "/usr/local/include"

# Headers we know must exist for any device.c compile to succeed; used
# as canary files the probe checks. `libgvml_element_wise.h` declares
# every GVML op Tenon's emitter actually references, including the
# logical ones (gvml_xor_16, gvml_or_16, gvml_and_16, gvml_not_16)
# alongside the arithmetic eltwise ops -- this SDK has no separate
# `libgvml_logical.h`, and nothing in the generated `device.c` chain
# pulls one in. A single canary on the eltwise header is sufficient.
_GVML_CANARY_HEADERS = ("gsi/libgvml_element_wise.h",)

# Files copied verbatim from the example-gvml template. `Common/` is
# handled separately (copytree).
_COPY_FILES = ("gsi_dma.c", "gsi_dma.h", "gsi_device_profiling.h")

# APU v1 element width today; report 12 §3 and the `_32K` constant.
_SUPPORTED_DTYPE_NAMES = ("uint16", "int16", "float16")


def _template_dir() -> Path:
    return Path(os.environ.get("TENON_APU_V1_TEMPLATE_DIR", _DEFAULT_TEMPLATE_DIR))


def _toolchain_base() -> str:
    return os.environ.get("TENON_APU_V1_TOOLCHAIN_BASE", _DEFAULT_TOOLCHAIN_BASE)


def _gvml_include_root() -> str:
    """Return the directory under which `<gsi/libgvml_*.h>` headers live.

    Resolution order:
      1. env var `TENON_APU_V1_GVML_INCLUDE_ROOT` (override; not validated here).
      2. `_DEFAULT_GVML_INCLUDE_ROOT` (= `/usr/local/include`) -- the path
         GSI's stock `Common/common.mk` ships with for `product=x86_64`.

    Validation is intentionally deferred to `_assert_gvml_sdk_present()`;
    this getter is pure.
    """
    return os.environ.get(
        "TENON_APU_V1_GVML_INCLUDE_ROOT",
        _DEFAULT_GVML_INCLUDE_ROOT,
    )


def _assert_gvml_sdk_present() -> None:
    """Raise FileNotFoundError if any GVML SDK canary header is missing.

    Called by the build harness right before `make` runs, so that a
    misconfigured host produces a single readable error instead of a
    make-stderr blob. Skip gates in tests should call
    `_gvml_sdk_available()` (below) so that PASS/SKIP routing stays in
    one place.
    """
    root = Path(_gvml_include_root())
    for rel in _GVML_CANARY_HEADERS:
        canary = root / rel
        if not canary.is_file():
            raise FileNotFoundError(
                f"GVML SDK not found: expected {canary} (set "
                f"TENON_APU_V1_GVML_INCLUDE_ROOT to override)."
            )


def _gvml_sdk_available() -> bool:
    """Non-raising counterpart of `_assert_gvml_sdk_present()`."""
    root = Path(_gvml_include_root())
    return all((root / rel).is_file() for rel in _GVML_CANARY_HEADERS)


# --------------------------------------------------------------------- #
# Role / IO table
# --------------------------------------------------------------------- #


def _role_io_table(
    compiled: "Compiled",
    inputs: dict,
    output_specs: dict,
) -> tuple[list[str], list[str]]:
    """Return (input_role_names, output_role_names) in deterministic order.

    Inputs first (sorted alphabetically), then outputs (sorted). The
    same order is used in struct.h, device.c L4 pointer decls, and host.c
    argv parsing.
    """
    in_roles = sorted(inputs.keys())
    out_roles = sorted(output_specs.keys())
    # Sanity assertion on supported dtypes.
    for name, arr in inputs.items():
        dtype_str = str(getattr(arr, "dtype", "")) if arr is not None else ""
        if dtype_str and dtype_str not in _SUPPORTED_DTYPE_NAMES:
            raise ValueError(
                f"APU v1 build: input {name!r} has dtype {dtype_str!r}; "
                f"only {_SUPPORTED_DTYPE_NAMES} are supported on this backend."
            )
    for name, (_shape, dtype) in output_specs.items():
        dtype_str = str(dtype)
        if dtype_str not in _SUPPORTED_DTYPE_NAMES:
            raise ValueError(
                f"APU v1 build: output {name!r} has dtype {dtype_str!r}; "
                f"only {_SUPPORTED_DTYPE_NAMES} are supported on this backend."
            )
    return in_roles, out_roles


# --------------------------------------------------------------------- #
# Emitters for the generated files
# --------------------------------------------------------------------- #


def _emit_makefile(lab_name: str) -> str:
    base = _toolchain_base()
    return (
        f"GNU_TOOLCHAIN_FOR_ARC_BASE := {base}\n"
        "export PATH:=${GNU_TOOLCHAIN_FOR_ARC_BASE}/bin:${PATH}\n"
        "\n"
        f"lab_name := {lab_name}\n"
        "TOP_DIR  := $(shell pwd)\n"
        "include $(TOP_DIR)/Common/common.mk\n"
    )


def _emit_struct_h(in_roles: list[str], out_roles: list[str]) -> str:
    fields = []
    for role in in_roles + out_roles:
        fields.append(f"        uint64_t mem_hndl_{role};")
    body = "\n".join(fields)
    return (
        "#ifndef DATA_STRUCT_H\n"
        "#define DATA_STRUCT_H\n"
        "\n"
        "#ifdef __cplusplus\n"
        "extern \"C\"\n"
        "{\n"
        "#endif /* __cplusplus */\n"
        "\n"
        "#include <stdint.h>\n"
        "    struct program_data\n"
        "    {\n"
        f"{body}\n"
        "        uint16_t mac_lut[256];\n"
        "    } __attribute__((packed));\n"
        "\n"
        "    struct program_cmd\n"
        "    {\n"
        "        char buffer[64];\n"
        "        union\n"
        "        {\n"
        "            struct program_data data;\n"
        "        } __attribute__((packed));\n"
        "    } __attribute__((packed));\n"
        "\n"
        "#ifdef __cplusplus\n"
        "}\n"
        "#endif /* __cplusplus */\n"
        "\n"
        "#endif /* STRUCT_H */\n"
    )


# Map workload-side memref names to the canonical L4 pointer C names the
# APUv1Ctx emits into self.cmds. Mirrors `APUv1Ctx._L4_PTR_BY_ROLE` but
# routed by memref *name* (since that's what the build harness sees from
# the user). Conservative default for any unmatched name: pick the next
# free pointer slot.
def _l4_ptr_for_role(role: str, is_output: bool) -> str:
    if is_output:
        return "out_L4ptr"
    # Inputs: "x"-ish memref names get inp_L4ptr; "W"/"y"-ish get wgt_L4ptr.
    low = role.lower()
    if low.endswith("_w") or low == "w" or "wgt" in low or "weight" in low:
        return "wgt_L4ptr"
    return "inp_L4ptr"


def _emit_device_c(
    compiled: "Compiled",
    in_roles: list[str],
    out_roles: list[str],
) -> str:
    # L4 pointer decls -- one per role, named to match the canonical
    # APUv1Ctx._L4_PTR_BY_ROLE strings the emitted body references.
    l4_decls: list[str] = []
    used_ptrs: set[str] = set()

    # First pass: use the ctx's iter_l4_roles() to find pointers that
    # actually appear in the body, so we can avoid unused-variable
    # warnings.
    ctx_l4_roles: list[str] = []
    try:
        ctx_l4_roles = list(compiled.layout_ctx.iter_l4_roles())  # type: ignore[attr-defined]
    except Exception:
        ctx_l4_roles = []

    # Fall back: scan compiled.cmds for the L4 ptr substrings.
    if not ctx_l4_roles:
        for ptr in ("inp_L4ptr", "wgt_L4ptr", "out_L4ptr"):
            if any(ptr in line for line in compiled.cmds):
                ctx_l4_roles.append(ptr)

    # Build a stable input-role -> ptr mapping using the same heuristic
    # as APUv1Ctx (inp/wgt by role string heuristic; out for outputs).
    ptr_to_field: dict[str, str] = {}
    for role in in_roles:
        ptr = _l4_ptr_for_role(role, is_output=False)
        # Avoid collisions: if "inp_L4ptr" is taken, fall through to "wgt_L4ptr".
        if ptr in ptr_to_field:
            ptr = "wgt_L4ptr" if ptr == "inp_L4ptr" else "inp_L4ptr"
        ptr_to_field[ptr] = f"mem_hndl_{role}"
    for role in out_roles:
        ptr_to_field["out_L4ptr"] = f"mem_hndl_{role}"

    # Emit decls only for the L4 pointers the body references (drops
    # unused-variable warnings); always emit out_L4ptr if we have any
    # output role.
    decl_ptr_order = ["inp_L4ptr", "wgt_L4ptr", "out_L4ptr"]
    for ptr in decl_ptr_order:
        if ptr in ptr_to_field and (ptr in ctx_l4_roles or ptr == "out_L4ptr"):
            field = ptr_to_field[ptr]
            l4_decls.append(
                f"    uint16_t *{ptr} = (uint16_t *)"
                f"gal_mem_handle_to_apu_ptr(data->{field});"
            )
            used_ptrs.add(ptr)

    # SPEC-018: unconditionally declare the popcount LUT pointer used by
    # the sv_lookup MAC expansion (`gvml_lookup_16(..., mac_lut_ptr, 256)`).
    # The decl is always emitted to keep device.c shape stable across
    # mode="sv" / mode="sv_lookup"; the (void) guard silences
    # -Wunused-variable when the body never names it.
    l4_decls.append(
        "    const uint16_t *mac_lut_ptr = (const uint16_t *)data->mac_lut;"
    )
    l4_decls.append("    (void)mac_lut_ptr;")

    # VR alias decls -- pulled from the ctx via iter_vr_aliases(); fall
    # back to the canonical pair the declarative-mode body needs.
    vr_aliases: list[tuple[str, str]] = []
    try:
        vr_aliases = list(compiled.layout_ctx.iter_vr_aliases())  # type: ignore[attr-defined]
    except Exception:
        vr_aliases = []
    if not vr_aliases:
        vr_aliases = [("vrs", "GVML_VR16_0"), ("mac_tmp_vr", "GVML_VR16_1")]
    vr_decls = [
        f"    enum gvml_vr16 {c_name} = {enum_name};"
        for c_name, enum_name in vr_aliases
    ]

    body_lines = ["    " + ln for ln in compiled.cmds]

    header = (
        "#include <gsi/libsys/assert.h>\n"
        "#include <gsi/libsys.h>\n"
        "#include <gsi/libgal.h>\n"
        "#include <gsi/gal-fast-funcs.h>\n"
        "#include <gsi/libgvml_memory.h>\n"
        "#include <gsi/libgvml_element_wise.h>\n"
        "#include <gsi/libgvml_debug.h>\n"
        "\n"
        "#include \"struct.h\"\n"
        "#include \"gsi_dma.h\"\n"
        "#include <gsi_device_profiling.h>\n"
        "\n"
        "PROF_VAR(total);\n"
        "\n"
        "static void prof_init(void)\n"
        "{\n"
        "    arc_counters_init();\n"
        "    PROF_INIT(total);\n"
        "}\n"
        "\n"
        "static void prof_print(void)\n"
        "{\n"
        "    PROF_PRINT(total);\n"
        "}\n"
        "\n"
        "static int my_kernel(struct program_data *data) {\n"
        "\n"
        "    prof_init();\n"
        "    PROF_START(total);\n"
        "\n"
    )
    decls = "\n".join(l4_decls) + ("\n\n" if l4_decls else "")
    vr_block = "\n".join(vr_decls) + "\n\n"
    pre_body = "    gal_set_l2dma_dma_mode(GAL_L2DMA_MODE_DIRECT);\n\n"
    body = "\n".join(body_lines) + "\n"
    footer = (
        "\n    PROF_END(total);\n"
        "\n    prof_print();\n"
        "\n    return 0;\n"
        "}\n"
        "\n"
        "GAL_TASK_ENTRY_POINT(apu_kernel_task, in, out)\n"
        "{\n"
        "    struct program_cmd *cmd = (struct program_cmd *)in;\n"
        "    gsi_info(\"\\nRunning tenon program!\\n\");\n"
        "    gvml_init_once();\n"
        "    return my_kernel(&cmd->data);\n"
        "}\n"
    )
    return header + decls + vr_block + pre_body + body + footer


def _dtype_byte_count(shape: tuple, dtype) -> int:
    """Total byte count for shape/dtype, robust to dtype passed as np.dtype
    or as the string name."""
    import numpy as np

    n = 1
    for d in shape:
        n *= int(d)
    return int(n * np.dtype(dtype).itemsize)


def _emit_host_c(
    in_roles: list[str],
    out_roles: list[str],
    input_byte_sizes: dict[str, int],
    output_byte_sizes: dict[str, int],
) -> str:
    """Emit a host.c that reads inputs from raw binary files, copies
    them to L4, runs the kernel, and writes outputs back to raw binary
    files. Argv order is `<bin> <in_role0_path> ... <out_role0_path> ...`.
    """
    all_roles = in_roles + out_roles

    # Argv parsing: argv[1..N] are file paths, in role order.
    argv_decls = []
    for i, role in enumerate(all_roles):
        argv_decls.append(f"    const char *path_{role} = argv[{i + 1}];")

    # Allocate one combined L4 buffer (mirrors the example-gvml stitched
    # layout). For each role, gdl_add_to_mem_handle chains from the
    # previous handle by the previous role's size.
    alloc_block_lines = [
        f"    const uint64_t sz_{role} = {sz}ULL;"
        for role, sz in list(input_byte_sizes.items()) + list(output_byte_sizes.items())
    ]
    total_size_expr = " + ".join(f"sz_{r}" for r in all_roles)
    alloc_block_lines.append(
        f"    const uint64_t io_total = {total_size_expr};"
    )

    # Chained handle assignments: first role from base ptr, each next
    # role offset by prior role's size.
    chain_lines = [
        f"    struct program_cmd cmd = {{ .data.mem_hndl_{all_roles[0]} = "
        "input_dev_bufs, };"
    ]
    for i, role in enumerate(all_roles[1:], start=1):
        prev = all_roles[i - 1]
        chain_lines.append(
            f"    ret = gdl_add_to_mem_handle(&cmd.data.mem_hndl_{role}, "
            f"cmd.data.mem_hndl_{prev}, sz_{prev});\n"
            f"    if (ret) goto CLEAN_UP;"
        )

    # Per-input host->dev copy via fread of the raw .bin file.
    copy_to_dev_lines = []
    for role in in_roles:
        copy_to_dev_lines.append(
            f"    {{\n"
            f"        FILE *f = fopen(path_{role}, \"rb\");\n"
            f"        if (!f) {{ gsi_error(\"open %s\\n\", path_{role}); "
            f"ret = -1; goto CLEAN_UP; }}\n"
            f"        void *buf = malloc(sz_{role});\n"
            f"        size_t n = fread(buf, 1, sz_{role}, f);\n"
            f"        fclose(f);\n"
            f"        if (n != sz_{role}) {{ free(buf); "
            f"gsi_error(\"short read %s\\n\", path_{role}); "
            f"ret = -1; goto CLEAN_UP; }}\n"
            f"        ret = gdl_mem_cpy_to_dev(cmd.data.mem_hndl_{role}, "
            f"buf, sz_{role});\n"
            f"        free(buf);\n"
            f"        if (ret) goto CLEAN_UP;\n"
            f"    }}"
        )

    # Per-output dev->host copy + fwrite.
    copy_from_dev_lines = []
    for role in out_roles:
        copy_from_dev_lines.append(
            f"    {{\n"
            f"        void *buf = malloc(sz_{role});\n"
            f"        ret = gdl_mem_cpy_from_dev(buf, cmd.data.mem_hndl_{role}, "
            f"sz_{role});\n"
            f"        if (ret) {{ free(buf); goto CLEAN_UP; }}\n"
            f"        FILE *f = fopen(path_{role}, \"wb\");\n"
            f"        if (!f) {{ free(buf); "
            f"gsi_error(\"create %s\\n\", path_{role}); "
            f"ret = -1; goto CLEAN_UP; }}\n"
            f"        size_t n = fwrite(buf, 1, sz_{role}, f);\n"
            f"        fclose(f);\n"
            f"        free(buf);\n"
            f"        if (n != sz_{role}) {{ "
            f"gsi_error(\"short write %s\\n\", path_{role}); "
            f"ret = -1; goto CLEAN_UP; }}\n"
            f"    }}"
        )

    argv_decls_str = "\n".join(argv_decls)
    alloc_block_str = "\n".join(alloc_block_lines)
    chain_str = "\n".join(chain_lines)
    copy_to_dev_str = "\n".join(copy_to_dev_lines)
    copy_from_dev_str = "\n".join(copy_from_dev_lines)

    return (
        "#include <string.h>\n"
        "#include <stdio.h>\n"
        "#include <stdint.h>\n"
        "#include <stdlib.h>\n"
        "#include <time.h>\n"
        "\n"
        "#include <gsi/libgdl.h>\n"
        "#include <gsi/libsys.h>\n"
        "#include <gsi/gsi_sim_config.h>\n"
        "\n"
        "GDL_TASK_DECLARE(apu_kernel_task);\n"
        "\n"
        "#include \"struct.h\"\n"
        "#include \"gsi_dma.h\"\n"
        "\n"
        "static int run_tenon_kernel(gdl_context_handle_t ctx_id, int argc, char *argv[])\n"
        "{\n"
        "    int ret;\n"
        "    gdl_mem_handle_t dev_cmd_buf = GDL_MEM_HANDLE_NULL;\n"
        "    gdl_mem_handle_t input_dev_bufs = GDL_MEM_HANDLE_NULL;\n"
        "\n"
        f"{argv_decls_str}\n"
        "\n"
        f"{alloc_block_str}\n"
        "\n"
        "    input_dev_bufs = gdl_mem_alloc_aligned(ctx_id, io_total, "
        "GDL_CONST_MAPPED_POOL, GDL_ALIGN_32);\n"
        "    if (gdl_mem_handle_is_null(input_dev_bufs)) {\n"
        "        gsi_error(\"gdl_mem_alloc() failed (%lu bytes)\\n\", io_total);\n"
        "        ret = gsi_status(ENOMEM);\n"
        "        goto CLEAN_UP;\n"
        "    }\n"
        "\n"
        f"{chain_str}\n"
        "\n"
        f"{copy_to_dev_str}\n"
        "\n"
        "    uint64_t cmd_buf_size = sizeof(cmd);\n"
        "    dev_cmd_buf = gdl_mem_alloc_aligned(ctx_id, cmd_buf_size, "
        "GDL_CONST_MAPPED_POOL, GDL_ALIGN_32);\n"
        "    if (gdl_mem_handle_is_null(dev_cmd_buf)) {\n"
        "        gsi_error(\"gdl_mem_alloc() cmd failed (%lu bytes)\\n\", cmd_buf_size);\n"
        "        ret = gsi_status(ENOMEM);\n"
        "        goto CLEAN_UP;\n"
        "    }\n"
        "    /* SPEC-018: populate the MAC popcount LUT used by gvml_lookup_16\n"
        "     * for the sv_lookup mode of binary MAC. The LUT lives inline in\n"
        "     * cmd.data.mac_lut and is copied to the device with the rest of\n"
        "     * the cmd struct. */\n"
        "    for (unsigned k = 0; k < 256; ++k) {\n"
        "        unsigned c = 0;\n"
        "        for (unsigned b = 0; b < 8; ++b) if ((k >> b) & 1u) ++c;\n"
        "        cmd.data.mac_lut[k] = (uint16_t)c;\n"
        "    }\n"
        "\n"
        "    ret = gdl_mem_cpy_to_dev(dev_cmd_buf, &cmd, cmd_buf_size);\n"
        "    if (ret) goto CLEAN_UP;\n"
        "\n"
        "    ret = gdl_run_task_timeout(\n"
        "        ctx_id,\n"
        "        GDL_TASK(apu_kernel_task),\n"
        "        dev_cmd_buf,\n"
        "        GDL_MEM_HANDLE_NULL,\n"
        "        GDL_TEMPORARY_DEFAULT_MEM_BUF,\n"
        "        GDL_TEMPORARY_DEFAULT_MEM_BUF_SIZE,\n"
        "        GDL_TEMPORARY_DEFAULT_CORE_INDEX,\n"
        "        NULL,\n"
        "        0,\n"
        "        GDL_USER_MAPPING);\n"
        "    if (ret) goto CLEAN_UP;\n"
        "\n"
        f"{copy_from_dev_str}\n"
        "\n"
        "CLEAN_UP:\n"
        "    if (!gdl_mem_handle_is_null(dev_cmd_buf)) gdl_mem_free(dev_cmd_buf);\n"
        "    if (!gdl_mem_handle_is_null(input_dev_bufs)) gdl_mem_free(input_dev_bufs);\n"
        "    return ret;\n"
        "}\n"
        "\n"
        "enum { NUM_CTXS = 1 };\n"
        "static struct gsi_sim_contexts g_ctxs[NUM_CTXS] = {\n"
        "    {\n"
        "        .apu_count = 1,\n"
        "        .apucs_per_apu = 4,\n"
        "        .mem_size = 0x40000000,\n"
        "    }\n"
        "};\n"
        "\n"
        "int main(int argc, char *argv[])\n"
        "{\n"
        "    uint32_t num_ctxs;\n"
        "    struct gdl_context_desc contexts_desc[GDL_MAX_NUM_CONTEXTS];\n"
        "\n"
        "    int ret = gsi_libsys_init(\"tenon apu program\", true);\n"
        "    if (ret) gsi_fatal(\"gsi_libsys_init(): %s\", gsi_status_errorstr(ret));\n"
        "\n"
        "    gsi_sim_create_simulator(NUM_CTXS, g_ctxs);\n"
        "\n"
        "    ret = gdl_init();\n"
        "    if (ret) gsi_fatal(\"gdl_init(): %s\", gsi_status_errorstr(ret));\n"
        "\n"
        "    ret = gdl_context_count_get(&num_ctxs);\n"
        "    if (ret) gsi_fatal(\"gdl_context_count_get(): %s\", gsi_status_errorstr(ret));\n"
        "\n"
        "    ret = gdl_context_desc_get(contexts_desc, num_ctxs);\n"
        "    if (ret) gsi_fatal(\"gdl_context_desc_get(): %s\", gsi_status_errorstr(ret));\n"
        "\n"
        "    gdl_context_handle_t valid_ctx_id = 0;\n"
        "    uint32_t ctx;\n"
        "    for (ctx = 0; ctx < num_ctxs; ++ctx) {\n"
        "        if (contexts_desc[ctx].status == GDL_CONTEXT_READY) {\n"
        "            valid_ctx_id = contexts_desc[ctx].ctx_id;\n"
        "            break;\n"
        "        }\n"
        "    }\n"
        "    if (ctx == num_ctxs) gsi_fatal(\"no valid context\");\n"
        "\n"
        "    const long long unsigned int const_mapped_size_req = 3LL * 1024L * 1024L * 1024L;\n"
        "    long long unsigned int const_mapped_size_recv = 0, dynamic_mapped_size_recv = 0;\n"
        "    ret = gdl_context_alloc(valid_ctx_id, const_mapped_size_req, "
        "&const_mapped_size_recv, &dynamic_mapped_size_recv);\n"
        "    if (ret) gsi_fatal(\"gdl_context_alloc(): %s\", gsi_status_errorstr(ret));\n"
        "\n"
        "    ret = run_tenon_kernel(valid_ctx_id, argc, argv);\n"
        "\n"
        "    gdl_context_free(valid_ctx_id);\n"
        "    gdl_exit();\n"
        "    gsi_libsys_exit();\n"
        "    return ret;\n"
        "}\n"
    )


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #


def gen_apu_v1_low_mode_project(
    *,
    dst_dir,
    compiled: "Compiled",
    inputs: dict,
    output_specs: dict,
    lab_name: str = "tenon-apu-v1-kernel",
) -> Path:
    """Emit a buildable ARC project tree and return its directory.

    See `experiments/allo/.orchestrator/specs/017-apu-v1-run-spec.md` §2.

    Parameters
    ----------
    dst_dir : Path | str
        Destination project directory. Created if missing; files are
        overwritten on reuse.
    compiled : Compiled
        Artifact from `compile_for_target(apu_v1_target, trace)`.
        `compiled.cmds` must be a `list[str]`.
    inputs : dict[str, np.ndarray]
        Workload-side memref name -> host data. Used to size the L4
        buffer for each input role and to validate dtype.
    output_specs : dict[str, tuple[shape, dtype]]
        Workload-side memref name -> (shape, dtype). Allocated on L4
        and copied back after the kernel runs.
    lab_name : str
        Becomes the Makefile `lab_name`; binary lives at
        `build/debug/<lab_name>`.
    """
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    template = _template_dir()
    if not template.exists():
        raise FileNotFoundError(
            f"APU v1 build: template dir missing at {template}"
        )

    # Step 1: copy support files.
    common_src = template / "Common"
    common_dst = dst / "Common"
    if common_dst.exists():
        shutil.rmtree(common_dst)
    shutil.copytree(common_src, common_dst)
    for fname in _COPY_FILES:
        src = template / fname
        if src.exists():
            shutil.copy2(src, dst / fname)

    # Step 2: validate IO and determine role order.
    in_roles, out_roles = _role_io_table(compiled, inputs, output_specs)

    # Step 3: compute byte sizes for host.c allocations.
    input_byte_sizes: dict[str, int] = {}
    for role in in_roles:
        arr = inputs[role]
        # np.ndarray has .nbytes; tolerate plain bytes/buffer too.
        nb = getattr(arr, "nbytes", None)
        if nb is None:
            raise ValueError(
                f"APU v1 build: input {role!r} is not a numpy array."
            )
        input_byte_sizes[role] = int(nb)
    output_byte_sizes: dict[str, int] = {}
    for role in out_roles:
        shape, dtype = output_specs[role]
        output_byte_sizes[role] = _dtype_byte_count(shape, dtype)

    # Step 4: write Makefile, struct.h, device.c, host.c.
    (dst / "Makefile").write_text(_emit_makefile(lab_name))
    (dst / "struct.h").write_text(_emit_struct_h(in_roles, out_roles))
    (dst / "device.c").write_text(_emit_device_c(compiled, in_roles, out_roles))
    (dst / "host.c").write_text(
        _emit_host_c(in_roles, out_roles, input_byte_sizes, output_byte_sizes)
    )

    return dst.resolve()
