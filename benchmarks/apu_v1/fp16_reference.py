# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional installed-GSI reference checks; never used by the compiler.

Run ``python -m benchmarks.apu_v1.fp16_reference --exhaustive --out DIR``
to check every multiply operand pair and both orders of every special-value
operand against every binary16 pattern for add/subtract. No board is used.
The shared object is built locally from the user's SDK and is not distributed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from allo.pim import apu_v1_fp16 as model
from allo.spmw_simenv import gvml_include_root


BRIDGE = r"""
#include <stddef.h>
#include <stdint.h>
#include <gsi/refgvml_element_wise.h>
#include <gsi/refgvml_iv.h>
void _gsi_log(int level, const char *format, ...) {(void)level; (void)format;}
void tenon_ref_binary(const uint16_t *a, const uint16_t *b,
                      uint16_t *out, size_t n, int op) {
    for (size_t i = 0; i < n; ++i)
        out[i] = op == 0 ? refgvml_mul_f16(a[i], b[i]) :
                 op == 1 ? refgvml_add_f16(a[i], b[i]) : refgvml_sub_f16(a[i], b[i]);
}
void tenon_ref_groups(uint16_t *out, const uint16_t *in, int log_group) {
    refgvml_add_subgrps_f16_grp(out, in, log_group, REFGVML_P2_1, 0);
}
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReferenceUnavailable(RuntimeError):
    """The optional local SDK or C compiler was not found."""


class Reference:
    """A locally linked vector wrapper around the scalar SDK oracle."""

    def __init__(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        include = Path(
            os.environ.get(
                "TENON_APU_V1_REF_INCLUDE", str(Path(gvml_include_root()) / "gsi/ref")
            )
        )
        configured = os.environ.get("TENON_APU_V1_REF_LIBRARY")
        candidates = (
            [Path(configured)]
            if configured
            else sorted(Path("/usr/local/lib").glob("**/gsi/libgsigvml_ref.a"))
        )
        compiler = shutil.which(os.environ.get("CC", "cc"))
        headers = [
            include / "gsi" / name
            for name in ("refgvml.h", "refgvml_element_wise.h", "refgvml_iv.h")
        ]
        if (
            not compiler
            or len(candidates) != 1
            or not candidates[0].is_file()
            or not all(p.is_file() for p in headers)
        ):
            raise ReferenceUnavailable(
                "GSI reference SDK/CC missing or ambiguous: set "
                "TENON_APU_V1_REF_LIBRARY and TENON_APU_V1_REF_INCLUDE "
                "(directory containing gsi/refgvml*.h)"
            )
        library = candidates[0]
        source, shared = root / "bridge.c", root / "reference.so"
        source.write_text(BRIDGE)
        command = [
            compiler,
            "-shared",
            "-fPIC",
            "-O3",
            f"-I{include}",
            str(source),
            str(library),
            "-lm",
            "-o",
            str(shared),
        ]
        built = subprocess.run(command, capture_output=True, text=True, check=False)
        (root / "build.stdout").write_text(built.stdout)
        (root / "build.stderr").write_text(built.stderr)
        if built.returncode:
            raise RuntimeError(f"GSI reference bridge build failed: {built.stderr}")
        self.library = ctypes.CDLL(str(shared))
        pointer = np.ctypeslib.ndpointer(np.uint16, flags="C_CONTIGUOUS")
        self.library.tenon_ref_binary.argtypes = [
            pointer,
            pointer,
            pointer,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.library.tenon_ref_groups.argtypes = [pointer, pointer, ctypes.c_int]
        self.provenance = {
            "command": command,
            "vendor_library": str(library),
            "vendor_library_sha256": sha256(library),
            "headers": {str(p): sha256(p) for p in headers},
            "bridge_sha256": sha256(source),
            "model_sha256": sha256(model.__file__),
            "numpy": np.__version__,
        }

    def binary(self, left, right, operation):
        left, right = np.broadcast_arrays(left, right)
        left = np.ascontiguousarray(left, dtype=np.uint16)
        right = np.ascontiguousarray(right, dtype=np.uint16)
        result = np.empty_like(left)
        self.library.tenon_ref_binary(left, right, result, left.size, operation)
        return result

    def groups(self, bits, group):
        bits = np.ascontiguousarray(bits, dtype=np.uint16)
        if bits.size != 32768:
            raise ValueError("the SDK vector reference requires exactly 32768 lanes")
        output = np.zeros_like(bits)
        self.library.tenon_ref_groups(output, bits, group.bit_length() - 1)
        return output[::group].copy()


def compare(reference, left, right, operation):
    functions = (model.mul_f16, model.add_f16, model.sub_f16)
    expected = reference.binary(left, right, operation)
    actual = functions[operation](left.view(np.float16), right.view(np.float16)).view(
        np.uint16
    )
    mismatch = np.flatnonzero((actual != expected).reshape(-1))
    a, b = np.broadcast_arrays(left, right)
    return {
        "pairs": int(expected.size),
        "mismatches": int(mismatch.size),
        "examples": [
            [int(a.flat[i]), int(b.flat[i]), int(expected.flat[i]), int(actual.flat[i])]
            for i in mismatch[:16]
        ],
        "reference_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "model_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rows-per-chunk", type=int, default=32)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(
            "use a fresh evidence directory; existing evidence is never overwritten"
        )
    if args.workers < 1 or not 1 <= args.rows_per_chunk <= 65536:
        parser.error("workers and rows-per-chunk must be positive; chunk rows <= 65536")
    args.out.mkdir(parents=True)
    ref = Reference(args.out / "reference")
    (args.out / "provenance.json").write_text(
        json.dumps(ref.provenance, indent=2) + "\n"
    )
    started = time.monotonic()
    all_bits = np.arange(65536, dtype=np.uint16)
    special = all_bits[((all_bits >> 10) & 31 == 0) | ((all_bits >> 10) & 31 == 31)]
    rng = np.random.default_rng(9213)
    totals = {}
    with (args.out / "chunks.jsonl").open("w") as stream:

        def record(label, item):
            stream.write(json.dumps({"check": label, **item}) + "\n")
            stream.flush()
            total = totals.setdefault(label, {"pairs": 0, "mismatches": 0})
            total["pairs"] += item["pairs"]
            total["mismatches"] += item["mismatches"]

        for op in range(3):
            a = rng.integers(0, 65536, 1 << 22, dtype=np.uint16)
            b = rng.integers(0, 65536, a.size, dtype=np.uint16)
            record(f"random_{op}", compare(ref, a, b, op))
        if args.exhaustive:
            for op, domain, orders in [
                (0, all_bits, (False,)),
                (1, special, (False, True)),
                (2, special, (False, True)),
            ]:
                for reverse in orders:
                    label = f"exhaustive_{op}_{'reverse' if reverse else 'forward'}"

                    def chunk(start):
                        rows = domain[start : start + args.rows_per_chunk]
                        left, right = rows[:, None], all_bits[None, :]
                        if reverse:
                            left, right = right, left
                        result = compare(ref, left, right, op)
                        return {
                            "domain_start": start,
                            "domain_rows": len(rows),
                            **result,
                        }

                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        for item in pool.map(
                            chunk, range(0, len(domain), args.rows_per_chunk)
                        ):
                            record(label, item)
                            if item["domain_start"] % 2048 == 0:
                                print(
                                    label,
                                    item["domain_start"],
                                    totals[label],
                                    flush=True,
                                )
        for log_group in range(16):
            group = 1 << log_group
            for kind in (
                "all_bits_low",
                "all_bits_high",
                "finite_random",
                "arbitrary_random",
            ):
                if kind.startswith("all_bits"):
                    bits = (
                        all_bits[:32768] if kind.endswith("low") else all_bits[32768:]
                    )
                elif kind == "finite_random":
                    bits = rng.normal(size=32768).astype(np.float16).view(np.uint16)
                else:
                    bits = rng.integers(0, 65536, 32768, dtype=np.uint16)
                expected = ref.groups(bits, group)
                actual = model.add_subgrps_f16_grp(bits.view(np.float16), group).view(
                    np.uint16
                )
                record(
                    f"groups_{group}_{kind}",
                    {
                        "pairs": int(actual.size),
                        "mismatches": int(np.count_nonzero(actual != expected)),
                    },
                )
    report = {
        "checks": totals,
        "exhaustive": args.exhaustive,
        "special_operand_patterns": int(special.size),
        "elapsed_seconds": time.monotonic() - started,
        "status": "FAIL" if any(t["mismatches"] for t in totals.values()) else "PASS",
    }
    if args.exhaustive:
        assert totals["exhaustive_0_forward"]["pairs"] == 1 << 32
        assert all(
            totals[f"exhaustive_{op}_{order}"]["pairs"] == 4096 * 65536
            for op in (1, 2)
            for order in ("forward", "reverse")
        )
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
