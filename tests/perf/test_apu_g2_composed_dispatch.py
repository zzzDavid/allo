# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the device template against recording SDK stubs, without a card.

This checks primitive order and timing boundaries, not SDK arithmetic or ticks.
The real-card numerical/performance checks remain separate obligations.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

from allo.pim import apu_g2_runtime


@pytest.fixture(scope="module")
def dispatch_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("device-template recording check requires a C++ compiler")
    root = tmp_path_factory.mktemp("composed-dispatch")
    headers = {
        "device/vector_core/gsi_libgal.h": """
#pragma once
#include <cstdint>
#define GSI_LIBRARY_ENTRY_POINT(name, arg) int name(arg)
#define GSI_STATUS_SUCCESS 0
#define GSI_STATUS_GENERAL_ERROR 1
""",
        "transport_gsi_library.h": """
#define GSI_LIBRARY_ENTRY_POINT_FUNCTION_PROTOTYPE(name, type) int name(type)
""",
        "g2_64vl_types.h": """
#pragma once
namespace gsi {
enum class G2_TYPES { INT, UINT };
namespace g2_64vl {
struct L1Vectors {
  L1Vectors(unsigned, G2_TYPES, unsigned, unsigned, unsigned) {}
};
struct MmbVectors {
  MmbVectors(unsigned, G2_TYPES, unsigned, unsigned) {}
};
struct MmbVectors_seg0 : MmbVectors { using MmbVectors::MmbVectors; };
struct MmbVectors_seg1 : MmbVectors { using MmbVectors::MmbVectors; };
}}
""",
        "gsi_arc_cache.h": """
#include <cstddef>
void gsi_arc_dcache_flush_mlines(void *, size_t);
""",
        "gsi_perf.h": """
void gsi_perf_start();
uint64_t gsi_perf_end();
""",
    }
    for name, contents in headers.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    template = Path(apu_g2_runtime.__file__).parent / "templates" / "apu_g2"
    # Include the distributed production source, without rewriting its body.
    probe = root / "probe.cc"
    probe.write_text(
        r"""
#include <cstdlib>
#include <iostream>
#include "apu_g2_composed_dot.cc"

void event(const char *name) { std::cout << name << '\n'; }
void gsi_perf_start() { event("start"); }
uint64_t gsi_perf_end() { event("end"); return 123; }
void gsi_arc_dcache_flush_mlines(void *, size_t) { event("flush"); }
namespace gsi::g2_64vl {
void copy(const L1Vectors &, const MmbVectors &) { event("load"); }
void copy(const MmbVectors &, const L1Vectors &) { event("store"); }
void add(const MmbVectors_seg0 &, const MmbVectors_seg1 &,
         const MmbVectors &) { event("add"); }
void mul(const MmbVectors_seg0 &, const MmbVectors_seg0 &,
         const MmbVectors_seg1 &) { event("mul"); }
void sum(const MmbVectors_seg0 &, uint32_t,
         const MmbVectors_seg1 &) { event("sum"); }
void copy_from_odd_to_even_vectors(const MmbVectors &, const MmbVectors &) {
  event("odd_to_even");
}
void seu_barrier() { event("barrier"); }
}
int main(int argc, char **argv) {
  if (argc != 5) return 2;
  ApuG2ComposedDotParams p{};
  p.epilogue = std::strtoul(argv[1], nullptr, 10);
  p.repetitions = std::strtoul(argv[2], nullptr, 10);
  p.lhs_signed = p.rhs_signed = p.out_signed = p.auxiliary_signed =
      std::strtoul(argv[3], nullptr, 10);
  p.log_reduction = std::strtoul(argv[4], nullptr, 10);
  p.lhs_bits = p.rhs_bits = 3;
  p.dot_bits = 6 + p.log_reduction;
  p.auxiliary_bits = p.epilogue == 2 ? p.dot_bits : 2;
  p.out_bits = p.dot_bits + (p.epilogue == 0 ? 0 : p.epilogue == 1 ? 3 : 1);
  ApuG2ComposedDotTimings timings{};
  p.timings_l5 = reinterpret_cast<uintptr_t>(&timings);
  const int rc = tenon_apu_g2_composed_dot(reinterpret_cast<uintptr_t>(&p));
  std::cout << "result " << rc << ' ' << timings.pipeline_ticks << ' '
            << timings.final_pipeline_ticks << '\n';
}
"""
    )
    binary = root / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{root}",
            f"-I{template}",
            f"-I{template / 'device'}",
            str(probe),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return binary


def test_complete_primitive_sequences_stay_inside_both_timed_windows(dispatch_probe):
    reduction = ["load", "load", "mul", "store", "load", "sum"]
    epilogues = (
        ["store"],
        [
            "store",
            "load",
            "load",
            "mul",
            "store",
            "load",
            "odd_to_even",
            "add",
            "store",
        ],
        ["store", "load", "load", "add", "store"],
    )
    for epilogue, tail in enumerate(epilogues):
        for repetitions, signed, log_reduction in ((1, 0, 1), (7, 1, 8)):
            result = subprocess.run(
                [
                    str(dispatch_probe),
                    *map(str, (epilogue, repetitions, signed, log_reduction)),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            body = reduction + tail
            assert result.stdout.splitlines() == (
                ["start"]
                + body * repetitions
                + ["barrier", "end", "start"]
                + body
                + ["barrier", "end", "flush", "result 0 123 123"]
            )


def test_invalid_task_never_starts_a_timing_window(dispatch_probe):
    for args in ((3, 7, 1, 8), (0, 0, 1, 8), (0, 1, 2, 8), (0, 1, 1, 0)):
        result = subprocess.run(
            [str(dispatch_probe), *map(str, args)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.stdout == "result 1 0 0\n"
