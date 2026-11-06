<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Third-party notices

Allo (this package) is licensed under the Apache License 2.0 (`LICENSE`).
The distributions also carry the third-party components below; their
license and notice texts are reproduced verbatim under
`THIRD_PARTY_LICENSES/` and are installed with the wheel under
`allo-<version>.dist-info/licenses/`.  Upstream notices inside individual
files are retained as found.  The inventory of what the wheel's native
files embed was established from their symbols (`nm -C`) and strings, not
inferred from the dynamic dependencies alone; the method and the counts are
in the release ledger (`docs/release/VALIDATION.md`, rounds 88 and 89).

## Components and texts

| Component | What is shipped | Where | License | Text |
|---|---|---|---|---|
| LLVM / MLIR (`llvm-project`, the `main` branch the bindings were built from; the shared library is versioned `22.0git`) | the MLIR Python bindings (`ir.py`, `passmanager.py`, `execution_engine.py`, `rewrite.py`, `dialects/`, `extras/`, `runtime/`, the TableGen-generated `_*_gen.py` modules), the C API headers under `include/`, the aggregate C API library `libAlloMLIRAggregateCAPI.so` and the `_mlir*.so` extension modules; LLVM's own Support code these contain (the Itanium and Microsoft demanglers, SipHash, SHA-1 as adapted by LLVM) is under this license | the wheel, `allo/_mlir/` (copied by `setup.py`'s `build_ext` from the CMake build; not in the sdist, which rebuilds them) | Apache License 2.0 with LLVM Exceptions | `THIRD_PARTY_LICENSES/LLVM-LICENSE.TXT` |
| LLVM / MLIR dialect scaffolding retained in Allo's own dialect sources | `mlir/lib/Dialect/AlloOps.cpp` and `mlir/lib/Dialect/AlloDialect.cpp` keep the LLVM file header under which they were derived (the other `mlir/` sources are Allo's, Apache-2.0) | the sdist, `mlir/` | Apache License 2.0 with LLVM Exceptions | `THIRD_PARTY_LICENSES/LLVM-LICENSE.TXT` |
| xxHash (Yann Collet), as `llvm::xxHash64` / `xxh3_64bits` in LLVM Support | compiled into `libAlloMLIRAggregateCAPI.so` (symbols `llvm::xxHash64`, `llvm::xxh3_64bits`) | the wheel | BSD 2-Clause | `THIRD_PARTY_LICENSES/XXHASH-LICENSE.txt` (the notice at the head of `llvm/lib/Support/xxhash.cpp`) |
| Unicode, Inc. UTF conversion code, as `llvm::ConvertUTF*` in LLVM Support | compiled into `libAlloMLIRAggregateCAPI.so` (symbols `llvm::ConvertUTF8toUTF16`, `llvm::ConvertUTF16toUTF8`, ...) | the wheel | Unicode copyright and permission notice | `THIRD_PARTY_LICENSES/CONVERTUTF-UNICODE-NOTICE.txt` (the notice at the head of `llvm/lib/Support/ConvertUTF.cpp`) |
| Henry Spencer's regex library, as `llvm_regcomp` / `llvm_regexec` / `llvm_regfree` in LLVM Support | compiled into `libAlloMLIRAggregateCAPI.so` | the wheel | Henry Spencer's license (`llvm/lib/Support/COPYRIGHT.regex`) | `THIRD_PARTY_LICENSES/LLVM-REGEX-COPYRIGHT.txt` |
| BLAKE3 (the BLAKE3 team), as `llvm::BLAKE3` / `llvm_blake3_*` in LLVM Support | compiled into `libAlloMLIRAggregateCAPI.so` | the wheel | CC0 1.0, or alternatively Apache License 2.0 (`llvm/lib/Support/BLAKE3/LICENSE`) | `THIRD_PARTY_LICENSES/BLAKE3-LICENSE.txt` |
| MD5 (Alexander Peslyak, "Solar Designer"), as `llvm::MD5` in LLVM Support | compiled into `libAlloMLIRAggregateCAPI.so` | the wheel | public domain, with the fallback terms in the notice | `THIRD_PARTY_LICENSES/MD5-NOTICE.txt` (the notice at the head of `llvm/lib/Support/MD5.cpp`) |
| nanobind (Wenzel Jakob) | statically embedded in every `allo/_mlir/_mlir_libs/_*.so` extension module (`nanobind::` symbols in all twenty). Two origins: the nineteen `_mlir*` modules come from the shared LLVM build, whose cache names nanobind 2.10.2 (`NB_DIR`); the `_allo` module is built by `setup.py`'s CMake step against the nanobind of the building environment (`setup_requires=["nanobind>=2.9,<3.0"]`; 2.13.0 in the round 88 and 89 builds, `mlir/build/CMakeCache.txt` `NB_DIR`), so a future supported build embeds whatever version that range resolves | the wheel | BSD 3-Clause (the text is identical across 2.10.2 and 2.13.0) | `THIRD_PARTY_LICENSES/NANOBIND-LICENSE.txt` |
| robin-map (Thibaut Goetghebuer-Planchon), nanobind's `ext/robin_map` dependency | template instantiations compiled into every `allo/_mlir/_mlir_libs/_*.so` extension module (`tsl::detail_robin_hash::` symbols in all twenty) | the wheel | MIT | `THIRD_PARTY_LICENSES/ROBIN-MAP-LICENSE.txt` (the notice at the head of `tsl/robin_map.h`) |
| uPIMulator (VIA-Research, commit `870d916334e9ff0b190f555f951a9ec3c4257781`) | the hunks of `simulators/uPIMulator/tenon-upimulator.patch` that modify upstream's own files (CMake, README and benchmark files under `golang/uPIMulator/benchmark/`), plus Tenon's own hunks | the sdist, `simulators/uPIMulator/` | MIT (upstream `LICENSE`, sha256 `73de95273ba877b7c8b1037d8c478179fd58ff97169da6683bc10d88b79fd16d`) | `THIRD_PARTY_LICENSES/UPIMULATOR-LICENSE.txt` |
| UPMEM DPU runtime (`dpu-rt`, "Copyright 2020 UPMEM", BSD-style), as bundled by upstream uPIMulator under `golang/uPIMulator/sdk/` | the hunks of the same patch that modify the runtime sources `sdk/stdlib/stdio.c` and `sdk/syslib/dpuruntime.h`, the runtime's CMake files under `sdk/`, and the new file `sdk/misc/stdoutBufferState.c`, which Tenon wrote against those sources and placed under UPMEM's retained notice | the sdist, `simulators/uPIMulator/` | BSD 3-Clause (UPMEM's `dpu-rt/LICENSE`, identical text in the UPMEM SDK 2023.1.0 and referenced by the files' headers) | `THIRD_PARTY_LICENSES/UPMEM-DPU-RT-LICENSE.txt` |

## Not shipped, by design

* Samsung PIMSimulator sources and the five Tenon patches to them
  (Samsung's notice forbids redistribution; `simulators/PIMSimulator/README.md`,
  release ledger P-04). The package ships only Tenon's own conductor sources
  under `simulators/PIMSimulator/src/`, `apply.sh` and a `Sconstruct` whose
  structure follows upstream's (listed as an open question of P-04).
* GSI Gemini-I SDK files (`example-gvml`): staged from the user's SDK at
  build time by `allo/spmw_apu_v1_build.py`, recorded in a per-project
  `THIRD_PARTY.txt`, never copied into this package (release ledger V1-08).
* The system libraries the aggregate library links dynamically (`libz`,
  `libzstd`, the C and C++ runtimes) are not bundled; they come from the
  host.

## Checksums of the bundled texts (sha256)

* `THIRD_PARTY_LICENSES/BLAKE3-LICENSE.txt`: `6a94bedb8b707ed97f6e310d0d015ab14e0683ffa0a612b02958581b9cc9fc0e`
* `THIRD_PARTY_LICENSES/CONVERTUTF-UNICODE-NOTICE.txt`: `34f4f9bd10889e31554b8d07edccc41a9d26b6b138da9024020eee6ea6989029`
* `THIRD_PARTY_LICENSES/LLVM-LICENSE.TXT`: `8d85c1057d742e597985c7d4e6320b015a9139385cff4cbae06ffc0ebe89afee`
* `THIRD_PARTY_LICENSES/LLVM-REGEX-COPYRIGHT.txt`: `0424e57d4303164dc59a8509c20dae0518b853692e5c2b0e98b11816fdbc97c7`
* `THIRD_PARTY_LICENSES/MD5-NOTICE.txt`: `b95327f2bd0adacaed56b02916fc90a9be2ca2ab2324332c6f1217414d5383cb`
* `THIRD_PARTY_LICENSES/NANOBIND-LICENSE.txt`: `f305091a4e57805870b217a28ea4b1f16610eece8457d2370f822a9696b75836`
* `THIRD_PARTY_LICENSES/ROBIN-MAP-LICENSE.txt`: `ecbbd4f8369e99a884dc41bf6b3f79a2c1efdadcd2a5ecc03947c09c78759bd1`
* `THIRD_PARTY_LICENSES/UPIMULATOR-LICENSE.txt`: `73de95273ba877b7c8b1037d8c478179fd58ff97169da6683bc10d88b79fd16d`
* `THIRD_PARTY_LICENSES/UPMEM-DPU-RT-LICENSE.txt`: `8f49c23ec6464b86586dde2e0dc2694494d20ff28a45f7337e756b2eda0b498f`
* `THIRD_PARTY_LICENSES/XXHASH-LICENSE.txt`: `99c01ca52b22f5716cb82abf6be14ede0c0dfcdfdcb25f2e138794ea42b761b6`
