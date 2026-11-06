<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Third-party notices

Allo (this package) is licensed under the Apache License 2.0 (`LICENSE`).
The distributions also carry the third-party components below; their
license texts are reproduced verbatim under `THIRD_PARTY_LICENSES/` and are
installed with the wheel under `allo-<version>.dist-info/licenses/`.
Upstream notices inside individual files are retained as found.

| Component | What is shipped | Where | License | Text |
|---|---|---|---|---|
| LLVM / MLIR (`llvm-project`, the `main` branch the bindings were built from; the shared library is versioned `22.0git`) | the MLIR Python bindings (`ir.py`, `passmanager.py`, `execution_engine.py`, `rewrite.py`, `dialects/`, `extras/`, `runtime/`, the TableGen-generated `_*_gen.py` modules), the C API headers under `include/`, the aggregate C API library `libAlloMLIRAggregateCAPI.so` and the `_mlir*.so` extension modules | the wheel, `allo/_mlir/` (copied by `setup.py`'s `build_ext` from the CMake build; not in the sdist, which rebuilds them) | Apache License 2.0 with LLVM Exceptions | `THIRD_PARTY_LICENSES/LLVM-LICENSE.TXT` |
| LLVM / MLIR dialect scaffolding retained in Allo's own dialect sources | `mlir/lib/Dialect/AlloOps.cpp` and `mlir/lib/Dialect/AlloDialect.cpp` keep the LLVM file header under which they were derived (the other `mlir/` sources are Allo's, Apache-2.0) | the sdist, `mlir/` | Apache License 2.0 with LLVM Exceptions | `THIRD_PARTY_LICENSES/LLVM-LICENSE.TXT` |
| nanobind 2.10.2 (Wenzel Jakob) | statically embedded in every `allo/_mlir/_mlir_libs/_*.so` extension module by the MLIR Python binding build | the wheel | BSD 3-Clause | `THIRD_PARTY_LICENSES/NANOBIND-LICENSE.txt` |
| uPIMulator (VIA-Research, commit `870d916334e9ff0b190f555f951a9ec3c4257781`) | `simulators/uPIMulator/tenon-upimulator.patch`: Tenon's own hunks and modified hunks of upstream CMake, README and benchmark files (which bundle the UPMEM BSD-style DPU runtime sources under upstream's `sdk/`) | the sdist, `simulators/uPIMulator/` | MIT (upstream `LICENSE`, sha256 `73de95273ba877b7c8b1037d8c478179fd58ff97169da6683bc10d88b79fd16d`) | `THIRD_PARTY_LICENSES/UPIMULATOR-LICENSE.txt` |

Not shipped, by design:

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

Checksums of the bundled texts (sha256): LLVM `8d85c1057d742e597985c7d4e6320b015a9139385cff4cbae06ffc0ebe89afee`;
nanobind `f305091a4e57805870b217a28ea4b1f16610eece8457d2370f822a9696b75836`; uPIMulator `73de95273ba877b7c8b1037d8c478179fd58ff97169da6683bc10d88b79fd16d`.
