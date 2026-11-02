<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# aim_simulator build environment for Tenon (SK hynix AiM backend)

Tenon's AiM backend (`allo/spmw_aim.py`) replays its command trace on the
Ramulator 2.0 based AiM simulator. The upstream tree needs no source change;
it ships no build environment, and its binary needs glibc >= 2.38, so Tenon
provides only this `Dockerfile` (Ubuntu 24.04, g++-12, CMake).

## Upstream

* Repository: <https://github.com/arkhadem/aim_simulator>
* Commit: `0f28a07bdb83e42b9305ad3d45410ebd3aa2c091` (2025-07-22)
* License: MIT, "Copyright (c) 2025 University of Michigan" (`LICENSE` in the
  checkout; authors Yufeng Gu and Alireza Khadem, ASPLOS 2025 "PIM is All You
  Need"). Tenon vendors no simulator code.

## Build

```bash
git clone https://github.com/arkhadem/aim_simulator.git
git -C aim_simulator checkout 0f28a07bdb83e42b9305ad3d45410ebd3aa2c091
cp $ALLO/simulators/aim_simulator/Dockerfile aim_simulator/
docker build -t tenon-aim-simulator-build:0f28a07 aim_simulator
docker run --rm -u $(id -u):$(id -g) -v $PWD/aim_simulator:/work tenon-aim-simulator-build:0f28a07 \
    bash -c "mkdir -p build && cd build && cmake .. && make -j8"
# smoke test: memory_system_cycles: 689
docker run --rm -u $(id -u):$(id -g) -v $PWD/aim_simulator:/work tenon-aim-simulator-build:0f28a07 \
    bash -c "cd /work && ./build/ramulator2 -f test/example.yaml -t test/all_isr.trace"
export AIM_SIMULATOR_ROOT=$PWD/aim_simulator TENON_AIM_SIM_IMAGE=tenon-aim-simulator-build:0f28a07
```

`-u $(id -u):$(id -g)` keeps the build tree owned by you. CMake downloads
third-party dependencies into `build/_deps` during configure, so the build
needs network access. Any tag works; Tenon reads it from
`TENON_AIM_SIM_IMAGE` (default `aim-simulator-build`, the authors' image).

## Verification

```bash
cd $ALLO
python -m pytest tests/pim/test_gemv_all_targets.py -k aim   # 2 passed; AiM GEMV 256x256 BF16: 222 cycles
```

The model is timing only (no data path); `test/example.yaml` fixes 32
channels and Tenon rejects AiM targets whose geometry differs from it.
Recorded runs: `docs/release/simulators.md`.
