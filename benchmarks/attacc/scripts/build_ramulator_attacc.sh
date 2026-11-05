#!/bin/bash
# Build Ramulator2-AttAcc natively from its MIT sources (Ramulator 2.0 with
# the AttAcc HBM3-PIM model, as packaged under DCC's src/src/Ramulator2/AttAcc:
# CMakeLists.txt, src/, ext/{yaml-cpp,spdlog,argparse} vendored, LICENSE).
# Use it when the host cannot run a prebuilt binary (glibc) and has no
# container: it needs CMake >= 3.14 and a C++20 compiler (gcc >= 11).
#
#   build_ramulator_attacc.sh <AttAcc source dir> <output dir> [CXX [CC]]
#
# Output: <output dir>/build/ramulator2 (the binary) and
# <output dir>/src/libramulator.so (CMake writes the library into the source
# copy); point ATTACC_RAMULATOR and ATTACC_RAMULATOR_LIBDIR at them.  The
# sources are copied once into <output dir>/src so a read-only checkout works.
set -eu
SRC=${1:?AttAcc source dir (DCC src/src/Ramulator2/AttAcc)}
OUT=${2:?output dir}
CXX=${3:-${CXX:-g++}}
CC=${4:-${CC:-gcc}}
[ -f "$SRC/CMakeLists.txt" ] && [ -d "$SRC/ext/yaml-cpp" ] || { echo "not an AttAcc Ramulator2 tree: $SRC" >&2; exit 2; }
mkdir -p "$OUT"
[ -d "$OUT/src" ] || rsync -a --exclude build --exclude '*.so' "$SRC/" "$OUT/src/"
rm -rf "$OUT/build"
cmake -S "$OUT/src" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX"
cmake --build "$OUT/build" -j "${JOBS:-8}"
ls -la "$OUT/build/ramulator2" "$OUT/src/libramulator.so"
sha256sum "$OUT/build/ramulator2" "$OUT/src/libramulator.so"
echo "export ATTACC_RAMULATOR=$OUT/build/ramulator2"
echo "export ATTACC_RAMULATOR_LIBDIR=$OUT/src"
