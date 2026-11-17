#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# The Gemini-I (APU v1) board validation of an installed candidate (round 147, ISSUES X-47): the
# vector programs through the default selection with the expert and archived paper-era Tenon lanes
# interleaved, the forced alternative plan and the cost perturbation, each stage once, from the
# installed wheel with the benchmark drivers copied out of the checkout, under the production
# launcher's board lock.  Every prerequisite failure (package origin, assets, scratch headroom) stops
# the run before any launch, explicitly; each stage has its own headroom gate and a stage whose gate
# fails is recorded as a failed stage, not skipped silently; a prior attempt's directory is never
# overwritten; the driver preserves a session the host's scratch cuts short as incomplete.
#
#   g1_board.sh <evidence-dir>
#
# Expects <dir>/pkg/{venv,wheel,revision.txt} (a symlink to another round's pkg is fine; the package
# origin is compared by resolved path) and the drivers under <dir>/testtree (or G1_TESTTREE).
# Environment: G1_MEASURED (vector rows, default 10), G1_FORCED_MEASURED (forced plan and
# perturbation, default 5), G1_DRY_RUN=1 (prerequisites and the commands only, under
# <dir>/board-dry-run; no launch), TENON_ARTIFACTS (the evaluation assets), G1_TMP_PER_LAUNCH_BYTES
# (the retained-log/build/evidence reserve per launch, default 1 GiB: up to seven
# 64 MiB raw reads, their normalized evidence and build/output scratch). G1_TMP_PATH
# actually selects TMPDIR; it is not a pretend relocation of /tmp. Bulk profile
# copies use TENON_APU_V1_PROFILE_DIR (default <run>/profiles). All actual storage
# filesystems are budgeted; /tmp's small vendor session log is checked separately
# by the runtime. Place the evidence directory on a filesystem with room.
set -u
E=${1:?evidence dir}; [[ "$E" = /* ]] || { echo "evidence directory must be absolute" >&2; exit 1; }; P=$E/pkg; T=${G1_TESTTREE:-$E/testtree}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MEASURED=${G1_MEASURED:-10}; FORCED_MEASURED=${G1_FORCED_MEASURED:-5}
PER_LAUNCH=${G1_TMP_PER_LAUNCH_BYTES:-1073741824}
DRY=${G1_DRY_RUN:-0}
B=$E/board; [ "$DRY" = "1" ] && B=$E/board-dry-run
if [ -e "$B" ]; then
  echo "refusing to run: $B already exists (a prior attempt's evidence is never overwritten; use a fresh evidence directory)" >&2
  exit 1
fi
export PYTHONSAFEPATH=1 PYTHONPATH=$T TMPDIR=${G1_TMP_PATH:-$B/tmp}
export TENON_APU_V1_PROFILE_DIR=${TENON_APU_V1_PROFILE_DIR:-$B/profiles}
export TENON_ARTIFACTS=${TENON_ARTIFACTS:-/work/shared/users/phd/nz264/spmw-for-pim/tenon-artifacts}
unset LLVM_BUILD_DIR
PYV=$P/venv/bin/python
source "$HERE/stage_lib.sh"
mkdir -p "$TMPDIR" "$TENON_APU_V1_PROFILE_DIR" "$B/scratch" "$B/run" || exit 1
cd "$B/run" || exit 1
milestone_begin $B/stages
echo "host $(hostname) date $(date -Is) commit $(cat $P/revision.txt) pkg $(readlink -f $P) drivers $(readlink -f $T) measured $MEASURED forced $FORCED_MEASURED dry run $DRY" | tee -a $B/stages/SUMMARY.txt
sha256sum $P/wheel/allo-0.5-*.whl | tee -a $B/stages/SUMMARY.txt
abort() { echo "$1 $(date -Is)" | tee -a $B/stages/SUMMARY.txt >&2; milestone_finish; exit 1; }
GUARD='from allo.pim.apu_v1_program import ledag_scratch_headroom_problem as f; p = f(); print("guard:", p); assert p is None, p'
# Prerequisites: a failure here ends the run before any launch.
stage module_origin -- $PYV -c "import allo, benchmarks, os, sys; pkg = os.path.realpath('$P'); print('allo from', allo.__file__); print('benchmarks from', benchmarks.__file__); assert os.path.realpath(allo.__file__).startswith(pkg + '/venv/'), (os.path.realpath(allo.__file__), pkg); assert os.path.realpath(benchmarks.__file__).startswith(os.path.realpath('$T') + '/'); assert not any(p.startswith('$HERE'.rsplit('/docs/',1)[0]) for p in sys.path)" || abort "prerequisite module_origin failed; nothing launched"
stage assets_present -- $PYV -c "import os; root = os.environ['TENON_ARTIFACTS']; cases = ['matrix_multiply', 'histogram']; missing = [c for c in cases if not os.path.isdir(os.path.join(root, 'apuv1', 'tenon', c))]; print('assets', root, 'cases', cases); assert not missing, missing" || abort "prerequisite assets_present failed; nothing launched"
stage headroom_before_board -- bash -c "set -e; df -B1 --output=avail /tmp | tail -1; $PYV -c '$GUARD'" || abort "prerequisite headroom_before_board failed; nothing launched"
# The sequence's scratch budget: two vector cases x three lanes x (one warm-up + MEASURED), plus the
# forced plan and the perturbation (one lane each, one warm-up + FORCED_MEASURED).
LAUNCHES=$(( 2 * 3 * (1 + MEASURED) + 2 * (1 + FORCED_MEASURED) ))
stage scratch_budget -- "$PYV" -I "$HERE/g1_storage_budget.py" "$LAUNCHES" "$PER_LAUNCH" "$B" "$TMPDIR" "$TENON_APU_V1_PROFILE_DIR" || abort "prerequisite scratch_budget failed (actual retained-profile, build and evidence storage); nothing launched"
# Dry runs do not contact the board. Real sequences prove daemon storage access
# and measure the copy time once before launching any pre-registered sample.
if [ "$DRY" != "1" ]; then
  stage transport_preflight -- "$PYV" "$HERE/g1_transport_preflight.py" "$B/transport-preflight.json" || abort "prerequisite transport_preflight failed; nothing launched"
  SECONDS_PER_LAUNCH=$("$PYV" -c 'import json,sys; print(json.load(open(sys.argv[1]))["seconds_per_launch"])' "$B/transport-preflight.json") || abort "cannot read fresh transport time budget; nothing launched"
fi
# run NAME DRIVER-ARGS...: the stage's own headroom gate, then the driver; a gate failure is a
# failed stage (the evidence is incomplete), recorded, and the following stages still try their gates.
run() {
  local name=$1 launches=$2; shift 2
  if [ "$DRY" = "1" ]; then
    echo "DRY RUN: would launch $name with: $PYV -m benchmarks.apu_v1.paper_rows $* --scratch $E/board/scratch/$name --out $E/board/apu_v1_$name.json" | tee -a $B/stages/SUMMARY.txt
    return 0
  fi
  local failed_gate=""
  if ! stage headroom_before_$name -- "$PYV" -c "$GUARD"; then
    failed_gate=headroom_before_$name
  elif ! stage storage_before_$name -- "$PYV" -I "$HERE/g1_storage_budget.py" "$launches" "$PER_LAUNCH" "$B" "$TMPDIR" "$TENON_APU_V1_PROFILE_DIR"; then
    failed_gate=storage_before_$name
  fi
  if [ -z "$failed_gate" ]; then
    local stage_seconds=$((600 + launches * SECONDS_PER_LAUNCH))
    echo "stage $name timeout: 600 + $launches x $SECONDS_PER_LAUNCH = $stage_seconds seconds" | tee -a $B/stages/SUMMARY.txt
    stage $name -- timeout "$stage_seconds" $PYV -m benchmarks.apu_v1.paper_rows "$@" --scratch $B/scratch/$name --out $B/apu_v1_$name.json
  else
    stage $name -- bash -c 'echo "NOT LAUNCHED: gate $1 failed; see its stage log for the actual storage path"; exit 1' _ "$failed_gate"
  fi
  echo "/tmp after $name: $(df -B1 --output=avail /tmp | tail -1) bytes free $(date -Is)" | tee -a $B/stages/SUMMARY.txt
}
run vector_rows "$((2 * 3 * (1 + MEASURED)))" --measured $MEASURED --expert --historical-binary --cases matrix_multiply histogram
run forced_acc4 "$((1 + FORCED_MEASURED))" --measured $FORCED_MEASURED --cases matrix_multiply --layout temporal_dma_coalescing_broadcast_friendly_acc4 --label forced_acc4
run perturbed_dma "$((1 + FORCED_MEASURED))" --measured $FORCED_MEASURED --cases matrix_multiply --perturb MICRO_DMA_L4_L1_32K=0.01 --label perturbed_dma
milestone_finish
