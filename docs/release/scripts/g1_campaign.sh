#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# The pre-registered Gemini-I (APU v1) paper-row campaign from an installed artifact (rounds 142-144,
# ISSUES X-46): every prerequisite failure stops the campaign before any launch; each session runs
# only the rows that still owe complete sessions under the per-row quota, counting the complete rows
# of the existing session files (G1_EXISTING) and of this campaign's own sessions (g1_pool.py's
# freshly generated plan; a planner failure stops the campaign instead of choosing hardware work
# from a stale or absent plan); a session whose headroom gate fails is skipped and recorded; every
# session's record is kept; a prior attempt's directory is never overwritten; the maintained pooler
# pools every complete row of every file under the fixed rule at the end.
#
#   g1_campaign.sh <evidence-dir>
#
# Expects <dir>/pkg/{venv,wheel,revision.txt} (a symlink to another round's pkg is fine; the package
# origin is compared by resolved path) and the benchmark drivers copied out of the checkout under
# <dir>/testtree (or G1_TESTTREE).  Environment: G1_ROW_QUOTA (default
# "kmeans=3,word_count=3,linear_regression=3": complete rows per case including the existing files),
# G1_EXISTING (whitespace-separated session JSON files already collected at this candidate, every one
# of which must exist: give explicit absolute file names, not a quoted brace pattern, which the shell
# does not expand inside a variable), G1_MAX_SESSIONS (attempts at most, default 6), G1_MEASURED
# (default 10), G1_DRY_RUN=1 (prerequisites and the plan only, under <dir>/campaign-dry-run; no
# launch), TENON_ARTIFACTS (the evaluation assets), G1_TMP_PER_LAUNCH_BYTES (the host scratch each
# launch consumes, default 60000000: rounds 136 and 146 measured 54 to 56 MB per launch of the board
# host's /tmp; a session is launched only when its whole budget is free, since the 8 MiB production
# guard protects one attempt, not a session, X-47), G1_TMP_PATH (the scratch filesystem, default /tmp).
set -u
E=${1:?evidence dir}; P=$E/pkg; T=${G1_TESTTREE:-$E/testtree}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROW_QUOTA=${G1_ROW_QUOTA:-"kmeans=3,word_count=3,linear_regression=3"}
MAX_SESSIONS=${G1_MAX_SESSIONS:-6}; MEASURED=${G1_MEASURED:-10}
PER_LAUNCH=${G1_TMP_PER_LAUNCH_BYTES:-60000000}; TMP_PATH=${G1_TMP_PATH:-/tmp}
DRY=${G1_DRY_RUN:-0}
EXISTING=(${G1_EXISTING:-})
CAMP=$E/campaign; [ "$DRY" = "1" ] && CAMP=$E/campaign-dry-run
if [ -e "$CAMP" ]; then
  echo "refusing to run: $CAMP already exists (a prior attempt's evidence is never overwritten; use a fresh evidence directory)" >&2
  exit 1
fi
export PYTHONSAFEPATH=1 PYTHONPATH=$T TMPDIR=$CAMP/tmp
export TENON_ARTIFACTS=${TENON_ARTIFACTS:-/work/shared/users/phd/nz264/spmw-for-pim/tenon-artifacts}
unset LLVM_BUILD_DIR
PYV=$P/venv/bin/python
source "$HERE/stage_lib.sh"
mkdir -p $CAMP/tmp $CAMP/scratch $CAMP/run && cd $CAMP/run
milestone_begin $CAMP/stages
echo "host $(hostname) date $(date -Is) commit $(cat $P/revision.txt) row quota [$ROW_QUOTA] existing [${EXISTING[*]:-}] max sessions $MAX_SESSIONS dry run $DRY" | tee -a $CAMP/stages/SUMMARY.txt
sha256sum $P/wheel/allo-0.5-*.whl | tee -a $CAMP/stages/SUMMARY.txt
cases_of_quota() { echo "$ROW_QUOTA" | tr ',' '\n' | cut -d= -f1 | tr '\n' ' '; }
abort() { echo "$1 $(date -Is)" | tee -a $CAMP/stages/SUMMARY.txt >&2; milestone_finish; exit 1; }
# Prerequisites: a failure here ends the campaign before any launch.
stage module_origin -- $PYV -c "import allo, benchmarks, os, sys; pkg = os.path.realpath('$P'); print('allo from', allo.__file__); print('benchmarks from', benchmarks.__file__); assert os.path.realpath(allo.__file__).startswith(pkg + '/venv/'), (os.path.realpath(allo.__file__), pkg); assert os.path.realpath(benchmarks.__file__).startswith(os.path.realpath('$T') + '/'); assert not any(p.startswith('$HERE'.rsplit('/docs/',1)[0]) for p in sys.path)" || abort "prerequisite module_origin failed; nothing launched"
stage assets_present -- $PYV -c "import os; root = os.environ['TENON_ARTIFACTS']; cases = '$(cases_of_quota)'.split(); missing = [c for c in cases if not os.path.isdir(os.path.join(root, 'apuv1', 'tenon', c))]; print('assets', root, 'cases', cases); assert not missing, missing" || abort "prerequisite assets_present failed; nothing launched"
stage existing_sessions_present -- bash -c 'set -e; for f in "$@"; do if [ -f "$f" ]; then echo "existing: $f"; else echo "MISSING existing session file: $f" >&2; exit 1; fi; done; echo "${#@} existing session file(s)"' _ "${EXISTING[@]}" || abort "prerequisite existing_sessions_present failed (every G1_EXISTING file must exist: give explicit absolute file names); nothing launched"
# plan FILES...: pool the existing and given files into a fresh plan.  Prints the next session's
# cases (empty when every row is at its quota) and returns 0 when the pooler reported a valid
# planning outcome (0: at quota, 3: rows still owed); any other pooler status, or a missing fresh
# plan, returns 1 and prints nothing: the caller must stop rather than launch.
plan() {
  local files=("${EXISTING[@]}" "$@")
  rm -f $CAMP/plan.json
  if [ ${#files[@]} -eq 0 ]; then
    echo "$(cases_of_quota)"
    return 0
  fi
  local rc=0
  $PYV -I $HERE/g1_pool.py --out $CAMP/apu_v1_campaign_pooled.json --row-quota "$ROW_QUOTA" --measured $MEASURED --plan-out $CAMP/plan.json "${files[@]}" > $CAMP/plan.log 2>&1 || rc=$?
  if [ $rc -ne 0 ] && [ $rc -ne 3 ]; then
    echo "PLANNING FAILED: g1_pool.py exited $rc (see $CAMP/plan.log)" >&2
    return 1
  fi
  [ -f $CAMP/plan.json ] || { echo "PLANNING FAILED: no fresh plan was written" >&2; return 1; }
  $PYV -I -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))['next_session_cases']))" $CAMP/plan.json || { echo "PLANNING FAILED: the plan cannot be read" >&2; return 1; }
}
# The plan is computed in this shell (its exit status decides); the stage records the same plan.
next=$(plan) || abort "planning over the existing files failed; nothing launched"
stage initial_plan -- bash -c 'echo "next session cases: [$1]"; echo "remaining complete rows:"; if [ -f "$2" ]; then cat "$2"; echo; else echo "(no existing files: every row of the quota is owed)"; fi' _ "$next" $CAMP/plan.json || abort "the initial plan could not be recorded"
if [ "$DRY" = "1" ]; then
  if [ -z "$next" ]; then
    echo "DRY RUN: every row is at its quota over the existing files; nothing would be launched" | tee -a $CAMP/stages/SUMMARY.txt
  else
    echo "DRY RUN: would launch session 1 with: $PYV -m benchmarks.apu_v1.paper_rows --measured $MEASURED --expert --historical-binary --cases $next --scratch $E/campaign/scratch/session_1 --out $E/campaign/apu_v1_campaign_s1.json" | tee -a $CAMP/stages/SUMMARY.txt
  fi
  milestone_finish; exit 0
fi
stage headroom_before_campaign -- bash -c "set -e; df -B1 --output=avail /tmp | tail -1; env -u TMPDIR $PYV -c 'from allo.pim.apu_v1_program import ledag_scratch_headroom_problem as f; p = f(); print(\"guard (TMPDIR unset):\", p); assert p is None'; $PYV -c 'from allo.pim.apu_v1_program import ledag_scratch_headroom_problem as f; p = f(); print(\"guard (TMPDIR on the evidence tree):\", p); assert p is None'" || abort "prerequisite headroom_before_campaign failed; nothing launched"
files=()
for s in $(seq 1 $MAX_SESSIONS); do
  next=$(plan "${files[@]}") || abort "planning failed before session $s; no further session is launched"
  if [ -z "$next" ]; then echo "every row at its quota after $((s - 1)) session(s); no further session is launched $(date -Is)" | tee -a $CAMP/stages/SUMMARY.txt; break; fi
  echo "session $s plans the rows still owed: [$next] $(date -Is)" | tee -a $CAMP/stages/SUMMARY.txt
  # The session's scratch budget: its cases x three lanes x (one warm-up + MEASURED) launches.
  ncases=$(echo "$next" | wc -w); launches=$(( ncases * 3 * (1 + MEASURED) ))
  if stage headroom_before_session_$s -- bash -c "set -e; avail=\$(df -B1 --output=avail '$TMP_PATH' | tail -1); need=\$(( $launches * $PER_LAUNCH )); echo \"planned launches $launches x $PER_LAUNCH bytes = \$need needed; \$avail available in $TMP_PATH\"; [ \"\$avail\" -ge \"\$need\" ]; $PYV -c 'from allo.pim.apu_v1_program import ledag_scratch_headroom_problem as f; p = f(); print(\"guard:\", p); assert p is None'"; then
    stage session_$s -- timeout 7200 $PYV -m benchmarks.apu_v1.paper_rows --measured $MEASURED --expert --historical-binary --cases $next --scratch $CAMP/scratch/session_$s --out $CAMP/apu_v1_campaign_s$s.json
    [ -f $CAMP/apu_v1_campaign_s$s.json ] && files+=($CAMP/apu_v1_campaign_s$s.json)
  else
    echo "session $s NOT LAUNCHED: the headroom gate failed at $(date -Is) ($(df -B1 --output=avail /tmp | tail -1) bytes free in /tmp)" | tee -a $CAMP/stages/SUMMARY.txt
  fi
  echo "/tmp after session $s: $(df -B1 --output=avail /tmp | tail -1) bytes free $(date -Is)" | tee -a $CAMP/stages/SUMMARY.txt
done
all=("${EXISTING[@]}" "${files[@]}")
if [ ${#all[@]} -gt 0 ]; then
  stage pool_campaign -- $PYV -I $HERE/g1_pool.py --out $CAMP/apu_v1_campaign_pooled.json --row-quota "$ROW_QUOTA" --measured $MEASURED --plan-out $CAMP/plan.json "${all[@]}"
else
  stage pool_campaign -- false
fi
milestone_finish
