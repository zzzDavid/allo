#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# The pre-registered Gemini-I (APU v1) paper-row campaign from an installed artifact (rounds 142-143,
# ISSUES X-46): every prerequisite failure stops the campaign before any launch; each session runs
# only the rows that still owe complete sessions under the per-row quota, counting the complete rows
# of the existing session files (G1_EXISTING) and of this campaign's own sessions (g1_pool.py's
# plan), so no extra complete row is collected; a session whose headroom gate fails is skipped and
# recorded; every session's record is kept; the maintained pooler pools every complete row of every
# file under the fixed rule at the end.
#
#   g1_campaign.sh <evidence-dir>
#
# Expects <dir>/pkg/{venv,wheel,revision.txt} (a symlink to another round's pkg is fine; the package
# origin is compared by resolved path) and the benchmark drivers copied out of the checkout under
# <dir>/testtree (or G1_TESTTREE).  Environment: G1_ROW_QUOTA (default
# "kmeans=3,word_count=3,linear_regression=3": complete rows per case including the existing files),
# G1_EXISTING (space-separated session JSON files already collected at this candidate, pooled with
# the new ones and counted toward the quota), G1_MAX_SESSIONS (attempts at most, default 6),
# G1_MEASURED (default 10), G1_DRY_RUN=1 (prerequisites, plan and the first session's command only;
# no launch), TENON_ARTIFACTS (the evaluation assets).
set -u
E=${1:?evidence dir}; P=$E/pkg; T=${G1_TESTTREE:-$E/testtree}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROW_QUOTA=${G1_ROW_QUOTA:-"kmeans=3,word_count=3,linear_regression=3"}
MAX_SESSIONS=${G1_MAX_SESSIONS:-6}; MEASURED=${G1_MEASURED:-10}
EXISTING=(${G1_EXISTING:-})
export PYTHONSAFEPATH=1 PYTHONPATH=$T TMPDIR=$E/campaign/tmp
export TENON_ARTIFACTS=${TENON_ARTIFACTS:-/work/shared/users/phd/nz264/spmw-for-pim/tenon-artifacts}
unset LLVM_BUILD_DIR
PYV=$P/venv/bin/python
source "$HERE/stage_lib.sh"
mkdir -p $E/campaign/tmp $E/campaign/scratch $E/campaign/run && cd $E/campaign/run
milestone_begin $E/campaign/stages
echo "host $(hostname) date $(date -Is) commit $(cat $P/revision.txt) row quota [$ROW_QUOTA] existing [${EXISTING[*]:-}] max sessions $MAX_SESSIONS dry run ${G1_DRY_RUN:-0}" | tee -a $E/campaign/stages/SUMMARY.txt
sha256sum $P/wheel/allo-0.5-*.whl | tee -a $E/campaign/stages/SUMMARY.txt
cases_of_quota() { echo "$ROW_QUOTA" | tr ',' '\n' | cut -d= -f1 | tr '\n' ' '; }
# Prerequisites: a failure here ends the campaign before any launch.
stage module_origin -- $PYV -c "import allo, benchmarks, os, sys; pkg = os.path.realpath('$P'); print('allo from', allo.__file__); print('benchmarks from', benchmarks.__file__); assert os.path.realpath(allo.__file__).startswith(pkg + '/venv/'), (os.path.realpath(allo.__file__), pkg); assert os.path.realpath(benchmarks.__file__).startswith(os.path.realpath('$T') + '/'); assert not any(p.startswith('$HERE'.rsplit('/docs/',1)[0]) for p in sys.path)" || { milestone_finish; exit 1; }
stage assets_present -- $PYV -c "import os; root = os.environ['TENON_ARTIFACTS']; cases = '$(cases_of_quota)'.split(); missing = [c for c in cases if not os.path.isdir(os.path.join(root, 'apuv1', 'tenon', c))]; print('assets', root, 'cases', cases); assert not missing, missing" || { milestone_finish; exit 1; }
stage existing_sessions_present -- bash -c "for f in ${EXISTING[*]:-}; do test -f \$f && echo \$f; done; true" || { milestone_finish; exit 1; }
plan() {  # pool existing + new files, write the plan; prints the next session's cases (empty when done)
  local files=("${EXISTING[@]}" "$@")
  if [ ${#files[@]} -eq 0 ]; then echo "$(cases_of_quota)"; return; fi
  $PYV -I $HERE/g1_pool.py --out $E/campaign/apu_v1_campaign_pooled.json --row-quota "$ROW_QUOTA" --measured $MEASURED --plan-out $E/campaign/plan.json "${files[@]}" > $E/campaign/plan.log 2>&1
  $PYV -c "import json; print(' '.join(json.load(open('$E/campaign/plan.json'))['next_session_cases']))"
}
stage initial_plan -- bash -c "$(declare -f plan cases_of_quota); EXISTING=(${EXISTING[*]:-}); PYV=$PYV; HERE=$HERE; E=$E; ROW_QUOTA='$ROW_QUOTA'; MEASURED=$MEASURED; next=\$(plan); echo \"next session cases: [\$next]\"; cat $E/campaign/plan.json 2>/dev/null; true" || { milestone_finish; exit 1; }
if [ "${G1_DRY_RUN:-0}" = "1" ]; then
  next=$(plan)
  echo "DRY RUN: would launch session 1 with: $PYV -m benchmarks.apu_v1.paper_rows --measured $MEASURED --expert --historical-binary --cases $next --scratch $E/campaign/scratch/session_1 --out $E/campaign/apu_v1_campaign_s1.json" | tee -a $E/campaign/stages/SUMMARY.txt
  milestone_finish; exit 0
fi
stage headroom_before_campaign -- bash -c "df -B1 --output=avail /tmp | tail -1; env -u TMPDIR $PYV -c 'from allo.pim.apu_v1_program import ledag_scratch_headroom_problem as f; p = f(); print(\"guard (TMPDIR unset):\", p); assert p is None'; $PYV -c 'from allo.pim.apu_v1_program import ledag_scratch_headroom_problem as f; p = f(); print(\"guard (TMPDIR on the evidence tree):\", p); assert p is None'" || { milestone_finish; exit 1; }
files=()
for s in $(seq 1 $MAX_SESSIONS); do
  next=$(plan "${files[@]}")
  if [ -z "$next" ]; then echo "every row at its quota after $((s - 1)) session(s); no further session is launched $(date -Is)" | tee -a $E/campaign/stages/SUMMARY.txt; break; fi
  echo "session $s plans the rows still owed: [$next] $(date -Is)" | tee -a $E/campaign/stages/SUMMARY.txt
  if stage headroom_before_session_$s -- bash -c "df -B1 --output=avail /tmp | tail -1; $PYV -c 'from allo.pim.apu_v1_program import ledag_scratch_headroom_problem as f; p = f(); print(\"guard:\", p); assert p is None'"; then
    stage session_$s -- timeout 7200 $PYV -m benchmarks.apu_v1.paper_rows --measured $MEASURED --expert --historical-binary --cases $next --scratch $E/campaign/scratch/session_$s --out $E/campaign/apu_v1_campaign_s$s.json
    [ -f $E/campaign/apu_v1_campaign_s$s.json ] && files+=($E/campaign/apu_v1_campaign_s$s.json)
  else
    echo "session $s NOT LAUNCHED: the headroom gate failed at $(date -Is) ($(df -B1 --output=avail /tmp | tail -1) bytes free in /tmp)" | tee -a $E/campaign/stages/SUMMARY.txt
  fi
  echo "/tmp after session $s: $(df -B1 --output=avail /tmp | tail -1) bytes free $(date -Is)" | tee -a $E/campaign/stages/SUMMARY.txt
done
all=("${EXISTING[@]}" "${files[@]}")
if [ ${#all[@]} -gt 0 ]; then
  stage pool_campaign -- $PYV -I $HERE/g1_pool.py --out $E/campaign/apu_v1_campaign_pooled.json --row-quota "$ROW_QUOTA" --measured $MEASURED --plan-out $E/campaign/plan.json "${all[@]}"
else
  stage pool_campaign -- false
fi
milestone_finish
