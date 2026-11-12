#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# The pre-registered Gemini-I (APU v1) paper-row campaign from an installed artifact (round 142,
# ISSUES X-46): every prerequisite failure stops the campaign before any launch, a session whose
# headroom gate fails is skipped and recorded (never launched into an exhausted scratch), every
# session's record is kept, and the maintained pooler (g1_pool.py) pools the complete rows under the
# fixed rule.
#
#   g1_campaign.sh <evidence-dir>
#
# Expects <dir>/pkg/{venv,wheel,revision.txt} (a symlink to another round's pkg is fine; the package
# origin is compared by resolved path) and the benchmark drivers copied out of the checkout under
# <dir>/testtree (stage_lib's copy_tree_with_optional).  Environment: G1_SESSIONS (default 3),
# G1_QUOTA (default = G1_SESSIONS), G1_CASES (default "kmeans linear_regression word_count"),
# G1_MEASURED (default 10), TENON_ARTIFACTS (the evaluation assets).
set -u
E=${1:?evidence dir}; P=$E/pkg; T=$E/testtree
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SESSIONS=${G1_SESSIONS:-3}; QUOTA=${G1_QUOTA:-$SESSIONS}; MEASURED=${G1_MEASURED:-10}
CASES=${G1_CASES:-"kmeans linear_regression word_count"}
export PYTHONSAFEPATH=1 PYTHONPATH=$T TMPDIR=$E/campaign/tmp
export TENON_ARTIFACTS=${TENON_ARTIFACTS:-/work/shared/users/phd/nz264/spmw-for-pim/tenon-artifacts}
unset LLVM_BUILD_DIR
PYV=$P/venv/bin/python
source "$HERE/stage_lib.sh"
mkdir -p $E/campaign/tmp $E/campaign/scratch $E/campaign/run && cd $E/campaign/run
milestone_begin $E/campaign/stages
echo "host $(hostname) date $(date -Is) commit $(cat $P/revision.txt) sessions $SESSIONS quota $QUOTA cases [$CASES]" | tee -a $E/campaign/stages/SUMMARY.txt
sha256sum $P/wheel/allo-0.5-*.whl | tee -a $E/campaign/stages/SUMMARY.txt
# Prerequisites: a failure here ends the campaign before any launch.
stage module_origin -- $PYV -c "import allo, benchmarks, os, sys; pkg = os.path.realpath('$P'); print('allo from', allo.__file__); print('benchmarks from', benchmarks.__file__); assert os.path.realpath(allo.__file__).startswith(pkg + '/venv/'), (os.path.realpath(allo.__file__), pkg); assert os.path.realpath(benchmarks.__file__).startswith(os.path.realpath('$T') + '/'); assert not any(p.startswith('$HERE'.rsplit('/docs/',1)[0]) for p in sys.path)" || { milestone_finish; exit 1; }
stage assets_present -- $PYV -c "import os; root = os.environ['TENON_ARTIFACTS']; cases = '$CASES'.split(); missing = [c for c in cases if not os.path.isdir(os.path.join(root, 'apuv1', 'tenon', c))]; print('assets', root, 'cases', cases); assert not missing, missing" || { milestone_finish; exit 1; }
stage headroom_before_campaign -- bash -c "df -B1 --output=avail /tmp | tail -1; env -u TMPDIR $PYV -c 'from allo.pim.apu_v1_program import ledag_scratch_headroom_problem as f; p = f(); print(\"guard (TMPDIR unset):\", p); assert p is None'; $PYV -c 'from allo.pim.apu_v1_program import ledag_scratch_headroom_problem as f; p = f(); print(\"guard (TMPDIR on the evidence tree):\", p); assert p is None'" || { milestone_finish; exit 1; }
files=()
for s in $(seq 1 $SESSIONS); do
  if stage headroom_before_session_$s -- bash -c "df -B1 --output=avail /tmp | tail -1; $PYV -c 'from allo.pim.apu_v1_program import ledag_scratch_headroom_problem as f; p = f(); print(\"guard:\", p); assert p is None'"; then
    stage session_$s -- timeout 7200 $PYV -m benchmarks.apu_v1.paper_rows --measured $MEASURED --expert --historical-binary --cases $CASES --scratch $E/campaign/scratch/session_$s --out $E/campaign/apu_v1_campaign_s$s.json
    files+=($E/campaign/apu_v1_campaign_s$s.json)
  else
    echo "session $s NOT LAUNCHED: the headroom gate failed at $(date -Is) ($(df -B1 --output=avail /tmp | tail -1) bytes free in /tmp)" | tee -a $E/campaign/stages/SUMMARY.txt
  fi
  echo "/tmp after session $s: $(df -B1 --output=avail /tmp | tail -1) bytes free $(date -Is)" | tee -a $E/campaign/stages/SUMMARY.txt
done
if [ ${#files[@]} -gt 0 ]; then
  stage pool_campaign -- $PYV -I $HERE/g1_pool.py --out $E/campaign/apu_v1_campaign_pooled.json --quota $QUOTA --measured $MEASURED "${files[@]}"
else
  stage pool_campaign -- false
fi
milestone_finish
