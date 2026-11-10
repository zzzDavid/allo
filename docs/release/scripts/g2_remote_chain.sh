#!/bin/bash
# Gemini-II chain on zhang-x4 for a named candidate: wheel from the candidate's sdist, the card
# tests, then the ONE pre-registered validation session.   g2_remote_chain.sh <evidence-round-dir>
# Each step is a stage (stage_lib.sh): complete logs under <dir>/g2/stages, own exit statuses,
# nonzero exit if any required stage fails.
set -u
E=${1:?evidence round dir}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ALLO=${ALLO_CHECKOUT:-/work/shared/users/phd/nz264/spmw-for-pim/experiments/allo}
source "$HERE/stage_lib.sh"
mkdir -p $E/g2; milestone_begin $E/g2/stages
for i in $(seq 1 90); do [ -f $E/pkg/dist/allo-0.5.tar.gz ] && break; sleep 10; done
stage sdist_present -- test -f $E/pkg/dist/allo-0.5.tar.gz || { milestone_finish; exit 1; }
sleep 20
stage build_wheel -- "$HERE/g2_build_wheel.sh" $E || { milestone_finish; exit 1; }
stage copy_tests_and_benchmarks -- bash -c "rm -rf $E/g2/bench $E/g2/tests && mkdir -p $E/g2/bench $E/g2/tests && cp -r $ALLO/benchmarks $E/g2/bench/ && cp -r $ALLO/tests/pim $E/g2/tests/ && cp $ALLO/tests/conftest.py $E/g2/tests/"
stage card_tests -- "$HERE/g2_card_tests.sh" $E
stage candidate_session -- "$HERE/g2_session.sh" $E/g2 $E/g2_candidate_rows.json
milestone_finish
