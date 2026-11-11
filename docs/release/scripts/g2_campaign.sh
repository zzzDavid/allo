#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# The pre-registered Gemini-II confirmation campaign (ISSUES X-34 / X-37, PAPER_REPRODUCTION.md):
# at the final candidate, three further paired sessions of 20 rounds of all five rows from the
# SAME zhang-x4 wheel as that candidate's validation session, under the device lock, pooled per row
# with that session by g2_pool.py.  Run once; the status is fixed by the pooled result.
#   g2_campaign.sh <evidence-round-dir> <g2-dir-of-the-candidate-wheel> [sessions=3]
set -u
E=${1:?evidence round dir}; G=${2:?g2 dir of the candidate wheel}; N=${3:-3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/stage_lib.sh"
mkdir -p $E; milestone_begin $E/campaign-stages
echo "campaign of $N sessions from wheel $(sha256sum $G/wheel/allo-0.5-*.whl)" | tee -a $E/campaign-stages/SUMMARY.txt
for s in $(seq 1 $N); do
  stage session_$s -- "$HERE/g2_session.sh" $G $E/g2_campaign_s$s.json
done
milestone_finish
