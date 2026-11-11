#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Gemini-II card tests from the installed wheel (waits for the device lock).  g2_card_tests.sh <evidence-round-dir>
set -u
E=${1:?evidence round dir}/g2
ART=${TENON_ARTIFACTS:-/work/shared/users/phd/nz264/spmw-for-pim/tenon-artifacts}
cd /tmp
export PYTHONSAFEPATH=1 PATH=/opt/gsi/g2/bin/amd64:$PATH PYTHONPATH=$E/bench TENON_ARTIFACTS=$ART
echo "=== card tests (installed wheel) $(date -Is) wheel $(sha256sum $E/wheel/allo-0.5-*.whl)"
$E/venv/bin/python -c "import allo; print('allo from', allo.__file__)" || exit 1
flock -w 7200 /tmp/tenon-apu-g2.lock timeout 3000 $E/venv/bin/python -m pytest $E/tests/pim/test_apu_g2_hardware.py -s -rA -p no:cacheprovider --rootdir=$E/tests --basetemp=$E/pytest-tmp
