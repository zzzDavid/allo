#!/bin/bash
# ONE paired Gemini-II session of benchmarks.apu_g2.paper_rows (20 rounds, all five rows) from
# the installed wheel under the device lock.   g2_session.sh <g2-dir-holding-venv-and-bench> <out.json>
set -u
G=${1:?g2 dir}; OUT=${2:?output json}
ART=${TENON_ARTIFACTS:-/work/shared/users/phd/nz264/spmw-for-pim/tenon-artifacts}
cd /tmp
export PYTHONSAFEPATH=1 PATH=/opt/gsi/g2/bin/amd64:$PATH PYTHONPATH=$G/bench TENON_ARTIFACTS=$ART
echo "host $(hostname) date $(date -Is) wheel $(sha256sum $G/wheel/allo-0.5-*.whl)"
df -h /tmp | tail -1
$G/venv/bin/python -c "import allo, benchmarks; print('allo from', allo.__file__); print('benchmarks from', benchmarks.__file__)" || exit 1
flock -w 7200 /tmp/tenon-apu-g2.lock timeout 5400 $G/venv/bin/python -m benchmarks.apu_g2.paper_rows --rounds 20 --out "$OUT"; rc=$?
echo "session exit $rc $(date -Is)"; df -h /tmp | tail -1
[ $rc -eq 0 ] && grep -q '"status": "pass"' "$OUT"
