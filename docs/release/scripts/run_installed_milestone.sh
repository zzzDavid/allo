#!/bin/bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Installed-artifact milestone, local part (docs/release/VALIDATION.md):
# every stage from the wheel in a clean venv, outside the checkout (no
# PYTHONPATH into it; tests and benchmarks copied out), the Samsung simulator
# built from the wheel's bundled recipe and found through TENON_HOME.
#
#   run_installed_milestone.sh <evidence-round-dir>
#
# Expects <dir>/pkg/{venv,wheel,dist,revision.txt} from build_wheel.sh.  Each
# stage's complete output is kept in <dir>/stages/<name>.log with its own exit
# status in <name>.status; the script exits nonzero if any required stage
# fails (stage_lib.sh).  Every pytest stage runs the whole selection; the
# summary lines are read from the logs, not used as the verdict.
set -u
E=${1:?evidence round dir}; P=$E/pkg
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
S=$HERE/probes
ALLO=${ALLO_CHECKOUT:-/work/shared/users/phd/nz264/spmw-for-pim/experiments/allo}
SPMW_PY=${SPMW_PY:-/work/zhang-capra/users/nz264/miniforge3/envs/spmw/bin/python}
export TMPDIR=$E/tmp PYTHONSAFEPATH=1
unset PYTHONPATH LLVM_BUILD_DIR PIMSIMULATOR_ROOT TENON_SIMULATORS_DIR TENON_PIMSIM_RECIPE
export TENON_HOME=$E/tenon-home
export AIM_SIMULATOR_ROOT=${AIM_SIMULATOR_ROOT:-/work/shared/users/phd/nz264/spmw-for-pim/experiments/simulators/aim_simulator}
export TENON_AIM_RAMULATOR2=${TENON_AIM_RAMULATOR2:-/work/shared/users/phd/nz264/aim-native/build/ramulator2}
export TENON_UPMEM_SDK_HOME=${TENON_UPMEM_SDK_HOME:-/work/shared/users/phd/nz264/spmw-for-pim/experiments/simulators/upmem-sdk/upmem-2023.1.0-Linux-x86_64}
export TENON_ARTIFACTS=${TENON_ARTIFACTS:-/work/shared/users/phd/nz264/spmw-for-pim/tenon-artifacts}
PY=$P/venv/bin/python
source "$HERE/stage_lib.sh"
mkdir -p $E/tmp $E/installed-run && cd $E/installed-run
milestone_begin $E/stages
echo "host $(hostname) date $(date -Is) commit $(cat $P/revision.txt)" | tee -a $E/stages/SUMMARY.txt
sha256sum $P/wheel/allo-0.5-*.whl $P/dist/allo-0.5.tar.gz | tee -a $E/stages/SUMMARY.txt

stage module_origin -- $PY -c "import allo, allo.spmw_aim, allo.spmw_apu_v1, allo.spmw_apu_g2, allo.spmw_upmem, allo.spmw_attacc, allo.spmw_samsung, sys; print(allo.__file__); assert not any(p.startswith('$ALLO') for p in sys.path)"
# The installed distribution's own metadata must identify the Tenon fork (ISSUES P-07): the
# summary, home page, maintainer and project URLs read back from the installed dist-info.
stage distribution_metadata -- $PY -c "from importlib.metadata import metadata; m = metadata('allo'); print('Summary:', m['Summary']); print('Home-page:', m['Home-page']); print('Maintainer:', m['Maintainer']); urls = m.get_all('Project-URL') or []; print('Project-URL:', urls); assert 'Tenon' in m['Summary'] and 'processing-in-memory' in m['Summary'], m['Summary']; assert m['Home-page'] == 'https://github.com/zzzDavid/allo/tree/tenon', m['Home-page']; assert 'tenon' in (m['Maintainer'] or ''), m['Maintainer']; assert any('cornell-zhang/allo' in u for u in urls) and any('STATUS.md' in u for u in urls), urls"
# The dependency versions the stages below run under are recorded from the interpreter, not
# assumed from the venv's name (round 120: the "release NumPy" venv had installed unpinned
# numpy 2.5.4 while the ledger said 2.3.5).  TENON_MILESTONE_NUMPY, when set, is asserted.
stage environment_versions -- $PY -c "import sys, numpy, allo; print('python', sys.version.split()[0]); print('numpy', numpy.__version__); print('allo', allo.__file__); expected = '${TENON_MILESTONE_NUMPY:-}'; assert not expected or numpy.__version__ == expected, (numpy.__version__, expected)"
stage samsung_verify -- $PY -m allo.pim.samsung_simulator verify
stage samsung_locate_before_build -- bash -c "$PY -m allo.pim.samsung_simulator locate; rc=\$?; [ \$rc -eq 1 ]"   # 1 expected: nothing built yet
stage samsung_build -- $PY -m allo.pim.samsung_simulator build --jobs 16
stage samsung_locate_after_build -- $PY -m allo.pim.samsung_simulator locate
stage samsung_driver_identity -- bash -c "objcopy --strip-all --remove-section=.note.gnu.build-id $TENON_HOME/simulators/PIMSimulator/pim_driver $E/tmp/stripped.bin && sha256sum $E/tmp/stripped.bin"
stage probe_bundled_samsung -- $PY $S/bundled_samsung_probe.py
for s in fresh_install_check fresh_sim_check aim_check upmem_check; do
  stage probe_$s -- timeout 1800 $PY $S/$s.py
done

# tests and benchmarks copied out of the checkout (mandatory); pytest.ini and a root
# conftest.py are optional (this tree has neither); a failed copy fails the milestone.
stage copy_test_tree -- copy_tree_with_optional $E/testtree $ALLO tests benchmarks -- pytest.ini conftest.py
T=$E/testtree; cd $T; export PYTHONPATH=$T
stage golden_rows -- timeout 3600 $PY -m pytest -q -p no:cacheprovider tests/pim/test_paper_golden.py -k "not apu_v1" -s
stage golden_rows_paper_full -- timeout 5400 $PY -m pytest -q -p no:cacheprovider tests/pim/test_paper_golden.py -k "not apu_v1" -m paper_full
stage samsung_conductor_and_gemv_all_targets -- timeout 3600 $PY -m pytest -q -p no:cacheprovider tests/pim/samsung_hbm_pim/test_samsung_conductor_runs.py tests/pim/test_gemv_all_targets.py -k "not apu_v1"
stage cent_suite -- timeout 3600 $PY -m pytest -q -p no:cacheprovider tests/pim/test_cent_aim_workloads.py
stage cent_suite_paper_full -- timeout 5400 $PY -m pytest -q -p no:cacheprovider tests/pim/test_cent_aim_workloads.py -m paper_full
stage shared_contract_set -- timeout 3600 $PY -m pytest -q -p no:cacheprovider tests/pim/test_attacc_virtual.py tests/pim/test_attacc_orientation.py tests/pim/test_aim_allocation.py tests/pim/test_contraction_access_forms.py tests/pim/test_aim_orientation.py tests/pim/test_axis_roles_coordinates.py
# AttAcc required figures (G10): the six kernel rows on the natively built Ramulator2-AttAcc
# with the DCC trace generator (ATTACC_* below), command-by-command trace checks, the six decode
# configurations composed from these sessions, and the comparison with the frozen figures
# (docs/release/scripts/attacc_compare_frozen.py).  Traces, configurations and raw outputs are
# retained under $E/attacc/*.artifacts.  tests/pim/test_attacc_virtual.py (in the shared-contract
# stage above) runs no simulator and is not figure evidence.
export ATTACC_DCC_ROOT=${ATTACC_DCC_ROOT:-/work/shared/users/sg2682/project/pim-compilation/repro/dcc-attacc/src/src}
export ATTACC_RAMULATOR=${ATTACC_RAMULATOR:-/work/shared/users/phd/nz264/attacc-native/build/ramulator2}
export ATTACC_RAMULATOR_LIBDIR=${ATTACC_RAMULATOR_LIBDIR:-/work/shared/users/phd/nz264/attacc-native/src}
mkdir -p $E/attacc; cd $E/attacc
for row in qkv_b1 qkv_b4 proj_b1 proj_b4; do
  stage attacc_$row -- timeout 3600 $PY -m benchmarks.attacc.scripts.run_ramulator --row $row --fixed-baseline --artifacts $E/attacc/attacc_$row.artifacts --out $E/attacc/attacc_$row.json
done
for row in score_b1 score_b4; do
  stage attacc_$row -- timeout 5400 $PY -m benchmarks.attacc.scripts.run_ramulator --row $row --artifacts $E/attacc/attacc_$row.artifacts --out $E/attacc/attacc_$row.json
done
stage attacc_compose_e2e -- timeout 1800 $PY -m benchmarks.attacc.scripts.compose_e2e --sessions $E/attacc/attacc_qkv_b1.json $E/attacc/attacc_qkv_b4.json $E/attacc/attacc_proj_b1.json $E/attacc/attacc_proj_b4.json $E/attacc/attacc_score_b1.json $E/attacc/attacc_score_b4.json --out $E/attacc/attacc_e2e.json
stage attacc_compare_frozen -- $PY $HERE/attacc_compare_frozen.py --sessions-dir $E/attacc --inventory $ALLO/docs/release/paper_reproduction/inventory.csv --manifest ${TENON_ARTIFACTS}/attacc-llm-e2e/tenon/tenon_manifest.csv --out $E/attacc/attacc_compare_frozen.json
cd $T
# The APU v1 grouped-enumerator regressions (V1-20) and the grouped / APU v1 selection of the
# shared-semantics suite: the functional-reference stages below do not exercise that enumerator.
# The APU v1 build harness's iterator boundary (C-13, rounds 126 to 128) and the build
# configuration tests, so a candidate's evidence carries those regressions.
stage apu_v1_build_iterators_and_config -- timeout 1800 $PY -m pytest -q -p no:cacheprovider tests/perf/test_apu_v1_build_iterators.py tests/perf/test_apu_v1_build_config.py
stage apu_v1_grouped_mapping -- timeout 1800 $PY -m pytest -q -p no:cacheprovider tests/pim/test_apu_v1_grouped_mapping.py
stage apu_v1_shared_semantics_selection -- timeout 1800 $PY -m pytest -q -p no:cacheprovider tests/pim/test_shared_semantics.py -k "apu_v1 or grouped"
# The Samsung cost program's staging accounting (S-12, rounds 132 to 134): the unit tests and the
# production-path regression against the bundled conductor built above (the trace test must run,
# not skip), and the APU v1 measurement driver's scratch precondition (X-24, rounds 130 to 131)
# with its mocked headroom regressions; both suites are required stages of the milestone.
stage samsung_cost_staging_suites -- timeout 2400 $PY -m pytest -q -p no:cacheprovider tests/perf/test_samsung_performance.py tests/pim/samsung_hbm_pim/test_vector_staging.py -rA
stage samsung_staging_regressions_named -- bash -c "for t in test_vector_staging_shares_partition_the_conductors_vector_staging test_emitted_vector_staging_equals_the_conductors_work test_operand_order_does_not_change_the_staging_counts test_conductor_trace_matches_the_transcription; do grep -c \"PASSED.*\$t\" $E/stages/samsung_cost_staging_suites.log || exit 1; done"
stage measurement_drivers_suite -- timeout 1800 $PY -m pytest -q -p no:cacheprovider tests/perf/test_measurement_drivers.py -rA
stage headroom_regressions_named -- bash -c "grep -c 'PASSED.*test_ledag_drain_refuses_without_scratch_headroom' $E/stages/measurement_drivers_suite.log && grep -c 'PASSED.*separately_provisioned_extraction_directory' $E/stages/measurement_drivers_suite.log"
stage apu_v1_functional_reference -- timeout 1800 $PY -m pytest -q -p no:cacheprovider tests/perf/test_apu_v1_functional_reference.py -W error::DeprecationWarning
stage numpy24_venv -- bash -c "env -u PYTHONPATH $SPMW_PY -m venv --system-site-packages $E/venv-np24 && $E/venv-np24/bin/python -m pip install -q 'numpy>=2.4' && $E/venv-np24/bin/python -m pip install -q --no-deps --force-reinstall $P/wheel/allo-0.5-*.whl"
stage numpy24_versions_asserted -- $E/venv-np24/bin/python -c "import numpy, allo; print('numpy', numpy.__version__, 'allo', allo.__file__); major, minor = (int(x) for x in numpy.__version__.split('.')[:2]); assert (major, minor) >= (2, 4), numpy.__version__; assert '/venv-np24/' in allo.__file__, allo.__file__"
RELEASE_NUMPY=${TENON_RELEASE_NUMPY:-2.3.5}
stage release_numpy_venv -- bash -c "rm -rf $E/venv-release-numpy && env -u PYTHONPATH $SPMW_PY -m venv $E/venv-release-numpy && $E/venv-release-numpy/bin/python -m pip install -q 'numpy==$RELEASE_NUMPY' PyYAML sympy ml_dtypes rich tabulate xmltodict pandas pyparsing graphviz psutil pytest && $E/venv-release-numpy/bin/python -m pip install -q --no-deps --force-reinstall $P/wheel/allo-0.5-*.whl"
stage release_numpy_versions_asserted -- $E/venv-release-numpy/bin/python -c "import numpy, allo; print('numpy', numpy.__version__, 'allo', allo.__file__); assert numpy.__version__ == '$RELEASE_NUMPY', numpy.__version__; assert '/venv-release-numpy/' in allo.__file__, allo.__file__"
stage apu_v1_functional_reference_release_numpy -- timeout 1800 $E/venv-release-numpy/bin/python -m pytest -q -p no:cacheprovider tests/perf/test_apu_v1_functional_reference.py tests/pim/test_apu_v1_grouped_mapping.py -W error::DeprecationWarning
stage apu_v1_functional_reference_numpy24 -- timeout 1800 $E/venv-np24/bin/python -m pytest -q -p no:cacheprovider tests/perf/test_apu_v1_functional_reference.py "tests/perf/test_apu_v1_recognizers.py::test_temporal_plan_accumulates_nonzero_initial_output" "tests/pim/test_shared_semantics.py::test_local_accumulator_gemm_executes_on_apu_v1_functional_backend" -W error::DeprecationWarning
milestone_finish
