#!/bin/bash
# Build the candidate's sdist into a wheel on zhang-x4 and install it into a clean venv
# (the round 15 recipe; no card run here).   g2_build_wheel.sh <evidence-round-dir>
set -u
E=${1:?evidence round dir}/g2
SDIST=${1}/pkg/dist/allo-0.5.tar.gz
PY=${G2_PY:-/work/shared/users/phd/nz264/anaconda3/envs/allo-dev/bin/python}
ENV=${G2_ENV:-/work/shared/users/phd/nz264/anaconda3/envs/allo-dev}
LLVM=${LLVM_BUILD_DIR:-/work/shared/common/llvm-project-main/build-rhel8}
ZSTD=${G2_ZSTD:-/work/shared/users/phd/nz264/tenon-release-evidence/r9/zstd-install}
echo "host $(hostname) date $(date -Is) sdist $(sha256sum $SDIST)"
rm -rf $E/src $E/wheel $E/venv; mkdir -p $E/src $E/wheel
tar xzf $SDIST -C $E/src || exit 1
env -u PYTHONPATH $PY -m venv $E/venv || exit 1
env -u PYTHONPATH $E/venv/bin/python -m pip install -q ninja cmake > $E/venv-tools.log 2>&1 || { tail -5 $E/venv-tools.log; exit 1; }
export PATH=$E/venv/bin:$PATH
cd $E && CMAKE_PREFIX_PATH="$ZSTD:$ENV" LLVM_BUILD_DIR=$LLVM CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib -Wl,-rpath-link,$ENV/lib -L$ZSTD/lib -Wl,-rpath,$ZSTD/lib" NUM_THREADS=48 \
  env -u PYTHONPATH $PY -m pip wheel --no-deps --no-build-isolation -w $E/wheel $E/src/allo-0.5 > $E/wheel-build.log 2>&1 || { tail -30 $E/wheel-build.log; exit 1; }
sha256sum $E/wheel/allo-0.5-*.whl
env -u PYTHONPATH $E/venv/bin/python -m pip install -q numpy PyYAML sympy ml_dtypes rich tabulate xmltodict pandas pyparsing graphviz psutil pytest > $E/venv-install.log 2>&1 || { tail -5 $E/venv-install.log; exit 1; }
env -u PYTHONPATH $E/venv/bin/python -m pip install -q --no-deps --force-reinstall $E/wheel/allo-0.5-*.whl >> $E/venv-install.log 2>&1 || exit 1
rm -rf $E/src   # regenerable from the sdist; the shared quota is tight
echo WHEEL_OK > $E/status; date -Is
