# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical module-name shim for the FEATHER+ baseline.

The minisa package (`minisa/lowering.py`) and the test runners
(`tests/test_trace_input.py`, `tests/test_figure7_mapping.py`) import the
baseline API under the module name ``feather_minisa``. The actual baseline
source lives in ``feather_plus.py``. This shim re-exports the baseline so the
canonical name resolves, without duplicating or editing the baseline source.

The performance / deployable variants live in their own files
(``feather_plus_perf.py``, ``feather_plus_deploy.py``) and import the shared
helpers (``reverse_bits``, ``compute_birrd_params``, ``FeatherModule``) from
here so they stay in sync with the baseline.
"""

from feather_plus import *  # noqa: F401,F403
from feather_plus import (  # noqa: F401  explicit re-export for `from feather_minisa import X`
    PS,
    AR,
    AL,
    SW,
    reverse_bits,
    compute_birrd_params,
    get_feather_full_matrix_top,
    FeatherModule,
    build_feather_simulator,
    schedule_feather_hls,
    build_feather_hls,
    run_sequential_gemm_layers,
)
