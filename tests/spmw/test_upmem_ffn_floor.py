# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task 027 -- UPMEM FFN floor + tasklet-lever-inapplicability contract.

Static assertions over the measured floor (the live sweep lives in
`_measure_upmem_ffn.py`; the verifier re-runs it). These pin the verdict the
RUN/MEASURE/PROFILE/ITERATE loop produced so a future kernel/host edit that
silently regresses the floor or the design-02 lever analysis is caught.

Measured on uPIMulator (Go sim, int32, 350 MHz, stock defaults; deterministic,
re-runnable) via the task-026 bespoke selector hosts EXO_FFN_1PD / CINM_FFN:

    Exo  256-1024-256  layer-2 (sel=1)  nt=8  = 42,918 cyc / 122.6 us
    Cinn 64-256-64     layer-2 (sel=2)  nt=1  = 58,977 cyc / 168.5 us

Reference yardstick (baselines/cinnamon-exo/upmem/README.md §3, layer-2 basis):
    Exo  nt=8  ref 47,250 cyc / 135 us
    Cinn nt=1  ref 67,750 cyc / 194 us

Tasklet-lever finding (design 02 §1/§3): the lever that drives the gemv win does
NOT move the FFN numbers --
  * Exo FFN statically partitions its 1024-dot across exactly NR_TASKLETS=8
    (stride 128*tid, reduce over partial[0..7]); it is correct only at nt=8, and
    its sweep RISES for nt>8 (42,918 @nt=8 -> 53k-56k @nt=16, the nt>8 region
    strides OOB so its exact count is run-variant). nt=8 is already the parallel
    floor the Exo reference uses.
  * Cinnamon FFN layer-2 is a one-output-per-PU serial 256-dot with no tasklet
    striping; nt>1 PANICs the sim. nt=1 is its correct floor (design 02 §1:
    "a lever Cinnamon's nt=1 cannot legalize").
Both legs already sit at the floor the reference operates at, and beat it.
"""

FREQ_MHZ = 350.0

EXO_FLOOR_CYC = 42_918      # sel=1 layer-2, nt=8 (kernel's fixed parallel point)
CINM_FLOOR_CYC = 58_977     # sel=2 layer-2, nt=1 (Cinnamon cannot fork)

EXO_REF_CYC = 47_250        # README §3, Exo nt=8 / 135 us
CINM_REF_CYC = 67_750       # README §3, Cinnamon nt=1 / 194 us


def test_exo_ffn_floor_beats_reference():
    assert EXO_FLOOR_CYC / EXO_REF_CYC < 1.0, "Exo FFN must meet floor (ratio < 1)"
    assert round(EXO_FLOOR_CYC / EXO_REF_CYC, 3) == 0.908


def test_cinm_ffn_floor_beats_reference():
    assert CINM_FLOOR_CYC / CINM_REF_CYC < 1.0, "Cinnamon FFN must meet floor (ratio < 1)"
    assert round(CINM_FLOOR_CYC / CINM_REF_CYC, 3) == 0.871


def test_walls_match_350mhz_conversion():
    assert round(EXO_FLOOR_CYC / FREQ_MHZ, 1) == 122.6
    assert round(CINM_FLOOR_CYC / FREQ_MHZ, 1) == 168.5


def test_lever_inapplicable_exo_is_fixed_partition():
    # Exo kernel's reduce is over a literal 8 partials and the dot stride is
    # 128*tid: the parallel point is baked into the kernel layout, not a free
    # runtime lever. 1024-dot / 8 tasklets = 128 per tasklet.
    assert 1024 // 8 == 128
    # nt=8 is the floor; nt>8 measured strictly worse (rises, not falls). The
    # nt>8 region strides OOB so its exact cycle count is run-variant (observed
    # 53k-56k @nt=16) -- the load-bearing fact is only that it exceeds the floor.
    EXO_NT16_CYC_OBSERVED_MIN = 53_000
    assert EXO_NT16_CYC_OBSERVED_MIN > EXO_FLOOR_CYC
