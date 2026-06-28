# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-2 measurement validation for the Mortise capacity sweep (task 004).

Static guards on the `experiments/scripts/R26_mortise_capacity_sweep.py` harness
(design 06; report-26 §7b). All run on the VIRTUAL backend, sim-free.

Proven here (the Phase-2 receipts):

1. **Sim-free, PROVEN not asserted.** The virtual runner is dispatch-identical
   to `_run_virtual` (imports no simulator path), and the harness's subprocess
   tripwire fires on a real subprocess (negative control) -- so a sweep that
   completes inside it provably invoked no PIMSimulator / uPIMulator / Docker /
   ramulator.
2. **Crossover / ordering.** B*(phi=1)=2; the ceiling collapses monotonically
   as capacity falls (full residency >> half residency).
3. **Ablation.** `cost_flavor="unlimited"` (phi=1) makes the resident-arm
   whole-program C-independent -- the capacity spread vanishes.
4. **Sensitivity band.** Over a representative corner set of the §4
   uncertain-constant band, B*(phi=1) is invariant and full>half holds at every
   corner (no ordering flip in the plausible band).
5. **Fair cross-target.** Identical workload, argmin-selected layouts
   (>=2 candidates/target), faithful methodology: Mortise(C=T_w) == Samsung
   (anchor), Mortise(C=T_w/2) degrades.

The harness is imported as a module; these tests call its functions directly so
the verifier re-runs them under pytest (a subset of corners for speed -- the full
81-corner band is the harness's `main()` artifact).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# The harness needs the regalloc kill-switch (Mortise has no capacity table;
# the residency flag rides `extra`, so argmin is unaffected). Set before import.
os.environ.setdefault("SPMW_DISABLE_REGALLOC", "1")

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import R26_mortise_capacity_sweep as R  # noqa: E402

from _mortise_target import build_mortise_target, _T_W_FULL  # noqa: E402


# --------------------------------------------------------------------- #
# 1. Sim-free, proven
# --------------------------------------------------------------------- #


def test_virtual_runner_is_dispatch_identical_to_sim_free_runner():
    """The 'virtual' backend dispatches to `_run_virtual`, which delegates to
    `spmw_cost_model.evaluate` -- no simulator path imported."""
    R.assert_sim_free_runner()  # raises on mismatch


def test_subprocess_tripwire_has_teeth():
    """Negative control: the tripwire MUST abort on a real subprocess call, so
    a sweep that completes inside it proves no sim ran."""
    import subprocess

    with R._SubprocessTripwire():
        with pytest.raises(AssertionError, match="SIM TRIPWIRE"):
            subprocess.run(["echo", "should-not-run"])
    # Restored after the context exits.
    r = subprocess.run(["echo", "ok"], capture_output=True, text=True)
    assert r.stdout.strip() == "ok"


def test_full_sweep_runs_under_tripwire_without_firing():
    """A small sweep under the tripwire completes -> no subprocess fired ->
    no simulator invoked (the proven no-sim guarantee)."""
    with R._SubprocessTripwire():
        table = R.sweep("faithful", batches=(1, 2), phis=(1.0, 0.5))
    assert table[1.0][2]["speedup"] > 1.0   # resident wins at B=2, phi=1


# --------------------------------------------------------------------- #
# 2. Crossover / ordering
# --------------------------------------------------------------------- #


def test_crossover_bstar_is_2_at_full_capacity():
    bstar = R.crossover_bstar(build_mortise_target(), "faithful")
    assert bstar == 2, bstar


def test_ceiling_collapses_with_capacity():
    full = R.asymptotic_speedup(build_mortise_target(), "faithful")
    half = R.asymptotic_speedup(
        build_mortise_target(resident_cap_elems=_T_W_FULL // 2), "faithful")
    assert full > half > 1.0, (full, half)
    # The collapse is large (the headline finding), not marginal.
    assert full / half > 3.0, (full, half)


# --------------------------------------------------------------------- #
# 3. Ablation
# --------------------------------------------------------------------- #


def test_ablation_collapses_all_capacity_spread():
    unlimited = R.sweep("unlimited", batches=(8,), phis=(1.0, 0.5, 0.1))
    resident_vals = {unlimited[phi][8]["resident_cyc"] for phi in unlimited}
    assert len(resident_vals) == 1, resident_vals      # feature off -> flat
    faithful = R.sweep("faithful", batches=(8,), phis=(1.0, 0.5, 0.1))
    faithful_vals = {faithful[phi][8]["resident_cyc"] for phi in faithful}
    assert len(faithful_vals) == 3, faithful_vals      # feature on -> spread


# --------------------------------------------------------------------- #
# 4. Sensitivity band (representative corners, for speed)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pv,es,rs,crf",
    [
        (0.5, 0.5, 0.5, 1),   # all-low corner
        (2.0, 2.0, 2.0, 8),   # all-high corner
        (0.5, 2.0, 0.5, 8),   # mixed
        (2.0, 0.5, 2.0, 1),   # mixed
    ],
)
def test_sensitivity_corner_preserves_ordering_and_bstar(pv, es, rs, crf):
    name = f"band_test_{pv}_{es}_{rs}_{crf}"
    R._register_band_flavor(name, scatter=pv, mac=es, gather_rd=rs, crf=crf)
    full = build_mortise_target()
    half = build_mortise_target(resident_cap_elems=_T_W_FULL // 2)
    # B* invariant at full capacity.
    assert R.crossover_bstar(full, name) == 2
    # Ordering (full > half) holds at this corner (no flip in the band).
    assert R.asymptotic_speedup(full, name) > R.asymptotic_speedup(half, name)


# --------------------------------------------------------------------- #
# 5. Fair cross-target
# --------------------------------------------------------------------- #


def test_fair_cross_target_identical_workload_argmin_layouts():
    rows, batches = R.fair_cross_target(batches=(2, 8, 32))
    # Every target has >=2 argmin candidates (fairness: no hand-picked layout).
    for label, per_B in rows.items():
        assert per_B[2]["n_candidates"] >= 2, (label, per_B[2])
    # Anchor identity: Mortise(C=T_w) == Samsung for the IDENTICAL workload.
    for B in batches:
        assert (rows["mortise(C=T_w)"][B]["cycles"]
                == rows["samsung_hbm_pim"][B]["cycles"]), B
    # The lever is the only difference: halving C degrades the same workload.
    assert (rows["mortise(C=T_w/2)"][32]["cycles"]
            > rows["mortise(C=T_w)"][32]["cycles"])


if __name__ == "__main__":
    failures = 0
    import inspect

    for nm, fn in list(globals().items()):
        if nm.startswith("test_") and callable(fn):
            sig = inspect.signature(fn)
            try:
                if sig.parameters:
                    # the only parametrized test: run one representative corner
                    fn(0.5, 2.0, 0.5, 8)
                else:
                    fn()
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {nm}: {e}")
    print("STATIC PASSED" if not failures else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)
