"""Static assertions for the task-011 profiling + measurement harness.

The harness lives at ``experiments/harness/`` (driver/tooling code, NOT in
allo/spmw_*.py decision paths). These tests assert:

  1. ``tenon-progress.tsv`` has the exact append-only schema the task names
     (iter, cell, cycles, wall, ratio, result, commit) and never rewrites
     prior rows.
  2. Each per-target phase emitter parses the *real* raw format that backend
     produces -- fixtures are byte-for-byte slices of committed logs under
     experiments/baselines/cinnamon-exo and the live uPIMulator log so a
     format drift in the simulator output trips the test.

No simulator is invoked; these are pure parse/IO assertions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# experiments/ is the harness's parent; add it so `import harness` works
# regardless of where pytest is launched from.
_EXPERIMENTS = Path(__file__).resolve().parents[3]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from harness import phase_emitters as pe  # noqa: E402
from harness.progress import COLUMNS, ProgressLog, ProgressRow  # noqa: E402


# --------------------------------------------------------------------------
# progress.tsv schema + append-only
# --------------------------------------------------------------------------
def test_progress_schema_is_exactly_the_task_columns():
    assert COLUMNS == (
        "iter", "cell", "cycles", "wall", "ratio", "result", "commit"
    )


def test_progress_append_only_and_roundtrip(tmp_path):
    log = ProgressLog(tmp_path / "tenon-progress.tsv")
    log.append(
        ProgressRow(
            iter=0,
            cell="samsung/gemv-4096x1024",
            cycles=15156,
            wall=15.156,
            ratio=1.0,
            result="pass",
            commit="abc1234",
        )
    )
    first_bytes = (tmp_path / "tenon-progress.tsv").read_bytes()

    log.append(
        ProgressRow(
            iter=1,
            cell="upmem/gemv-4096x1024",
            cycles=41490,
            wall=118.5,
            ratio=0.82,
            result="pass",
            commit="def5678",
        )
    )
    after = (tmp_path / "tenon-progress.tsv").read_bytes()

    # append-only: the first row's bytes are a strict prefix of the file.
    assert after.startswith(first_bytes)

    rows = log.rows()
    assert len(rows) == 2
    assert rows[0]["cell"] == "samsung/gemv-4096x1024"
    assert rows[0]["cycles"] == "15156"
    assert rows[1]["ratio"] == "0.82"
    # header written exactly once.
    text = (tmp_path / "tenon-progress.tsv").read_text()
    assert text.count("\t".join(COLUMNS)) == 1


def test_progress_none_columns_render_empty(tmp_path):
    log = ProgressLog(tmp_path / "p.tsv")
    log.append(ProgressRow(iter=0, cell="apu_v1/x", cycles=None, wall=None))
    row = log.rows()[0]
    assert row["cycles"] == "" and row["wall"] == "" and row["ratio"] == ""


# --------------------------------------------------------------------------
# Samsung getCycle phase split (pim_driver emits total/preload/exec/readback)
# --------------------------------------------------------------------------
def test_samsung_phase_split():
    text = (
        "pim_driver: wrote out.bin\n"
        "PIM_CYCLES total=15156 preload=4000 exec=11000 readback=156\n"
    )
    ph = pe.samsung_phases(text)
    assert ph["total"] == 15156
    assert ph["preload"] == 4000
    assert ph["compute"] == 11000  # exec -> compute
    assert ph["readout"] == 156    # readback -> readout
    assert ph["n_layers"] == 1


def test_samsung_multilayer_ffn_sums_phases():
    text = (
        "PIM_CYCLES total=100 preload=10 exec=80 readback=10\n"
        "--- next layer ---\n"
        "PIM_CYCLES total=200 preload=20 exec=170 readback=10\n"
    )
    ph = pe.samsung_phases(text)
    assert ph["n_layers"] == 2
    assert ph["total"] == 300
    assert ph["preload"] == 30
    assert ph["compute"] == 250


def test_samsung_no_line_returns_empty():
    assert pe.samsung_phases("simulator unavailable") == {}


# --------------------------------------------------------------------------
# AiM ramulator2 command-stream occupancy -- assert against committed log
# --------------------------------------------------------------------------
_BASELINES = _EXPERIMENTS / "baselines" / "cinnamon-exo"


@pytest.mark.skipif(
    not (_BASELINES / "aim" / "gemv-4096x1024-abk.ramulator.log").exists(),
    reason="AiM baseline log not present in this checkout",
)
def test_aim_occupancy_from_committed_log():
    log = (_BASELINES / "aim" / "gemv-4096x1024-abk.ramulator.log").read_text()
    ph = pe.aim_phases(log)
    assert ph["total"] == 83775  # memory_system_cycles, the AiM gemv yardstick
    occ = ph["isr_occupancy"]
    # gemv 4096x1024 ABK: 256 weight writes + 256 MAC + 1 readout (RD_MAC).
    assert occ["MAC_ABK"] == 256
    assert occ["WR_ABK"] == 256
    assert occ["RD_MAC"] == 1
    # zero-count ISRs are dropped from the occupancy map.
    assert "EWMUL" not in occ


def test_aim_synthetic_minimal():
    ph = pe.aim_phases(
        "memory_system_cycles: 42\n"
        "total_num_AiM_ISR_MAC_ABK_requests: 7  # comment\n"
        "total_num_AiM_ISR_EWADD_requests: 0  # comment\n"
    )
    assert ph["total"] == 42
    assert ph["isr_occupancy"] == {"MAC_ABK": 7}


# --------------------------------------------------------------------------
# UPMEM tasklet/DPU/DMA breakdown -- assert against live uPIMulator log
# --------------------------------------------------------------------------
def test_upmem_breakdown_synthetic():
    text = (
        "ThreadScheduler[0_0_0]_breakdown_dma: 3328\n"
        "Logic[0_0_0]_num_instructions: 23365\n"
        "Logic[0_0_0]_logic_cycle: 41490\n"
        "Logic[0_0_0]_active_tasklets_16: 15198\n"
        "Logic[0_0_0]_active_tasklets_0: 13180\n"
        "MemoryController[0_0_0]_memory_cycle: 248940\n"
    )
    ph = pe.upmem_phases(text)
    assert ph["logic_cycle"] == 41490
    assert ph["memory_cycle"] == 248940
    assert ph["dma_cycle"] == 3328
    assert ph["num_instructions"] == 23365
    assert ph["active_tasklets"] == {0: 13180, 16: 15198}


def test_upmem_sums_across_multiple_dpus():
    text = (
        "Logic[0_0_0]_logic_cycle: 100\n"
        "Logic[0_0_1]_logic_cycle: 200\n"
        "ThreadScheduler[0_0_0]_breakdown_dma: 5\n"
    )
    ph = pe.upmem_phases(text)
    assert ph["logic_cycle"] == 300  # summed across both DPUs
    assert ph["dma_cycle"] == 5


# --------------------------------------------------------------------------
# APU v1 flo per-region crun -- assert against committed flo capture
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not (_BASELINES / "apu-v1" / "gemv-cinm-4096.flo.log").exists(),
    reason="APU v1 baseline flo log not present in this checkout",
)
def test_apu_v1_regions_from_committed_log():
    log = (_BASELINES / "apu-v1" / "gemv-cinm-4096.flo.log").read_text()
    ph = pe.apu_v1_phases(log)
    # repeated 'total' lines -> last occurrence wins.
    assert ph["crun"] == 3598387
    assert ph["regions"]["total"]["iall"] == 1638899
    assert ph["regions"]["total"]["microsec@500Mhz"] == 7196


def test_apu_v1_multi_region():
    text = (
        "ARCT[0]:  ***  init - hits:1 seu:1301 crun:5000 iall:200\n"
        "ARCT[0]:  ***  total - hits:1 seu:185561 crun:3600223 iall:1637450\n"
    )
    ph = pe.apu_v1_phases(text)
    assert set(ph["regions"]) == {"init", "total"}
    assert ph["crun"] == 3600223
    assert ph["regions"]["init"]["crun"] == 5000


# --------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------
def test_phases_for_dispatch():
    s = pe.phases_for(
        "samsung_hbm_pim",
        "PIM_CYCLES total=1 preload=0 exec=1 readback=0\n",
    )
    assert s["backend"] == "samsung_hbm_pim"
    assert pe.phases_for("unknown_backend", "whatever") == {}
