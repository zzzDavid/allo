# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shell-path checks of the maintained Gemini-I board-validation driver
`docs/release/scripts/g1_board.sh` (round 147, ISSUES X-47): a failed
prerequisite (package origin, assets, scratch headroom) stops the run before
any driver invocation; the dry run names the three stages' commands and
launches nothing; a prior attempt's directory is never overwritten; a stage
whose own headroom gate fails is a recorded failed stage and the driver is
not invoked for it.  The sandbox stands in for the installed artifact (a
`pkg/venv` whose interpreter sees a stub `allo` package and a stub
`benchmarks` tree whose driver records its invocations)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "docs" / "release" / "scripts" / "g1_board.sh"


def _sandbox(root: Path, *, headroom="None", driver_exit=0) -> dict:
    pkg = root / "pkg"
    (pkg / "venv" / "bin").mkdir(parents=True)
    lib = pkg / "venv" / "lib"
    (lib / "allo" / "pim").mkdir(parents=True)
    (lib / "allo" / "__init__.py").write_text("")
    (lib / "allo" / "pim" / "__init__.py").write_text("")
    # The stub guard reports a problem once the stub driver has left its mark
    # when ``headroom`` says so (scratch lost between stages).
    (lib / "allo" / "pim" / "apu_v1_program.py").write_text(
        "import os\n"
        "def ledag_scratch_headroom_problem():\n"
        f"    return {headroom}\n"
    )
    python = pkg / "venv" / "bin" / "python"
    python.write_text(
        "#!/bin/sh\n"
        f'export PYTHONPATH="{lib}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    python.chmod(0o755)
    (pkg / "revision.txt").write_text("f57c840\n")
    (pkg / "wheel").mkdir()
    (pkg / "wheel" / "allo-0.5-sandbox.whl").write_bytes(b"not a wheel")
    tree = root / "testtree"
    (tree / "benchmarks" / "apu_v1").mkdir(parents=True)
    (tree / "benchmarks" / "__init__.py").write_text("")
    (tree / "benchmarks" / "apu_v1" / "__init__.py").write_text("")
    calls = root / "driver-calls.txt"
    (tree / "benchmarks" / "apu_v1" / "paper_rows.py").write_text(
        "import sys\n"
        f"open({str(calls)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        f"sys.exit({driver_exit})\n"
    )
    assets = root / "assets"
    for case in ("matrix_multiply", "histogram"):
        (assets / "apuv1" / "tenon" / case).mkdir(parents=True)
    return {"pkg": pkg, "tree": tree, "assets": assets, "calls": calls}


def _run(sb, evidence: Path, *, dry=True, env_extra=None):
    evidence.mkdir(parents=True, exist_ok=True)
    if not (evidence / "pkg").exists():
        (evidence / "pkg").symlink_to(sb["pkg"])
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    tmp = evidence.parent / "tmp"
    tmp.mkdir(exist_ok=True)
    env.update(
        {
            "G1_TESTTREE": str(sb["tree"]),
            "TENON_ARTIFACTS": str(sb["assets"]),
            "TMPDIR": str(tmp),
            # the sequence budget is checked on the sandbox's filesystem, not
            # the host's /tmp (which may be full while the tests run)
            "G1_TMP_PATH": str(tmp),
        }
    )
    if dry:
        env["G1_DRY_RUN"] = "1"
    env.update(env_extra or {})
    result = subprocess.run(
        ["bash", str(SCRIPT), str(evidence)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    return result.returncode, result.stdout


def _driver_calls(sb):
    return sb["calls"].read_text().splitlines() if sb["calls"].exists() else []


def test_prerequisite_failures_invoke_no_driver(tmp_path):
    sb = _sandbox(tmp_path)
    # package origin: the drivers are not where G1_TESTTREE says
    code, out = _run(sb, tmp_path / "e1", env_extra={"G1_TESTTREE": str(tmp_path / "elsewhere")})
    assert code == 1 and "prerequisite module_origin failed; nothing launched" in out
    assert "DRY RUN" not in out and _driver_calls(sb) == []
    # assets: a vector case is missing
    code, out = _run(sb, tmp_path / "e2", env_extra={"TENON_ARTIFACTS": str(tmp_path / "no-assets")})
    assert code == 1 and "prerequisite assets_present failed; nothing launched" in out
    assert _driver_calls(sb) == []
    # scratch headroom: the installed guard reports a problem
    sb2 = _sandbox(tmp_path / "two", headroom="'/tmp has 0.4 MiB free'")
    code, out = _run(sb2, tmp_path / "e3")
    assert code == 1 and "prerequisite headroom_before_board failed; nothing launched" in out
    assert "guard: /tmp has 0.4 MiB free" in out and _driver_calls(sb2) == []


def test_dry_run_names_the_three_stages_and_refuses_to_overwrite(tmp_path):
    sb = _sandbox(tmp_path)
    code, out = _run(sb, tmp_path / "e4")
    assert code == 0, out
    lines = [l for l in out.splitlines() if "DRY RUN: would launch" in l]
    assert [l.split("would launch ")[1].split(" with:")[0] for l in lines] == [
        "vector_rows",
        "forced_acc4",
        "perturbed_dma",
    ]
    assert "--measured 10 --expert --historical-binary --cases matrix_multiply histogram" in lines[0]
    assert "--measured 5 --cases matrix_multiply --layout temporal_dma_coalescing_broadcast_friendly_acc4 --label forced_acc4" in lines[1]
    assert "--measured 5 --cases matrix_multiply --perturb MICRO_DMA_L4_L1_32K=0.01 --label perturbed_dma" in lines[2]
    assert "MILESTONE PASSED" in out and _driver_calls(sb) == []
    assert (tmp_path / "e4" / "board-dry-run" / "stages" / "SUMMARY.txt").is_file()
    code, out = _run(sb, tmp_path / "e4")
    assert code == 1 and "refusing to run" in out


def test_each_stage_runs_behind_its_own_gate_and_a_lost_gate_is_a_failed_stage(tmp_path):
    # Scratch is fine until the first stage's driver has run (its call log
    # exists), then gone: the later stages must not invoke the driver.
    sb = _sandbox(tmp_path)
    calls = sb["calls"]
    (sb["pkg"] / "venv" / "lib" / "allo" / "pim" / "apu_v1_program.py").write_text(
        "import os\n"
        "def ledag_scratch_headroom_problem():\n"
        f"    return '/tmp has 0.4 MiB free' if os.path.exists({str(calls)!r}) else None\n"
    )
    code, out = _run(sb, tmp_path / "e5", dry=False)
    assert code == 1, out
    invoked = _driver_calls(sb)
    assert len(invoked) == 1 and invoked[0].startswith(
        "--measured 10 --expert --historical-binary --cases matrix_multiply histogram"
    )
    assert "--scratch" in invoked[0] and "/board/scratch/vector_rows" in invoked[0]
    assert "stage vector_rows PASS" in out
    assert "stage headroom_before_forced_acc4 FAIL" in out and "stage forced_acc4 FAIL" in out
    assert "stage perturbed_dma FAIL" in out and "MILESTONE FAILED" in out
    assert out.count("NOT LAUNCHED: the headroom gate failed") >= 2
    # With scratch intact all three stages run the driver, in order.
    sb = _sandbox(tmp_path / "three")
    code, out = _run(sb, tmp_path / "e6", dry=False)
    assert code == 0, out
    invoked = _driver_calls(sb)
    assert len(invoked) == 3 and "MILESTONE PASSED" in out
    assert "--label forced_acc4" in invoked[1] and "--perturb MICRO_DMA_L4_L1_32K=0.01" in invoked[2]
    # A driver that exits nonzero (an incomplete or failed session) is a failed stage.
    sb = _sandbox(tmp_path / "four", driver_exit=1)
    code, out = _run(sb, tmp_path / "e7", dry=False)
    assert code == 1 and "stage vector_rows FAIL" in out and len(_driver_calls(sb)) == 3


def test_the_sequence_scratch_budget_is_a_prerequisite(tmp_path):
    """The whole sequence's scratch must be free before the first launch
    (round 152, X-47): 78 planned launches x the per-launch consumption."""
    import shutil

    sb = _sandbox(tmp_path)
    free = shutil.disk_usage(tmp_path).free
    # a per-launch figure that the 78 launches cannot fit: refused, nothing launched
    code, out = _run(sb, tmp_path / "e8", env_extra={"G1_TMP_PATH": str(tmp_path), "G1_TMP_PER_LAUNCH_BYTES": str(free * 2)})
    assert code == 1 and "prerequisite scratch_budget failed" in out and "nothing launched" in out
    assert "planned launches 78 x" in out and _driver_calls(sb) == []
    # a per-launch figure that fits: the dry run proceeds to the three commands
    code, out = _run(sb, tmp_path / "e9", env_extra={"G1_TMP_PATH": str(tmp_path), "G1_TMP_PER_LAUNCH_BYTES": "1"})
    assert code == 0 and "planned launches 78 x 1 bytes = 78 needed" in out
    assert out.count("DRY RUN: would launch") == 3
