# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shell-path checks of the maintained Gemini-I campaign driver
`docs/release/scripts/g1_campaign.sh` in its dry-run mode (round 144, ISSUES
X-46): a missing or malformed existing session file, or an incompatible one,
stops the campaign before any launch; a prior attempt's directory is never
overwritten; a valid incomplete plan names only the rows still owed; a
campaign at quota launches nothing.  The sandbox stands in for the installed
artifact (a `pkg/venv` whose interpreter sees a stub `allo` package and a
stub `benchmarks` tree); the pooler is the real one."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "docs" / "release" / "scripts" / "g1_campaign.sh"
FIXTURES = REPO / "tests" / "perf" / "test_release_g1_pooling_script.py"


@pytest.fixture(scope="module")
def fixtures():
    spec = importlib.util.spec_from_file_location("g1_fixtures", FIXTURES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sandbox(root: Path) -> dict:
    pkg = root / "pkg"
    (pkg / "venv" / "bin").mkdir(parents=True)
    lib = pkg / "venv" / "lib"
    (lib / "allo" / "pim").mkdir(parents=True)
    (lib / "allo" / "__init__.py").write_text("")
    (lib / "allo" / "pim" / "__init__.py").write_text("")
    (lib / "allo" / "pim" / "apu_v1_program.py").write_text(
        "def ledag_scratch_headroom_problem():\n    return None\n"
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
    (tree / "benchmarks").mkdir(parents=True)
    (tree / "benchmarks" / "__init__.py").write_text("")
    assets = root / "assets"
    for case in ("kmeans", "word_count", "linear_regression"):
        (assets / "apuv1" / "tenon" / case).mkdir(parents=True)
    return {"pkg": pkg, "tree": tree, "assets": assets}


def _run(root: Path, evidence: Path, existing, env_extra=None):
    sb = (
        _sandbox(root)
        if not (root / "pkg").exists()
        else {"pkg": root / "pkg", "tree": root / "testtree", "assets": root / "assets"}
    )
    evidence.mkdir(parents=True, exist_ok=True)
    if not (evidence / "pkg").exists():
        (evidence / "pkg").symlink_to(sb["pkg"])
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "G1_DRY_RUN": "1",
            "G1_TESTTREE": str(sb["tree"]),
            "G1_EXISTING": " ".join(str(f) for f in existing),
            "TENON_ARTIFACTS": str(sb["assets"]),
            "TMPDIR": str(root / "tmp"),
        }
    )
    (root / "tmp").mkdir(exist_ok=True)
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


def _sessions(fixtures, root: Path, sessions):
    files = []
    root.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(sessions, 1):
        f = root / f"apu_v1_campaign_s{i}.json"
        f.write_text(json.dumps(s))
        files.append(f)
    return files


def test_missing_existing_file_stops_before_any_launch(tmp_path):
    code, out = _run(tmp_path, tmp_path / "e1", [tmp_path / "does-not-exist.json"])
    assert code == 1
    assert "MISSING existing session file" in out and "DRY RUN" not in out
    assert "nothing launched" in out
    # the quoted brace pattern of the old recipe is a nonexistent file name, refused the same way
    code, out = _run(
        tmp_path, tmp_path / "e1b", [tmp_path / "apu_v1_campaign_s{1,2,3}.json"]
    )
    assert code == 1 and "MISSING existing session file" in out and "{1,2,3}" in out


def test_malformed_or_incompatible_existing_input_stops_the_campaign(
    fixtures, tmp_path
):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    code, out = _run(tmp_path, tmp_path / "e2", [bad])
    assert code == 1 and "PLANNING FAILED" in out and "DRY RUN" not in out
    sessions = [
        fixtures.make_session("a"),
        fixtures.make_session("b", allo="/other/venv/allo/__init__.py"),
    ]
    files = _sessions(fixtures, tmp_path / "s2", sessions)
    code, out = _run(tmp_path, tmp_path / "e2b", files)
    assert (
        code == 1
        and "PLANNING FAILED" in out
        and "exited 2" in out
        and "DRY RUN" not in out
    )


def test_a_prior_attempt_is_never_overwritten(fixtures, tmp_path):
    files = _sessions(fixtures, tmp_path / "s3", [fixtures.make_session("a")])
    code, out = _run(tmp_path, tmp_path / "e3", files)
    assert code == 0 and "DRY RUN: would launch" in out
    # a stale plan from the first attempt cannot be consumed: the directory is refused
    (tmp_path / "e3" / "campaign-dry-run" / "plan.json").write_text(
        json.dumps({"next_session_cases": ["kmeans"]})
    )
    code, out = _run(tmp_path, tmp_path / "e3", [tmp_path / "does-not-exist.json"])
    assert code == 1 and "refusing to run" in out and "DRY RUN" not in out


def test_a_valid_incomplete_plan_names_only_the_rows_still_owed(fixtures, tmp_path):
    first = fixtures.make_session("s1")
    first["records"][2]["problems"] = ["historical_binary measured 6: lost"]
    files = _sessions(fixtures, tmp_path / "s4", [first])
    code, out = _run(tmp_path, tmp_path / "e4", files)
    assert code == 0
    launch = [l for l in out.splitlines() if "DRY RUN: would launch" in l][0]
    assert "--cases kmeans word_count linear_regression " in launch
    plan = json.loads((tmp_path / "e4" / "campaign-dry-run" / "plan.json").read_text())
    assert plan["remaining_complete_rows"] == {
        "kmeans": 2,
        "word_count": 2,
        "linear_regression": 3,
    }
    files = _sessions(
        fixtures,
        tmp_path / "s4b",
        [first, fixtures.make_session("s2"), fixtures.make_session("s3")],
    )
    code, out = _run(tmp_path, tmp_path / "e4b", files)
    assert code == 0
    launch = [l for l in out.splitlines() if "DRY RUN: would launch" in l][0]
    assert (
        "--cases linear_regression " in launch
        and "kmeans" not in launch.split("--cases")[1].split("--scratch")[0]
    )


def test_a_campaign_at_quota_launches_nothing(fixtures, tmp_path):
    files = _sessions(
        fixtures, tmp_path / "s5", [fixtures.make_session(t) for t in ("a", "b", "c")]
    )
    code, out = _run(tmp_path, tmp_path / "e5", files)
    assert code == 0
    assert "every row is at its quota" in out and "would launch" not in out
    # with no existing files every row of the quota is owed
    code, out = _run(tmp_path, tmp_path / "e5b", [])
    assert code == 0 and "--cases kmeans word_count linear_regression " in out


def test_a_session_is_launched_only_within_its_scratch_budget(fixtures, tmp_path):
    """Round 152 (X-47): the session's planned launches x the per-launch
    consumption must be free; a session over budget is recorded as not
    launched and the driver is not invoked."""
    import shutil

    sb = _sandbox(tmp_path)
    free = shutil.disk_usage(tmp_path).free
    calls = tmp_path / "calls.txt"
    (sb["tree"] / "benchmarks" / "apu_v1").mkdir(parents=True, exist_ok=True)
    (sb["tree"] / "benchmarks" / "apu_v1" / "__init__.py").write_text("")
    (sb["tree"] / "benchmarks" / "apu_v1" / "paper_rows.py").write_text(
        f"import sys\nopen({str(calls)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\nsys.exit(1)\n"
    )
    # three rows owed: 99 launches planned for session 1; a budget it cannot meet
    code, out = _run(tmp_path, tmp_path / "e6", [], env_extra={
        "G1_DRY_RUN": "0", "G1_MAX_SESSIONS": "1",
        "G1_TMP_PATH": str(tmp_path), "G1_TMP_PER_LAUNCH_BYTES": str(free // 99 + 1),
    })
    assert "planned launches 99 x" in out and "session 1 NOT LAUNCHED: the headroom gate failed" in out
    assert not calls.exists()
    # within budget the session is launched (the stub driver fails, which is a failed stage)
    code, out = _run(tmp_path, tmp_path / "e7", [], env_extra={
        "G1_DRY_RUN": "0", "G1_MAX_SESSIONS": "1",
        "G1_TMP_PATH": str(tmp_path), "G1_TMP_PER_LAUNCH_BYTES": "1",
    })
    assert "planned launches 99 x 1 bytes = 99 needed" in out and calls.exists()
    assert "--cases kmeans word_count linear_regression" in calls.read_text()
