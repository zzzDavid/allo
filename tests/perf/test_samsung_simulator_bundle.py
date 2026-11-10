# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bundled Samsung PIMSimulator build path (`allo.pim.samsung_simulator`,
release issue P-04): the recipe is found from the package, its upstream copy
verifies against the manifest, a tampered copy is refused, the wheel's
build_py carries the recipe inside the package, the default build root is
discovered by `spmw_simenv.pimsim_root()` only once built, and (toolchain
permitting) the build produces a `pim_driver` reporting the geometry."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from allo import spmw_simenv
from allo.pim import samsung_simulator as bundle

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "simulators" / "PIMSimulator"
GEOMETRY = (
    "PIM_GEOMETRY channels=64 ranks=1 pim_blocks=8 banks=16 bank_groups=4 "
    "grf_a=8 grf_b=8 lanes=16 crf_words=32 cols_per_row=32 rows=16384"
)


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    """No simulator configured anywhere; TENON_HOME under tmp."""
    for variable in (
        "PIMSIMULATOR_ROOT",
        "TENON_SIMULATORS_DIR",
        "TENON_PIMSIM_RECIPE",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(spmw_simenv, "_CHECKOUT_SIMULATORS_DIR", tmp_path / "absent")
    home = tmp_path / "home"
    monkeypatch.setenv("TENON_HOME", str(home))
    return home


def test_recipe_dir_is_the_checkout_copy(monkeypatch):
    monkeypatch.delenv("TENON_PIMSIM_RECIPE", raising=False)
    assert bundle.recipe_dir() == RECIPE


def test_recipe_dir_override_and_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("TENON_PIMSIM_RECIPE", str(tmp_path / "nowhere"))
    with pytest.raises(bundle.BundleError, match="TENON_PIMSIM_RECIPE"):
        bundle.recipe_dir()
    copy = tmp_path / "copy"
    shutil.copytree(RECIPE, copy)
    monkeypatch.setenv("TENON_PIMSIM_RECIPE", str(copy))
    assert bundle.recipe_dir() == copy


def test_bundled_upstream_verifies_and_tampering_is_reported(tmp_path):
    assert bundle.verify_recipe(RECIPE) == []
    entries = bundle.read_manifest(RECIPE)
    assert len(entries) >= 60
    copy = tmp_path / "copy"
    shutil.copytree(RECIPE, copy)
    (copy / "upstream/src/PIMRank.cpp").write_text("// altered\n")
    (copy / "upstream/src/Burst.h").unlink()
    (copy / "upstream/extra.txt").write_text("stray\n")
    (copy / "patches/Burst.h.patch").unlink()
    problems = bundle.verify_recipe(copy)
    assert "differs from upstream 3703d1f: src/PIMRank.cpp" in problems
    assert "missing: src/Burst.h" in problems
    assert "not in the manifest: extra.txt" in problems
    assert "missing: patches/Burst.h.patch" in problems
    with pytest.raises(bundle.BundleError, match="does not match the pinned upstream"):
        bundle.assemble(copy, tmp_path / "root")


def test_assemble_copies_pristine_upstream_and_refuses_the_recipe_itself(tmp_path):
    root = bundle.assemble(RECIPE, tmp_path / "root")
    assert bundle.verify_upstream(RECIPE, root) == []
    assert (root / "ini/HBM2_samsung_2M_16B_x64.ini").is_file()
    with pytest.raises(bundle.BundleError, match="must not be inside the recipe"):
        bundle.assemble(RECIPE, RECIPE / "build")


def test_default_root_follows_tenon_home_and_is_used_only_once_built(isolated_home):
    expected = isolated_home / "simulators" / "PIMSimulator"
    assert spmw_simenv.default_pimsim_root() == expected
    assert spmw_simenv.pimsim_root() is None
    reason = spmw_simenv.samsung_unavailable_reason()
    assert "python -m allo.pim.samsung_simulator build" in reason
    assert str(expected) in reason
    expected.mkdir(parents=True)
    assert spmw_simenv.pimsim_root() is None  # a directory alone is not a build
    (expected / "pim_driver").write_text("")
    assert spmw_simenv.pimsim_root() == expected
    reason = spmw_simenv.samsung_unavailable_reason()
    assert reason is not None and "not executable" in reason
    os.chmod(expected / "pim_driver", 0o755)
    assert spmw_simenv.samsung_unavailable_reason() is None


def test_configured_locations_keep_precedence_over_the_default(
    isolated_home, monkeypatch, tmp_path
):
    built = isolated_home / "simulators" / "PIMSimulator"
    built.mkdir(parents=True)
    (built / "pim_driver").write_text("")
    monkeypatch.setenv("TENON_SIMULATORS_DIR", str(tmp_path / "sims"))
    assert spmw_simenv.pimsim_root() == tmp_path / "sims" / "PIMSimulator"
    monkeypatch.setenv("PIMSIMULATOR_ROOT", str(tmp_path / "explicit"))
    assert spmw_simenv.pimsim_root() == tmp_path / "explicit"


def test_xdg_data_home_default(monkeypatch, tmp_path):
    monkeypatch.delenv("TENON_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert spmw_simenv.tenon_data_home() == tmp_path / "xdg" / "tenon"
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert spmw_simenv.tenon_data_home() == Path.home() / ".local" / "share" / "tenon"


def test_cli_verify_and_locate(isolated_home):
    env = dict(os.environ, TENON_HOME=str(isolated_home))
    for variable in (
        "PIMSIMULATOR_ROOT",
        "TENON_SIMULATORS_DIR",
        "TENON_PIMSIM_RECIPE",
    ):
        env.pop(variable, None)
    verify = subprocess.run(
        [sys.executable, "-m", "allo.pim.samsung_simulator", "verify"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "upstream files verified" in verify.stdout
    assert bundle.UPSTREAM_COMMIT in verify.stdout
    locate = subprocess.run(
        [sys.executable, "-m", "allo.pim.samsung_simulator", "locate"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )
    # The checkout's simulators directory is still visible to a subprocess
    # (only the in-process monkeypatch hides it); either way the default
    # root is reported and the recipe is found.
    assert "recipe " in locate.stdout and "default root " in locate.stdout
    assert str(isolated_home / "simulators" / "PIMSimulator") in locate.stdout


def test_build_py_bundles_the_recipe_inside_the_package(tmp_path):
    """The wheel carries simulators/PIMSimulator as allo/pim/simulators/
    PIMSimulator (setup.py BuildPy), without upstream build outputs."""
    # setuptools is the only prerequisite; its absence is detected narrowly
    # before the build so that a build failure is a failure, not a skip.
    if importlib.util.find_spec("setuptools") is None:
        pytest.skip("setuptools is not installed")
    build_lib = tmp_path / "lib"
    result = subprocess.run(
        [
            sys.executable,
            "setup.py",
            "--quiet",
            "build_py",
            "--build-lib",
            str(build_lib),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"setup.py build_py exited with {result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}"
    )
    copied = build_lib / "allo" / "pim" / "simulators" / "PIMSimulator"
    assert (copied / "apply.sh").is_file()
    assert (copied / "NOTICE.md").is_file()
    assert bundle.verify_recipe(copied) == []
    present = {
        p.relative_to(copied).as_posix() for p in copied.rglob("*") if p.is_file()
    }
    source = {
        p.relative_to(RECIPE).as_posix() for p in RECIPE.rglob("*") if p.is_file()
    }
    assert present == source
    assert not [p for p in present if p.startswith(("bin/", "libdramsim/"))]


def _toolchain_missing() -> str | None:
    for tool in ("patch", "g++"):
        if shutil.which(tool) is None:
            return f"{tool} is not installed"
    if shutil.which("scons") is None and not os.environ.get("TENON_SCONS"):
        if importlib.util.find_spec("SCons") is None:
            return "SCons is not installed"
    return None


def test_build_produces_pim_driver_with_the_geometry(
    isolated_home, monkeypatch, tmp_path
):
    """The full path from the bundled sources: assemble, patch, build, and
    `pim_driver --describe` reports the simulator geometry; the result is
    discovered through the default root.  Compiler temporaries go to a
    writable TMPDIR so a full /tmp does not fail the build."""
    missing = _toolchain_missing()
    if missing:
        pytest.skip(missing)
    scratch = tmp_path / "cc-tmp"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    root = bundle.build(jobs=min(16, os.cpu_count() or 1))
    assert root == isolated_home / "simulators" / "PIMSimulator"
    assert bundle.describe(root) == GEOMETRY
    assert spmw_simenv.pimsim_root() == root
    assert spmw_simenv.samsung_unavailable_reason() is None
    for rel in bundle.RUNTIME_FILES:
        assert (root / rel).is_file(), rel
    assert (root / "libdramsim" / "libdramsim2.a").is_file()
    # A rebuild starts from pristine upstream again and re-applies the patches.
    (root / "src/PIMRank.cpp").write_text("// altered\n")
    again = bundle.build(jobs=min(16, os.cpu_count() or 1), skip_build=True)
    assert again == root
    assert bundle.verify_upstream(RECIPE, root) != []  # patched + Tenon files differ
    assert "// altered" not in (root / "src/PIMRank.cpp").read_text()
