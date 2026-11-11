# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release packaging and environment-discovery contract (release audit P-01..P-21).

The sdist must carry every non-Python file the package reads at runtime and
the MLIR sources its build_ext needs, and must not ship the repository-only
trees. Every external location goes through ``allo.spmw_simenv`` with an
environment override.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from allo import spmw_aim, spmw_simenv

REPO = Path(__file__).resolve().parents[2]

# Runtime assets read through __file__-relative paths, relative to REPO.
RUNTIME_ASSET_GLOBS = (
    "allo/pim/templates/apu_g2/*",
    "allo/pim/templates/apu_g2/device/*",
    "allo/autoscheduler/outline_loops.mlir",
    "allo/ir/template/*.mlir",
    "allo/harness/*/*",
    "allo/library/aie/kernels/*.cc",
    "allo/library/aie/modules/*.py",
)


def _runtime_assets() -> list[str]:
    assets = set()
    for pattern in RUNTIME_ASSET_GLOBS:
        for path in REPO.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                assets.add(path.relative_to(REPO).as_posix())
    return sorted(assets)


@pytest.fixture(scope="module")
def sdist_members(tmp_path_factory):
    if importlib.util.find_spec("setuptools") is None:
        pytest.skip("setuptools is not installed")
    out = tmp_path_factory.mktemp("sdist")
    if importlib.util.find_spec("build") is not None:
        command = [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(out),
            str(REPO),
        ]
    else:
        command = [
            sys.executable,
            "setup.py",
            "--quiet",
            "egg_info",
            "--egg-base",
            str(out),
            "sdist",
            "--dist-dir",
            str(out),
        ]
    result = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, timeout=300, check=False
    )
    if result.returncode != 0:
        pytest.skip(f"sdist build backend unavailable: {result.stderr[-2000:]}")
    (archive,) = out.glob("*.tar.gz")
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        top_level = next(
            (n for n in names if n.endswith(".egg-info/top_level.txt")), None
        )
        top = tar.extractfile(top_level).read().decode().split() if top_level else None
    if top is None:
        on_disk = list(out.glob("*.egg-info/top_level.txt"))
        top = on_disk[0].read_text().split() if on_disk else None
    prefix = names[0].split("/", 1)[0] + "/"
    members = {n[len(prefix) :] for n in names if n.startswith(prefix)}
    return members, top


def test_sdist_ships_every_runtime_asset(sdist_members):
    members, _top = sdist_members
    assets = _runtime_assets()
    # 11 Gemini-II template files since the uint16 programs left (G2-11);
    # guard the glob.
    assert sum(a.startswith("allo/pim/templates/apu_g2/") for a in assets) >= 11
    missing = [asset for asset in assets if asset not in members]
    assert not missing, missing


def test_sdist_ships_mlir_sources_but_no_build_tree(sdist_members):
    members, _top = sdist_members
    assert "mlir/CMakeLists.txt" in members
    assert "requirements.txt" in members
    assert any(m.startswith("mlir/lib/") and m.endswith(".cpp") for m in members)
    leaked = [
        m
        for m in members
        if m.startswith(
            (
                "mlir/build/",
                "allo/_mlir/",
                "tests/",
                "examples/",
                "benchmarks/",
                "playground/",
            )
        )
    ]
    assert not leaked, leaked[:10]


def test_only_allo_is_a_top_level_package(sdist_members):
    _members, top = sdist_members
    if top is None:
        pytest.skip("sdist carries no egg-info top_level.txt")
    # "mlir" is the CMakeExtension's name, which build_ext installs as
    # allo/_mlir rather than as a top-level module.
    assert set(top) <= {"allo", "mlir"} and "allo" in top


# --------------------------------------------------------------------- #
# spmw_simenv discovery
# --------------------------------------------------------------------- #


def test_simulator_roots_follow_tenon_simulators_dir(monkeypatch, tmp_path):
    for variable in ("PIMSIMULATOR_ROOT", "AIM_SIMULATOR_ROOT"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TENON_SIMULATORS_DIR", str(tmp_path))
    assert spmw_simenv.pimsim_root() == tmp_path / "PIMSimulator"
    assert spmw_simenv.aim_root() == tmp_path / "aim_simulator"
    monkeypatch.setenv("PIMSIMULATOR_ROOT", str(tmp_path / "pim"))
    assert spmw_simenv.pimsim_root() == tmp_path / "pim"


def test_unconfigured_simulator_location_names_the_variable(monkeypatch, tmp_path):
    for variable in ("PIMSIMULATOR_ROOT", "AIM_SIMULATOR_ROOT", "TENON_SIMULATORS_DIR"):
        monkeypatch.delenv(variable, raising=False)
    # An installed copy has no checkout-relative simulators directory, and
    # the bundled simulator's default root is not built here.
    monkeypatch.setattr(spmw_simenv, "_CHECKOUT_SIMULATORS_DIR", tmp_path / "absent")
    monkeypatch.setenv("TENON_HOME", str(tmp_path / "home"))
    assert spmw_simenv.simulators_dir() is None
    assert spmw_simenv.pimsim_root() is None
    reason = spmw_simenv.samsung_unavailable_reason()
    assert "PIMSIMULATOR_ROOT" in reason and "TENON_SIMULATORS_DIR" in reason
    # ... and names the bundled build command and where it installs.
    assert "python -m allo.pim.samsung_simulator build" in reason
    assert str(tmp_path / "home" / "simulators" / "PIMSimulator") in reason
    monkeypatch.setattr(spmw_simenv, "docker_image_unavailable_reason", lambda _n: None)
    reason = spmw_simenv.aim_unavailable_reason()
    assert "AIM_SIMULATOR_ROOT" in reason and "TENON_SIMULATORS_DIR" in reason


def test_aim_image_override(monkeypatch):
    # These are the Docker executor's tests: a native binary configured in
    # the environment (TENON_AIM_RAMULATOR2) would bypass it.
    monkeypatch.delenv("TENON_AIM_RAMULATOR2", raising=False)
    monkeypatch.delenv("TENON_AIM_SIM_IMAGE", raising=False)
    assert spmw_simenv.aim_sim_image() == "aim-simulator-build"
    monkeypatch.setenv("TENON_AIM_SIM_IMAGE", "my-aim")
    probed = []
    monkeypatch.setattr(
        spmw_simenv,
        "docker_image_unavailable_reason",
        lambda name: probed.append(name) or "stop",
    )
    spmw_simenv.aim_unavailable_reason()
    assert probed == ["my-aim"]


def test_aim_run_never_writes_into_the_simulator_checkout(monkeypatch, tmp_path):
    # These are the Docker executor's tests: a native binary configured in
    # the environment (TENON_AIM_RAMULATOR2) would bypass it.
    monkeypatch.delenv("TENON_AIM_RAMULATOR2", raising=False)
    root = tmp_path / "aim_simulator"
    (root / "test").mkdir(parents=True)
    (root / "test" / "example.yaml").write_text("")
    before = sorted(root.rglob("*"))
    seen = {}

    def fake_run(command, **_kwargs):
        mount = command[command.index("-v", command.index("-v") + 1) + 1]
        host_dir, _, _ = mount.partition(":/trace")
        seen["command"] = command
        seen["trace"] = (Path(host_dir) / "segment.trace").read_text()
        return subprocess.CompletedProcess(
            command, 0, stdout=b"memory_system_cycles: 7\n", stderr=b""
        )

    monkeypatch.setenv("TENON_AIM_SIM_IMAGE", "my-aim")
    monkeypatch.setattr(spmw_aim, "_aim_unavailable_reason", lambda: None)
    monkeypatch.setattr(spmw_aim, "_aim_root", lambda: root)
    monkeypatch.setattr(spmw_aim, "_check_simulator_geometry", lambda *_a: None)
    monkeypatch.setattr(spmw_aim.subprocess, "run", fake_run)
    compiled = SimpleNamespace(
        target=None, cmds=[], runtime_segments=(("seg", 3, ["AiM EOC"]),)
    )
    result = spmw_aim._run_aim(compiled)
    assert result.cycles == 21
    assert "my-aim" in seen["command"]
    assert any(part.endswith(":/trace:ro") for part in seen["command"])
    assert seen["trace"] == "AiM EOC\n"
    assert sorted(root.rglob("*")) == before


def test_apu_g2_locations_go_through_simenv(monkeypatch, tmp_path):
    for variable in (
        "TENON_APU_G2_SDK_ROOT",
        "TENON_APU_G2_GSI_TOOL",
        "TENON_APU_G2_DEVICE",
    ):
        monkeypatch.delenv(variable, raising=False)
    assert spmw_simenv.apu_g2_sdk_root() == Path("/opt/gsi")
    assert spmw_simenv.apu_g2_device() == "apu-00"
    assert all(
        str(path).startswith("/opt/gsi/")
        for path in spmw_simenv.apu_g2_required_sdk_paths()
    )
    assert spmw_simenv.apu_g2_unsupported_override_reason() is None

    tool = tmp_path / "gsi_tool"
    monkeypatch.setenv("TENON_APU_G2_GSI_TOOL", str(tool))
    assert spmw_simenv.apu_g2_gsi_tool() == str(tool)

    # The shipped templates compile in /opt/gsi and open device 0: other
    # values must fail loudly instead of probing a different installation.
    monkeypatch.setenv("TENON_APU_G2_SDK_ROOT", str(tmp_path))
    assert spmw_simenv.apu_g2_required_sdk_paths()[0].is_relative_to(tmp_path)
    assert "TENON_APU_G2_SDK_ROOT" in spmw_simenv.apu_g2_unsupported_override_reason()
    from allo.pim import apu_g2_runtime

    with pytest.raises(RuntimeError, match="TENON_APU_G2_SDK_ROOT"):
        apu_g2_runtime._require_hardware_stack()
    monkeypatch.delenv("TENON_APU_G2_SDK_ROOT")
    monkeypatch.setenv("TENON_APU_G2_DEVICE", "apu-01")
    assert "TENON_APU_G2_DEVICE" in spmw_simenv.apu_g2_unsupported_override_reason()


def test_apu_g2_runtime_has_no_hard_coded_sdk_paths():
    source = (REPO / "allo" / "pim" / "apu_g2_runtime.py").read_text()
    assert "/opt/gsi" not in source
    assert 'which("gsi_tool")' not in source


def _fake_upmem_root(tmp_path: Path) -> Path:
    root = tmp_path / "uPIMulator"
    slot = root / "benchmark" / spmw_simenv.upmem_benchmark_slot() / "dpu"
    slot.mkdir(parents=True)
    (slot / "CMakeLists.txt").write_text("")
    build = root / "sdk" / "build"
    build.mkdir(parents=True)
    (build / "x.o").write_bytes(b"not the campaign sdk")
    return root


def test_upmem_user_build_bypasses_snapshot_and_manifest(monkeypatch, tmp_path):
    root = _fake_upmem_root(tmp_path)
    binary = tmp_path / "uPIMulator.bin"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr(spmw_simenv, "docker_image_unavailable_reason", lambda _n: None)
    monkeypatch.setenv("TENON_UPMEM_PROVENANCE_DIR", str(tmp_path / "no-archive"))
    monkeypatch.setenv("UPIMULATOR_ROOT", str(root))
    monkeypatch.setenv("UPIMULATOR_BIN", str(binary))
    manifest = spmw_simenv.sdk_build_manifest_sha256(root)
    assert manifest != spmw_simenv.UPMEM_SDK_BUILD_MANIFEST_SHA256

    assert spmw_simenv.upmem_user_build()
    assert spmw_simenv.upmem_unavailable_reason() is None
    assert spmw_simenv.ensure_upimulator_root() == root
    assert (
        spmw_simenv.upmem_sdk_build_identity(manifest) == f"user-build:{manifest[:12]}"
    )

    # Without UPIMULATOR_BIN the run is the pinned route: the drifted
    # manifest is refused and the message names the user-build variables.
    monkeypatch.delenv("UPIMULATOR_BIN")
    monkeypatch.setattr(spmw_simenv, "upimulator_bin", lambda: binary)
    assert not spmw_simenv.upmem_user_build()
    reason = spmw_simenv.upmem_unavailable_reason()
    assert "pinned campaign build" in reason and "UPIMULATOR_BIN" in reason
    with pytest.raises(RuntimeError, match="UPIMULATOR_BIN"):
        spmw_simenv.upmem_sdk_build_identity(manifest)
    pinned = spmw_simenv.UPMEM_SDK_BUILD_MANIFEST_SHA256
    assert spmw_simenv.upmem_sdk_build_identity(pinned) == "pinned-campaign"


def test_upmem_missing_binary_names_both_routes(monkeypatch, tmp_path):
    for variable in ("UPIMULATOR_ROOT", "UPIMULATOR_BIN"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TENON_UPMEM_PROVENANCE_DIR", str(tmp_path))
    reason = spmw_simenv.upmem_unavailable_reason()
    assert "UPIMULATOR_BIN" in reason and "TENON_ARTIFACTS" in reason


def test_no_user_home_literals_in_the_package():
    offenders = []
    for path in (REPO / "allo").rglob("*.py"):
        if "_mlir" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/home/nz264" in text or "/work/shared/users" in text:
            offenders.append(path.relative_to(REPO).as_posix())
    assert not offenders, offenders


def test_paper_golden_artifacts_default_comes_from_simenv(monkeypatch, tmp_path):
    monkeypatch.setenv("TENON_ARTIFACTS", str(tmp_path))
    source = (REPO / "tests" / "pim" / "test_paper_golden.py").read_text()
    assert "/home/" not in source
    assert spmw_simenv.tenon_artifacts_root() == tmp_path
    monkeypatch.delenv("TENON_ARTIFACTS")
    assert spmw_simenv.tenon_artifacts_root() == (
        Path(os.path.expanduser("~")) / "shared" / "tenon-artifacts"
    )


def test_install_requires_is_the_runtime_list_only():
    """P-13: dev tooling and direct-URL wheels stay out of install_requires."""
    import runpy

    setup = runpy.run_path(str(REPO / "setup.py"), run_name="allo_setup")
    runtime = setup["parse_requirements"](str(REPO / "requirements.txt"))
    names = {re.split(r"[<>=@ ]", line)[0].lower() for line in runtime}
    assert not names & {"pytest", "black", "pylint", "clang-format", "gurobipy", "past"}
    assert {"numpy", "sympy", "ml_dtypes", "pyyaml"} <= names
    dev = (REPO / "requirements-dev.txt").read_text().splitlines()
    assert "-r requirements.txt" in dev
    with pytest.raises(RuntimeError, match="requirements-dev.txt"):
        setup["parse_requirements"](str(REPO / "requirements-dev.txt"))


def test_old_cmake_error_names_the_pip_fix(monkeypatch):
    """P-20: a system cmake older than 3.20 fails with the pip install hint."""
    import runpy

    setup = runpy.run_path(str(REPO / "setup.py"), run_name="allo_setup")
    monkeypatch.setattr(
        subprocess, "check_output", lambda *args, **kwargs: "cmake version 3.16.3\n"
    )
    command = setup["CMakeBuild"].__new__(setup["CMakeBuild"])
    command.extensions = []
    with pytest.raises(RuntimeError, match="pip install cmake"):
        command.run()


def test_upmem_calibration_table_regenerates():
    """P-14: the committed regenerator reproduces PHYSICAL_CALIBRATION_FACTORS."""
    import runpy

    from allo.pim.costs import upmem as upmem_costs

    script = runpy.run_path(
        str(REPO / "scripts" / "regenerate_upmem_calibration.py"), run_name="regen"
    )
    table = script["factors"]()
    assert table == upmem_costs.PHYSICAL_CALIBRATION_FACTORS
    assert list(table) == list(upmem_costs.PHYSICAL_CALIBRATION_FACTORS)


def test_distribution_metadata_identifies_the_tenon_fork():
    """P-07: the distribution's own metadata names the Tenon fork, the PIM
    targets it adds, its publishing repository and its upstream; the package
    name and version line stay upstream Allo's by decision (installed in
    place of upstream Allo)."""
    result = subprocess.run(
        [
            sys.executable,
            "setup.py",
            "--quiet",
            "--name",
            "--version",
            "--description",
            "--url",
            "--maintainer",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    name, version, description, url, maintainer = result.stdout.strip().splitlines()[
        -5:
    ]
    assert name == "allo" and version == "0.5"
    assert "Tenon" in description and "processing-in-memory" in description
    for target in ("Samsung", "AiM", "UPMEM", "Gemini-I", "Gemini-II", "AttAcc"):
        assert target in description, target
    assert url == "https://github.com/zzzDavid/allo/tree/tenon"
    assert "zzzDavid/allo" in maintainer and "tenon" in maintainer
