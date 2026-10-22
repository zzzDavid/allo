# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""External simulator, toolchain and device locations for SPMW backends.

This module is the single owner of every path outside the repository that a
PIM backend reads, and of every availability probe. Each location has an
environment override; defaults are repository-relative, a system install
path, or derived from the user's home directory.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

_SIMULATORS_DIR = Path(__file__).resolve().parents[2] / "simulators"
_DEFAULT_PIMSIM_ROOT = _SIMULATORS_DIR / "PIMSimulator"
_DEFAULT_AIM_ROOT = _SIMULATORS_DIR / "aim_simulator"
_DEFAULT_UPMEM_SDK_IMAGE = "bongjoonhyun/upimulator"
_DEFAULT_UPMEM_BENCHMARK_SLOT = "GENERIC"
_UPIMULATOR_SNAPSHOT_NAME = "source-snapshot.tar"
_UPIMULATOR_SNAPSHOT_SUBDIR = Path("golang") / "uPIMulator"
_STAGED_MARKER = ".tenon-staged"
_STAGED_BENCHMARK_CMAKE = (
    "cmake_minimum_required(VERSION 3.16)\n"
    "project(benchmark)\n"
    "add_subdirectory(GENERIC)\n"
)

# sha256 over the ``sha256sum`` lines (``<hex>  ./<rel>``) of every ``*.o``
# under the staged root's ``sdk/build``, sorted by relative path. It pins the
# patched DPU SDK build that produced the archived UPMEM paper cycles; another
# SDK build shifts them (VA 114567 instead of 114534).
UPMEM_SDK_BUILD_MANIFEST_SHA256 = (
    "d45e285c34b5ab53e02f9d8a7289725112a3a25c0c3ed8a8a15d47c41173963a"
)

_DEFAULT_APU_V1_TOOLCHAIN_BASE = (
    "/usr/local/gsi-apu/13.7.1/ubuntu_20_04/"
    "arc_gnu_2021.09-release_elf32_le_linux_no_sdata"
)
_DEFAULT_GVML_INCLUDE_ROOT = "/usr/local/include"
_GSI_PCI_VENDOR_ID = "0x1e4c"
_PCI_DEVICES_DIR = Path("/sys/bus/pci/devices")

_DOCKER_PROBE_TIMEOUT_S = 10


def pimsim_root() -> Path:
    return Path(os.environ.get("PIMSIMULATOR_ROOT", str(_DEFAULT_PIMSIM_ROOT)))


def aim_root() -> Path:
    return Path(os.environ.get("AIM_SIMULATOR_ROOT", str(_DEFAULT_AIM_ROOT)))


def tenon_artifacts_root() -> Path:
    override = os.environ.get("TENON_ARTIFACTS")
    if override is not None:
        return Path(override)
    return Path.home() / "shared" / "tenon-artifacts"


def upimulator_provenance_dir() -> Path:
    """Archived uPIMulator binary and source snapshot of the UPMEM paper runs."""
    override = os.environ.get("TENON_UPMEM_PROVENANCE_DIR")
    if override is not None:
        return Path(override)
    return tenon_artifacts_root() / "upmem" / "provenance" / "upimulator"


def upimulator_snapshot() -> Path:
    return upimulator_provenance_dir() / _UPIMULATOR_SNAPSHOT_NAME


_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def _file_sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _SHA256_CACHE.get(key)
    if cached is None:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        cached = digest.hexdigest()
        _SHA256_CACHE[key] = cached
    return cached


def upimulator_root() -> Path:
    """uPIMulator Go project root holding ``benchmark/`` and ``sdk/``.

    Default: the provenance source snapshot staged under
    ``upmem_scratch_dir()`` by :func:`ensure_upimulator_root`.
    """
    override = os.environ.get("UPIMULATOR_ROOT")
    if override is not None:
        return Path(override)
    sha = _file_sha256(upimulator_snapshot())
    return upmem_scratch_dir() / f"root-{sha[:12]}" / _UPIMULATOR_SNAPSHOT_SUBDIR


def upimulator_bin() -> Path:
    override = os.environ.get("UPIMULATOR_BIN")
    if override is not None:
        return Path(override)
    return upimulator_provenance_dir() / "uPIMulator"


def upimulator_sha256() -> str:
    return _file_sha256(upimulator_bin())


def sdk_build_manifest_sha256(root: Path) -> str:
    """Recompute the pinned SDK object manifest hash for a uPIMulator root."""
    build = Path(root) / "sdk" / "build"
    objects = sorted(
        path.relative_to(build).as_posix() for path in build.rglob("*.o") if path.is_file()
    )
    text = "".join(
        f"{hashlib.sha256((build / rel).read_bytes()).hexdigest()}  ./{rel}\n"
        for rel in objects
    )
    return hashlib.sha256(text.encode()).hexdigest()


def ensure_upimulator_root() -> Path:
    """Stage the provenance snapshot as the uPIMulator root, once.

    Extracts ``source-snapshot.tar`` with mtimes preserved (so the Docker
    ``ninja`` sees the prebuilt SDK as current), cuts ``benchmark/CMakeLists``
    to the ``GENERIC`` slot the snapshot ships, and writes a marker holding the
    tar sha256. Docker later writes root-owned files here, so the root is
    reused and never deleted. With ``UPIMULATOR_ROOT`` set nothing is staged.
    """
    if "UPIMULATOR_ROOT" in os.environ:
        return upimulator_root()
    root = upimulator_root()
    stage = root.parents[len(_UPIMULATOR_SNAPSHOT_SUBDIR.parts) - 1]
    snapshot = upimulator_snapshot()
    sha = _file_sha256(snapshot)
    marker = stage / _STAGED_MARKER
    if marker.is_file() and marker.read_text().strip() == sha:
        return root
    scratch = upmem_scratch_dir()
    scratch.mkdir(parents=True, exist_ok=True)
    with (scratch / "root.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if marker.is_file() and marker.read_text().strip() == sha:
            return root
        stage.mkdir(parents=True, exist_ok=True)
        with tarfile.open(snapshot) as archive:
            if hasattr(tarfile, "tar_filter"):
                archive.extractall(stage, filter="tar")
            else:
                archive.extractall(stage)
        (root / "benchmark" / "CMakeLists.txt").write_text(_STAGED_BENCHMARK_CMAKE)
        marker.write_text(sha + "\n")
    return root


def upmem_sdk_image() -> str:
    """Docker image whose DPU SDK compiles the benchmark slot."""
    return os.environ.get("TENON_UPMEM_SDK_IMAGE", _DEFAULT_UPMEM_SDK_IMAGE)


def upmem_benchmark_slot() -> str:
    """Directory name under ``benchmark/`` that Tenon writes DPU programs into."""
    return os.environ.get("TENON_UPMEM_BENCHMARK_SLOT", _DEFAULT_UPMEM_BENCHMARK_SLOT)


def upmem_scratch_dir() -> Path:
    override = os.environ.get("TENON_UPMEM_SCRATCH")
    if override is not None:
        return Path(override)
    return Path(tempfile.gettempdir()) / "tenon-upmem"


def apu_v1_toolchain_base() -> Path:
    """ARC GNU toolchain install root, without the ``arc-snps-elf/`` suffix."""
    return Path(
        os.environ.get("TENON_APU_V1_TOOLCHAIN_BASE", _DEFAULT_APU_V1_TOOLCHAIN_BASE)
    )


def apu_v1_template_dir() -> Path:
    """GSI ``example-gvml`` project that APU v1 builds copy their harness from."""
    override = os.environ.get("TENON_APU_V1_TEMPLATE_DIR")
    if override is not None:
        return Path(override)
    return Path.home() / "shared" / "accelerator-hub" / "gsi-apu" / "example-gvml"


def gvml_include_root() -> str:
    """Return the directory under which `<gsi/libgvml_*.h>` headers live.

    Resolution order:
      1. env var `TENON_APU_V1_GVML_INCLUDE_ROOT` (override; not validated here).
      2. `_DEFAULT_GVML_INCLUDE_ROOT` (= `/usr/local/include`) -- the path
         GSI's stock `Common/common.mk` ships with for `product=x86_64`.

    Validation is intentionally deferred to `_assert_gvml_sdk_present()`;
    this getter is pure.
    """
    return os.environ.get(
        "TENON_APU_V1_GVML_INCLUDE_ROOT",
        _DEFAULT_GVML_INCLUDE_ROOT,
    )


def apu_v1_pci_node() -> Path | None:
    """sysfs node of the GSI board: the override, else a PCI vendor-ID scan."""
    override = os.environ.get("TENON_APU_V1_PCI_NODE")
    if override is not None:
        return Path(override)
    try:
        devices = sorted(_PCI_DEVICES_DIR.iterdir())
    except OSError:
        return None
    for device in devices:
        try:
            vendor = (device / "vendor").read_text().strip()
        except OSError:
            continue
        if vendor == _GSI_PCI_VENDOR_ID:
            return device
    return None


def docker_available() -> bool:
    return shutil.which("docker") is not None


def docker_image_unavailable_reason(name: str) -> str | None:
    """None when docker answers and image ``name`` is present."""
    if not docker_available():
        return "docker is not on PATH"
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", name],
            capture_output=True,
            timeout=_DOCKER_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"docker daemon did not answer within {_DOCKER_PROBE_TIMEOUT_S} s"
    except (subprocess.SubprocessError, OSError) as error:
        return f"docker probe failed: {error}"
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        if "Cannot connect to the Docker daemon" in stderr:
            return "docker daemon is not running"
        return f"docker image {name!r} is missing"
    return None


def docker_image_exists(name: str) -> bool:
    return docker_image_unavailable_reason(name) is None


def samsung_unavailable_reason() -> str | None:
    driver = pimsim_root() / "pim_driver"
    if not driver.exists():
        return f"pim_driver not found at {driver} (set PIMSIMULATOR_ROOT)"
    if not os.access(driver, os.X_OK):
        return f"pim_driver at {driver} is not executable"
    return None


def aim_unavailable_reason() -> str | None:
    reason = docker_image_unavailable_reason("aim-simulator-build")
    if reason is not None:
        return reason
    yaml_cfg = aim_root() / "test" / "example.yaml"
    if not yaml_cfg.exists():
        return f"AiM config missing at {yaml_cfg} (set AIM_SIMULATOR_ROOT)"
    return None


def upmem_unavailable_reason() -> str | None:
    binary = upimulator_bin()
    if not binary.exists():
        return f"uPIMulator binary not found at {binary} (set UPIMULATOR_BIN)"
    if not os.access(binary, os.X_OK):
        return f"uPIMulator binary at {binary} is not executable (set UPIMULATOR_BIN)"
    reason = docker_image_unavailable_reason(upmem_sdk_image())
    if reason is not None:
        return f"{reason} (set TENON_UPMEM_SDK_IMAGE)"
    if "UPIMULATOR_ROOT" not in os.environ:
        snapshot = upimulator_snapshot()
        if not snapshot.is_file():
            return (
                f"uPIMulator source snapshot missing at {snapshot} "
                "(set TENON_UPMEM_PROVENANCE_DIR or UPIMULATOR_ROOT)"
            )
    try:
        root = ensure_upimulator_root()
    except (OSError, tarfile.TarError) as error:
        return f"uPIMulator root staging failed: {error} (set UPIMULATOR_ROOT)"
    slot = root / "benchmark" / upmem_benchmark_slot() / "dpu" / "CMakeLists.txt"
    if not slot.is_file():
        return (
            f"uPIMulator benchmark slot missing at {slot} "
            "(set UPIMULATOR_ROOT or TENON_UPMEM_BENCHMARK_SLOT)"
        )
    manifest = sdk_build_manifest_sha256(root)
    if manifest != UPMEM_SDK_BUILD_MANIFEST_SHA256:
        return (
            f"UPMEM SDK build under {root / 'sdk' / 'build'} has manifest "
            f"{manifest[:12]}, not the pinned campaign build "
            f"{UPMEM_SDK_BUILD_MANIFEST_SHA256[:12]} (set UPIMULATOR_ROOT)"
        )
    return None


def apu_v1_unavailable_reason() -> str | None:
    """Return a short human-readable reason the APU v1 path can't run,
    or None if all preconditions (ARC toolchain dir, example-gvml
    template dir, GSI PCI sysfs node, GVML SDK headers) are present.
    """
    arc_base = apu_v1_toolchain_base()
    if not arc_base.exists():
        return (
            f"ARC toolchain missing at {arc_base} "
            "(set TENON_APU_V1_TOOLCHAIN_BASE)"
        )
    template = apu_v1_template_dir()
    if not template.exists():
        return (
            f"example-gvml template missing at {template} "
            "(set TENON_APU_V1_TEMPLATE_DIR)"
        )
    pci = apu_v1_pci_node()
    if pci is None or not pci.exists():
        where = pci if pci is not None else f"vendor {_GSI_PCI_VENDOR_ID}"
        return f"GSI device not present at {where} (set TENON_APU_V1_PCI_NODE)"
    # GVML SDK headers: a partial install (eltwise present, logical
    # absent) makes the build fail mid-way; check both canaries up-front
    # so the run path raises SimulatorUnavailable instead of a RuntimeError
    # from `make`.
    from .spmw_apu_v1_build import _gvml_sdk_available

    if not _gvml_sdk_available():
        return (
            f"GVML SDK headers missing under {gvml_include_root()} "
            "(set TENON_APU_V1_GVML_INCLUDE_ROOT)"
        )
    return None
