# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""External simulator, toolchain and device locations for SPMW backends.

This module is the single owner of every path outside the package that a PIM
backend reads, and of every availability probe. Every location has an
environment override. Defaults are a system install path, the simulator
checkouts of the containing Tenon repository (only when the package runs from
that checkout), or a lab home-directory layout kept as a documented fallback.
The same table is in ``docs/release/INSTALL.md``.

Variables (meaning; default when unset):

``TENON_SIMULATORS_DIR``
    Directory holding the ``PIMSimulator/`` and ``aim_simulator/`` checkouts.
    Default: ``<repo>/experiments/simulators`` when the package runs from the
    Tenon repository checkout, otherwise unset.
``PIMSIMULATOR_ROOT``
    Samsung PIMSimulator tree containing ``pim_driver``.
    Default: ``$TENON_SIMULATORS_DIR/PIMSimulator``.
``AIM_SIMULATOR_ROOT``
    aim_simulator tree (``build/ramulator2``, ``test/example.yaml``).
    Default: ``$TENON_SIMULATORS_DIR/aim_simulator``.
``TENON_AIM_SIM_IMAGE``
    Docker image that runs ``ramulator2``. Default: ``aim-simulator-build``.
``UPIMULATOR_ROOT``
    uPIMulator Go project root (``benchmark/``, ``sdk/``). Default: the
    provenance snapshot staged under ``$TENON_UPMEM_SCRATCH``.
``UPIMULATOR_BIN``
    uPIMulator binary. Default: ``$TENON_UPMEM_PROVENANCE_DIR/uPIMulator``.
``TENON_UPMEM_SDK_IMAGE``
    Docker image holding the UPMEM DPU SDK. Default: ``bongjoonhyun/upimulator``
    (a locally built image; see ``docs/release/simulators.md``).
``TENON_UPMEM_BENCHMARK_SLOT``
    ``benchmark/<slot>`` directory Tenon writes ``task.c`` into.
    Default: ``GENERIC``.
``TENON_UPMEM_SCRATCH``
    Staged roots, run directories and locks. Default: ``<tmpdir>/tenon-upmem``.
``TENON_UPMEM_PROVENANCE_DIR``
    Archived campaign binary and source snapshot.
    Default: ``$TENON_ARTIFACTS/upmem/provenance/upimulator``.
``TENON_ARTIFACTS``
    Tenon evaluation-artifact archive (unpublished).
    Default: ``~/shared/tenon-artifacts`` (authors' lab layout; fallback only).
``TENON_APU_V1_TOOLCHAIN_BASE``
    ARC GNU toolchain root, without ``arc-snps-elf/``. Default:
    ``/usr/local/gsi-apu/13.7.1/ubuntu_20_04/arc_gnu_2021.09-release_elf32_le_linux_no_sdata``.
``TENON_APU_V1_GVML_INCLUDE_ROOT``
    Directory holding ``gsi/libgvml_*.h``. Default: ``/usr/local/include``.
``TENON_APU_V1_TEMPLATE_DIR``
    GSI ``example-gvml`` project (ships with the GSI SDK). Default:
    ``~/shared/accelerator-hub/gsi-apu/example-gvml`` (lab layout; fallback only).
``TENON_APU_V1_PCI_NODE``
    sysfs node of the Gemini-I board. Default: PCI scan for vendor ``0x1e4c``.
``TENON_APU_G2_SDK_ROOT``
    GSI Gemini-II SDK root. Default: ``/opt/gsi``.
``TENON_APU_G2_GSI_TOOL``
    ``gsi_tool`` executable. Default: ``gsi_tool`` on ``PATH``.
``TENON_APU_G2_DEVICE``
    Gemini-II card probed with ``gsi_tool info``. Default: ``apu-00``.

UPMEM has two routes. The *pinned-campaign* route (the default) uses the
archived uPIMulator binary, stages its source snapshot unless
``UPIMULATOR_ROOT`` is set, and requires the DPU SDK object manifest to equal :data:`UPMEM_SDK_BUILD_MANIFEST_SHA256`; it reproduces the
paper cycles. Setting both ``UPIMULATOR_ROOT`` and ``UPIMULATOR_BIN`` selects
the *user-build* route: nothing is staged, the manifest is not checked, and
:func:`upmem_sdk_build_identity` labels the run ``user-build:<manifest12>``.

The Gemini-II project templates (``allo/pim/templates/apu_g2``) compile in
``/opt/gsi`` and device 0, so ``TENON_APU_G2_SDK_ROOT`` and
``TENON_APU_G2_DEVICE`` values other than the defaults are rejected by
:func:`apu_g2_unsupported_override_reason` rather than probing one location
and building or running against another.
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

# ``experiments/allo/allo/spmw_simenv.py`` -> ``experiments/simulators``; only
# meaningful when the package runs from the Tenon repository checkout.
_CHECKOUT_SIMULATORS_DIR = Path(__file__).resolve().parents[2] / "simulators"
_PIMSIM_DIRNAME = "PIMSimulator"
_AIM_DIRNAME = "aim_simulator"
_DEFAULT_AIM_SIM_IMAGE = "aim-simulator-build"
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
# The Gemini-II CMake templates and host programs hard-code these two values.
_DEFAULT_APU_G2_SDK_ROOT = "/opt/gsi"
_DEFAULT_APU_G2_DEVICE = "apu-00"
_APU_G2_REQUIRED_SDK_FILES = (
    Path("include/64vl/g2_64vl/g2_64vl_types.h"),
    Path("g2/vector_core/lib/libg2_64vl.a"),
    Path("g2/vector_core/bin/g2_md_update.py"),
    Path("lib/libg2_device_core.a"),
)
_GSI_PCI_VENDOR_ID = "0x1e4c"
_PCI_DEVICES_DIR = Path("/sys/bus/pci/devices")

_DOCKER_PROBE_TIMEOUT_S = 10


def simulators_dir() -> Path | None:
    """Directory holding the simulator checkouts, or None when unknown."""
    override = os.environ.get("TENON_SIMULATORS_DIR")
    if override is not None:
        return Path(override)
    if _CHECKOUT_SIMULATORS_DIR.is_dir():
        return _CHECKOUT_SIMULATORS_DIR
    return None


def _simulator_root(variable: str, dirname: str) -> Path | None:
    override = os.environ.get(variable)
    if override is not None:
        return Path(override)
    base = simulators_dir()
    return None if base is None else base / dirname


def pimsim_root() -> Path | None:
    """Samsung PIMSimulator tree, or None when no location is configured."""
    return _simulator_root("PIMSIMULATOR_ROOT", _PIMSIM_DIRNAME)


def aim_root() -> Path | None:
    """SK hynix aim_simulator tree, or None when no location is configured."""
    return _simulator_root("AIM_SIMULATOR_ROOT", _AIM_DIRNAME)


def aim_sim_image() -> str:
    """Docker image whose ``/work`` mount runs ``./build/ramulator2``."""
    return os.environ.get("TENON_AIM_SIM_IMAGE", _DEFAULT_AIM_SIM_IMAGE)


def tenon_artifacts_root() -> Path:
    """Tenon evaluation-artifact archive (paper provenance; not published).

    The default is the authors' lab layout, kept only as a fallback; set
    ``TENON_ARTIFACTS`` elsewhere.
    """
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


def upmem_user_build() -> bool:
    """True on the user-build route: ``UPIMULATOR_ROOT`` and ``UPIMULATOR_BIN`` set.

    The user-build route runs a user's own uPIMulator + DPU SDK build: the
    provenance snapshot is not staged and the SDK object manifest is not
    checked against the campaign pin.
    """
    return "UPIMULATOR_ROOT" in os.environ and "UPIMULATOR_BIN" in os.environ


UPMEM_SDK_BUILD_PINNED = "pinned-campaign"


def upmem_sdk_build_identity(manifest_sha256: str) -> str:
    """Run-record label of the DPU SDK build whose manifest is ``manifest_sha256``.

    ``"pinned-campaign"`` on the pinned route, where any other manifest raises
    ``RuntimeError`` (the paper cycles depend on that exact build);
    ``"user-build:<first 12 hex digits>"`` on the :func:`upmem_user_build`
    route. Pass the value of :func:`sdk_build_manifest_sha256` computed after
    the DPU build.
    """
    if upmem_user_build():
        return f"user-build:{manifest_sha256[:12]}"
    if manifest_sha256 != UPMEM_SDK_BUILD_MANIFEST_SHA256:
        raise RuntimeError(
            f"UPMEM SDK build manifest {manifest_sha256[:12]} drifted from the "
            f"pinned campaign build {UPMEM_SDK_BUILD_MANIFEST_SHA256[:12]}; set "
            "UPIMULATOR_ROOT and UPIMULATOR_BIN to run a user build instead"
        )
    return UPMEM_SDK_BUILD_PINNED


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
    """GSI ``example-gvml`` project that APU v1 builds copy their harness from.

    The default is the authors' lab layout, kept only as a fallback; the GSI
    example project ships with the GSI SDK and must be provided by the user
    through ``TENON_APU_V1_TEMPLATE_DIR``.
    """
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


def apu_g2_sdk_root() -> Path:
    """GSI Gemini-II SDK root (headers, ARC libraries, ``g2_md_update.py``)."""
    return Path(os.environ.get("TENON_APU_G2_SDK_ROOT", _DEFAULT_APU_G2_SDK_ROOT))


def apu_g2_required_sdk_paths() -> tuple[Path, ...]:
    root = apu_g2_sdk_root()
    return tuple(root / relative for relative in _APU_G2_REQUIRED_SDK_FILES)


def apu_g2_gsi_tool() -> str | None:
    """``gsi_tool`` executable used to check card ownership, or None."""
    override = os.environ.get("TENON_APU_G2_GSI_TOOL")
    if override is not None:
        return override
    return shutil.which("gsi_tool")


def apu_g2_device() -> str:
    """Gemini-II card name passed to ``gsi_tool info``."""
    return os.environ.get("TENON_APU_G2_DEVICE", _DEFAULT_APU_G2_DEVICE)


def apu_g2_unsupported_override_reason() -> str | None:
    """Reason a Gemini-II override cannot be honoured by the shipped templates.

    The CMake templates under ``allo/pim/templates/apu_g2`` compile in
    ``/opt/gsi`` and the host programs open ``DeviceConfig{0, 0}``; probing a
    different SDK root or card than the one built and run against would
    validate the wrong installation.
    """
    if Path(apu_g2_sdk_root()) != Path(_DEFAULT_APU_G2_SDK_ROOT):
        return (
            f"TENON_APU_G2_SDK_ROOT={apu_g2_sdk_root()} is not supported yet: "
            f"the Gemini-II project templates hard-code {_DEFAULT_APU_G2_SDK_ROOT}"
        )
    if apu_g2_device() != _DEFAULT_APU_G2_DEVICE:
        return (
            f"TENON_APU_G2_DEVICE={apu_g2_device()} is not supported yet: the "
            f"Gemini-II host programs open device 0 ({_DEFAULT_APU_G2_DEVICE})"
        )
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
    root = pimsim_root()
    if root is None:
        return (
            "PIMSimulator location is not configured "
            "(set PIMSIMULATOR_ROOT or TENON_SIMULATORS_DIR)"
        )
    driver = root / "pim_driver"
    if not driver.exists():
        return f"pim_driver not found at {driver} (set PIMSIMULATOR_ROOT)"
    if not os.access(driver, os.X_OK):
        return f"pim_driver at {driver} is not executable"
    return None


def aim_unavailable_reason() -> str | None:
    reason = docker_image_unavailable_reason(aim_sim_image())
    if reason is not None:
        return reason
    root = aim_root()
    if root is None:
        return (
            "aim_simulator location is not configured "
            "(set AIM_SIMULATOR_ROOT or TENON_SIMULATORS_DIR)"
        )
    yaml_cfg = root / "test" / "example.yaml"
    if not yaml_cfg.exists():
        return f"AiM config missing at {yaml_cfg} (set AIM_SIMULATOR_ROOT)"
    return None


def upmem_unavailable_reason() -> str | None:
    binary = upimulator_bin()
    if not binary.exists():
        return (
            f"uPIMulator binary not found at {binary} (set UPIMULATOR_ROOT and "
            "UPIMULATOR_BIN for a user build, or TENON_ARTIFACTS / "
            "TENON_UPMEM_PROVENANCE_DIR for the pinned campaign build)"
        )
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
    if upmem_user_build():
        return None
    manifest = sdk_build_manifest_sha256(root)
    if manifest != UPMEM_SDK_BUILD_MANIFEST_SHA256:
        return (
            f"UPMEM SDK build under {root / 'sdk' / 'build'} has manifest "
            f"{manifest[:12]}, not the pinned campaign build "
            f"{UPMEM_SDK_BUILD_MANIFEST_SHA256[:12]} (set UPIMULATOR_ROOT and "
            "UPIMULATOR_BIN to run a user build)"
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
