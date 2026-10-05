# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""External simulator, toolchain and device locations for SPMW backends.

This module is the single owner of every path outside the repository that a
PIM backend reads, and of every availability probe. Each location has an
environment override; defaults are repository-relative, a system install
path, or derived from the user's home directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_SIMULATORS_DIR = Path(__file__).resolve().parents[2] / "simulators"
_DEFAULT_PIMSIM_ROOT = _SIMULATORS_DIR / "PIMSimulator"
_DEFAULT_AIM_ROOT = _SIMULATORS_DIR / "aim_simulator"

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
