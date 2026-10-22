# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run paths raise `SimulatorUnavailable` instead of returning cycles=None."""

import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from allo import spmw_apu_v2, spmw_samsung, spmw_simenv
from allo.spmw_codegen import (
    Compiled,
    SimulatorUnavailable,
    simulator_unavailable_reason,
)


def test_unknown_target_always_has_a_reason():
    assert simulator_unavailable_reason("not_a_target") is not None


def test_samsung_run_raises_when_pim_driver_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(spmw_simenv, "pimsim_root", lambda: tmp_path)
    reason = simulator_unavailable_reason("samsung_hbm_pim")
    assert reason is not None and "pim_driver" in reason

    with pytest.raises(SimulatorUnavailable) as failure:
        spmw_samsung._run_samsung(SimpleNamespace())
    assert failure.value.target_name == "samsung_hbm_pim"
    assert failure.value.reason == reason


def test_docker_timeout_is_reported_separately_from_missing_image(monkeypatch):
    monkeypatch.setattr(spmw_simenv, "docker_available", lambda: True)

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=10)

    monkeypatch.setattr(spmw_simenv.subprocess, "run", timeout)
    assert simulator_unavailable_reason("apu_v2") == (
        "docker daemon did not answer within 10 s"
    )

    def missing(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=(), returncode=1, stdout=b"", stderr=b"No such image"
        )

    monkeypatch.setattr(spmw_simenv.subprocess, "run", missing)
    assert simulator_unavailable_reason("apu_v2") == (
        "docker image 'gsi-g2-l1sim' is missing"
    )
    with pytest.raises(SimulatorUnavailable):
        spmw_apu_v2._run_apu_v2(SimpleNamespace(cmds=[]))


def test_run_without_a_hook_raises_not_implemented():
    compiled = Compiled(SimpleNamespace(name="no_runner"), None, [], None)
    with pytest.raises(NotImplementedError, match="no run hook"):
        compiled.run()


def test_promotion_lifecycle_is_gone():
    import allo
    import allo.pim.schedule_search as schedule_search

    with pytest.raises(ImportError):
        __import__("allo.pim.schedule_promotion")
    assert not hasattr(allo, "PromotionEvidence")
    assert not hasattr(schedule_search, "guarded_schedule_activation")


def test_apu_v1_pci_node_scans_vendor_id(monkeypatch, tmp_path):
    other = tmp_path / "0000:00:01.0"
    board = tmp_path / "0000:41:00.0"
    for device, vendor in ((other, "0x8086"), (board, "0x1e4c")):
        device.mkdir()
        (device / "vendor").write_text(vendor + "\n")
    monkeypatch.delenv("TENON_APU_V1_PCI_NODE", raising=False)
    monkeypatch.setattr(spmw_simenv, "_PCI_DEVICES_DIR", tmp_path)
    assert spmw_simenv.apu_v1_pci_node() == board

    (board / "vendor").write_text("0x10de\n")
    assert spmw_simenv.apu_v1_pci_node() is None

    monkeypatch.setenv("TENON_APU_V1_PCI_NODE", str(other))
    assert spmw_simenv.apu_v1_pci_node() == other


def test_apu_v1_locations_have_env_overrides(monkeypatch, tmp_path):
    from allo import spmw_apu_v1_build

    monkeypatch.setenv("TENON_APU_V1_TEMPLATE_DIR", str(tmp_path / "tpl"))
    monkeypatch.setenv("TENON_APU_V1_TOOLCHAIN_BASE", str(tmp_path / "arc"))
    assert spmw_apu_v1_build._template_dir() == tmp_path / "tpl"
    assert spmw_apu_v1_build._toolchain_base() == f"{tmp_path / 'arc'}/arc-snps-elf/"

    monkeypatch.delenv("TENON_APU_V1_TEMPLATE_DIR")
    assert spmw_simenv.apu_v1_template_dir().is_relative_to(pathlib.Path.home())
