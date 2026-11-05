# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The UPMEM DPU SDK executor: the Docker image or a host-installed SDK run
through a user-namespace bind at the image's path (``TENON_UPMEM_SDK_HOME``).
Pure command-construction and availability-gate checks; nothing is built."""

from pathlib import Path

import pytest

from allo import spmw_simenv


def test_docker_build_command_is_unchanged(monkeypatch):
    monkeypatch.delenv("TENON_UPMEM_SDK_HOME", raising=False)
    monkeypatch.setenv("TENON_UPMEM_SDK_IMAGE", "tenon-upimulator-sdk:test")
    command = spmw_simenv.upmem_sdk_build_command(
        Path("/scratch/root"), ["--num_dpus", "1", "--benchmark", "GENERIC"]
    )
    assert command == [
        "docker",
        "run",
        "--rm",
        "-v",
        "/scratch/root:/root/uPIMulator",
        "tenon-upimulator-sdk:test",
        "python3",
        "/root/uPIMulator/benchmark/build.py",
        "--num_dpus",
        "1",
        "--benchmark",
        "GENERIC",
    ]
    assert spmw_simenv.upmem_sdk_executor_label() == "docker:tenon-upimulator-sdk:test"


def test_native_build_command_binds_the_root_at_the_images_path(monkeypatch, tmp_path):
    sdk = tmp_path / "upmem-2023.1.0-Linux-x86_64"
    (sdk / "bin").mkdir(parents=True)
    monkeypatch.setenv("TENON_UPMEM_SDK_HOME", str(sdk))
    command = spmw_simenv.upmem_sdk_build_command(
        Path("/scratch/root"), ["--num_tasklets", "12"]
    )
    assert command[:4] == ["unshare", "-Urm", "bash", "-c"]
    script = command[4]
    # the same script at the same path as the Docker route, the root bound there
    assert "mount -t tmpfs none /root" in script
    assert 'mount --bind "$1" /root/uPIMulator' in script
    assert "python3 /root/uPIMulator/benchmark/build.py" in script
    assert 'PATH="$2/bin:$PATH"' in script and 'UPMEM_HOME="$2"' in script
    assert command[5:] == [
        "tenon-upmem-sdk",
        "/scratch/root",
        str(sdk),
        "--num_tasklets",
        "12",
    ]
    assert spmw_simenv.upmem_sdk_executor_label() == f"native-sdk:{sdk}"


def test_native_executor_gate_names_the_missing_piece(monkeypatch, tmp_path):
    sdk = tmp_path / "sdk"
    (sdk / "bin").mkdir(parents=True)
    reason = spmw_simenv.upmem_sdk_native_unavailable_reason(sdk)
    assert reason is not None and "compiler not found" in reason
    compiler = sdk / "bin" / spmw_simenv.UPMEM_SDK_COMPILER
    compiler.write_text("#!/bin/sh\n")
    reason = spmw_simenv.upmem_sdk_native_unavailable_reason(sdk)
    assert reason is not None and "not executable" in reason
    compiler.chmod(0o755)
    monkeypatch.setattr(
        spmw_simenv.shutil,
        "which",
        lambda tool: None if tool == "ninja" else "/usr/bin/" + tool,
    )
    reason = spmw_simenv.upmem_sdk_native_unavailable_reason(sdk)
    assert reason is not None and reason.startswith("ninja is not on PATH")


def test_availability_prefers_the_native_executor_when_configured(
    monkeypatch, tmp_path
):
    """With ``TENON_UPMEM_SDK_HOME`` set the Docker image is not consulted;
    an unusable native SDK is reported as such."""
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    monkeypatch.setenv("TENON_UPMEM_SDK_HOME", str(sdk))
    monkeypatch.setattr(spmw_simenv, "upimulator_bin", lambda: tmp_path / "uPIMulator")
    (tmp_path / "uPIMulator").write_text("")
    (tmp_path / "uPIMulator").chmod(0o755)
    called = []
    monkeypatch.setattr(
        spmw_simenv,
        "docker_image_unavailable_reason",
        lambda name: called.append(name) or "docker daemon is not running",
    )
    reason = spmw_simenv.upmem_unavailable_reason()
    assert (
        reason is not None
        and "compiler not found" in reason
        and "TENON_UPMEM_SDK_HOME" in reason
    )
    assert called == []
