# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free coverage for the shared APU v1 build-mode selection."""

import pytest

import allo.spmw_apu_v1_build as apu_v1_build
from allo.pim.apu_v1_vector_runtime import _build_manifest


def test_apu_v1_build_defaults_to_release(monkeypatch, tmp_path):
    monkeypatch.delenv("TENON_APU_V1_BUILD_MODE", raising=False)

    build = apu_v1_build._apu_v1_build_config()

    assert build.mode == "release"
    assert build.make_command == ("make", "mode=release")
    assert build.binary_path(tmp_path, "kernel") == (
        tmp_path / "build" / "release" / "kernel"
    )
    assert "mode ?= release\n" in apu_v1_build._emit_makefile("kernel")
    assert _build_manifest("kernel")["build"] == ["make", "mode=release"]
    assert _build_manifest("kernel")["binary"] == "build/release/kernel"


def test_apu_v1_debug_build_is_an_explicit_consistent_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TENON_APU_V1_BUILD_MODE", "debug")

    build = apu_v1_build._apu_v1_build_config()

    assert build.make_command == ("make", "mode=debug")
    assert build.binary_path(tmp_path, "kernel") == (
        tmp_path / "build" / "debug" / "kernel"
    )
    assert "mode ?= debug\n" in apu_v1_build._emit_makefile("kernel")
    assert _build_manifest("kernel")["binary"] == "build/debug/kernel"


@pytest.mark.parametrize("mode", ["", "Release", "profile", " release"])
def test_apu_v1_build_rejects_unknown_modes(monkeypatch, mode):
    monkeypatch.setenv("TENON_APU_V1_BUILD_MODE", mode)

    with pytest.raises(ValueError, match="TENON_APU_V1_BUILD_MODE must be one of"):
        apu_v1_build._apu_v1_build_config()
