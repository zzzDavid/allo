# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free coverage for the shared APU v1 build-mode selection."""

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


# --------------------------------------------------------------------------- #
# V1-08: only the needed GSI SDK files are copied, each with its origin
# --------------------------------------------------------------------------- #

import os
import stat

import pytest


def _fake_example_gvml(root):
    for relative in apu_v1_build._TEMPLATE_FILES + (
        "Common/arc.tcf",
        "Common/arc_module_example.lcf",
        "Common/memory.x",
        "Common/.clang-format",
        "device.c",
        "host.c",
        "README.md",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"gsi sdk file {relative}\n")
        if relative.endswith(".sh"):
            path.chmod(0o755)
    return root


def test_template_staging_copies_only_the_build_inputs_with_their_origin(
    monkeypatch, tmp_path
):
    template = _fake_example_gvml(tmp_path / "example-gvml")
    monkeypatch.setenv("TENON_APU_V1_TEMPLATE_DIR", str(template))
    project = tmp_path / "project"
    project.mkdir()
    copied = apu_v1_build.stage_apu_v1_template(project)

    staged = sorted(
        str(path.relative_to(project)) for path in project.rglob("*") if path.is_file()
    )
    assert staged == sorted((*apu_v1_build._TEMPLATE_FILES, "THIRD_PARTY.txt"))
    assert [relative for relative, _ in copied] == list(apu_v1_build._TEMPLATE_FILES)
    for script in (
        "fix-gvml-lib-for-the-gnu-toolchain-for-arc.sh",
        "generate-gnu-linker-command-file-for-cores-1-2-3.sh",
    ):
        assert os.stat(project / "Common" / script).st_mode & stat.S_IXUSR

    notice = (project / "THIRD_PARTY.txt").read_text()
    assert "TENON_APU_V1_TEMPLATE_DIR" in notice
    assert "does not redistribute" in notice
    assert f"source: {template.resolve()}" in notice
    for relative, digest in copied:
        assert f"{relative}  sha256={digest}" in notice
    assert "TENON_APU_V1_TEMPLATE_DIR" in apu_v1_build.__doc__
    assert "does not redistribute" in apu_v1_build.__doc__


def test_missing_template_names_the_variable_and_the_sdk(monkeypatch, tmp_path):
    monkeypatch.setenv("TENON_APU_V1_TEMPLATE_DIR", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError) as error:
        apu_v1_build.stage_apu_v1_template(tmp_path / "project")
    message = str(error.value)
    assert "TENON_APU_V1_TEMPLATE_DIR" in message
    assert "GSI SDK example-gvml" in message
    assert str(tmp_path / "absent") in message

    template = _fake_example_gvml(tmp_path / "partial")
    (template / "Common" / "memory.x.template").unlink()
    monkeypatch.setenv("TENON_APU_V1_TEMPLATE_DIR", str(template))
    with pytest.raises(FileNotFoundError, match="Common/memory.x.template is missing"):
        apu_v1_build.stage_apu_v1_template(tmp_path / "project")


def test_installed_example_gvml_provides_every_build_input():
    template = apu_v1_build._template_dir()
    if not template.is_dir():
        pytest.skip(f"GSI example-gvml template not installed at {template}")
    for relative in apu_v1_build._TEMPLATE_FILES:
        assert (template / relative).is_file(), relative
