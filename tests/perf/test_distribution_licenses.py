# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The third-party license texts the distributions carry (release ledger
P-25, round 88): the wheel bundles the LLVM/MLIR Python bindings and
extension modules (Apache-2.0 with LLVM Exceptions) with nanobind embedded
(BSD 3-Clause), and the sdist carries two dialect sources under the LLVM
header and the uPIMulator patch (MIT).  The texts live under
``THIRD_PARTY_LICENSES/``, are inventoried in ``THIRD_PARTY_NOTICES.md`` and
are packaged into both forms through ``setup.py`` and ``MANIFEST.in``."""

import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LICENSES = ROOT / "THIRD_PARTY_LICENSES"
UPIMULATOR_UPSTREAM_SHA256 = (
    "73de95273ba877b7c8b1037d8c478179fd58ff97169da6683bc10d88b79fd16d"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundled_texts_are_the_upstream_licenses():
    llvm = (LICENSES / "LLVM-LICENSE.TXT").read_text()
    assert "Apache License" in llvm and "Version 2.0, January 2004" in llvm
    assert "---- LLVM Exceptions to the Apache 2.0 License ----" in llvm
    nanobind = (LICENSES / "NANOBIND-LICENSE.txt").read_text()
    assert "Wenzel Jakob" in nanobind
    assert "Redistributions of source code must retain" in nanobind
    assert "Neither the name of the copyright holder" in nanobind
    upimulator = LICENSES / "UPIMULATOR-LICENSE.txt"
    assert "Permission is hereby granted, free of charge" in upimulator.read_text()
    assert _sha256(upimulator) == UPIMULATOR_UPSTREAM_SHA256


def test_notices_inventory_names_every_bundled_text_with_its_checksum():
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    for path in sorted(LICENSES.iterdir()):
        assert f"THIRD_PARTY_LICENSES/{path.name}" in notices, path.name
        assert _sha256(path) in notices, f"checksum of {path.name} not recorded"
    for component in ("LLVM / MLIR", "nanobind 2.10.2", "uPIMulator"):
        assert component in notices
    # What is deliberately not shipped is stated too.
    assert "Samsung PIMSimulator" in notices and "GSI Gemini-I SDK" in notices


def test_setup_and_manifest_package_the_texts_into_both_forms():
    setup = (ROOT / "setup.py").read_text()
    block = re.search(r"license_files=\[(.*?)\]", setup, re.S)
    assert block, "setup.py declares no license_files"
    assert '"LICENSE"' in block.group(1)
    assert '"THIRD_PARTY_NOTICES.md"' in block.group(1)
    assert '"THIRD_PARTY_LICENSES/*"' in block.group(1)
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "include THIRD_PARTY_NOTICES.md" in manifest
    assert "graft THIRD_PARTY_LICENSES" in manifest
    # The retained upstream notices stay in the dialect sources.
    for name in ("mlir/lib/Dialect/AlloOps.cpp", "mlir/lib/Dialect/AlloDialect.cpp"):
        assert "Apache-2.0 WITH LLVM-exception" in (ROOT / name).read_text(), name


def test_every_bundled_text_is_tracked_by_git():
    """The first round 88 build shipped only the LLVM text: a ``*.txt`` ignore
    rule had kept the nanobind and uPIMulator files out of the commit, and
    ``git archive`` packages the commit.  Every file in the directory must be
    tracked, so a dropped text fails here before a build."""
    tracked = subprocess.run(
        ["git", "ls-files", "THIRD_PARTY_LICENSES"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    present = sorted(f"THIRD_PARTY_LICENSES/{p.name}" for p in LICENSES.iterdir())
    assert sorted(tracked) == present, f"tracked {sorted(tracked)} vs present {present}"
    assert len(present) == 3
