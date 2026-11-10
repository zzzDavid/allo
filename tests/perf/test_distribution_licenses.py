# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The third-party license texts the distributions carry (release ledger
P-25, round 88): the wheel bundles the LLVM/MLIR Python bindings and
extension modules (Apache-2.0 with LLVM Exceptions) with nanobind embedded
(BSD 3-Clause), the sdist carries two dialect sources under the LLVM
header and the uPIMulator patch (MIT), and both carry the bundled Samsung
PIMSimulator sources (Samsung's notice; DRAMSim2 parts BSD 3-Clause; P-04).
The texts live under ``THIRD_PARTY_LICENSES/``, are inventoried in
``THIRD_PARTY_NOTICES.md`` and are packaged into both forms through
``setup.py`` and ``MANIFEST.in``."""

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


EXPECTED_TEXTS = {
    "LLVM-LICENSE.TXT": (
        "Version 2.0, January 2004",
        "---- LLVM Exceptions to the Apache 2.0 License ----",
    ),
    "NANOBIND-LICENSE.txt": (
        "Wenzel Jakob",
        "Redistributions of source code must retain",
    ),
    "UPIMULATOR-LICENSE.txt": ("Permission is hereby granted, free of charge",),
    "ROBIN-MAP-LICENSE.txt": ("MIT License", "Thibaut Goetghebuer-Planchon"),
    "XXHASH-LICENSE.txt": ("Yann Collet", "BSD 2-Clause License"),
    "CONVERTUTF-UNICODE-NOTICE.txt": (
        "1991-2015 Unicode, Inc.",
        "this copyright and permission notice appear",
    ),
    "LLVM-REGEX-COPYRIGHT.txt": ("Henry Spencer",),
    "BLAKE3-LICENSE.txt": ("CC0 1.0", "Apache License"),
    "MD5-NOTICE.txt": ("Alexander Peslyak", "placed in the public domain"),
    "UPMEM-DPU-RT-LICENSE.txt": (
        "Copyright (c) 2020, UPMEM",
        "Redistribution and use in source and binary forms",
    ),
    "PIMSIMULATOR-LICENSE.txt": (
        "Copyright (C) 2021 Samsung Electronics Co. LTD",
        "express written permission of Samsung Electronics",
        "non-commercial, personal or academic, research purpose",
    ),
    "DRAMSIM2-LICENSE.txt": (
        "Copyright (c) 2010-2011, Elliott Cooper-Balis",
        "University of Maryland",
        "Redistribution and use in source and binary forms",
    ),
}


def test_bundled_texts_are_the_upstream_licenses():
    present = sorted(p.name for p in LICENSES.iterdir())
    assert present == sorted(EXPECTED_TEXTS), present
    for name, phrases in EXPECTED_TEXTS.items():
        text = (LICENSES / name).read_text()
        for phrase in phrases:
            assert phrase in text, (name, phrase)
    assert _sha256(LICENSES / "UPIMULATOR-LICENSE.txt") == UPIMULATOR_UPSTREAM_SHA256


def test_notices_inventory_names_every_bundled_text_with_its_checksum():
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    for path in sorted(LICENSES.iterdir()):
        assert f"THIRD_PARTY_LICENSES/{path.name}" in notices, path.name
        assert _sha256(path) in notices, f"checksum of {path.name} not recorded"
    for component in (
        "LLVM / MLIR",
        "xxHash",
        "Unicode, Inc.",
        "Henry Spencer",
        "BLAKE3",
        "MD5",
        "nanobind",
        "robin-map",
        "uPIMulator",
        "UPMEM DPU runtime",
    ):
        assert component in notices, component
    # nanobind has two origins; neither is presented as the only one.
    assert "2.10.2" in notices and "2.13.0" in notices
    # The UPMEM runtime hunks are distinguished from the MIT components.
    assert "stdoutBufferState.c" in notices
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
    assert len(present) == len(EXPECTED_TEXTS) == 12
