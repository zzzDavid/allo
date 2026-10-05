# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""`import allo` must not load the PIM compiler stack (spec 005 P4)."""

import subprocess
import sys

import allo

_EAGER_FORBIDDEN = (
    "allo.pim",
    "allo.compiler",
    "allo.spmw_codegen",
    "allo.spmw_autoschedule",
    "allo.perf",
)


def test_import_allo_loads_no_pim_modules():
    probe = (
        "import sys, allo\n"
        f"forbidden = {_EAGER_FORBIDDEN!r}\n"
        "print('\\n'.join(sorted(m for m in sys.modules "
        "if any(m == f or m.startswith(f + '.') for f in forbidden))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_every_lazy_export_resolves():
    for name in allo._LAZY_EXPORTS:
        assert getattr(allo, name) is not None, name


def test_lazy_exports_are_listed():
    assert "compile" in dir(allo)
    assert "CompiledWorkload" in dir(allo)


def test_compile_is_the_compiler_entry_point():
    import allo.compiler

    assert allo.compile is allo.compiler.compile
