# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bundled Samsung PIMSimulator: locate, verify, assemble and build.

Tenon ships the sources its Samsung HBM-PIM simulator route needs
(``simulators/PIMSimulator/`` of the repository, ``allo/pim/simulators/
PIMSimulator/`` of an installed wheel; ``NOTICE.md`` there states the terms):
a pristine copy of upstream SAITPublic/PIMSimulator at the pinned commit
(``upstream/``, hashed in ``UPSTREAM_MANIFEST.sha256``), five patches to
upstream files (``patches/``), Tenon's conductor sources and build script,
and ``apply.sh``, which applies the patches, installs the Tenon files and
runs SCons.

``build`` copies ``upstream/`` into a build root, runs ``apply.sh`` on it and
checks the resulting ``pim_driver --describe``.  The default root is
:func:`allo.spmw_simenv.default_pimsim_root` (``$TENON_HOME/simulators/
PIMSimulator``), which :func:`allo.spmw_simenv.pimsim_root` discovers
without any further configuration; ``PIMSIMULATOR_ROOT`` and
``TENON_SIMULATORS_DIR`` keep precedence over it.

Command line::

    python -m allo.pim.samsung_simulator build [--root DIR] [--jobs N]
                                               [--skip-build] [--scons EXE]
    python -m allo.pim.samsung_simulator verify
    python -m allo.pim.samsung_simulator locate
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from allo import spmw_simenv

UPSTREAM_URL = "https://github.com/SAITPublic/PIMSimulator"
UPSTREAM_COMMIT = "3703d1f19c8f027360cc33a3243eb271e3bb6898"
MANIFEST_NAME = "UPSTREAM_MANIFEST.sha256"
PATCHES = (
    "PIMKernel.h.patch",
    "PIMKernel.cpp.patch",
    "PIMCmdGen.h.patch",
    "PIMCmdGen.cpp.patch",
    "Burst.h.patch",
)
OVERLAY = (
    "Sconstruct",
    "src/pim_driver.cc",
    "src/tests/TenonConductor.h",
    "src/tests/TenonConductor.inc",
    "src/tests/TemporalReduce.inc",
)
# Files pim_driver reads at run time, relative to the build root (the
# conductor runs with the root as its working directory).
RUNTIME_FILES = ("ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini")
# Upstream build outputs that a copy of the recipe must never carry.
BUILD_OUTPUTS = ("bin", "libdramsim", "pim_driver", "sim", "*.o", "*.a", "*.os")

_RECIPE_SUBPATH = Path("simulators") / "PIMSimulator"
# Installed wheel: allo/pim/simulators/PIMSimulator (copied by setup.py's
# BuildPy); checkout or editable install: <repo>/simulators/PIMSimulator.
_RECIPE_CANDIDATES = (
    Path(__file__).resolve().parent / _RECIPE_SUBPATH,
    Path(__file__).resolve().parents[2] / _RECIPE_SUBPATH,
)


class BundleError(RuntimeError):
    """The bundled recipe is missing, incomplete or altered, or a build step
    failed; the message says which and what to do."""


def recipe_dir() -> Path:
    """The bundled recipe directory (``TENON_PIMSIM_RECIPE`` overrides)."""
    override = os.environ.get("TENON_PIMSIM_RECIPE")
    candidates = [Path(override).expanduser()] if override else list(_RECIPE_CANDIDATES)
    for candidate in candidates:
        if (candidate / "apply.sh").is_file() and (candidate / "upstream").is_dir():
            return candidate
    looked = ", ".join(str(c) for c in candidates)
    raise BundleError(
        "the bundled PIMSimulator recipe was not found (looked in: "
        f"{looked}); reinstall the package, or set TENON_PIMSIM_RECIPE to a "
        "copy of simulators/PIMSimulator"
    )


def read_manifest(recipe: Path) -> dict[str, str]:
    """``relative path -> sha256`` of the pristine upstream files."""
    manifest = recipe / MANIFEST_NAME
    if not manifest.is_file():
        raise BundleError(f"{manifest} is missing; the recipe copy is incomplete")
    entries: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        if len(digest) != 64 or not rel:
            raise BundleError(f"{manifest}: malformed line {line!r}")
        entries[rel] = digest
    if not entries:
        raise BundleError(f"{manifest} lists no files")
    return entries


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_upstream(recipe: Path, tree: Path | None = None) -> list[str]:
    """Problems with ``tree`` (default: the recipe's ``upstream/``) against
    the manifest: missing files, files whose content differs, and files the
    manifest does not list.  Empty when the tree is the pristine upstream."""
    tree = recipe / "upstream" if tree is None else tree
    expected = read_manifest(recipe)
    problems = []
    for rel, digest in sorted(expected.items()):
        path = tree / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif _sha256(path) != digest:
            problems.append(f"differs from upstream {UPSTREAM_COMMIT[:7]}: {rel}")
    present = {
        p.relative_to(tree).as_posix()
        for p in tree.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    for rel in sorted(present - set(expected)):
        problems.append(f"not in the manifest: {rel}")
    return problems


def verify_recipe(recipe: Path) -> list[str]:
    """Problems with the recipe copy itself: the upstream tree against the
    manifest, and the presence of the patches, overlay files and notice."""
    problems = verify_upstream(recipe)
    for rel in [f"patches/{p}" for p in PATCHES] + list(OVERLAY) + ["NOTICE.md"]:
        if not (recipe / rel).is_file():
            problems.append(f"missing: {rel}")
    for rel in RUNTIME_FILES:
        if not (recipe / "upstream" / rel).is_file():
            problems.append(f"missing runtime file: upstream/{rel}")
    return problems


def assemble(recipe: Path, root: Path) -> Path:
    """Copy the pristine upstream sources into ``root`` (created if needed;
    existing files are overwritten, so a rebuild starts from upstream again
    and ``apply.sh`` re-applies the patches).  Returns ``root``."""
    problems = verify_recipe(recipe)
    if problems:
        shown = "\n  ".join(problems[:20])
        more = "" if len(problems) <= 20 else f"\n  ... {len(problems) - 20} more"
        raise BundleError(
            f"the bundled recipe at {recipe} does not match the pinned upstream "
            f"{UPSTREAM_COMMIT} ({len(problems)} problem(s)):\n  {shown}{more}"
        )
    root = Path(root).expanduser().resolve()
    if root == recipe.resolve() or recipe.resolve() in root.parents:
        raise BundleError(f"the build root {root} must not be inside the recipe")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        recipe / "upstream",
        root,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return root


def _scons_command(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("TENON_SCONS"):
        return os.environ["TENON_SCONS"]
    if shutil.which("scons"):
        return "scons"
    if importlib.util.find_spec("SCons") is None:  # pip's scons package
        raise BundleError(
            "SCons is not available: install it (python -m pip install scons, "
            "or your distribution's scons package) or pass --scons EXE"
        )
    return f"{sys.executable} -m SCons"


def _check_toolchain(skip_build: bool) -> None:
    if shutil.which("patch") is None:
        raise BundleError("patch(1) is not installed; the recipe applies unified diffs")
    if not skip_build and shutil.which("g++") is None and shutil.which("c++") is None:
        raise BundleError("no C++ compiler (g++ or c++) on PATH; pim_driver is C++14")


def apply_and_build(
    recipe: Path,
    root: Path,
    *,
    jobs: int | None = None,
    skip_build: bool = False,
    scons: str | None = None,
    log=None,
) -> str:
    """Run ``apply.sh`` on ``root``: patches, Tenon files and, unless
    ``skip_build``, the SCons build and ``pim_driver --describe``.  Returns
    the combined output; raises :class:`BundleError` on failure."""
    _check_toolchain(skip_build)
    env = dict(os.environ)
    env["TENON_SKIP_BUILD"] = "1" if skip_build else "0"
    if not skip_build:
        env["TENON_SCONS"] = _scons_command(scons)
    if jobs:
        env["TENON_BUILD_JOBS"] = str(int(jobs))
    env.pop("TENON_PIMSIM_PATCHES", None)
    proc = subprocess.run(
        ["bash", str(recipe / "apply.sh"), str(root), str(recipe / "patches")],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    output = proc.stdout + proc.stderr
    if log is not None:
        log.write(output)
    if proc.returncode != 0:
        raise BundleError(
            f"apply.sh exited with {proc.returncode} building {root}:\n{output[-4000:]}"
        )
    return output


def describe(root: Path) -> str:
    """The ``PIM_GEOMETRY`` line of ``pim_driver --describe`` at ``root``."""
    driver = Path(root) / "pim_driver"
    if not driver.is_file() or not os.access(driver, os.X_OK):
        raise BundleError(f"no executable pim_driver at {driver}")
    proc = subprocess.run(
        [str(driver), "--describe"],
        capture_output=True,
        text=True,
        cwd=str(root),
        timeout=120,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("PIM_GEOMETRY"):
            return line.strip()
    raise BundleError(
        f"pim_driver --describe at {root} printed no PIM_GEOMETRY line "
        f"(exit {proc.returncode}):\n{(proc.stdout + proc.stderr)[-2000:]}"
    )


def build(
    root: Path | None = None,
    *,
    jobs: int | None = None,
    skip_build: bool = False,
    scons: str | None = None,
    log=None,
) -> Path:
    """Assemble, patch and build the bundled simulator at ``root`` (default
    :func:`allo.spmw_simenv.default_pimsim_root`).  Returns the root, which
    then holds ``pim_driver`` and the files it reads."""
    recipe = recipe_dir()
    target = spmw_simenv.default_pimsim_root() if root is None else Path(root)
    target = assemble(recipe, target)
    apply_and_build(
        recipe, target, jobs=jobs, skip_build=skip_build, scons=scons, log=log
    )
    if not skip_build:
        describe(target)
        for rel in RUNTIME_FILES:
            if not (target / rel).is_file():
                raise BundleError(f"{target / rel} is missing after the build")
    return target


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m allo.pim.samsung_simulator",
        description=(
            "Build the bundled Samsung PIMSimulator (pim_driver) for Tenon's "
            "Samsung HBM-PIM simulator route, or inspect the bundle."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser(
        "build",
        help="copy the bundled upstream sources into a root, patch and build",
    )
    b.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "build root (default: $TENON_HOME/simulators/PIMSimulator, which "
            "pimsim_root() discovers; another root needs PIMSIMULATOR_ROOT)"
        ),
    )
    b.add_argument("--jobs", type=int, default=None, help="SCons parallelism")
    b.add_argument(
        "--skip-build",
        action="store_true",
        help="assemble and patch the tree without compiling",
    )
    b.add_argument("--scons", default=None, help="SCons executable to use")
    sub.add_parser("verify", help="check the bundled sources against the manifest")
    sub.add_parser("locate", help="print the recipe and the discovered simulator root")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            recipe = recipe_dir()
            problems = verify_recipe(recipe)
            print(f"recipe {recipe}")
            print(f"upstream {UPSTREAM_URL} {UPSTREAM_COMMIT}")
            if problems:
                print("\n".join(problems))
                return 1
            print(
                f"{len(read_manifest(recipe))} upstream files verified; patches and overlay present"
            )
            return 0
        if args.command == "locate":
            try:
                print(f"recipe {recipe_dir()}")
            except BundleError as exc:
                print(f"recipe not found: {exc}")
            root = spmw_simenv.pimsim_root()
            print(f"default root {spmw_simenv.default_pimsim_root()}")
            reason = spmw_simenv.samsung_unavailable_reason()
            print(f"pimsim_root {root if root is not None else '(none)'}")
            print("available" if reason is None else f"unavailable: {reason}")
            return 0 if reason is None else 1
        root = build(
            args.root,
            jobs=args.jobs,
            skip_build=args.skip_build,
            scons=args.scons,
            log=sys.stdout,
        )
        print(f"PIMSimulator root {root}")
        if not args.skip_build:
            print(describe(root))
            default = spmw_simenv.default_pimsim_root().resolve()
            if root.resolve() != default:
                print(f"export PIMSIMULATOR_ROOT={root}")
            else:
                print(
                    "discovered by allo.spmw_simenv.pimsim_root() without configuration"
                )
        return 0
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
