<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Release checklist: index clean-up commands

Commands that change the git index are listed here for the maintainer to
run; the release rounds do not run them. Paths are relative to the
repository named in each heading. Prepared 2026-11-02 against `tenon` at
`63ea0e8` (audit P-18; P-04 for the containing repository).

## 1. This repository (`experiments/allo`): stray tracked files (P-18)

```bash
cd experiments/allo
git rm --cached experiments/scripts/verifier_probe_residency_panic.py   # agent debugging probe, referenced nowhere
git rm --cached tests/pim/.apu_v1.lock                                  # empty board lock; *.lock is ignored and the lock is created at run time
git commit -m "[Release] Untrack an agent probe and a committed lock file"
# then, if the working copies are not wanted either:
rm experiments/scripts/verifier_probe_residency_panic.py tests/pim/.apu_v1.lock
rmdir experiments/scripts experiments 2>/dev/null || true
```

Checked before listing: `git grep` finds no reference to either file outside
the release documents; `.gitignore` already ignores `*.lock`.

Kept on purpose (audit section 6 listed them; they are not removed):

| File | Reason |
|---|---|
| `playground/int8_gemm.py`, `playground/mat_vec_test.py` | present in upstream `origin/main`; removing them would diverge from upstream Allo |
| `scripts/prepare_upmem_tenon_campaign.py` | tested by `tests/perf/test_prepare_upmem_tenon_campaign.py`, cited in `allo/spmw_upmem.py` |
| `scripts/run_pimsim_layout_trace.py` | used by `tests/pim/samsung_hbm_pim/layout_swizzle/test_layout_swizzle_probe.py` |
| `allo/_mlir` (symlink) | the editable install reaches the bindings through it (`INSTALL.md` section 2) |

Untracked files that are not part of the release (delete or keep locally;
they are not in the index, so `git rm` does not apply, but `git add -A`
would pick some of them up): `scripts/verifier_check_*.py` (six agent check
scripts), `np_W*.txt`, `simple*.lp`, `simple2.dot`, `*.prj/`, `tmp/`,
`tests/pim/apu_v1/vector_gemm/{RESULTS.md,results.json}` (run outputs).
Build release artifacts from a clean checkout (`INSTALL.md` section 2).

## 2. New files to add in this repository

These are new in the 2026-11-02 round and untracked until added:

```bash
git add requirements-dev.txt scripts/regenerate_upmem_calibration.py docs/release/RELEASE_CHECKLIST.md
```

`requirements-dev.txt` needs the `.gitignore` exception added in the same
round (`*.txt` is ignored; `!requirements.txt` and `!requirements-dev.txt`
re-include the two requirement files).

## 3. Containing repository: simulator overlays (P-05, P-06)

```bash
cd <containing repository>
git add experiments/simulators/setup/PIMSimulator/README.md \
        experiments/simulators/setup/PIMSimulator/apply.sh \
        experiments/simulators/setup/PIMSimulator/patches \
        experiments/simulators/setup/PIMSimulator/src \
        experiments/simulators/setup/PIMSimulator/Sconstruct \
        experiments/simulators/setup/uPIMulator/README.md \
        experiments/simulators/setup/uPIMulator/apply.sh \
        experiments/simulators/setup/uPIMulator/tenon-upimulator.patch \
        experiments/simulators/setup/apply.sh \
        experiments/simulators/README.md
```

Licensing caveat before publishing the Samsung overlay: the
`setup/PIMSimulator/patches/*.patch` hunks carry context lines from
Samsung-licensed files (`PIMKernel.h/.cpp`, `PIMCmdGen.h/.cpp`, `Burst.h`).
Whether Samsung's notice permits distributing such a patch is open
(`audit/packaging.md` section 9, question 2).

## 4. Containing repository: Samsung-licensed sources under `dev/` (P-04)

Only relevant if the containing repository is ever made public. These
tracked files are modified Samsung PIMSimulator sources carrying Samsung's
no-redistribution notice:

```bash
cd <containing repository>
git rm --cached dev/06292026-samsung-general-instr-codegen/pimsim-conductor/src/tests/KernelAddrGen.h \
                dev/06292026-samsung-general-instr-codegen/pimsim-conductor/src/tests/PIMKernel.cpp \
                dev/06292026-samsung-general-instr-codegen/pimsim-conductor/src/tests/PIMKernel.h
```

Untracking does not remove them from history; a public copy needs a history
rewrite or a fresh export that never contained them.
