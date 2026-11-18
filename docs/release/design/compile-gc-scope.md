<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# Compile-time cycle collection (C-38)

The frontend's two explicit `gc.collect()` calls run after loop construction.
They release Python wrapper cycles, not source-program effects. A full sweep
at every loop also scans unrelated objects retained by the calling process.
Round 213 established this dependence with identical emitted commands: adding
two million live cyclic lists added 35.21 seconds, 34.49 of them in GC.

The compile entry now scopes a context-local loop collection policy. During
`allo.compile`, the loop boundaries collect generation zero. The outer exit,
including an exception exit, performs one full collection to reclaim cycles
that survived or were promoted during compilation. The ordinary customizer
outside this scope keeps its original full collections. Automatic GC remains
enabled or disabled as the caller chose, with unchanged thresholds. This
batches the explicit old-generation scans; it does not disable collection.
Nested compiles share the outer sweep, while other threads retain their own
policy through a `ContextVar`. No process-wide `gc` function is replaced.

A blanket freeze/thaw was considered and rejected. CPython 3.12 automatically
moves immortal objects to the permanent generation during collection, so a
nonzero `get_freeze_count()` does not prove that the caller called `freeze()`.
`unfreeze()` also thaws the entire permanent generation; Python exposes no
selective thaw that could restore a pre-existing caller-owned set. The chosen
policy never freezes or unfreezes anything. References:
[Python GC API](https://docs.python.org/3.12/library/gc.html),
[CPython 3.12 `update_refs`](https://github.com/python/cpython/blob/v3.12.13/Modules/gcmodule.c).

MLIR's Python operation wrapper destructor invalidates an attached wrapper;
the owning module retains the operation. It destroys detached operations it
owns. Live module/context references must survive a sweep; dead wrapper cycles
must eventually release their context. Tests cover nursery collection of new
context cycles, full reclamation of promoted context cycles on exceptional
exit, and verification/execution of an artifact after the scope has exited.
The latter then drops all artifact references and verifies context release.
Neither the emitted IR nor the selected-program contract depends on timely
execution of an unreachable wrapper's destructor.

Validation rule, written before final candidate measurements: run the complete
non-hardware `tests/perf` and `tests/pim` collection in one process from the
installed wheel, after the installed milestone. The seven existing Samsung
120-second compilation guards must all pass unchanged; no subprocess isolation
or relaxed deadline is allowed. A paired heap diagnostic separately records
wall/CPU/GC time and command/selection identity. It distinguishes GC from
non-GC work, but cannot reconstruct unrecorded phase times from the original
failed run. The final collection must meet this rule before C-38 closes.

Candidate 603d5c38 meets that rule: the installed single-process collection has
1,863 passes, zero failures and 23 export-list skips, with all seven unchanged
guards passing. Early default compile bodies are 55.24–55.70 seconds; late
same-shape/policy bodies are 54.27–54.94 seconds. Their non-GC times remain
51.54–51.99 and 51.01–51.57 seconds respectively. The body timing excludes the
outer sweep; the original test guard includes it. The old collection-context
probe measured two compiles and two simulations, not one guarded compile.
See VALIDATION and `r214-final/c38-long-process-audit.json` for exact scopes,
raw timings, retained failures and the corrected evidence-harness ownership.
This closes the observed issue, without claiming a universal latency bound.
