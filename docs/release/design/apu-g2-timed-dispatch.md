<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# Gemini-II invariant epilogue dispatch (round 208)

The composed-dot task formerly called `pipeline` on every timed repetition.
That function loaded the immutable task epilogue and dispatched to identity,
pair-affine or accumulate after every reduction. The retained r185 ARC binary
confirms this was real repeated control work: the loop calls `pipeline` at
0x1000415a, branches back at 0x10004164, and the callee loads the epilogue
at offset 0x3c. This is an optimization opportunity, not an established cause
of the historical X-34/X-37 timing differences.

The maintained task now selects a specialized repetition loop once per
timed window. The host supplies constant parameters to a synchronous task;
the parameter block is separate from L1 operands and the allocated L5 timing
result. Legality, view construction, arithmetic, transfers and epilogue
bodies are unchanged. Each iteration still issues the entire primitive
sequence. Epilogue selection remains **after** `gsi_perf_start`, and every
window still ends with `seu_barrier` before `gsi_perf_end`. The separate
single-body correctness execution remains inside its own timing window.
No device work, precision, layout, repetition or timing scope is removed.

The specialization domain is the existing three legal ABI epilogues. It is
independent of workload names, shapes or paper results. The shared pipeline
still selects the same recipe, layout and packing; only the generic device
implementation of its repeated execution changes. Recipe cost features and
empirical coefficients remain unchanged. Those coefficients retain their
original calibration provenance; this change does not relabel old timings
as measurements of the new wrapper or guarantee improved cost accuracy.

Evidence under
`/work/zhang-capra/users/nz264/tenon-local-tmp/tenon-release-evidence/r208`:

- `g2-dispatch-control-flow.json` and the task disassembly: the remote ARC
  build has three repetition loops, with 7/11/15 primitive calls respectively
  for identity/accumulate/pair-affine and no epilogue reload in those loops.
  Both dispatches follow their timer start; both barriers precede timer end.
- `g2-dispatch-tests-fixed.log`: seven focused tests passed. Recording SDK
  stubs execute the actual distributed C++ template, checking full primitive
  sequences in both timing windows and rejection before timing for invalid
  tasks. These checks do not model SDK arithmetic or device ticks. The first
  harness compilation failure is retained in `g2-dispatch-tests.log`.
- `g2-draft-functional.log`: eight installed working-candidate tests passed
  on zhang-x4, with exact numerical checks for all three epilogues, a nearby
  two-launch GEMM, pre-call alias values and fused/separate layouts. The
  device-host lock is `/tmp/tenon-apu-g2-apu-00.lock`. Retained builds and
  outputs are in the remote `r208/g2-dispatch-candidate/functional` directory.
  These are functional checks, not paired paper-comparison samples.

Working-candidate identity: base 8a8c794 plus the sole production delta in
`apu_g2_composed_dot.cc`, SHA256
`de7b1a87f2546896b04116e14e512da5aaaaa532b7ec333c8f996c685f3f2b04`.
The source archive SHA256 is
`cec6abe9f72cfa936587dc3538c796370530835cd9563c565de109ed16500912`;
the separately built zhang-x4 wheel SHA256 is
`61e16c13bb07172d0e93a8bbaa20e066e6b32183b05fd8755e1a5823c6825722`.
This is a working-candidate validation artifact, not a published release.

## Bounded performance confirmation, recorded before paired measurements

After verifying that the installed working candidate contains exactly the
committed production change, use the unchanged `g2_session.sh` and
`g2_pool.py` protocol. Run exactly four sessions: one validation session and
three confirmation sessions, all five required kernels in each, one warm-up
per lane followed by twenty interleaved paired rounds, with twenty pipeline
repetitions. Re-run the same archived GTML baseline, retaining its existing
binary attestation and exact-output checks. No old Tenon candidate is run.

Pool all four complete sessions at this candidate (80 measured samples per
case) using the existing raw-tick, fixture, layout and program validation.
Keep every partial/failed attempt, with no automatic replacement session or
favourable-sample choice. An incomplete campaign remains incomplete even if
the valid subset is fast. The functional checks above are not pooled.
The old fixed campaign remains separately recorded, with X-34/X-37 OPEN at
its original candidate; it is not pooled, discarded or relabelled.

The acceptance rule is unchanged: each pooled median must be no greater
than its frozen paper value, with exact outputs and comparable scope. Report
quartiles, session medians, raw samples and paired baseline ratios; neither
aggregate gains nor a confidence interval crossing zero establishes parity
for a failing row. No added tolerance or post-result extension is permitted.
If the new candidate fails, retain that result and investigate a concrete
mechanism before any further performance work. Independent source review
and the remaining release gates are still required.
