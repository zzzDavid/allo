<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Tenon/CENT SK hynix AiM comparison

This package builds the 14 typed Tenon programs that correspond to the frozen
CENT vendor cases (`workloads.py::CENT_CASES`). Each program compiles through
`allo.compile(build_case(case_id), build_aim_target())` and runs on ramulator2
in Docker. The paper cycle counts are asserted by
`tests/pim/test_paper_golden.py`. The archived campaign producer lives in the
artifact bundle (`tenon-artifacts/skhynix/scripts/run_campaign.py`).

The SK hynix Ramulator2 model used here is timing-only. It does not consume
tensor payloads or produce numerical outputs. Consequently the bundle records
this limitation alongside a deterministic typed semantic/work-invariant
manifest; it claims measured cycle fidelity, not simulator-backed numerical
correctness.

Attention layout selection is shape- and residency-driven. QK selects packed
batch rows when aligned head reductions fit a bank row; SV selects
channel-batched output groups when the cache reserves `max_seq_len` reduction
storage. The retained compiler manifest records the generic policy, selected
layout, reason, row ownership, and logical work. No case ID or canonical model
dimension participates in that selection.

`compiler_logical_work_coverage` reconciles command spans per typed operation
and derives physical capacity from operation sizes, vector lanes, mask fanout,
and target banks or bank groups. `command_shape_signature` excludes addresses
and command order, so legal linear layouts may differ physically without
hiding an under-emitted logical tensor.

The frozen CENT attention traces issue V-cache `WR_ABK` writes only to the
first 8-channel group (1/4 replica coverage). They remain untouched as vendor
evidence. Tenon materializes all 32 typed producer channels (4/4 coverage), so
the affected cases intentionally contain 1,024 rather than 256 `WR_ABK`
commands. The bundle records this inventory delta, the expected and observed
ownership coordinate counts, and the resulting cycles instead of claiming
physical-work equality. As with all timing-model evidence, a system-wide
`WR_GB` mask establishes channel/work ownership but cannot prove that the
masked channels carry distinct numerical replica values.
