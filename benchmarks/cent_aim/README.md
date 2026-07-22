<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Tenon/CENT SK hynix AiM comparison

This package builds the 14 typed Tenon programs that correspond to the frozen
CENT vendor cases, compiles every program twice, and refuses to measure unless
the complete traces and manifests are byte-identical. Each trace must contain
exactly one trailing `AiM EOC`.

Run the campaign from the Allo repository root:

```bash
python -m benchmarks.cent_aim.run_campaign \
  --simulator-root /path/to/aim_simulator \
  --vendor-root /path/to/skhynix/vendor \
  --output-root /path/to/new/evidence-directory
```

The runner requires the simulator checkout at commit
`0f28a07bdb83e42b9305ad3d45410ebd3aa2c091`. It resolves the requested Docker
tag to a content-addressed `sha256:` image ID and uses that immutable ID for all
runs. The simulator and evidence directories are mounted read-only and network
access is disabled. The output directory must be absent or empty.

By default, the campaign also derives legal contraction-reuse candidates from
`target.mac_reg.slots`, compiles each candidate twice, and measures it twice on
the same simulator. `--skip-reuse-calibration` omits this diagnostic phase; no
calibration cycle constant is embedded in the compiler or harness.

Verify the completed campaign independently:

```bash
python -m benchmarks.cent_aim.verify_campaign \
  /path/to/evidence-directory \
  --vendor-root /path/to/skhynix/vendor \
  --tenon-root . \
  --simulator-root /path/to/aim_simulator
```

`--tenon-root` and `--simulator-root` are optional during verification because
the bundle retains a compiler-source snapshot, config, and content hashes.
Supplying them additionally requires the current local files to match the
measurement. The vendor root remains required because the comparison table is
cryptographically joined to its retained raw logs rather than copying or
silently trusting vendor cycle constants.

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

Evidence does not equate opcode counts with semantic completeness. For every
typed operation, the verifier reconciles command spans and derives physical
capacity from operation sizes, vector lanes, mask fanout, and target banks or
bank groups. It also retains canonical vendor/Tenon command-shape signatures
and their deltas while excluding addresses and command order, so legal linear
layouts may differ physically without hiding an under-emitted logical tensor.

The frozen CENT attention traces issue V-cache `WR_ABK` writes only to the
first 8-channel group (1/4 replica coverage). They remain untouched as vendor
evidence. Tenon materializes all 32 typed producer channels (4/4 coverage), so
the affected cases intentionally contain 1,024 rather than 256 `WR_ABK`
commands. The bundle records this inventory delta, the expected and observed
ownership coordinate counts, and the resulting cycles instead of claiming
physical-work equality. As with all timing-model evidence, a system-wide
`WR_GB` mask establishes channel/work ownership but cannot prove that the
masked channels carry distinct numerical replica values.
