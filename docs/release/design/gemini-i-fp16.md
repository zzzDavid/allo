<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# Gemini-I binary16 arithmetic

Gemini-I `float16` uses the GVML target primitives. `strict` preserves source
operation order and each primitive's target rounding. It does not promise
IEEE round-to-nearest-even multiplication. The public result's `numerics`
metadata states this on functional, analytical and device paths. The default
is an explicit decision for the user; there is no separate IEEE-product
policy in the existing reduction-order plumbing. Neither a CPU replacement
nor a tolerance exception is used to make the device conform to IEEE.

## Primitive definitions

`allo.pim.apu_v1_fp16` is the maintained NumPy model; it needs no vendor SDK.
It covers every binary16 encoding, including signed zeros, subnormals,
normal values, infinities, signaling/quiet NaNs and both NaN signs.

For multiply, normalize finite significands to an explicit leading one
(including subnormals). Multiply the two 11-bit significands exactly. Round
the resulting integer to nearest-even at bit 10 **before** normalizing a
product that reaches two. Any subsequent right shift, including an
underflow shift into the subnormal range, truncates. Overflow gives signed
infinity. The result sign is the operand-sign XOR. Zero times infinity and
any NaN operand give NaN magnitude `0x7fff` with that XOR sign.

For finite representable exact products this agrees with IEEE. Inexact
finite products can be one binary16 step smaller in magnitude than IEEE RNE;
the underflow, overflow and NaN rules above are part of the contract too.
This is a target primitive definition, not a compiler-selected approximation.

Addition is IEEE nearest-even for finite operands (including signed-zero
rules and overflow). Invalid infinity addition and NaN inputs produce NaN
magnitude `0x7fff`: sign follows the operand with the larger unsigned
magnitude encoding, or the AND of the signs on a tie. Subtraction flips the
second operand's sign and applies addition. A binary64 intermediate can
represent every finite binary16 sum exactly, so the NumPy add model has one
final RNE conversion; explicit bit handling supplies the target NaN rule.

The emitted `gvml_add_subgrps_f16_grp` form has one element per subgroup,
a power-of-two group, and destination subgroup zero. Its selected heads
reduce **adjacent pairs at each level**. Other destination lanes are scratch
and are not the modeled output ABI. The contraction model pads incomplete
groups with zero, applies that tree, and adds heads in ascending group order.
Strict contractions instead multiply and add in source reduction order.
Both start from the same staged initial output and retained alias snapshots.

## Derivation and independent checks

The optional harness `benchmarks/apu_v1/fp16_reference.py` compiles an original
small wrapper against the user's installed `libgsigvml_ref.a`, calling
`refgvml_mul_f16`, `refgvml_add_f16`, `refgvml_sub_f16`, and
`refgvml_add_subgrps_f16_grp` directly. `_gsi_log` is a no-op link stub only.
No vendor library, header or compiled bridge is bundled. Paths are discoverable
or configurable through `TENON_APU_V1_REF_LIBRARY` and
`TENON_APU_V1_REF_INCLUDE`. Missing SDK/compiler permits a test skip; a build
or comparison error fails. Reproduce the complete check with:

```sh
python -m benchmarks.apu_v1.fp16_reference --exhaustive --out /absolute/fresh/directory
```

Round 211 independently re-derived the formulas and directly checked the
subgroup instruction. `r211/exhaustive-v1` reports zero differences for all
4,294,967,296 multiply pairs; all 4,096 special operand encodings against
all 65,536 encodings in both orders for add and subtract (1,073,741,824 pairs);
4,194,304 additional random pairs per primitive; and 64 full-vector subgroup
checks over all group powers 1 through 32,768, all encodings and random data.
Chunk ranges, comparison hashes, local SDK/header identities and the model
hash are retained. This is SDK-reference evidence, not board execution.

Independent recomputation from the SDK also matches all eight retained r210
strict/regrouped board outputs exactly (`r211/retained-board-comparison.json`).
The independent IEEE source-order oracle remains in the tests. The K=1 probe
(seed 300, 1,024 products) gives the same 66 IEEE differences. This evidence
explains the prior failures; fresh installed confirmation below independently agrees.

## Frozen paper and scope

The lock was rechecked: 73/73 files at paper commit
`30b6f49e64583b5d301d2a5e8b902498f5578e4e`. Section V-B,
`sections/5-evaluation.tex:97–101`, reports the 4096x256 source-order
experiment; its signed-power-of-two operands times one have exact products.
The model predicts all 4,096 IEEE-oracle outputs exactly in
`test_ieee_product_difference_and_exact_product_paper_inputs`.
A separate installed-model prediction (`r211/paper-numerics-prediction.json`)
also gives the historical regrouped result: 859 of 4,096 outputs equal the
source-order reference, versus 4,096 for strict. This is a functional
prediction, not a new device measurement.
`sections/6-discussion.tex:27–30` discusses permission to regroup and the
limits of exact-integer probes. Neither establishes arbitrary-product IEEE
multiplication on Gemini-I. The paper is unchanged; CLAIMS E11 retains that
experiment's bounded claim. This arithmetic definition applies throughout
the existing supported FP16 domain, not only to those exact-product inputs.

Integration audit: the ordinary-source vector route uses the retained
contraction accesses for both orders; low-level `execute_numpy` uses the same
primitive model. `spmw_apu_v1`'s other FP16 sites validate/pack data, not
compute host products; its grouped route reports the same target numerics.
The separate scalar C11 record route rejects half types in the C emitter,
so it has no hidden IEEE FP16 functional execution path.

## Predeclared board confirmation

Before launching, the expected result is **zero bit mismatches**, including
NaN encodings. From a fresh installed wheel run both strict and reassociating
plans at `(M,K,seed)` = `(512,256,706)`, `(384,200,707)`, `(640,96,9211)`, and
`(1000,300,9212)`, with the existing normal-input recipe (scale 0.2).
Run one additional 32x1 by 1x32 outer product whose operands contain signed
zeros, subnormals, normal extremes, overflow/underflow pairs, infinities and
NaNs. Keep the initial negative-zero outputs and compare every uint16 bit
pattern. The test retains inputs, actual outputs, expected outputs and the
independent IEEE comparison. Runtime evidence retains generated source,
binary, packing, fresh profiles and device outputs.

Use the unchanged ten-minute, 15-second, 16-MiB residual-space window and
read-only transport preflight before the nine launches. No xfail, tolerance
change or failed-sample replacement is allowed. Any disagreement is a defect
to diagnose against device ground truth, followed by fresh-input confirmation.
Existing performance campaigns are fixed and must not be resampled. V1-16 is a documented target-arithmetic boundary after the confirmation below.


## Installed confirmation at e0c568a

The protocol above was committed before launch. Wheel SHA256
`c4a13fa7b1c5699b4d9cccf44fad4ef9904dcf01be082c044d1f4a581767bd56`
was installed outside the checkout. `r211/board` completed all nine launches:

| M × K | Seed | Strict mismatches | Regrouped mismatches | IEEE source-order differences |
| --- | ---: | ---: | ---: | ---: |
| 512 × 256 | 706 | 0 / 512 | 0 / 512 | 259 |
| 384 × 200 | 707 | 0 / 384 | 0 / 384 | 194 |
| 640 × 96 | 9211 | 0 / 640 | 0 / 640 | 301 |
| 1000 × 300 | 9212 | 0 / 1000 | 0 / 1000 | 518 |

The 32x1 by 1x32 special-value outer product had zero mismatches of 1,024
encodings, including NaN payloads/signs. Five tests passed, no xfails or
skips. The independent inspection binds every logical result to retained
raw device bytes, generated source/binary/input hashes and exactly one fresh
profile record (`r211/board-evidence-inspection.json`). Every launch used two
reads; no extra flush-lag retry was needed. This is evidence for these runs,
not a universal firmware flush-time bound. Firmware `220.13.500.7` and
ledag-ssh `100.12.0.1.1000.25` are recorded in the copied-log manifests.

The unchanged 41-reading, ten-minute window held 170,713,088 free bytes with
no drop, guard failure or leaked reserve. The real transport preflight took
48.44 seconds. The installed model's complete SDK validation also passed
(`r211/exhaustive-installed`, 114.57 seconds); contiguous chunk coverage and
installed model/SDK hashes were independently rechecked. The eight archived
r210 outputs remain distinct explanatory evidence. No performance campaign
was rerun or repooled, and no arithmetic tolerance or accepted input class
was changed.
