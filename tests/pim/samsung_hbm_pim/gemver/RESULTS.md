# gemver on samsung_hbm_pim (SMALL)

- correctness: **CYCLES-ONLY** -- samsung/gemver: multi-stage kernel out of the pure-MAC REDUCE paradigm (gemver mixes ELTWISE rank-1 + GEMV); a real REDUCE pass over stage-0 gives cycles, but the composed numerics are unchecked (no chain recipe). Honest CYCLES-ONLY, never a fabricated PASS.
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'N': 120}
- cycles: 4435
- source: `PIMSimulator@bin-sha256:121250edea00`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/gemver/test_gemver_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-30T11:53:52.996721
- tenon_commit: `1af5ccd18a995ba016570ea802403affd91cad2a`

Tier-1 multi-output; Samsung reports cycles only (no functional readback on the faithful path) -> CYCLES-ONLY at the GEMV design point; shape the sim cannot express -> BLOCKED-SIM. | SPEC-05 multi-stage, no chain recipe (mixed ELTWISE+REDUCE); honest CYCLES-ONLY.
