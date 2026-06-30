# syrk on samsung_hbm_pim (SMALL)

- correctness: **CYCLES-ONLY** -- samsung/syrk: multi-stage kernel out of the pure-MAC REDUCE paradigm (gemver mixes ELTWISE rank-1 + GEMV); a real REDUCE pass over stage-0 gives cycles, but the composed numerics are unchecked (no chain recipe). Honest CYCLES-ONLY, never a fabricated PASS.
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 60, 'N': 80}
- cycles: 4435
- source: `PIMSimulator@bin-sha256:121250edea00`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/syrk/test_syrk_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-30T01:05:26.652500
- tenon_commit: `ee1e52a539d6385a60d149ca238b592d9763103e`

Tier-1 single-output; Samsung reports cycles only (no output array) -> CYCLES-ONLY at the GEMV design point; a shape the reference sim cannot express -> BLOCKED-SIM. | SPEC-05 multi-stage, no chain recipe (mixed ELTWISE+REDUCE); honest CYCLES-ONLY.
