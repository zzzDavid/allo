# gramschmidt on samsung_hbm_pim (SMALL)

- correctness: **CYCLES-ONLY** -- samsung/gramschmidt: multi-stage kernel out of the pure-MAC REDUCE paradigm (gemver mixes ELTWISE rank-1 + GEMV); a real REDUCE pass over stage-0 gives cycles, but the composed numerics are unchecked (no chain recipe). Honest CYCLES-ONLY, never a fabricated PASS.
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 60, 'N': 80}
- cycles: 4435
- source: `PIMSimulator@bin-sha256:121250edea00`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/gramschmidt/test_gramschmidt_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-30T12:35:11.542895
- tenon_commit: `837aab5ab7def05b5255b98cf66eee65362bb546`

Tier-2 Gram contraction; real REDUCE cycles with honest CYCLES-ONLY numerics. | SPEC-05 multi-stage, no chain recipe (mixed ELTWISE+REDUCE); honest CYCLES-ONLY.
