# 2mm on apu_v1 (SMALL)

- correctness: **CYCLES-ONLY** -- apu_v1/2mm: real-device run (real cycles) -- but apu_v1 fp16-MAC is OUT-OF-PARADIGM (planner SPEC-018b ruling): the silicon runs BINARY XNOR-popcount MAC (the validated bmatmul_sv/sv_lookup paradigm); a general fp16 multiply-accumulate is a DIFFERENT compute kernel (no gvml fp16-MAC primitive; the autoscheduler/cost model know only the binary {sv,sv_lookup} modes), an architectural boundary (SPEC-018b, noted for a ceiling task), NOT a fix-in-session gap. The board runs the binary LUT-MAC stub so the output cannot match the fp16 ref -- CYCLES-ONLY, never a FAIL and never a fabricated fp16 PASS
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'P': 40, 'R': 50, 'Q': 70, 'S': 80}
- cycles: 3609445
- source: `apu_v1_device@zhang-capra-xcel.ece.cornell.edu/gsi-13.7.1`
- run_cmd: `python -m pytest tests/pim/apu_v1/2mm/test_two_mm_apu_v1.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T11:20:40.308432
- tenon_commit: `ed217633e316c4b45d19368b4cfadd61c028aa21`

Tier-1 real-device; APU v1 surfaces real numerics from the board. The emitted declarative MAC is the SPEC-018 popcount-LUT pattern, so the board runs real cycles but a general fp16 GEMV output is the SPEC-018b TODO -> CYCLES-ONLY (real cycles), never a fabricated PASS.
