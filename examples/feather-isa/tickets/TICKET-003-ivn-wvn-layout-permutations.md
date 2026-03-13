---
id: TICKET-003
title: Apply IVN/WVN layout order permutations in crossbar
status: open
priority: medium
---

# TICKET-003: Apply IVN/WVN Layout Order Permutations in Crossbar

## Problem

The MINISA ISA defines 6 layout order permutations (ORDER_012 through ORDER_210) for
both SetIVNLayout and SetWVNLayout instructions. These orders control how the M/K/N
dimensions are mapped to the physical buffer layout. Currently, the order field is
encoded in instructions but the crossbar always uses ORDER_012 indexing.

## Why It Matters

- Different layout orders enable different data reuse patterns in the PE array
- ORDER_012 is only one of 6 possible mappings — the others may be optimal for
  certain workload shapes or memory access patterns
- Full MINISA compliance requires supporting all 6 permutations

## Root Cause

The `crossbar_load` kernel computes input/weight indices assuming a fixed dimensional
ordering. Supporting all 6 permutations requires 6 cases (or a general permutation
formula) in the crossbar index computation for both iActs and weights.

## Relevant Code

- `feather_minisa.py`: `crossbar_load` kernel — index computation for `iAct_buf` and `wgt_buf`
- `minisa/isa.py`: `SetIVNLayout` and `SetWVNLayout` — order field definitions
- `minisa/isa.py`: `ORDER_012` through `ORDER_210` constants

## Acceptance Criteria

- [ ] All 6 IVN layout orders produce correct GEMM output
- [ ] All 6 WVN layout orders produce correct GEMM output
- [ ] Crossbar index computation handles order field from decoded instruction
- [ ] Existing ORDER_012 tests still pass
- [ ] At least one test verifying a non-012 order for both IVN and WVN
