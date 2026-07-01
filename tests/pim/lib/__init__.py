# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite shared library (written once; spec Answer 1).

Kernel folders import ONLY from here: `lib.targets` (re-exported constructors),
`lib.cost` (calibration-profile binding), `lib.runner` (cross-target compile + provenance),
`lib.reference` (validated numpy refs + verdict), `lib.shapes` (psize.json),
and backend execution/reporting helpers after each leaf visibly calls
`allo.compile`.
No kernel folder declares hardware, capacity, cost constants, or dimension
literals.
"""
