# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite: shared single-output Tier-1 @allo.work specs.

Each workload is declared ONCE here (target-independent) and imported by the
per-target kernel-folder tests (`tests/pim/<target>/<kernel>/test_<k>.py`). This
honors 'lib written once': a workload is not re-declared per backend. Dimensions
come from `lib.shapes` (psize.json), never pasted.
"""
