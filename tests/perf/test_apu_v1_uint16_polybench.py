# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""All 30 APU v1 PolyBench programs have exact uint16 MLIR/C semantics."""

from pathlib import Path
import sys

import numpy as np

import allo
from allo.pim.targets import build_apu_v1_target


sys.path.insert(0, str(Path(__file__).parents[1] / "pim"))
from lib.apu_v1 import (
    NAMES,
    build_program,
    get_case,
)  # pylint: disable=wrong-import-position


def test_all_apu_v1_polybench_cases_compile_and_match_uint16_reference():
    target = build_apu_v1_target()
    for name in NAMES:
        case = get_case(name)
        source = case.make_inputs()
        expected = case.run_reference(source)
        arrays = {key: np.array(value, copy=True) for key, value in source.items()}

        compiled = allo.compile(build_program(case), target, backend="functional")
        artifact = compiled.compiled.artifact
        returned = artifact.compile()(*(arrays[item.name] for item in case.arguments))
        returned = (
            ()
            if returned is None
            else (returned if isinstance(returned, tuple) else (returned,))
        )
        extra = dict(
            zip(
                (item.name for item in case.results if item.name not in arrays),
                returned,
            )
        )

        assert all(value.dtype == np.dtype(np.uint16) for value in arrays.values())
        for result in case.results:
            observed = (
                arrays[result.name] if result.name in arrays else extra[result.name]
            )
            np.testing.assert_array_equal(observed, expected[result.name], err_msg=name)
