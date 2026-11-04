# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The APU v1 scalar manifest is read from the retained IR (release gate G9).

`discover_apu_v1_hybrid_manifest` walks the retained MLIR module through the
IR bindings: formals and their types, named allocations and their fills,
``func.call`` operations in program order and each callee's loads and stores
on its formals.  These tests pin the ABI identities, call order, access
directions, initialization and intermediate lifetimes it derives, on
programs with returned locals, nested calls, in/out arguments and unsigned
locals, and on renamed neighbours.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32, int32, uint16
from allo.pim import apu_v1_hybrid
from allo.pim.apu_v1_program import CompiledAPUv1Program
from allo.pim.targets import build_apu_v1_target


def returned_local(x: int32[8]) -> int32[8]:
    y: int32[8] = 0
    scratch: int32[8] = 0
    for i in range(8):
        scratch[i] = x[i] * 3
    for i in range(8):
        y[i] = scratch[i] + 1
    return y


def helper(a: int32[8], b: int32[8]):
    for i in range(8):
        b[i] = a[i] * 2


def finish(t: int32[8], z: int32[8]):
    for i in range(8):
        z[i] = t[i] + z[i]


def nested_calls(x: int32[8], z: int32[8]):
    tmp: int32[8] = 0
    helper(x, tmp)
    finish(tmp, z)


def scale(p: int32[8], q: int32[8]):
    for i in range(8):
        q[i] = p[i] * 2


def accumulate(u: int32[8], w: int32[8]):
    for i in range(8):
        w[i] = u[i] + w[i]


def renamed_nested_calls(left: int32[8], total: int32[8]):
    buffer: int32[8] = 0
    scale(left, buffer)
    accumulate(buffer, total)


def inout_scalar(a: float32[4], c: float32[4]):
    for i in range(4):
        c[i] = c[i] + a[i] * 2.0


def unsigned_local(v: uint16[16]) -> uint16[4]:
    out: uint16[4] = 0
    for i in range(16):
        out[i % 4] = out[i % 4] + v[i]
    return out


def _manifest(kernel, result_names=()):
    program = allo.APUv1Program(
        (allo.APUv1Phase(kernel, result_names=result_names),), name=kernel.__name__
    )
    compiled = CompiledAPUv1Program(
        program, build_apu_v1_target(), backend="functional"
    )
    return compiled.hybrid_manifest


def _values(manifest):
    return {value.name: value for value in manifest.values}


def test_returned_local_buffers_are_outputs_and_zero_initialized_intermediates():
    manifest = _manifest(returned_local, result_names=("y",))
    values = _values(manifest)
    assert values["x"].program_input and not values["x"].program_output
    assert values["y"].program_output and not values["y"].intermediate
    assert values["y"].zero_initialized  # the fill of the returned local
    assert values["scratch"].intermediate and values["scratch"].zero_initialized
    (phase,) = manifest.phases
    (region,) = phase.regions
    assert region.kind == "scalar" and region.function == "returned_local"
    assert [(o.value, o.access) for o in region.operands] == [
        ("x", "read"),
        ("y", "readwrite"),
    ]
    assert phase.inputs == ("x",) and phase.outputs == ("y",)


def test_nested_calls_keep_order_access_directions_and_dependencies():
    manifest = _manifest(nested_calls)
    (phase,) = manifest.phases
    first, second = phase.regions
    assert (first.function, second.function) == ("helper", "finish")
    assert [(o.value, o.access) for o in first.operands] == [
        ("x", "read"),
        ("tmp", "write"),
    ]
    assert [(o.value, o.access) for o in second.operands] == [
        ("tmp", "read"),
        ("z", "readwrite"),
    ]
    assert first.produced_intermediates == ("tmp",)
    assert second.consumed_intermediates == ("tmp",)
    assert second.dependencies == ("region0",)
    (barrier,) = phase.barriers
    assert barrier.values == ("tmp",) and barrier.reason == "data_dependency"
    assert second.barrier == barrier.id
    assert _values(manifest)["tmp"].intermediate
    assert phase.inputs == ("x", "z") and phase.outputs == ("z",)


def test_renamed_nested_program_has_the_same_structure():
    """Value, function and loop names do not shape the manifest: a renamed
    neighbour gives the same regions, accesses, dependencies and barriers."""

    def normalized(manifest):
        (phase,) = manifest.phases
        return [
            (
                region.ordinal,
                [o.access for o in region.operands],
                [o.position for o in region.operands],
                region.dependencies,
                len(region.produced_intermediates),
                len(region.consumed_intermediates),
                region.barrier,
            )
            for region in phase.regions
        ], [b.reason for b in phase.barriers]

    assert normalized(_manifest(nested_calls)) == normalized(
        _manifest(renamed_nested_calls)
    )


def test_inout_argument_is_both_input_and_output():
    manifest = _manifest(inout_scalar)
    values = _values(manifest)
    assert values["a"].program_input and not values["a"].program_output
    assert values["c"].program_input and values["c"].program_output
    (phase,) = manifest.phases
    (region,) = phase.regions
    assert [(o.value, o.access) for o in region.operands] == [
        ("a", "read"),
        ("c", "readwrite"),
    ]


def test_unsigned_local_keeps_its_signedness_from_the_retained_ir():
    manifest = _manifest(unsigned_local, result_names=("out",))
    values = _values(manifest)
    # The artifact's ABI spells unsigned types ``ui<bits>``; the formal and
    # the returned local agree with it.
    assert values["v"].dtype == "ui16"
    assert values["out"].dtype == "ui16" and values["out"].program_output


def test_discovery_reads_the_ir_not_its_text():
    """No regex over the printed module: the only pattern left validates
    scalar element type spellings."""
    import re

    patterns = [
        value for value in vars(apu_v1_hybrid).values() if isinstance(value, re.Pattern)
    ]
    assert patterns == [apu_v1_hybrid._DTYPE_RE]
    for name in ("_function_regions", "_type_parts", "_FUNC", "_FORMAL", "_CALL"):
        assert not hasattr(apu_v1_hybrid, name)
