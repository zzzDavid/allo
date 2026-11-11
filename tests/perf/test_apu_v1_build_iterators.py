# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""C-13 (round 127): the APU v1 build harness falls back to its canonical
L4 pointer and VR alias declarations only when the context lacks the
iterator; an iterator that exists is invoked outside the guard, so an
AttributeError raised inside it, eagerly or lazily, propagates instead of
silently replacing a selected alias in the emitted device source."""

from types import SimpleNamespace

import pytest

from allo.spmw_apu_v1_build import _emit_device_c

CMDS = ["gvml_cpy_16(vr0, vr0);"]


def _compiled(ctx):
    return SimpleNamespace(layout_ctx=ctx, cmds=list(CMDS), trace=None)


def _alias_line(source, name="vr0"):
    return next(
        line for line in source.splitlines() if f"enum gvml_vr16 {name} =" in line
    )


class _Good:
    def iter_l4_roles(self):
        return []

    def iter_vr_aliases(self):
        return [("vr0", "GVML_VR16_9")]


class _Missing:
    pass


class _BrokenL4(_Good):
    def iter_l4_roles(self):
        raise AttributeError("broken internal l4 state")


class _BrokenVR(_Good):
    def iter_vr_aliases(self):
        raise AttributeError("broken internal allocation state")


class _BrokenGenerator(_Good):
    def iter_vr_aliases(self):
        yield ("vr0", "GVML_VR16_9")
        raise AttributeError("broken lazy allocation state")


def test_noncanonical_alias_from_the_context_is_preserved():
    source = _emit_device_c(_compiled(_Good()), [], [])
    assert "GVML_VR16_9" in _alias_line(source)


def test_absent_iterators_fall_back_to_the_canonical_declarations():
    source = _emit_device_c(_compiled(_Missing()), [], [])
    assert "GVML_VR16_0" in _alias_line(source)


@pytest.mark.parametrize("ctx", [_BrokenL4, _BrokenVR, _BrokenGenerator])
def test_internal_attribute_errors_propagate(ctx):
    with pytest.raises(AttributeError, match="broken"):
        _emit_device_c(_compiled(ctx()), [], [])
