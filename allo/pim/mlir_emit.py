"""Emit phase-2 `pim_target` MLIR from an ``allo.pim.Target`` (or the
``_PendingLevel`` tree an ``@allo.target`` chain leaves on
``target.tn_root``).

This is a pure textual emitter: no Python bindings for the new dialect
exist yet, and phase 2 of the MLIR dialect is deliberately text-first
(the round-trip is proved via the standalone ``pim-opt`` binary).  The
emitter reads the decorator tree and walks it recursively, writing:

* ``pim_target.describe @<target_name> {...}``           target root
* ``pim_target.unit @<axis> mapping [...] mode "..."``   one per level
* ``pim_target.memory @<name> {...}``                    inside its owning unit
* ``pim_target.op @<name> {...}``                        inside its owning unit
* ``pim_target.stream @<name> ...``                      inside its owning unit
* ``pim_target.cost {...}``                              once per level that carried ``tn.cost``

The emitted text is intended to be fed to the standalone ``pim-opt``
driver (built from ``experiments/E6_mlir_dialect_phase1/``).  A
round-trip test under ``tests/pim/test_mlir_phase2_roundtrip.py`` walks
all five backend targets (Samsung, AiM, UPMEM, APU v1, APU v2),
produces text, feeds the text to ``pim-opt``, and asserts byte-identity
after a single canonicalization pass.

Design notes
------------

* The ``_PendingLevel`` tree is the richest source: it retains the
  axis name, the mapping (list of extents — Phase 2 emits one unit op
  per extent when ``len(extents) > 1`` is ever supported, matching
  ``allo.unit``'s chain-expansion rule), the mode, per-level memories /
  ops / streams / cost_attrs, and children.  All attributes supported
  by the Python dataclasses are forwarded verbatim via the MLIR
  attribute dictionary.
* Memory / Op ``extras`` (kwargs that were not in the dataclass
  signature) are preserved via ``__dict__``; unknown scalar values are
  emitted best-effort (`int`, `float`, `str`, `bool` are supported —
  anything richer is stringified).
* We only emit the ``pim_target.describe`` symbol for the target root;
  host memories / host ops attach to describe via the attribute dict.
  Units attach to the describe via a ``parent`` symbol reference.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional

from .target import Memory, Op, Target


# ---------------------------------------------------------------------------
# Attribute formatting helpers
# ---------------------------------------------------------------------------


def _mlir_str(s: Any) -> str:
    """Emit a quoted MLIR string attribute value."""
    s = str(s)
    # MLIR strings accept the usual \n / \t / \" escapes.
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


class _RawAttr:
    """Wrapper for a value that should be emitted verbatim (e.g. an MLIR
    symbol reference ``@foo``, which is not a plain string)."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


def _mlir_scalar(v: Any) -> str:
    """Best-effort format of a single Python value as an MLIR attribute.

    * ``_RawAttr`` -> emit its ``text`` verbatim
    * ``bool`` -> ``true`` / ``false``
    * ``int``  -> ``<v> : i64``
    * ``float``-> ``<v> : f64``
    * ``str``  -> ``"..."``
    * tuple/list -> ``[a, b, c]`` (nested scalars)
    * ``None`` -> ``unit``
    * otherwise stringify.
    """
    if isinstance(v, _RawAttr):
        return v.text
    if v is None:
        return "unit"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v} : i64"
    if isinstance(v, float):
        # MLIR accepts decimals; use repr for round-trip-friendliness.
        return f"{v!r} : f64"
    if isinstance(v, str):
        return _mlir_str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_mlir_scalar(x) for x in v) + "]"
    return _mlir_str(repr(v))


def _format_attr_dict(kv: Mapping[str, Any]) -> str:
    """Format ``{k=..., ...}`` MLIR attribute syntax. Empty -> ``{}``."""
    if not kv:
        return "{}"
    parts = []
    # Sort to get deterministic output — the dialect printer already
    # alphabetizes, so matching here avoids a diff on first round-trip.
    for k in sorted(kv.keys()):
        parts.append(f"{k} = {_mlir_scalar(kv[k])}")
    return "{" + ", ".join(parts) + "}"


def _memory_attrs(m: Memory) -> dict:
    """Pull the structured fields off a Memory plus any extras."""
    d = {
        "capacity_bytes": int(m.capacity_bytes),
        "lanes": int(m.lanes),
        "dtype": str(m.dtype),
        "parallel_units": int(m.parallel_units),
        "scope": str(m.scope),
    }
    # Extras live on __dict__ but are filtered so we don't duplicate the
    # dataclass fields.
    known = set(d.keys()) | {"name"}
    for k, v in m.__dict__.items():
        if k in known:
            continue
        d[k] = v
    return d


def _op_attrs(o: Op) -> dict:
    d = {
        "lanes": int(o.lanes),
        "latency": int(o.latency),
        "throughput": float(o.throughput),
        "cycles_per_elem": float(o.cycles_per_elem),
        "energy_pJ": float(o.energy_pJ),
    }
    if o.emit:
        d["emit"] = str(o.emit)
    known = set(d.keys()) | {"name", "emit"}
    for k, v in o.__dict__.items():
        if k in known:
            continue
        d[k] = v
    return d


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def target_to_mlir(t: Target) -> str:
    """Emit the ``pim_target`` MLIR text for a built ``Target``.

    The ``_PendingLevel`` tree stored under ``t.tn_root`` (populated by
    ``@allo.target``) provides the level structure; we walk it.  If
    ``tn_root`` is missing (e.g. the target was built directly via
    ``build_from_grid``), we fall back to emitting just the describe op
    plus flat memory/op symbols — matching the phase-1 surface.  The
    round-trip test drives only decorator-built targets, so the fallback
    is a safety net rather than a supported path.
    """
    lines: List[str] = []
    lines.append("module {")
    describe_attrs = {"parallel_units": int(t.parallel_units)}
    for k, v in (t.caps or {}).items():
        describe_attrs[f"cap_{k}"] = bool(v)
    lines.append(f"  pim_target.describe @{t.name} {_format_attr_dict(describe_attrs)}")

    # Emit host-scope memories and ops flat under the target (they are
    # not part of the user-declared grid — they sit on the synthetic
    # host root in Report 11 §3.7).
    for name, m in t.memories.items():
        if m.scope == "host":
            attrs = _memory_attrs(m)
            attrs["parent"] = _RawAttr(f"@{_sanitize(t.name)}")
            lines.append(f"  pim_target.memory @{_sanitize(name)} {_format_attr_dict(attrs)}")
    for name, o in t.ops.items():
        if name.startswith("host."):
            attrs = _op_attrs(o)
            attrs["parent"] = _RawAttr(f"@{_sanitize(t.name)}")
            attrs["is_host"] = True
            lines.append(
                f'  pim_target.op @{_sanitize(name)} {_format_attr_dict(attrs)}'
            )

    root = getattr(t, "tn_root", None)
    if root is not None:
        _emit_level(root, parent_name=t.name, indent=2, lines=lines,
                    device_memory_names=_device_memory_names(t))
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Level walker
# ---------------------------------------------------------------------------


def _emit_level(level, *, parent_name: str, indent: int,
                lines: List[str],
                device_memory_names: set) -> None:
    """Emit a single ``_PendingLevel`` as one or more nested unit ops
    (one per extent entry) plus its memory/op/stream/cost children."""
    pad = " " * indent
    axis = _sanitize(level.axis)
    extents = list(level.extents)
    mode = level.mode
    mapping_str = "[" + ", ".join(str(int(e)) for e in extents) + "]"
    lines.append(
        f'{pad}pim_target.unit @{axis} mapping {mapping_str} '
        f'mode {_mlir_str(mode)} parent @{parent_name} {{'
    )
    inner = indent + 2

    # Level-bound memories.
    for m in level.memories:
        # Skip host-scope memories here — they were already emitted at
        # the describe level. This shouldn't normally hit (host-scope
        # memories are passed through `host_memories=` on @tn.target),
        # but guard in case a user attaches one inside the grid.
        if m.scope == "host":
            continue
        attrs = _memory_attrs(m)
        attrs["parent"] = _RawAttr(f"@{axis}")
        lines.append(
            f'{" "*inner}pim_target.memory '
            f'@{_sanitize(m.name)} {_format_attr_dict(attrs)}'
        )

    # Level-bound ops.
    for o in level.ops:
        if o.name.startswith("host."):
            continue
        attrs = _op_attrs(o)
        attrs["parent"] = _RawAttr(f"@{axis}")
        lines.append(
            f'{" "*inner}pim_target.op '
            f'@{_sanitize(o.name)} {_format_attr_dict(attrs)}'
        )

    # Streams are MIMD-only per Rule R4, but the decorator-side check
    # already enforced that — emit whatever the tree carries.
    for s in level.streams:
        _emit_stream(s, parent_axis=axis, indent=inner, lines=lines)

    # Per-level cost — only emit when non-empty.
    if level.cost_attrs:
        cost_attrs = {}
        for k, v in level.cost_attrs.items():
            cost_attrs[k] = v if not callable(v) else repr(v)
        lines.append(f'{" "*inner}pim_target.cost {_format_attr_dict(cost_attrs)}')

    # Recurse into children.
    for child in level.children:
        _emit_level(child, parent_name=axis, indent=inner, lines=lines,
                    device_memory_names=device_memory_names)

    lines.append(f"{pad}}}")


def _emit_stream(stream: Mapping[str, Any], *, parent_axis: str,
                 indent: int, lines: List[str]) -> None:
    name = _sanitize(stream.get("name", "unnamed_stream"))
    element_type = stream.get("T", stream.get("element_type", "i32"))
    capacity = int(stream.get("N", stream.get("capacity", 0)))
    topology = str(stream.get("topology", "neighbor"))
    # element_type may be a Python type (e.g. int, uint32 alias) or a
    # string; stringify it to something MLIR can parse.
    element_type_str = _stream_type_to_mlir(element_type)
    pad = " " * indent
    lines.append(
        f'{pad}pim_target.stream @{name} '
        f'element_type {element_type_str} N {capacity} '
        f'topology {_mlir_str(topology)} parent @{parent_axis}'
    )


def _stream_type_to_mlir(t: Any) -> str:
    """Map a Python / allo dtype description to an MLIR type keyword."""
    if isinstance(t, type):
        # int -> i64, float -> f64 (rough but serviceable).
        if t is int:
            return "i64"
        if t is float:
            return "f64"
        if t is bool:
            return "i1"
        return "i32"
    if hasattr(t, "__name__"):  # numpy dtypes, aliases
        n = t.__name__
    else:
        n = str(t)
    # Accept aliases like "uint32" / "int16" / "fp16" / "f32".
    n_lower = n.lower()
    mapping = {
        "uint8": "i8", "int8": "i8",
        "uint16": "i16", "int16": "i16",
        "uint32": "i32", "int32": "i32",
        "uint64": "i64", "int64": "i64",
        "fp16": "f16", "float16": "f16",
        "fp32": "f32", "float32": "f32",
        "fp64": "f64", "float64": "f64",
        "f16": "f16", "f32": "f32", "f64": "f64",
        "bf16": "bf16",
        "bool": "i1",
    }
    return mapping.get(n_lower, "i32")


def _device_memory_names(t: Target) -> set:
    return {name for name, m in t.memories.items() if m.scope != "host"}


def _sanitize(name: str) -> str:
    """Make a Python identifier MLIR-friendly. MLIR symbol names
    accept letters, digits, and ``._-$``; we replace anything else with
    an underscore and prefix with an underscore if the first char is a
    digit.
    """
    if not name:
        return "_"
    out = []
    for i, ch in enumerate(name):
        if ch.isalnum() or ch in "._-$":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    if s[0].isdigit():
        s = "_" + s
    return s


__all__ = ["target_to_mlir"]
