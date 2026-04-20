# Backend target descriptions

Every file here returns an `allo.pim.Target` via a `build_*()` function. The
compiler (`allo.pim.lower`, `allo.pim.execute`) consumes that `Target` object
regardless of how it was constructed, so both construction forms below are
interchangeable.

## Two construction forms

**Flat form** — call `t.memory(...)`, `t.op(...)`, `t.pattern(...)` directly
on a `Target`. One file, three tables. Easiest for small targets and quick
edits; no structural information about how levels nest.

```python
from allo.pim.target import Target

def build_simple() -> Target:
    t = Target("my_target", parallel_units=1024)
    t.cap(has_mac=True)
    t.memory("bank", capacity_bytes=32 << 20, scope="bank",
             parallel_units=t.parallel_units)
    t.op("my.add", lanes=16, latency=10, emit="...")
    t.pattern(match=..., lower=..., name="...")
    return t
```

Backends still shipping in flat form: `tenon_pim_v0.py`, `newton.py`,
`apu_v1.py`, `apu_v2.py`.

**Grid-tree form** — build a nested tree of `Grid` and `Leaf` nodes and flatten
it with `build_from_grid(...)`. Tree shape matches the physical hierarchy
(channels → banks → GRF, or DPUs → tasklets, …); `parallel_units` for each
memory is computed from the subtree extent above it. This is the canonical
form (report 07 §2).

```python
from allo.pim.target import Memory, Op, Leaf, grid, build_from_grid

def build_simple():
    leaf = Leaf(
        memory=[Memory("bank", capacity_bytes=32 << 20, scope="bank")],
        ops=[Op("my.add", lanes=16, latency=10, emit="...")],
    )
    t = build_from_grid(
        "my_target",
        grid((64, 16), ["channel", "bank"], child=leaf),
        caps=dict(has_mac=True),
    )
    t.pattern(match=..., lower=..., name="...")
    return t
```

Backends shipping in grid-tree form: `samsung.py`, `aim.py`, `upmem.py`.

## API reference (grid-tree)

- `Leaf(memory=None, ops=None)` — leaf-level memories and ops. Both args
  accept a single object or a list.
- `Grid(shape, name, child, memory=None, sync=None, broadcast_ops=None,
  ops=None)` — one level of the tree. `shape` is an `int` extent, or a tuple
  (e.g. `(64, 16)`) that expands into a chain of single-extent levels with
  names from the `name` list. `child` is the next `Grid` or a `Leaf`.
- `grid(shape, name, child, **kw)` — convenience factory, equivalent to
  `Grid(...)`.
- `build_from_grid(name, root, caps=None, host_memories=None, host_ops=None)
  -> Target` — walks the tree, stamps each attached `Memory` with
  `parallel_units = product of extents from root to its level`, copies every
  op into `target.ops`, prepends `host_memories`, appends `host_ops`, and
  applies `caps`. Returns a `Target` the rest of the compiler consumes as-is.

The module is also exposed as `spmw` for the style in report 07 §2.3:
`from allo.pim import spmw` then `spmw.grid(...)`, `spmw.Leaf(...)`.

## Which form to use

- **Pick grid-tree** when the target has a clear nested hierarchy — cost
  rollup and capability-gap lifting (report 07 §2.6) become structural.
- **Pick flat** for a shallow single-level target (e.g. a sketch SRAM-CIM)
  where the tree is degenerate anyway.

Both forms produce the same `Target` type, so `allo.pim.lower` / `execute` /
`analyze_cost` are unchanged. You can migrate a backend at any time; a
byte-identical regression harness for the three DRAM-PIM backends lives at
`experiments/E3_grid_tree_refactor/dump_backends.py`.
