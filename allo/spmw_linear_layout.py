# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW linear-layout algebra.

A `LinearLayout` is an F2-linear map from a tuple of named *input-dim*
bit-vectors to a tuple of named *output-dim* bit-vectors. The autoscheduler
uses `LinearLayout` to construct candidate `Placement`s algebraically (e.g.
the Samsung HBM-PIM bank swizzle from Zhou et al. ASPLOS '26 §5.4) rather
than hand-listing them.

Layouts are stored in *basis form* — a dict keyed by input-dim name; each
value is a list of basis vectors, one per bit of that input dim. Each basis
vector is a tuple of integers, one per output dim, interpreted as a bitmask
over that output dim. See `experiments/allo/spec 013` §A for the matrix
view and §B for the API contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .spmw_target import MemoryRef, Register, UnitId


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _ilog2_exact(n: int) -> int:
    """Return k such that 2**k == n. Raise ValueError otherwise."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"size {n!r} is not a positive power of 2")
    k = 0
    while (1 << k) < n:
        k += 1
    return k


# --------------------------------------------------------------------- #
# LinearLayout
# --------------------------------------------------------------------- #


class LinearLayout:
    """F2-linear map between named bit-vector dims.

    See module docstring + spec 013 §A/§B for semantics. Layouts are
    enumerator-ephemeral: built inside `_*_enumerate`, materialised to
    `Register`/`MemoryRef` via `materialise_handle`, then discarded
    before reaching codegen.
    """

    def __init__(
        self,
        bases: dict[str, list[tuple[int, ...]]],
        out_dims: tuple[str, ...],
        out_sizes: tuple[int, ...] | None = None,
    ):
        if len(set(out_dims)) != len(out_dims):
            raise ValueError(f"out_dims must be unique, got {out_dims!r}")
        n_out = len(out_dims)

        # Normalise bases — coerce inner items to tuples of ints.
        norm_bases: dict[str, list[tuple[int, ...]]] = {}
        for name, vecs in bases.items():
            norm_vecs: list[tuple[int, ...]] = []
            for v in vecs:
                t = tuple(int(x) for x in v)
                if len(t) != n_out:
                    raise ValueError(
                        f"basis vector for input dim {name!r} has length "
                        f"{len(t)}; expected {n_out} (one entry per out_dim)"
                    )
                for x in t:
                    if x < 0:
                        raise ValueError(
                            f"basis entries must be non-negative, got {x} in "
                            f"input dim {name!r}"
                        )
                norm_vecs.append(t)
            norm_bases[name] = norm_vecs

        if out_sizes is None:
            inferred = [1] * n_out
            for vecs in norm_bases.values():
                for v in vecs:
                    for j, x in enumerate(v):
                        # next pow-2 ≥ x+1, but at least 1
                        if x >= inferred[j]:
                            target = 1
                            while target <= x:
                                target <<= 1
                            inferred[j] = target
            out_sizes = tuple(inferred)
        else:
            out_sizes = tuple(int(s) for s in out_sizes)
            if len(out_sizes) != n_out:
                raise ValueError(
                    f"out_sizes has {len(out_sizes)} entries; expected {n_out}"
                )
            for s in out_sizes:
                _ilog2_exact(s)  # validates power of 2
            # Validate basis entries fit within out_sizes.
            for name, vecs in norm_bases.items():
                for v in vecs:
                    for j, x in enumerate(v):
                        if x >= out_sizes[j]:
                            raise ValueError(
                                f"basis entry {x} for input dim {name!r} "
                                f"exceeds out_sizes[{j}]={out_sizes[j]}"
                            )

        self.bases: dict[str, list[tuple[int, ...]]] = norm_bases
        self.out_dims: tuple[str, ...] = tuple(out_dims)
        self.out_sizes: tuple[int, ...] = out_sizes

    # ------------------------------------------------------------------ #
    # Class constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def identity(
        cls,
        dims: dict[str, int],
        out_dims: tuple[str, ...] | None = None,
    ) -> "LinearLayout":
        """Identity layout — input dim `d` of size 2^k maps to output dim
        `d` via the k basis vectors (1, 2, 4, ..., 2^(k-1)). Input dims
        whose name is not in `out_dims` get all-zero basis vectors (they
        do not contribute to any output). `out_dims` defaults to
        `tuple(dims)`.
        """
        if out_dims is None:
            out_dims = tuple(dims)
        n_out = len(out_dims)
        out_idx = {name: i for i, name in enumerate(out_dims)}
        out_sizes = [1] * n_out
        bases: dict[str, list[tuple[int, ...]]] = {}
        for name, size in dims.items():
            k = _ilog2_exact(size)
            vecs: list[tuple[int, ...]] = []
            j = out_idx.get(name)
            for bit in range(k):
                vec = [0] * n_out
                if j is not None:
                    vec[j] = 1 << bit
                vecs.append(tuple(vec))
            bases[name] = vecs
            if j is not None and size > out_sizes[j]:
                out_sizes[j] = size
        # If an output dim has no contributing input dim, give it size 1.
        return cls(bases=bases, out_dims=out_dims, out_sizes=tuple(out_sizes))

    @classmethod
    def zero(
        cls,
        in_dims: dict[str, int],
        out_dims: tuple[str, ...],
        out_sizes: tuple[int, ...],
    ) -> "LinearLayout":
        """All-zero map. Each input bit produces a zero basis vector."""
        n_out = len(out_dims)
        bases: dict[str, list[tuple[int, ...]]] = {}
        for name, size in in_dims.items():
            k = _ilog2_exact(size)
            bases[name] = [tuple([0] * n_out) for _ in range(k)]
        return cls(bases=bases, out_dims=out_dims, out_sizes=out_sizes)

    @classmethod
    def optimal_swizzle(
        cls,
        base: "LinearLayout",
        *,
        vec_dims: tuple[str, ...],
        bank_dims: tuple[str, ...],
        segment_dims: tuple[str, ...],
    ) -> "LinearLayout":
        """Find the smallest XOR-augmentation of `base` that makes the
        segment-dim image non-zero on the bank-output dims.

        Implements Zhou et al. ASPLOS '26 §5.4. For Samsung GEMV with
        base = identity({"grf":8,"bank":16,"tile":2}, out=("grf","bank"))
        and `segment_dims=("tile",)`, this returns a layout whose
        `bases["tile"]` = `[(0, 1)]` — i.e. `tile_parity ⊕ bank_bit_0`.
        """
        if not segment_dims:
            return cls(
                bases={k: list(v) for k, v in base.bases.items()},
                out_dims=base.out_dims,
                out_sizes=base.out_sizes,
            )
        for d in segment_dims:
            if d not in base.bases:
                raise ValueError(
                    f"optimal_swizzle: segment dim {d!r} not in base.bases "
                    f"(known: {sorted(base.bases)})"
                )

        bank_idxs = []
        for bd in bank_dims:
            if bd not in base.out_dims:
                raise ValueError(
                    f"optimal_swizzle: bank dim {bd!r} not in base.out_dims "
                    f"({base.out_dims})"
                )
            bank_idxs.append(base.out_dims.index(bd))

        # vec_dims are tracked for documentation / future use — for the
        # current Samsung case they pin the GRF (input) dim that must
        # remain conflict-free, which is automatic when the swizzle only
        # touches segment-dim bits. Validate they exist.
        for vd in vec_dims:
            if vd not in base.bases:
                raise ValueError(
                    f"optimal_swizzle: vec dim {vd!r} not in base.bases "
                    f"(known: {sorted(base.bases)})"
                )

        # Build a mutable copy of bases.
        new_bases = {k: [list(v) for v in vecs] for k, vecs in base.bases.items()}

        def _is_conflict_free() -> bool:
            tmp = cls(
                bases={k: [tuple(v) for v in vecs] for k, vecs in new_bases.items()},
                out_dims=base.out_dims,
                out_sizes=base.out_sizes,
            )
            return tmp.describes_conflict_free(
                bank_dims=bank_dims, varying_inputs=segment_dims
            )

        if _is_conflict_free():
            # Already conflict-free — return a clean copy.
            return cls(
                bases={k: [tuple(v) for v in vecs] for k, vecs in new_bases.items()},
                out_dims=base.out_dims,
                out_sizes=base.out_sizes,
            )

        # Search smallest XOR-augmentation: for each (segment_dim, bit, bank_dim,
        # bank_bit), try setting that bit and check conflict freedom. Pick the
        # first success in deterministic order (matches spec example).
        for sd in segment_dims:
            for bit_i in range(len(new_bases[sd])):
                for bd_idx, bd in zip(bank_idxs, bank_dims):
                    for bank_bit in range(_ilog2_exact(base.out_sizes[bd_idx])):
                        mask = 1 << bank_bit
                        new_bases[sd][bit_i][bd_idx] ^= mask
                        if _is_conflict_free():
                            return cls(
                                bases={
                                    k: [tuple(v) for v in vecs]
                                    for k, vecs in new_bases.items()
                                },
                                out_dims=base.out_dims,
                                out_sizes=base.out_sizes,
                            )
                        # revert
                        new_bases[sd][bit_i][bd_idx] ^= mask

        raise ValueError(
            "optimal_swizzle: no single-bit XOR-augmentation makes the layout "
            "conflict-free; larger search required (not implemented)"
        )

    # ------------------------------------------------------------------ #
    # Core operations
    # ------------------------------------------------------------------ #

    def apply(self, **inputs: int) -> tuple[int, ...]:
        """Evaluate `L(inputs)` over F2: XOR-accumulate the basis vectors
        for each set bit of each input dim.
        """
        n_out = len(self.out_dims)
        out = [0] * n_out
        for name, value in inputs.items():
            if name not in self.bases:
                raise KeyError(
                    f"apply: input dim {name!r} not in this layout "
                    f"(known: {sorted(self.bases)})"
                )
            vecs = self.bases[name]
            v = int(value)
            for bit_i, vec in enumerate(vecs):
                if (v >> bit_i) & 1:
                    for j in range(n_out):
                        out[j] ^= vec[j]
        return tuple(out)

    def matrix(self) -> "np.ndarray":
        """Return the concrete F2 matrix view. Rows are concatenated by
        out_dim (each out_dim contributes log2(out_size) rows, ordered
        bit-0 first). Cols are concatenated by input-dim insertion order
        (each input dim contributes len(bases[name]) cols, ordered bit-0
        first). Entries are 0/1 uint8.
        """
        row_widths = [_ilog2_exact(s) for s in self.out_sizes]
        n_rows = sum(row_widths)
        col_chunks = [(name, len(vecs)) for name, vecs in self.bases.items()]
        n_cols = sum(c for _, c in col_chunks)
        M = np.zeros((n_rows, n_cols), dtype=np.uint8)
        col_base = 0
        for name, n in col_chunks:
            for bit_i, vec in enumerate(self.bases[name]):
                row_base = 0
                for j, width in enumerate(row_widths):
                    word = vec[j]
                    for b in range(width):
                        if (word >> b) & 1:
                            M[row_base + b, col_base + bit_i] = 1
                    row_base += width
            col_base += n
        return M

    def compose(self, other: "LinearLayout") -> "LinearLayout":
        """Return `other ∘ self`: outputs of `self` feed inputs of `other`.

        For this to type-check, every `self.out_dim` must appear as an
        input dim of `other` with matching size (so the bit-vector
        plumbing lines up).
        """
        # Build a basis-form composition: for each input dim of self with
        # basis vector v (a tuple of integers over self.out_dims), feed v
        # as an input vector to `other` and collect the result over
        # other.out_dims.
        for name in self.out_dims:
            if name not in other.bases:
                raise ValueError(
                    f"compose: self.out_dims contains {name!r} but "
                    f"other.bases does not (other has {sorted(other.bases)})"
                )
            expected_size = self.out_sizes[self.out_dims.index(name)]
            other_bits = len(other.bases[name])
            if (1 << other_bits) != expected_size:
                raise ValueError(
                    f"compose: dim {name!r} has size {expected_size} on self "
                    f"but {1 << other_bits} on other"
                )

        new_bases: dict[str, list[tuple[int, ...]]] = {}
        for in_name, vecs in self.bases.items():
            new_vecs: list[tuple[int, ...]] = []
            for vec in vecs:
                # Feed vec into `other` keyed by self.out_dims.
                kwargs = {
                    self.out_dims[j]: vec[j] for j in range(len(self.out_dims))
                }
                new_vecs.append(other.apply(**kwargs))
            new_bases[in_name] = new_vecs
        return LinearLayout(
            bases=new_bases,
            out_dims=other.out_dims,
            out_sizes=other.out_sizes,
        )

    def product(self, other: "LinearLayout") -> "LinearLayout":
        """Disjoint block-diagonal combine. Input dims of self and other
        must not overlap; output dims are concatenated.
        """
        overlap = set(self.bases) & set(other.bases)
        if overlap:
            raise ValueError(
                f"product: input dims overlap: {sorted(overlap)}"
            )
        if set(self.out_dims) & set(other.out_dims):
            raise ValueError(
                "product: output dims overlap: "
                f"{set(self.out_dims) & set(other.out_dims)}"
            )

        new_out_dims = tuple(self.out_dims) + tuple(other.out_dims)
        new_out_sizes = tuple(self.out_sizes) + tuple(other.out_sizes)
        n_self = len(self.out_dims)
        n_other = len(other.out_dims)
        new_bases: dict[str, list[tuple[int, ...]]] = {}
        # Self's basis vectors: extend with zeros on other side.
        for name, vecs in self.bases.items():
            new_bases[name] = [tuple(list(v) + [0] * n_other) for v in vecs]
        # Other's basis vectors: prepend zeros on self side.
        for name, vecs in other.bases.items():
            new_bases[name] = [tuple([0] * n_self + list(v)) for v in vecs]
        return LinearLayout(
            bases=new_bases,
            out_dims=new_out_dims,
            out_sizes=new_out_sizes,
        )

    def sublayout(self, input_dims: tuple[str, ...]) -> "LinearLayout":
        """Project onto a subset of input dims (keeps the same outputs)."""
        for d in input_dims:
            if d not in self.bases:
                raise KeyError(f"sublayout: input dim {d!r} not in this layout")
        new_bases = {d: [tuple(v) for v in self.bases[d]] for d in input_dims}
        return LinearLayout(
            bases=new_bases,
            out_dims=self.out_dims,
            out_sizes=self.out_sizes,
        )

    def invert(self) -> "LinearLayout":
        """Invert this layout over F2.

        Requires the matrix to be square and full-rank — i.e. total input
        bits == total output bits and the columns are F2-linearly
        independent. Returns a new LinearLayout whose input/output dims
        and sizes are *swapped*: the original input dims become the
        output dims and vice versa.
        """
        n_in_bits = sum(len(v) for v in self.bases.values())
        n_out_bits = sum(_ilog2_exact(s) for s in self.out_sizes)
        if n_in_bits != n_out_bits:
            raise ValueError(
                f"invert: matrix is {n_out_bits}x{n_in_bits}, not square"
            )

        M = self.matrix()  # shape (n_out_bits, n_in_bits)
        n = n_in_bits
        # Augment with identity, run Gauss-Jordan over F2.
        aug = np.concatenate([M, np.eye(n, dtype=np.uint8)], axis=1)
        for col in range(n):
            # find pivot row
            pivot = -1
            for r in range(col, n):
                if aug[r, col]:
                    pivot = r
                    break
            if pivot == -1:
                raise ValueError("invert: matrix is not full-rank over F2")
            if pivot != col:
                tmp = aug[col].copy()
                aug[col] = aug[pivot]
                aug[pivot] = tmp
            for r in range(n):
                if r != col and aug[r, col]:
                    aug[r] ^= aug[col]
        inv = aug[:, n:]

        # Build the new LinearLayout from `inv`.
        # New input dims = self.out_dims (sizes = self.out_sizes).
        # New output dims = self.bases keys (sizes = 2**len(basis)).
        new_in_dims = list(self.out_dims)
        new_in_sizes = list(self.out_sizes)
        new_out_dims = tuple(self.bases.keys())
        new_out_sizes = tuple(2 ** len(self.bases[k]) for k in new_out_dims)

        row_widths = [_ilog2_exact(s) for s in new_out_sizes]

        new_bases: dict[str, list[tuple[int, ...]]] = {
            d: [] for d in new_in_dims
        }
        col_base = 0
        for in_name, in_size in zip(new_in_dims, new_in_sizes):
            k = _ilog2_exact(in_size)
            for bit_i in range(k):
                col = col_base + bit_i
                # Pack the column into per-out-dim bitmasks.
                vec = []
                row_base = 0
                for width in row_widths:
                    word = 0
                    for b in range(width):
                        if inv[row_base + b, col]:
                            word |= 1 << b
                    vec.append(word)
                    row_base += width
                new_bases[in_name].append(tuple(vec))
            col_base += k

        return LinearLayout(
            bases=new_bases,
            out_dims=new_out_dims,
            out_sizes=new_out_sizes,
        )

    def is_surjective(self) -> bool:
        """True iff the F2 rank equals the total number of output bits."""
        M = self.matrix().astype(np.uint8)
        rows, cols = M.shape
        # Gaussian elimination over F2.
        rank = 0
        r = 0
        for c in range(cols):
            pivot = -1
            for rr in range(r, rows):
                if M[rr, c]:
                    pivot = rr
                    break
            if pivot == -1:
                continue
            if pivot != r:
                tmp = M[r].copy()
                M[r] = M[pivot]
                M[pivot] = tmp
            for rr in range(rows):
                if rr != r and M[rr, c]:
                    M[rr] ^= M[r]
            rank += 1
            r += 1
            if r == rows:
                break
        return rank == rows

    def describes_conflict_free(
        self,
        *,
        bank_dims: tuple[str, ...] | None = None,
        varying_inputs: tuple[str, ...],
        out_dim: str | None = None,
    ) -> bool:
        """For every non-zero vector v in span(varying_inputs), is the
        image of v projected onto `bank_dims` non-zero?

        For convenience, `out_dim=...` (singular) is accepted as a
        synonym for `bank_dims=(out_dim,)`.
        """
        if bank_dims is None:
            if out_dim is None:
                raise TypeError(
                    "describes_conflict_free: pass bank_dims=(...) or out_dim=..."
                )
            bank_dims = (out_dim,)
        bank_idxs = [self.out_dims.index(d) for d in bank_dims]

        # Enumerate all nonzero (varying_inputs) vectors. Each input dim
        # contributes len(bases[d]) bits.
        total_bits = sum(len(self.bases[d]) for d in varying_inputs)
        for code in range(1, 1 << total_bits):
            bit = 0
            kwargs: dict[str, int] = {}
            for d in varying_inputs:
                nb = len(self.bases[d])
                val = (code >> bit) & ((1 << nb) - 1)
                kwargs[d] = val
                bit += nb
            out = self.apply(**kwargs)
            # Project onto bank_dims — if all zero, conflict.
            if all(out[j] == 0 for j in bank_idxs):
                return False
        return True

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"LinearLayout(bases={self.bases!r}, "
            f"out_dims={self.out_dims!r}, out_sizes={self.out_sizes!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LinearLayout):
            return NotImplemented
        return (
            self.bases == other.bases
            and self.out_dims == other.out_dims
            and self.out_sizes == other.out_sizes
        )


# --------------------------------------------------------------------- #
# Handle materialisation
# --------------------------------------------------------------------- #


def materialise_handle(
    layout: "LinearLayout",
    *,
    target: Any,
    out_dim: str,
    fixed: dict[str, int] | None = None,
    symbol_table: dict[str, Any] | None = None,
    handle_table: dict[str, Any] | None = None,
) -> Any:
    """Evaluate `layout` symbolically at a chosen output coordinate and
    return the target handle that owns the resulting slot.

    Semantics:
      - For each input dim of `layout`:
          * If in `fixed`, the concrete int is used.
          * Otherwise, the dim is bound to `symbol_table[dim]`. If absent,
            it defaults to a fresh `UnitId(level=N)` where N is 1-indexed
            by input-dim insertion order — sufficient for single-axis
            cases but enumerators with multi-axis trees should pass
            `symbol_table` explicitly.
      - Each basis vector's `out_dim`-th entry is treated as a scalar
        multiplier on the bound symbol/value; contributions sum.
      - For the canonical Samsung case (out_dim="bank",
        bases["bank"]=[(0,1),(0,2),(0,4),(0,8)], bases["tile"]=[(0,1)]),
        passing `symbol_table={"bank": 2*pid}` and `fixed={"tile": 0}`
        yields `idx = 2*pid + 0`. That SymExpr lowers to "EVEN_BANK" via
        `spmw_codegen._bank_parity`.

    The "scalar multiplier" semantics is a deliberate simplification of
    the full F2 algebra: it produces the SymExpr forms the codegen
    classifier (`_bank_parity`) expects. Callers that need full F2 bit
    expansion can use `layout.apply(...)` directly; this helper is the
    pragmatic bridge between LinearLayout and the current
    Register/MemoryRef handle plumbing.

    `handle_table[out_dim]` overrides the auto-resolved store. Default
    auto-resolve: `target.<out_dim>` then `target.<out_dim>s` (e.g.
    "bank" -> `target.banks`).
    """
    fixed = dict(fixed or {})
    symbol_table = dict(symbol_table or {})

    if out_dim not in layout.out_dims:
        raise KeyError(
            f"materialise_handle: out_dim {out_dim!r} not in layout.out_dims "
            f"({layout.out_dims})"
        )
    out_j = layout.out_dims.index(out_dim)

    # Resolve store handle.
    if handle_table is not None and out_dim in handle_table:
        store = handle_table[out_dim]
    else:
        store = getattr(target, out_dim, None)
        if store is None:
            store = getattr(target, out_dim + "s", None)
        if store is None:
            raise KeyError(
                f"materialise_handle: no target handle found for "
                f"out_dim {out_dim!r}; supply via handle_table=..."
            )

    # Allocate default symbols for any unfixed dim not in symbol_table.
    auto_level = 1
    for in_name in layout.bases.keys():
        if in_name in fixed or in_name in symbol_table:
            continue
        symbol_table[in_name] = UnitId(level=auto_level, unit=None)
        auto_level += 1

    # Build the index expression as a sum of per-input contributions.
    #
    # For each input dim, examine its basis vectors' entries for the
    # selected output dim. The contribution semantics are:
    #
    #   - Identity-on-this-output: basis bit i contributes mask 2^i (so
    #     the column is the canonical (1, 2, 4, ..., 2^(k-1))). Binding
    #     the input to a symbol yields contribution = symbol * scale
    #     where scale is 1 (== the symbol IS this output coordinate).
    #     For a fixed integer input, contribution = int_value.
    #
    #   - Scaled identity: column is (s, 2s, 4s, ...) for some s. Same
    #     as above but contribution = symbol * s (fixed: int_value * s).
    #
    #   - Single-bit input (k == 1): one mask value m. For symbol s the
    #     contribution is s * m; for fixed value v in {0, 1} it is v * m.
    #     The Samsung "tile" swizzle is exactly this case (m = 1).
    #
    #   - Other arbitrary columns: only fixed inputs are supported here
    #     (we cannot decompose an arbitrary symbol into its F2 bits and
    #     re-XOR them onto a SymExpr). A fixed input falls back to
    #     XOR-of-masks (per F2 algebra), which on disjoint bits matches
    #     plain addition; same form `_bank_parity` recognises.
    idx_expr: Any = 0
    for in_name, vecs in layout.bases.items():
        masks = [vec[out_j] for vec in vecs]
        if all(m == 0 for m in masks):
            continue

        # Detect scaled-identity pattern: masks = (s, 2s, 4s, ...) for
        # some positive s with all distinct bit-positions.
        scale = None
        if masks and masks[0] != 0:
            s = masks[0]
            if all(masks[i] == (s << i) for i in range(len(masks))):
                scale = s

        if in_name in fixed:
            v = int(fixed[in_name])
            # Direct: XOR the basis masks for set bits. On a properly-
            # constructed (scaled-)identity column with disjoint bits,
            # this XOR equals addition.
            contribution_int = 0
            for bit_i, m in enumerate(masks):
                if (v >> bit_i) & 1:
                    contribution_int ^= m
            if contribution_int == 0:
                continue
            contribution: Any = contribution_int
        else:
            sym = symbol_table[in_name]
            if scale is not None:
                # Identity-style column: contribution is scale * sym.
                contribution = sym if scale == 1 else scale * sym
            elif len(masks) == 1:
                # Single-bit input dim with arbitrary mask m.
                m = masks[0]
                contribution = sym if m == 1 else m * sym
            else:
                raise NotImplementedError(
                    f"materialise_handle: input dim {in_name!r} has a "
                    f"non-(scaled-)identity column ({masks!r}) and is "
                    f"not in `fixed`; cannot decompose a symbolic input "
                    f"into F2 bits. Pass a fixed value or extend the helper."
                )

        if isinstance(idx_expr, int) and idx_expr == 0:
            idx_expr = contribution
        else:
            idx_expr = idx_expr + contribution

    if isinstance(store, Register):
        return store
    if hasattr(store, "__getitem__"):
        return store[idx_expr]
    return store
