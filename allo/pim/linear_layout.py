"""Linear layouts over F2 — the Triton abstraction, as Python.

A LinearLayout is an 𝔽₂-linear map from a tuple of hardware-location
bit-vectors to a tuple of logical-tensor-coordinate bit-vectors, stored as a
list of basis vectors per named input dimension. Each basis vector gives the
output tuple produced when exactly one input bit is set.

  L(x1, x2, …) = Σ_i  bases[dim_i][bit_of(x_i)]         (XOR in 𝔽₂)

The field is 𝔽₂ so + is XOR and · is AND. All layout parameters (channels,
banks, lanes, tasklets) are powers of two, which is exactly the regime where
this algebra applies without padding.

Operations supported (matching Triton's LinearLayout.h):

  apply(**inputs)                   evaluate L at a point
  compose(other)                    matrix multiply: (other ∘ self)(x) = other(self(x))
  sublayout(input_dims, output_dims)  project onto a subset of named dims
  is_surjective()                   does the image cover the full output space?

Naming conventions (our PIM analogue of Triton's register/lane/warp/block):

  input  dims : 'element' (within a bank-burst), 'lane' (tasklet / simd lane),
                'bank', 'channel', 'dpu', 'rank', 'tile', …
  output dims : 'dim0', 'dim1' (tensor coordinates), or 'offset' for memory.

Dimensions that have sizes that aren't powers of two are not supported (matches
Triton). Zero-width dims are allowed (empty basis list).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _log2_exact(n: int) -> int:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"size must be a positive power of two, got {n}")
    return n.bit_length() - 1


def _bits_of(x: int, nbits: int) -> List[int]:
    return [(x >> i) & 1 for i in range(nbits)]


# ----------------------------------------------------------------------------
# LinearLayout
# ----------------------------------------------------------------------------

@dataclass
class LinearLayout:
    """𝔽₂-linear layout stored as per-input-dim basis lists.

    Fields:
      bases      : ordered dict  input_dim_name -> list of output tuples.
                   `bases[dim][i]` is the output tuple produced by setting
                   input bit i of `dim` to 1 (with all other bits zero).
                   len(bases[dim]) == log2(size of input dim).
      out_dims   : ordered list of output dim names; each output tuple
                   matches their order. Sizes come from the max value along
                   each output axis after enumeration, rounded up to the
                   next power of two.
    """
    bases: Dict[str, List[Tuple[int, ...]]] = field(default_factory=dict)
    out_dims: Tuple[str, ...] = ()

    # -- constructors -------------------------------------------------------

    @classmethod
    def empty(cls, out_dims: Sequence[str] = ("offset",)) -> "LinearLayout":
        return cls(bases={}, out_dims=tuple(out_dims))

    @classmethod
    def identity1d(cls, input_name: str, size: int,
                   out_name: str = "offset") -> "LinearLayout":
        """Contiguous placement: input bit i -> output bit i."""
        n = _log2_exact(size)
        return cls(bases={input_name: [(1 << i,) for i in range(n)]},
                   out_dims=(out_name,))

    # -- shape accessors ----------------------------------------------------

    def in_dims(self) -> Tuple[str, ...]:
        return tuple(self.bases.keys())

    def in_size(self, dim: str) -> int:
        return 1 << len(self.bases[dim])

    def out_size(self, out: str) -> int:
        """Power-of-two cover of the max value seen along this output dim."""
        idx = self.out_dims.index(out)
        m = 0
        for dim in self.bases:
            for tup in self.bases[dim]:
                if tup[idx] > m:
                    m = tup[idx]
        if m == 0:
            return 1
        return 1 << (m.bit_length())

    # -- core evaluation ----------------------------------------------------

    def apply(self, **inputs) -> Tuple[int, ...]:
        out = [0] * len(self.out_dims)
        for dim, val in inputs.items():
            if dim not in self.bases:
                if val != 0:
                    raise KeyError(f"unknown input dim {dim!r}")
                continue
            bits = _bits_of(val, len(self.bases[dim]))
            for i, b in enumerate(bits):
                if b:
                    tup = self.bases[dim][i]
                    for j, v in enumerate(tup):
                        out[j] ^= v
        return tuple(out)

    # -- structural ops -----------------------------------------------------

    def compose(self, other: "LinearLayout") -> "LinearLayout":
        """Return `other ∘ self`.  Self's outputs feed other's inputs by
        matching dim names. Requires self.out_dims ⊆ other.in_dims."""
        for d in self.out_dims:
            if d not in other.bases:
                raise ValueError(
                    f"compose: self output dim {d!r} not an input of other")
        new_bases: Dict[str, List[Tuple[int, ...]]] = {}
        for in_dim, vecs in self.bases.items():
            out_vecs: List[Tuple[int, ...]] = []
            for tup in vecs:
                # each output tuple from self acts as the input to other
                kwargs = {name: tup[idx]
                          for idx, name in enumerate(self.out_dims)}
                out_vecs.append(other.apply(**kwargs))
            new_bases[in_dim] = out_vecs
        return LinearLayout(bases=new_bases, out_dims=other.out_dims)

    def sublayout(self, input_dims: Optional[Iterable[str]] = None,
                  output_dims: Optional[Iterable[str]] = None) -> "LinearLayout":
        """Restrict to a subset of input/output dims. Useful for asking
        "what does just the bank + channel axes do?"."""
        in_keep = list(input_dims) if input_dims is not None else list(self.bases)
        out_keep = tuple(output_dims) if output_dims is not None else self.out_dims
        out_idx = [self.out_dims.index(d) for d in out_keep]
        new_bases = {d: [tuple(tup[i] for i in out_idx)
                         for tup in self.bases[d]]
                     for d in in_keep if d in self.bases}
        return LinearLayout(bases=new_bases, out_dims=out_keep)

    def product(self, other: "LinearLayout") -> "LinearLayout":
        """Block-diagonal product: combine two independent layouts whose input
        and output dim names don't overlap."""
        if set(self.bases) & set(other.bases):
            raise ValueError("product: input dims overlap")
        if set(self.out_dims) & set(other.out_dims):
            raise ValueError("product: output dims overlap")
        new_bases: Dict[str, List[Tuple[int, ...]]] = {}
        all_out = tuple(self.out_dims) + tuple(other.out_dims)
        pad_self = len(other.out_dims)
        pad_other = len(self.out_dims)
        for d, vecs in self.bases.items():
            new_bases[d] = [tup + (0,) * pad_self for tup in vecs]
        for d, vecs in other.bases.items():
            new_bases[d] = [(0,) * pad_other + tup for tup in vecs]
        return LinearLayout(bases=new_bases, out_dims=all_out)

    # -- analyses -----------------------------------------------------------

    def is_surjective(self) -> bool:
        """Does the image cover every output position in the product of
        per-output-dim sizes?  Computed via Gaussian elimination over 𝔽₂."""
        # Flatten all basis vectors into packed integers whose bits are all
        # the output-dim bits concatenated.
        offsets, total = [], 0
        for d in self.out_dims:
            s = max(1, self.out_size(d))
            nbits = _log2_exact(s)
            offsets.append(total)
            total += nbits
        rows: List[int] = []
        for vecs in self.bases.values():
            for tup in vecs:
                packed = 0
                for i, v in enumerate(tup):
                    packed |= v << offsets[i]
                rows.append(packed)
        # row-reduce over 𝔽₂; count non-zero pivots
        pivot_for_bit = {}
        for r in rows:
            v = r
            while v:
                lsb = v & -v
                if lsb in pivot_for_bit.values():
                    # reduce
                    for piv_row in rows:
                        if piv_row & lsb and piv_row != v:
                            v ^= piv_row
                            break
                    else:
                        break
                else:
                    pivot_for_bit[v.bit_length() - 1] = lsb
                    break
        return len(pivot_for_bit) == total

    def describes_conflict_free(self, contention_dim: str) -> bool:
        """Quick conflict check: for a fixed value of `contention_dim`
        (bank, say), do the other input dims' bases cover distinct output
        bits?  Used to sanity-check bank-interleaved placements."""
        others = [d for d in self.bases if d != contention_dim]
        seen_bits = 0
        for d in others:
            for tup in self.bases[d]:
                packed = 0
                for v in tup:
                    packed |= v
                if seen_bits & packed:
                    return False
                seen_bits |= packed
        return True

    # -- debug --------------------------------------------------------------

    def pretty(self) -> str:
        lines = [f"LinearLayout (out_dims={list(self.out_dims)}):"]
        for d, vecs in self.bases.items():
            lines.append(f"  {d} ({len(vecs)} bits):")
            for i, tup in enumerate(vecs):
                lines.append(f"    bit {i} -> {tup}")
        return "\n".join(lines)
