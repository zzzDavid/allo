# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduction-order policy: may partial sums of a reduction be regrouped?

``allo.compile(..., reduction_order=...)`` takes one of

``"strict"`` (default)
    Every accumulation keeps the source order and the source rounding: the
    accumulator is updated once per reduction iteration, in ascending
    iteration order, each multiply and add rounding to the accumulator's
    element type.  A schedule that splits a floating-point reduction into
    partial sums (bank fibers, SIMD lanes summed afterwards, tasklet or tile
    partials, tree reductions) is pruned before ranking.
``"reassociate"``
    Partial sums may be formed and combined in any grouping and order (and in
    a wider type).  This is the explicit permission the paper requires before
    a floating-point reduction is split.

Integer accumulation (``i*``/``ui*``/``si*``/``index``) is associative and
commutative modulo 2^width, so every grouping yields the same wrapped result;
a strict policy therefore still permits splitting an integer reduction.
The check reads the accumulator's MLIR element type; an accumulator whose
type cannot be read is treated as floating point (fail closed).

The policy is stamped on every accumulating ``MatchedOp`` as
``extra["reduction_policy"]`` (a :class:`ReductionPolicy`).  A match without a
stamp (a hand-built trace) is read under the default, ``"strict"``.
Semantics, per-backend consequences and tests:
``docs/release/design/reduction-policy.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STRICT = "strict"
REASSOCIATE = "reassociate"
REDUCTION_ORDERS = (STRICT, REASSOCIATE)
DEFAULT_REDUCTION_ORDER = STRICT
POLICY_KEY = "reduction_policy"


class ReductionOrderError(ValueError):
    """No schedule of the workload honours the requested reduction order."""


@dataclass(frozen=True)
class ReductionPolicy:
    """The reduction-order permission of one accumulating match."""

    order: str  # requested: "strict" | "reassociate"
    accumulator_type: str | None  # MLIR element type, e.g. "f16", "i32"

    @property
    def integer(self) -> bool:
        return is_integer_type(self.accumulator_type)

    @property
    def may_regroup(self) -> bool:
        """True when partial sums may be formed and recombined."""
        return self.order == REASSOCIATE or self.integer

    def describe(self) -> str:
        if self.order == REASSOCIATE:
            return "reduction_order='reassociate' permits regrouping partial sums"
        if self.integer:
            return (
                f"reduction_order='strict' on a {self.accumulator_type} accumulator: "
                "integer addition wraps modulo 2^width, so any grouping is exact"
            )
        return (
            f"reduction_order='strict' on a {self.accumulator_type or 'unknown-type'} "
            "accumulator: the source accumulation order and rounding must be kept"
        )


def normalize_reduction_order(value) -> str:
    """Validate a user-supplied ``reduction_order`` (None means the default)."""
    if value is None:
        return DEFAULT_REDUCTION_ORDER
    if value not in REDUCTION_ORDERS:
        raise ValueError(
            f"reduction_order must be one of {REDUCTION_ORDERS!r}, got {value!r}"
        )
    return value


_INTEGER_TYPE = re.compile(r"(?:[su]?i\d+|index)")
_MEMREF_ELEMENT = re.compile(r"memref<(?:[^x>]*x)*\s*([a-z]+\d*)")


def is_integer_type(element_type: str | None) -> bool:
    return bool(element_type) and bool(_INTEGER_TYPE.fullmatch(element_type))


def accumulator_element_type(match) -> str | None:
    """Element type of a match's loop-carried accumulator, or None."""
    for operand in getattr(match, "operands", ()):
        if getattr(operand, "is_loop_carried", False):
            found = _MEMREF_ELEMENT.match(getattr(operand, "memref_type", None) or "")
            return found.group(1) if found else None
    return None


def is_accumulating(match) -> bool:
    return any(
        getattr(operand, "is_loop_carried", False)
        for operand in getattr(match, "operands", ())
    )


def stamp_reduction_policy(trace_or_matches, reduction_order=None) -> None:
    """Stamp ``extra["reduction_policy"]`` on every accumulating match."""
    order = normalize_reduction_order(reduction_order)
    matches = getattr(trace_or_matches, "matches", trace_or_matches)
    for match in matches:
        if is_accumulating(match):
            match.extra[POLICY_KEY] = ReductionPolicy(
                order, accumulator_element_type(match)
            )


def reduction_policy(match) -> ReductionPolicy | None:
    """The stamped policy of an accumulating match (the default when absent);
    None for a match that does not accumulate."""
    if not is_accumulating(match):
        return None
    policy = match.extra.get(POLICY_KEY)
    if isinstance(policy, ReductionPolicy):
        return policy
    return ReductionPolicy(DEFAULT_REDUCTION_ORDER, accumulator_element_type(match))


def strict_policies(matches) -> list[ReductionPolicy]:
    """Policies of the accumulating matches that forbid regrouping."""
    out = []
    for match in matches:
        policy = reduction_policy(match)
        if policy is not None and not policy.may_regroup:
            out.append(policy)
    return out


def may_regroup(matches) -> bool:
    """True when no accumulating match forbids splitting its reduction."""
    return not strict_policies(matches)


def trace_reduction_order(trace) -> str | None:
    """The single order stamped on a trace's accumulating matches, if any."""
    orders = {
        policy.order
        for policy in (reduction_policy(m) for m in getattr(trace, "matches", ()))
        if policy is not None
    }
    return orders.pop() if len(orders) == 1 else None


def no_strict_program_error(target_name, matches, why) -> ReductionOrderError:
    """The error a backend raises when it has no order-preserving program."""
    policies = strict_policies(matches)
    types = sorted({p.accumulator_type or "unknown" for p in policies})
    return ReductionOrderError(
        f"reduction_order='strict': target {target_name!r} has no schedule that "
        f"keeps the source accumulation order of the {'/'.join(types)} reduction "
        f"({why}). Pass reduction_order='reassociate' to allow regrouping partial "
        "sums, or use an integer accumulator."
    )


# Matcher routes whose enumerators consult the policy themselves (they prune
# regrouping candidates and raise when none remains).  APU v1's enumerator
# prunes the regrouping vector plans of a plain-function contraction and
# refuses the grouped-tree program of a dataflow region under a strict float
# policy; Gemini-II lowers integer contractions only.
_POLICY_AWARE_MATCHER_ROUTES = frozenset(
    {"samsung_hbm_pim", "upmem", "apu_v1", "apu_v2"}
)

# Why the remaining matcher routes cannot honour a strict floating-point
# reduction.  Every program they emit regroups the reduction.
_REGROUPING_MATCHER_ROUTES = {
    "aim": (
        "every AiM MAC command reduces the 16 BF16 products of a burst through "
        "the bank's adder tree before accumulating, so no AiM program adds the "
        "products one at a time in source order"
    ),
}

APU_V1_GROUPED_ROUTE_REGROUPS = (
    "the APU v1 grouped matcher program of a dataflow region reduces each group "
    "with a GVML subgroup tree (gvml_add_subgrps); source-order plans exist only "
    "for a plain-function contraction (the vector plans)"
)


def require_matcher_route_policy(target, trace) -> None:
    """Fail before scheduling when the matcher route of ``target`` cannot keep
    a strict floating-point reduction in source order."""
    name = getattr(target, "name", None)
    if name in _POLICY_AWARE_MATCHER_ROUTES:
        return
    matches = list(getattr(trace, "matches", ()))
    if may_regroup(matches):
        return
    why = _REGROUPING_MATCHER_ROUTES.get(
        name,
        f"the {name!r} matcher route does not consult the reduction-order policy",
    )
    raise no_strict_program_error(name, matches, why)


__all__ = [
    "DEFAULT_REDUCTION_ORDER",
    "REASSOCIATE",
    "REDUCTION_ORDERS",
    "STRICT",
    "ReductionOrderError",
    "ReductionPolicy",
    "accumulator_element_type",
    "may_regroup",
    "no_strict_program_error",
    "APU_V1_GROUPED_ROUTE_REGROUPS",
    "normalize_reduction_order",
    "reduction_policy",
    "require_matcher_route_policy",
    "stamp_reduction_policy",
    "strict_policies",
    "trace_reduction_order",
]
