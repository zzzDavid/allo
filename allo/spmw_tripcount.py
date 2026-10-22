# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trip-count resolution for the cost path (design 04 §3).

`_parse_loop_bound` is a regex over
the affine upper-bound *string*: it returns an int for a literal bound,
the sole number when there is exactly one, else `None`. A symbolic bound
(`M*K//16`, `512*c0`, an outer-var-dependent bound) silently defaulted to
1 iteration -- exactly where the analytical estimate went wrong and the
simulator had to be the source of truth.

`resolve_trip_count` replaces that for the cost path in three escalating
tiers:

  1. Literal -- `int(text)` or sole-number regex (today's behaviour;
     covers the common GEMV case K=1024).
  2. Affine over operand shapes + mapping params -- bind affine
     symbols/dims to operand extents and unit-tree mapping fan-outs, then
     evaluate. Resolves `M*K//16`, `512*c0`, outer-var-dependent bounds.
  3. Genuinely dynamic -- return `None`. `compose` applies an explicit,
     declared fallback (not the silent `=1`) and marks the estimate
     `confidence="coarse"`.

This resolver lives in the cost layer and is consumed by `compose`, never
by the matcher: the matcher's `enclosing_loops` strings are the input,
unchanged. So D3 is purely additive to the cost path (design 04 §3.4).
"""

from __future__ import annotations

import re


def _parse_loop_bound(text: str) -> int | None:
    """Best-effort integer extraction from an affine-map upper-bound
    string (`"1024"`, `"() -> (1024)"`, ...). Returns None on failure.

    Tier-1 of `resolve_trip_count`; retained here (moved, not duplicated)
    as the literal fast path. Callers use it for the few
    callers that still want the pure literal parse.
    """
    try:
        return int(text.strip())
    except (ValueError, AttributeError):
        pass
    if not isinstance(text, str):
        return None
    nums = re.findall(r"\b(\d+)\b", text)
    if len(nums) == 1:
        return int(nums[0])
    return None


# Identifiers an affine bound may reference: bare names (symbols / dims /
# loop vars) we resolve against the env. We refuse anything outside this
# alphabet so a malformed bound cannot smuggle arbitrary Python into eval.
_AFFINE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_AFFINE_SAFE = re.compile(r"^[\sA-Za-z_0-9+\-*/()%,]*$")


def _strip_affine_wrapper(text: str) -> str:
    """Pull the body out of an affine-map string.

    Bounds arrive either as a bare expr (`"M * K // 16"`) or wrapped in
    affine-map syntax (`"(d0) -> (d0 * 512)"`, `"()[s0] -> (s0)"`). For
    the wrapped form we keep only the result tuple's interior.
    """
    arrow = text.rfind("->")
    if arrow != -1:
        text = text[arrow + 2 :]
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text


def _eval_affine(text: str, env: dict[str, int]) -> int | None:
    """Evaluate an affine-ish bound string against `env`.

    Tier-2. Supports +, -, *, // (Python floor-div), % and parentheses
    over integer-valued names bound in `env`. MLIR affine uses `floordiv`
    / `ceildiv` / `mod`; we normalise `floordiv`->`//`, `mod`->`%`, and
    `ceildiv(a,b)` is not emitted by the corpus so it is left unresolved
    (tier-3). Returns None when any referenced name is unbound or the
    expression is not in the safe alphabet.
    """
    body = _strip_affine_wrapper(text)
    if not body:
        return None
    # Normalise MLIR affine spelling to Python operators.
    body = re.sub(r"\bfloordiv\b", "//", body)
    body = re.sub(r"\bmod\b", "%", body)
    if "ceildiv" in body:
        return None
    if not _AFFINE_SAFE.match(body):
        return None
    # Every identifier left must be bound in env, else we cannot resolve.
    names = set(_AFFINE_TOKEN.findall(body))
    if not names <= set(env):
        return None
    try:
        value = eval(  # noqa: S307 -- alphabet-gated, names-restricted
            body, {"__builtins__": {}}, dict(env)
        )
    except (ValueError, ZeroDivisionError, TypeError, SyntaxError):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def resolve_trip_count(
    match,
    loop_idx: int = -1,
    *,
    shapes: dict[str, int] | None = None,
    mapping_env: dict[str, int] | None = None,
) -> int | None:
    """Resolve the trip count of `match.enclosing_loops[loop_idx]` (design
    04 §3.2 signature).

    `match` is a `MatchedOp`; `loop_idx` selects the enclosing loop (default
    -1 = innermost reduction loop, the common case the cost models price).
    Returns None when `match` has no such loop (so the caller applies its
    declared tier-3 fallback) or the bound is genuinely dynamic.

    Three tiers: literal / affine-over-operand-shapes+mapping-params /
    dynamic (None). The actual string resolution is `resolve_bound_text`;
    this is the `MatchedOp`-facing adapter the spec names.
    """
    loops = getattr(match, "enclosing_loops", None)
    if not loops:
        return None
    try:
        bound_text = loops[loop_idx][2]
    except (IndexError, TypeError):
        return None
    return resolve_bound_text(bound_text, shapes=shapes, mapping_env=mapping_env)


def resolve_bound_text(
    bound_text: str,
    *,
    shapes: dict[str, int] | None = None,
    mapping_env: dict[str, int] | None = None,
) -> int | None:
    """Resolve an affine upper-bound STRING to a concrete trip count.

    The string-level core of `resolve_trip_count`; used directly by
    `compose` for non-inner bounds (e.g. the `loops[:-1]` output-row
    products) where there is no single `loop_idx`.

    Tiers (design 04 §3.2):
      1. Literal -- bare or affine-wrapped integer.
      2. Affine over `shapes` (operand extents) + `mapping_env` (unit-tree
         fan-outs / resolved outer loop vars) -- bind and evaluate.
      3. Dynamic -- return None; `compose` applies a declared fallback and
         downgrades confidence.

    The two env dicts are merged (mapping_env wins on a name clash, since
    a mapping param is the tighter binding). Returns None when the bound
    is genuinely dynamic.
    """
    if not isinstance(bound_text, str):
        return None
    # Strict literal: a bare or affine-wrapped integer. This must take
    # precedence (the common GEMV `K=1024` case) but must NOT swallow an
    # expression that merely contains one number (e.g. `M * K floordiv 16`).
    body = _strip_affine_wrapper(bound_text)
    try:
        return int(body)
    except ValueError:
        pass
    # Tier 2: affine over operand shapes + mapping params. mapping_env wins
    # on a name clash (the tighter binding).
    env: dict[str, int] = {}
    if shapes:
        env.update(shapes)
    if mapping_env:
        env.update(mapping_env)
    if env:
        val = _eval_affine(bound_text, env)
        if val is not None:
            return val
    # Tier 1 fallback: sole-number heuristic (legacy literal parse) only
    # when no env resolution applied -- covers odd bound strings carrying
    # exactly one number and no resolvable symbol.
    nums = re.findall(r"\b(\d+)\b", body)
    if len(nums) == 1 and not _AFFINE_TOKEN.findall(body):
        return int(nums[0])
    return None
