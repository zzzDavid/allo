# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-neutral placement legality (paper Sec. III-A; release audit C-01).

The target description says where each operation may read and write: storage
declared on a unit is accessible from that unit and its descendants only,
an operation's ``src``/``dst`` patterns name the storage its operands may
occupy, and ``allo.move`` declares the only data paths.  A candidate
``Placement`` (memref -> storage handle) is legal for a matched operation when
it respects all of these.  Replacing the cost program cannot make an illegal
placement legal: the checks below read only the target tree, the matched
operation and the placement, never timing.

Rules, applied per matched operation ``m`` with target operation ``op``
(the placement may select another declared operation through
``operation_name``, e.g. AiM ``MAC_ABK``):

``declaration``
    ``op`` exists; its ``src`` arity fits its ``fn`` (one pattern per data
    parameter, or per non-accumulator parameter of an accumulating op, or a
    single pattern shared by every source); every handle its patterns name
    and every endpoint of a device move the plan uses is declared on the
    owning unit or an ancestor of it.  Host-scope moves are exempt: they are
    the declared host<->device transfers.
``access-scope``
    Every placed operand handle is declared on ``op.owner`` or an ancestor
    of it.  Storage under a sibling subtree or under the host is not
    readable by the operation.  A declared move does not change this: the
    plan's materialization moves only fill or drain the placed handle
    itself, so an operand that lives elsewhere must be placed on the
    in-scope destination of a declared move (rule ``move`` then requires
    that move).
``operand-pattern``
    Each source operand handle is admitted by its ``src`` pattern and the
    destination (the accumulator of an accumulating op, the stored result
    otherwise) by ``dst``.  Source parameters are matched up to the
    symmetries of ``fn`` under commutativity, because the matcher binds
    commuted operands (``x*W`` versus ``W*x``) to either parameter.
``move``
    A register-placed source that no other operation of the same work item
    produces needs a declared fill (a device move into it whose source is in
    the move owner's scope, or a host move into it); a register-placed
    result that no other operation consumes needs a declared drain.  The
    moves ``spmw_plan._moves_for_handles`` selects must be unambiguous.
``register-aliasing``
    An accumulating op's accumulator may not share a register with a
    distinct source value.  The accumulator is written at every reduction
    step while the sources are read at later steps, and a placement names a
    whole register (no slot offset), so the accumulate would overwrite a
    source still being read.  A non-accumulating op may write a source's
    register in place (each element is read before it is written).  Two
    distinct read-only sources on one register are a slot-materialization
    question left to the backend lowering.
``register-capacity``
    Each distinct value on a register occupies at least one slot, so the
    number of distinct values placed on it may not exceed ``Register.slots``.

Placements that name no handles (UPMEM's MLIR-to-C route) satisfy every rule
vacuously; their candidates come from the backend's own legal decision set.
"""

from __future__ import annotations

import inspect
import itertools
from dataclasses import dataclass

from .spmw_target import (
    Memory,
    MemoryRef,
    Register,
    _is_host_side,
    handle_owner,
    pattern_admits,
    pattern_handles,
    storage_key,
)

RULE_DECLARATION = "declaration"
RULE_SCOPE = "access-scope"
RULE_PATTERN = "operand-pattern"
RULE_MOVE = "move"
RULE_ALIASING = "register-aliasing"
RULE_CAPACITY = "register-capacity"


@dataclass(frozen=True)
class Violation:
    """One broken legality rule."""

    rule: str
    op: str
    func_name: str | None
    operand: str | None
    memref: str | None
    handle: str | None
    owner: str | None
    detail: str

    def __str__(self) -> str:
        where = f"{self.op}" + (f" in {self.func_name!r}" if self.func_name else "")
        parts = [f"[{self.rule}] {where}"]
        if self.operand is not None or self.memref is not None:
            parts.append(f"operand {self.operand!r} (memref {self.memref!r})")
        if self.handle is not None:
            parts.append(
                f"handle {self.handle}"
                + (f" owned by unit {self.owner!r}" if self.owner else "")
            )
        return ", ".join(parts) + f": {self.detail}"


@dataclass(frozen=True)
class LegalityVerdict:
    """Legal iff ``violations`` is empty."""

    violations: tuple = ()

    @property
    def legal(self) -> bool:
        return not self.violations

    def grouped(self) -> list[str]:
        """One line per distinct violation; the same violation in several
        work-item functions is reported once with the function count."""
        groups: dict[tuple, list] = {}
        for v in self.violations:
            key = (v.rule, v.op, v.operand, v.memref, v.handle, v.owner, v.detail)
            groups.setdefault(key, []).append(v.func_name)
        lines = []
        for key, funcs in groups.items():
            text = str(Violation(*key[:2], funcs[0], *key[2:]))
            named = [f for f in funcs if f]
            if len(named) > 1:
                text += f" (and {len(named) - 1} more work-item function(s))"
            lines.append(text)
        return lines

    def describe(self, limit: int | None = 8) -> str:
        items = self.grouped()
        shown = items if limit is None else items[:limit]
        text = "; ".join(shown)
        if limit is not None and len(items) > limit:
            text += f"; ... ({len(items) - limit} more)"
        return text

    def raise_if_illegal(self, context: str = "placement") -> None:
        if self.violations:
            raise IllegalPlacement(self, context)


class IllegalPlacement(ValueError):
    """A placement the target description does not permit."""

    def __init__(self, verdict: LegalityVerdict, context: str = "placement"):
        self.verdict = verdict
        self.violations = verdict.violations
        lines = verdict.grouped()
        super().__init__(
            f"illegal {context}: {len(lines)} violation(s) of the target "
            "description:\n  " + "\n  ".join(lines)
        )


# --------------------------------------------------------------------- #
# Handle and operation helpers
# --------------------------------------------------------------------- #


def _unwrap(handle):
    return getattr(handle, "home_handle", handle)


def _handle_text(handle) -> str:
    handle = _unwrap(handle)
    if isinstance(handle, (Register, Memory)):
        return f"{type(handle).__name__}({handle.name})"
    if isinstance(handle, MemoryRef):
        return f"MemoryRef({handle.memory.name}[{handle.idx!r}])"
    return repr(handle)


def _pattern_text(pattern) -> str:
    from .spmw_target import AnyOf, OrOf

    if isinstance(pattern, OrOf):
        return "or_(" + ", ".join(_pattern_text(a) for a in pattern.alternatives) + ")"
    if isinstance(pattern, AnyOf):
        cands = pattern.candidates
        if isinstance(cands, (Memory, MemoryRef, Register)):
            return f"any_({_pattern_text(cands)})"
        return "any_([" + ", ".join(_pattern_text(c) for c in cands) + "])"
    if isinstance(pattern, (Register, Memory)):
        return pattern.name
    if isinstance(pattern, MemoryRef):
        return f"{pattern.memory.name}[{pattern.idx!r}]"
    return repr(pattern)


def _owner_name(handle) -> str | None:
    owner = handle_owner(handle)
    return None if owner is None else owner.name


def _is_storage(handle) -> bool:
    return isinstance(_unwrap(handle), (Register, Memory, MemoryRef))


def _data_params(op) -> list[str]:
    """``fn`` parameters bound to workload values (not const/iv params)."""
    skip = set(getattr(op, "const_params", ())) | set(getattr(op, "iv_params", ()))
    return [p for p in inspect.signature(op.fn).parameters if p not in skip]


def operand_signature(op):
    """``(source_params, acc_param, src_pattern_of)`` for ``op``.

    ``acc_param`` is the accumulator (the last data parameter of an
    accumulating op, as the matcher's ``_acc_param``) or None.
    ``src_pattern_of`` maps each source parameter (and, when the target
    lists it among ``src``, the accumulator) to its declared pattern.
    Raises ValueError when ``src`` arity does not fit ``fn``.
    """
    data = _data_params(op)
    acc = data[-1] if op.accumulates and data else None
    sources = [p for p in data if p != acc]
    src = op.src
    if isinstance(src, (tuple, list)):
        if len(src) == len(data):
            pattern_of = dict(zip(data, src))
        elif acc is not None and len(src) == len(sources):
            pattern_of = dict(zip(sources, src))
        else:
            raise ValueError(
                f"op {op.name!r} declares {len(src)} src pattern(s) for data "
                f"parameters {data}" + (f" (accumulator {acc!r})" if acc else "")
            )
    else:
        pattern_of = {p: src for p in data}
    return sources, acc, pattern_of


def _canonical(node, rename):
    from .spmw_match_engine import (
        _COMMUTATIVE,
        _MIRRORED_PREDICATE,
        PBinOp,
        PCmp,
        PConst,
        PSelect,
        PUnary,
        PVar,
    )

    if isinstance(node, PVar):
        return ("var", rename.get(node.name, node.name))
    if isinstance(node, PConst):
        return ("const", repr(node.value))
    if isinstance(node, PBinOp):
        lhs, rhs = _canonical(node.lhs, rename), _canonical(node.rhs, rename)
        if node.op in _COMMUTATIVE:
            lhs, rhs = sorted((lhs, rhs), key=repr)
        return ("bin", node.op, lhs, rhs)
    if isinstance(node, PUnary):
        return ("un", node.fn, _canonical(node.arg, rename))
    if isinstance(node, PCmp):
        lhs, rhs = _canonical(node.lhs, rename), _canonical(node.rhs, rename)
        direct = ("cmp", node.pred, lhs, rhs)
        mirrored = ("cmp", _MIRRORED_PREDICATE[node.pred], rhs, lhs)
        return min(direct, mirrored, key=repr)
    if isinstance(node, PSelect):
        return (
            "sel",
            _canonical(node.cond, rename),
            _canonical(node.a, rename),
            _canonical(node.b, rename),
        )
    return ("other", repr(node))


def source_symmetries(op, sources) -> list[dict]:
    """Renamings of ``sources`` under which ``op.fn`` is unchanged modulo
    commutativity; always includes the identity."""
    cache = getattr(op, "_legality_symmetries", None)
    key = tuple(sources)
    if cache is not None and cache[0] == key:
        return cache[1]
    identity = {p: p for p in sources}
    found = [identity]
    if len(sources) > 1:
        from .spmw_match_engine import compile_op_pattern

        try:
            body = compile_op_pattern(op).body
        except ValueError:  # fn source unrecoverable: no symmetry assumed
            body = None
        if body is not None:
            reference = _canonical(body, {})
            for perm in itertools.permutations(sources):
                rename = dict(zip(sources, perm))
                if rename != identity and _canonical(body, rename) == reference:
                    found.append(rename)
    op._legality_symmetries = (key, found)
    return found


def check_op_declaration(op) -> list[Violation]:
    """Declaration-level legality of one ``allo.op``."""
    out = []
    try:
        _sources, _acc, pattern_of = operand_signature(op)
    except ValueError as error:
        return [
            Violation(
                RULE_DECLARATION, op.name, None, None, None, None, None, str(error)
            )
        ]
    patterns = list(pattern_of.values()) + [op.dst]
    for pattern in patterns:
        for handle in pattern_handles(pattern):
            owner = handle_owner(handle)
            if owner is not None and not owner.encloses(op.owner):
                out.append(
                    Violation(
                        RULE_DECLARATION,
                        op.name,
                        None,
                        None,
                        None,
                        _handle_text(handle),
                        owner.name,
                        f"declared operand storage is outside the subtree of the "
                        f"op's unit {op.owner.name!r}",
                    )
                )
    return out


def check_move_declaration(move) -> list[Violation]:
    """Declaration-level legality of one device ``allo.move`` (host-scope
    moves are the declared host<->device transfers and are exempt)."""
    if _is_host_side(move.owner):
        return []
    out = []
    for end, pattern in (("src", move.src), ("dst", move.dst)):
        for handle in pattern_handles(pattern):
            owner = handle_owner(handle)
            if owner is not None and not owner.encloses(move.owner):
                out.append(
                    Violation(
                        RULE_DECLARATION,
                        move.name,
                        None,
                        end,
                        None,
                        _handle_text(handle),
                        owner.name,
                        f"move {end} is outside the subtree of the move's unit "
                        f"{move.owner.name!r}",
                    )
                )
    return out


def check_target(target) -> LegalityVerdict:
    """Declaration-level legality of every op and device move of ``target``."""
    out = []
    for unit in target._walk():
        for op in unit.ops.values():
            out.extend(check_op_declaration(op))
        for move in unit.moves.values():
            out.extend(check_move_declaration(move))
    return LegalityVerdict(tuple(out))


def _declared_moves(target):
    for unit in target._walk():
        yield from unit.moves.values()


def _has_fill(target, handle) -> bool:
    for move in _declared_moves(target):
        if move.src is move.dst or not pattern_admits(move.dst, handle):
            continue
        if _is_host_side(move.owner) or not check_move_declaration(move):
            return True
    return False


def _has_drain(target, handle) -> bool:
    for move in _declared_moves(target):
        if move.src is move.dst or not pattern_admits(move.src, handle):
            continue
        if _is_host_side(move.owner) or not check_move_declaration(move):
            return True
    return False


# --------------------------------------------------------------------- #
# Placement legality
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlacedOperand:
    """One placed operand of a matched op: ``kind`` is ``"src"`` or
    ``"dst"`` (the accumulator of an accumulating op, the stored result
    otherwise)."""

    role: str
    memref: str
    handle: object
    kind: str


def _same_semantics(a, b) -> bool:
    from .spmw_match_engine import compile_op_pattern

    if a is b:
        return True
    try:
        pa, pb = compile_op_pattern(a), compile_op_pattern(b)
    except ValueError:
        return False
    rename = dict(zip(pb.param_names, pa.param_names))
    return len(pa.param_names) == len(pb.param_names) and _canonical(
        pa.body, {}
    ) == _canonical(pb.body, rename)


def selected_operation(target, match, properties):
    """The declared op a placement executes for ``match``.

    A placement's ``operation_name`` (AiM ``MAC`` versus ``MAC_ABK``) selects
    among ops with the matched op's semantics; other matches of the same
    work item (an activation fused after the MAC) keep their own op.
    """
    matched = target.op(match.target_op_name)
    name = properties.get("operation_name")
    if name is None or name == matched.name:
        return matched
    selected = target.op(name)
    return selected if _same_semantics(matched, selected) else matched


def placed_operands(op, match, placement) -> list[PlacedOperand]:
    """The operands of ``match`` that ``placement`` maps to a handle."""
    data = set(_data_params(op))
    acc = None
    if op.accumulates:
        params = _data_params(op)
        acc = params[-1] if params else None
    placements = getattr(placement, "placements", {}) or {}
    out = []
    for operand in match.operands:
        if operand.role not in data or operand.memref_name is None:
            continue
        handle = placements.get(operand.memref_name)
        if handle is None:
            continue
        kind = "dst" if operand.role == acc else "src"
        out.append(PlacedOperand(operand.role, operand.memref_name, handle, kind))
    if acc is None and match.result_memref_name is not None:
        handle = placements.get(match.result_memref_name)
        if handle is not None:
            out.append(PlacedOperand("dst", match.result_memref_name, handle, "dst"))
    return out


def _check_match(target, match, placement, properties, produced, consumed):
    out = []
    func = match.func_name
    try:
        op = selected_operation(target, match, properties)
    except KeyError as error:
        return [
            Violation(
                RULE_DECLARATION,
                match.target_op_name,
                func,
                None,
                None,
                None,
                None,
                f"target declares no such operation ({error})",
            )
        ]

    def violation(rule, item, detail):
        return Violation(
            rule,
            op.name,
            func,
            item.role,
            item.memref,
            _handle_text(item.handle),
            _owner_name(item.handle),
            detail,
        )

    out.extend(check_op_declaration(op))
    placed = placed_operands(op, match, placement)

    # (a) access scope.
    for item in placed:
        if not _is_storage(item.handle):
            out.append(
                violation(
                    RULE_SCOPE, item, "placed value is not a declared storage handle"
                )
            )
            continue
        owner = handle_owner(item.handle)
        if owner is None or not owner.encloses(op.owner):
            out.append(
                violation(
                    RULE_SCOPE,
                    item,
                    f"op {op.name!r} runs on unit {op.owner.name!r}; storage "
                    f"declared on {getattr(owner, 'name', None)!r} is outside "
                    "its ancestor chain, so the op cannot reach it",
                )
            )

    # (b) src/dst patterns, sources up to the symmetries of fn.
    try:
        sources, acc, pattern_of = operand_signature(op)
    except ValueError:
        sources, acc, pattern_of = [], None, None
    if pattern_of is not None:
        srcs = [item for item in placed if item.kind == "src"]
        best = None
        for rename in source_symmetries(op, sources):
            bad = [
                item
                for item in srcs
                if not pattern_admits(
                    pattern_of[rename.get(item.role, item.role)], item.handle
                )
            ]
            if best is None or len(bad) < len(best[1]):
                best = (rename, bad)
            if not bad:
                break
        if best is not None:
            rename, bad = best
            for item in bad:
                pattern = pattern_of[rename.get(item.role, item.role)]
                out.append(
                    violation(
                        RULE_PATTERN,
                        item,
                        f"not admitted by src pattern {_pattern_text(pattern)}",
                    )
                )
        for item in placed:
            if item.kind != "dst":
                continue
            if not pattern_admits(op.dst, item.handle):
                out.append(
                    violation(
                        RULE_PATTERN,
                        item,
                        f"not admitted by dst pattern {_pattern_text(op.dst)}",
                    )
                )
            if (
                acc is not None
                and acc in pattern_of
                and not pattern_admits(pattern_of[acc], item.handle)
            ):
                out.append(
                    violation(
                        RULE_PATTERN,
                        item,
                        f"accumulator not admitted by src pattern {_pattern_text(pattern_of[acc])}",
                    )
                )

    # (d) aliasing and capacity on registers.
    by_register: dict = {}
    for item in placed:
        if isinstance(_unwrap(item.handle), Register):
            by_register.setdefault(storage_key(item.handle), []).append(item)
    for items in by_register.values():
        register = _unwrap(items[0].handle)
        values = list(dict.fromkeys(item.memref for item in items))
        if len(values) > int(register.slots):
            out.append(
                violation(
                    RULE_CAPACITY,
                    items[0],
                    f"{len(values)} distinct values {values} placed on a register "
                    f"with {register.slots} slot(s)",
                )
            )
        if op.accumulates:
            accs = [item for item in items if item.kind == "dst"]
            for acc_item in accs:
                clobbered = sorted(
                    {item.memref for item in items if item.kind == "src"}
                    - {acc_item.memref}
                )
                if clobbered:
                    out.append(
                        violation(
                            RULE_ALIASING,
                            acc_item,
                            f"accumulator shares register {register.name!r} with "
                            f"distinct source value(s) {clobbered}; the reduction "
                            "would overwrite a source it still reads",
                        )
                    )

    # (c) declared moves for register-placed values.
    for item in placed:
        handle = _unwrap(item.handle)
        if not isinstance(handle, Register):
            continue
        if (
            item.kind == "src"
            and item.memref not in produced
            and not _has_fill(target, handle)
        ):
            out.append(
                violation(
                    RULE_MOVE,
                    item,
                    "no declared move fills this register with the source value",
                )
            )
        if (
            item.kind == "dst"
            and item.memref not in consumed
            and not _has_drain(target, handle)
        ):
            out.append(
                violation(
                    RULE_MOVE,
                    item,
                    "no declared move drains the result from this register",
                )
            )
    from .spmw_plan import _moves_for_handles

    for phase, kinds in (("pre", ("src", "dst")), ("post", ("dst",))):
        handles = [item.handle for item in placed if item.kind in kinds]
        try:
            moves = _moves_for_handles(target, handles, phase=phase)
        except ValueError as error:
            out.append(
                Violation(RULE_MOVE, op.name, func, None, None, None, None, str(error))
            )
            continue
        for move in moves:
            out.extend(check_move_declaration(move))
    return out


def check_placement(target, matches, placement) -> LegalityVerdict:
    """Legality of one candidate ``placement`` for the matches of one work
    item (one autoschedule bucket)."""
    from .spmw_autoschedule import derive_layout_properties

    matches = list(matches)
    try:
        properties = derive_layout_properties(target, placement)
    except ValueError as error:
        return LegalityVerdict(
            (
                Violation(
                    RULE_DECLARATION,
                    getattr(placement, "mode", "") or "placement",
                    None,
                    None,
                    None,
                    None,
                    None,
                    f"layout is not realizable: {error}",
                ),
            )
        )
    out = []
    for match in matches:
        # Values another op of the same work item produces need no fill;
        # values another op consumes need no drain.
        others = [m for m in matches if m is not match]
        produced = {
            o.memref_name for m in others for o in m.operands if o.is_loop_carried
        } | {m.result_memref_name for m in others if m.result_memref_name}
        consumed = {
            o.memref_name
            for m in others
            for o in m.operands
            if not o.is_loop_carried and o.memref_name != m.result_memref_name
        }
        out.extend(
            _check_match(target, match, placement, properties, produced, consumed)
        )
    return LegalityVerdict(tuple(dict.fromkeys(out)))


def check_layouts(target, trace, layouts) -> LegalityVerdict:
    """Legality of a per-kernel layout list (or one broadcast placement)
    against every autoschedule bucket of ``trace``."""
    from .spmw_autoschedule import Placement, _bucket_for_autoschedule

    buckets = _bucket_for_autoschedule(trace)
    if isinstance(layouts, Placement):
        layouts = [layouts] * len(buckets)
    layouts = list(layouts)
    if len(layouts) != len(buckets):
        # The caller reports the arity mismatch; nothing to pair here.
        return LegalityVerdict()
    out = []
    for (_scope, matches), placement in zip(buckets, layouts):
        out.extend(check_placement(target, matches, placement).violations)
    return LegalityVerdict(tuple(dict.fromkeys(out)))
