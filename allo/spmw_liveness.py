# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical whole-trace value identity and matcher liveness."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from .spmw_match import IRValueRef, MatchTrace, MatchedOp


@dataclass(frozen=True, order=True)
class MatcherValueId:
    """Alpha-renaming-invariant identity for one matcher-visible value."""

    origin: str
    group_id: int
    role: str
    ordinal: int = 0

    def __post_init__(self):
        if self.origin not in ("abi", "result"):
            raise ValueError("matcher value origin must be 'abi' or 'result'")
        if self.group_id < -1:
            raise ValueError("matcher value group_id must be at least -1")
        if not self.role:
            raise ValueError("matcher value role must be nonempty")
        if self.ordinal < 0:
            raise ValueError("matcher value ordinal must be non-negative")

    def manifest(self) -> dict:
        return {
            "origin": self.origin,
            "group_id": self.group_id,
            "role": self.role,
            "ordinal": self.ordinal,
        }


@dataclass
class LiveSpan:
    """Whole-trace live span for one canonical matcher value."""

    value_id: MatcherValueId
    first: tuple | None = None
    last: tuple | None = None
    search_scopes: tuple = ()
    group_ids: tuple[int, ...] = ()
    work_ids: tuple[tuple[int, ...], ...] = ()
    roles: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    crosses_kernel: bool = False
    crosses_workid: bool = False
    producer_func: tuple | None = None
    consumer_func: tuple | None = None

    @property
    def memref_name(self) -> str | None:
        return self.aliases[0] if self.aliases else None

    @property
    def role(self) -> str:
        return self.roles[0] if self.roles else self.value_id.role

    @property
    def func_names(self) -> tuple:
        return self.search_scopes


class TraceLiveness(Mapping[MatcherValueId, LiveSpan]):
    """Canonical liveness plus scoped occurrence-to-value bindings."""

    def __init__(
        self,
        spans: Mapping[MatcherValueId, LiveSpan],
        operand_values: Mapping[tuple[int, int], MatcherValueId],
        result_values: Mapping[int, MatcherValueId],
        match_alias_values: Mapping[int, Mapping[str, frozenset[MatcherValueId]]],
        alias_values: Mapping[str, frozenset[MatcherValueId]],
    ):
        self._spans = dict(spans)
        self._operand_values = dict(operand_values)
        self._result_values = dict(result_values)
        self._match_alias_values = {
            key: dict(values) for key, values in match_alias_values.items()
        }
        self._alias_values = dict(alias_values)

    def __getitem__(self, key: MatcherValueId) -> LiveSpan:
        return self._spans[key]

    def __iter__(self) -> Iterator[MatcherValueId]:
        return iter(self._spans)

    def __len__(self) -> int:
        return len(self._spans)

    def value_id_for_operand(
        self, match: MatchedOp, operand: int | str
    ) -> MatcherValueId | None:
        if isinstance(operand, str):
            indices = [
                index
                for index, binding in enumerate(match.operands)
                if binding.role == operand
            ]
            if len(indices) != 1:
                return None
            operand = indices[0]
        return self._operand_values.get((id(match), int(operand)))

    def value_id_for_result(self, match: MatchedOp) -> MatcherValueId | None:
        return self._result_values.get(id(match))

    def value_ids_for_memref(
        self,
        memref_name: str,
        matches: Sequence[MatchedOp] | None = None,
    ) -> tuple[MatcherValueId, ...]:
        if matches is None:
            values = self._alias_values.get(memref_name, frozenset())
        else:
            values = frozenset(
                value_id
                for match in matches
                for value_id in self._match_alias_values.get(id(match), {}).get(
                    memref_name, frozenset()
                )
            )
        return tuple(sorted(values))

    def span_for_memref(
        self,
        memref_name: str,
        matches: Sequence[MatchedOp] | None = None,
    ) -> LiveSpan | None:
        value_ids = self.value_ids_for_memref(memref_name, matches)
        if len(value_ids) != 1:
            return None
        return self._spans.get(value_ids[0])


def _result_role(match: MatchedOp) -> str:
    roles = [operand.role for operand in match.operands if operand.is_loop_carried]
    return roles[0] if len(roles) == 1 else "result"


def _append_unique(values: tuple, value) -> tuple:
    return values if value in values else values + (value,)


def trace_liveness(trace: MatchTrace) -> TraceLiveness:
    """Build liveness over canonical def-use and ABI value identities."""
    from .spmw_autoschedule import _bucket_for_autoschedule, _matcher_work_scope

    provisional_operands: dict[tuple[int, int], MatcherValueId] = {}
    provisional_results: dict[int, MatcherValueId] = {}
    provisional_aliases: dict[int, dict[str, set[MatcherValueId]]] = {}
    occurrences = []
    values_by_ref: dict[IRValueRef, set[MatcherValueId]] = {}
    refs_by_value: dict[MatcherValueId, set[IRValueRef]] = {}
    ordinals: dict[tuple[str, int, str], int] = {}
    group_order = []
    global_index = 0

    def allocate(origin, group_id, role):
        key = (origin, group_id, role)
        ordinal = ordinals.get(key, 0)
        ordinals[key] = ordinal + 1
        return MatcherValueId(origin, group_id, role, ordinal)

    def bind(value_id, value_ref):
        if value_ref is None:
            return
        if not isinstance(value_ref, IRValueRef):
            raise TypeError("matcher value identity must be an IRValueRef")
        existing = refs_by_value.setdefault(value_id, set())
        existing.add(value_ref)
        if len(existing) != 1:
            raise ValueError("one matcher value has conflicting retained identities")
        values_by_ref.setdefault(value_ref, set()).add(value_id)

    for search_scope, matches in _bucket_for_autoschedule(trace):
        if not matches:
            continue
        scope = _matcher_work_scope(matches[0])
        group_id = scope.group_id
        work_id = scope.work_id
        if group_id not in group_order:
            group_order.append(group_id)
        for match in matches:
            per_match = provisional_aliases.setdefault(id(match), {})
            result_name = match.result_memref_name
            result_id = None
            if result_name is not None:
                role = _result_role(match)
                result_id = allocate("result", group_id, role)
                provisional_results[id(match)] = result_id
                bind(result_id, match.result_value_ref)
            for operand_index, operand in enumerate(match.operands):
                alias = operand.memref_name
                if alias is None:
                    continue
                if operand.is_loop_carried and result_id is not None:
                    value_id = result_id
                else:
                    value_id = allocate("abi", group_id, operand.role)
                bind(value_id, operand.value_ref)
                provisional_operands[(id(match), operand_index)] = value_id
                per_match.setdefault(alias, set()).add(value_id)
                occurrences.append(
                    (
                        value_id,
                        alias,
                        operand.role,
                        search_scope,
                        group_id,
                        work_id,
                        global_index,
                        "operand",
                        bool(operand.is_loop_carried),
                        tuple(map(str, operand.indices)),
                    )
                )

            if result_name is not None and result_id is not None:
                per_match.setdefault(result_name, set()).add(result_id)
                occurrences.append(
                    (
                        result_id,
                        result_name,
                        "result",
                        search_scope,
                        group_id,
                        work_id,
                        global_index,
                        "result",
                        False,
                        (),
                    )
                )
            global_index += 1

    all_value_ids = {
        *provisional_operands.values(),
        *provisional_results.values(),
    }
    parents = {value_id: value_id for value_id in all_value_ids}

    def find(value_id):
        parent = parents[value_id]
        if parent != value_id:
            parents[value_id] = find(parent)
        return parents[value_id]

    def union(producer, consumer):
        producer_root = find(producer)
        consumer_root = find(consumer)
        if producer_root != consumer_root:
            parents[consumer_root] = producer_root

    for values in values_by_ref.values():
        values = tuple(values)
        for value_id in values[1:]:
            union(values[0], value_id)

    components = {}
    for value_id in all_value_ids:
        components.setdefault(find(value_id), []).append(value_id)
    group_positions = {group_id: index for index, group_id in enumerate(group_order)}
    canonical = {}
    for members in components.values():
        chosen = min(
            members,
            key=lambda value_id: (
                group_positions.get(value_id.group_id, len(group_positions)),
                value_id.origin != "result",
                value_id,
            ),
        )
        for value_id in members:
            canonical[value_id] = chosen

    spans: dict[MatcherValueId, LiveSpan] = {}
    alias_values: dict[str, set[MatcherValueId]] = {}
    span_events = {}
    for (
        provisional,
        alias,
        role,
        search_scope,
        group_id,
        work_id,
        site_index,
        event_kind,
        is_loop_carried,
        indices,
    ) in occurrences:
        value_id = canonical[provisional]
        site = (search_scope, work_id, site_index)
        span = spans.get(value_id)
        if span is None:
            span = LiveSpan(value_id=value_id, first=site, last=site)
            spans[value_id] = span
        else:
            span.last = site
        span.search_scopes = _append_unique(span.search_scopes, search_scope)
        span.group_ids = _append_unique(span.group_ids, group_id)
        span.work_ids = _append_unique(span.work_ids, work_id)
        span.roles = _append_unique(span.roles, role)
        span.aliases = _append_unique(span.aliases, alias)
        alias_values.setdefault(alias, set()).add(value_id)
        span_events.setdefault(value_id, []).append(
            (
                event_kind,
                is_loop_carried,
                search_scope,
                group_id,
                work_id,
                role,
                indices,
                site_index,
            )
        )

    exact_components = {canonical[value_id] for value_id in refs_by_value}
    for span in spans.values():
        events = span_events[span.value_id]
        result_scopes = tuple(
            dict.fromkeys(
                search_scope
                for kind, _carried, search_scope, _group, _work, _role, _indices, _site in events
                if kind == "result"
            )
        )
        consumer_scopes = tuple(
            dict.fromkeys(
                search_scope
                for kind, carried, search_scope, _group, _work, _role, _indices, _site in events
                if kind == "operand" and not carried
            )
        )
        result_groups = {
            group_id
            for kind, _carried, _scope, group_id, _work, _role, _indices, _site in events
            if kind == "result"
        }
        consumer_groups = {
            group_id
            for kind, carried, _scope, group_id, _work, _role, _indices, _site in events
            if kind == "operand" and not carried
        }
        producer_sites = [
            site_index
            for kind, _carried, _scope, _group, _work, _role, _indices, site_index in events
            if kind == "result"
        ]
        consumer_sites = [
            site_index
            for kind, carried, _scope, _group, _work, _role, _indices, site_index in events
            if kind == "operand" and not carried
        ]
        operand_signatures = {
            (role, indices)
            for kind, carried, _scope, _group, _work, role, indices, _site in events
            if kind == "operand" and not carried
        }
        is_exact = span.value_id in exact_components
        span.crosses_kernel = (
            is_exact
            and len(result_scopes) == 1
            and len(consumer_scopes) == 1
            and len(result_groups) == 1
            and len(consumer_groups) == 1
            and result_groups != consumer_groups
            and max(producer_sites) < min(consumer_sites)
        )
        span.crosses_workid = (
            is_exact
            and not result_scopes
            and len(span.group_ids) == 1
            and len(span.work_ids) > 1
            and len(operand_signatures) == 1
        )
        if span.crosses_kernel:
            span.producer_func = result_scopes[0]
            span.consumer_func = consumer_scopes[0]

    return TraceLiveness(
        spans,
        {key: canonical[value_id] for key, value_id in provisional_operands.items()},
        {key: canonical[value_id] for key, value_id in provisional_results.items()},
        {
            match_id: {
                alias: frozenset(canonical[value_id] for value_id in value_ids)
                for alias, value_ids in aliases.items()
            }
            for match_id, aliases in provisional_aliases.items()
        },
        {alias: frozenset(value_ids) for alias, value_ids in alias_values.items()},
    )


def crosses_boundary(span: LiveSpan) -> bool:
    return span.crosses_kernel or span.crosses_workid


def memref_span(
    liveness: TraceLiveness | None,
    memref_name: str,
    matches: Sequence[MatchedOp] | None = None,
) -> LiveSpan | None:
    """Resolve a diagnostic memref spelling only within an unambiguous scope."""
    if not liveness:
        return None
    if not isinstance(liveness, TraceLiveness):
        raise TypeError("matcher liveness must use canonical TraceLiveness")
    return liveness.span_for_memref(memref_name, matches)
