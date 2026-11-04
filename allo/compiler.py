# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public workload + target + cost compilation facade."""

# Public API spelling intentionally matches Python's built-in compiler verb.
# pylint: disable=redefined-builtin

from __future__ import annotations

import inspect
import math
import typing
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .ir.builder import _spmw_group_contract
from .perf import BoundCostSpec, CostSpec
from .pim.apu_v1_record_frontend import is_apu_v1_program, normalize_apu_v1_program
from .spmw_autoschedule import MatcherWorkScope, _retain_matcher_work_scope
from .spmw_codegen import RunResult, compile_for_target, source_backend_binding
from .spmw_liveness import MatcherValueId, trace_liveness
from .pim.apu_g2_record_frontend import is_apu_g2_record, normalize_apu_g2_record
from .spmw_pipeline import normalize_source
from .spmw_plan import BufferMetricManifest, launch_schedule
from .spmw_reduction_policy import (
    POLICY_KEY,
    normalize_reduction_order,
    require_matcher_route_policy,
)
from .spmw_resources import normalize_resource_budget, stamp_resource_budget
from .spmw_target import HostProgram


def _materialize_workload(workload):
    if callable(workload):
        return workload
    build = getattr(workload, "build", None)
    if callable(build):
        return build()
    raise TypeError("workload must be callable or expose a callable build()")


def _materialize_target(target):
    if hasattr(target, "name"):
        return target
    if callable(target):
        built = target()
        if hasattr(built, "name"):
            return built
    raise TypeError("target must be a Target or a zero-argument target builder")


def _resolve_cost(target, cost_spec):
    """Bind an executable cost program to this target instance."""
    if cost_spec is None:
        return None
    if isinstance(cost_spec, BoundCostSpec):
        if cost_spec.target is not target:
            raise ValueError("bound cost spec belongs to a different target instance")
        return cost_spec
    if not isinstance(cost_spec, CostSpec):
        raise TypeError("cost must be an executable CostSpec")
    return cost_spec.bind(target)


def _buffer_metrics(workload):
    """Extract static operand geometry for executable cost rules.

    Allo workloads commonly use postponed annotations, so resolve them in the
    defining module before inspecting ``TypeAnnotation.shape`` and dtype bits.
    Unknown annotations are simply omitted; cost programs remain free to use
    other event metrics.
    """
    try:
        annotations = inspect.get_annotations(workload, eval_str=True)
    except (NameError, TypeError):
        annotations = getattr(workload, "__annotations__", {}) or {}

    metrics = {}
    for name, annotation in annotations.items():
        if name == "return":
            continue
        shape = getattr(annotation, "shape", None)
        if shape is None:
            continue
        try:
            shape = tuple(int(extent) for extent in shape)
        except (TypeError, ValueError):
            continue
        dtype = getattr(annotation, "dtype", None)
        bits = int(getattr(dtype, "bits", 0) or 0)
        elements = math.prod(shape)
        entry = {"shape": shape, "elements": elements}
        if bits > 0:
            entry.update(dtype_bits=bits, bytes=(elements * bits + 7) // 8)
        metrics[name] = entry
    return metrics


def _host_move_buffer_role(record):
    """Recover the workload ABI role from a recorded or resolved host move."""
    role = getattr(record, "buffer_role", None)
    if role is not None:
        return role
    from .spmw_target import BufferToken, HandleToken

    for argument in getattr(record, "args", ()):
        if isinstance(argument, BufferToken):
            return getattr(argument, "base", argument.name)
        if isinstance(argument, str):
            return argument
        if not isinstance(argument, HandleToken):
            name = getattr(argument, "name", None)
            if name is not None:
                return name
    return None


def _buffer_metric_manifest(workload, trace, host_moves):
    """Bind annotation geometry to exact matcher-visible workload values."""
    metrics_by_name = _buffer_metrics(workload)
    source_value_refs = getattr(trace, "source_value_refs", None)
    if not source_value_refs:
        # Preserve the legacy path for externally constructed traces that do
        # not carry frontend source identity.
        return metrics_by_name

    liveness = trace_liveness(trace)
    values_by_ref = {}
    for match in trace.matches:
        for index, operand in enumerate(match.operands):
            value_id = liveness.value_id_for_operand(match, index)
            if operand.value_ref is not None and value_id is not None:
                values_by_ref.setdefault(operand.value_ref, set()).add(value_id)
        value_id = liveness.value_id_for_result(match)
        if match.result_value_ref is not None and value_id is not None:
            values_by_ref.setdefault(match.result_value_ref, set()).add(value_id)

    parameters = tuple(inspect.signature(workload).parameters)
    parameter_ordinals = {name: index for index, name in enumerate(parameters)}
    entries = {}
    parameter_values = {}
    for name, metrics in metrics_by_name.items():
        ordinal = parameter_ordinals[name]
        source_ref = source_value_refs.get(ordinal)
        candidates = values_by_ref.get(source_ref, set())
        if len(candidates) > 1:
            raise ValueError(
                f"workload buffer {name!r} has conflicting structural values"
            )
        value_id = (
            next(iter(candidates))
            if candidates
            else MatcherValueId("abi", -1, "argument", ordinal)
        )
        existing = entries.get(value_id)
        if existing is not None and existing != metrics:
            raise ValueError("one structural value has conflicting buffer metrics")
        entries[value_id] = metrics
        parameter_values[name] = value_id

    host_bindings = {}
    for index, record in enumerate(host_moves or ()):
        role = _host_move_buffer_role(record)
        if role not in parameter_values:
            raise ValueError(f"host transfer {index} lacks an exact ABI metric binding")
        host_bindings[index] = parameter_values[role]
    return BufferMetricManifest.create(entries, host_bindings=host_bindings)


def _retained_symbol_name(function):
    attributes = getattr(function, "attributes", {})
    if "sym_name" not in attributes:
        return None
    attribute = attributes["sym_name"]
    value = getattr(attribute, "value", None)
    if value is not None:
        return str(value)
    text = str(attribute)
    return text[1:-1] if len(text) >= 2 and text[0] == text[-1] == '"' else text


_SPMW_SCOPE_ATTRS = (
    "spmw.group_id",
    "spmw.work_id",
    "spmw.group_shape",
    "spmw.coalesced_axes",
    "spmw.group_fingerprint",
    "spmw.body_fingerprint",
)
_SPMW_ABI_VALUE_IDS_ATTR = "spmw.abi_value_ids"


def _retained_integer_attribute(attributes, name):
    attribute = attributes[name]
    return int(getattr(attribute, "value", attribute))


def _retained_integer_array_attribute(attributes, name):
    return tuple(
        int(getattr(attribute, "value", attribute)) for attribute in attributes[name]
    )


def _retained_string_attribute(attributes, name):
    attribute = attributes[name]
    value = getattr(attribute, "value", attribute)
    return str(value)


def _retained_matcher_work_scope(function):
    attributes = getattr(function, "attributes", {})
    present = tuple(name in attributes for name in _SPMW_SCOPE_ATTRS)
    if not any(present):
        return None
    if not all(present):
        raise ValueError("retained matcher scope attributes are incomplete")
    return MatcherWorkScope(
        group_id=_retained_integer_attribute(attributes, "spmw.group_id"),
        work_id=_retained_integer_array_attribute(attributes, "spmw.work_id"),
        group_shape=_retained_integer_array_attribute(attributes, "spmw.group_shape"),
        coalesced_axes=_retained_integer_array_attribute(
            attributes, "spmw.coalesced_axes"
        ),
    )


def _retained_function_arguments(function):
    arguments = getattr(function, "arguments", None)
    if arguments is not None:
        return tuple(arguments)
    return None


def _validated_retained_matcher_contract(module):
    body = getattr(module, "body", None)
    records = []
    saw_kernel = False
    for function in getattr(body, "operations", ()):
        operation = getattr(function, "operation", None)
        if getattr(operation, "name", None) != "func.func":
            continue
        attributes = getattr(function, "attributes", {})
        is_kernel = "df.kernel" in attributes
        retained_scope = _retained_matcher_work_scope(function)
        if retained_scope is not None and not is_kernel:
            raise ValueError("retained matcher scope is attached to a non-kernel")
        if not is_kernel:
            continue
        saw_kernel = True
        if retained_scope is None:
            raise ValueError("retained matcher adapter contract is incomplete")
        if _SPMW_ABI_VALUE_IDS_ATTR not in attributes:
            raise ValueError("retained matcher adapter lacks typed ABI identities")
        abi_value_ids = _retained_integer_array_attribute(
            attributes, _SPMW_ABI_VALUE_IDS_ATTR
        )
        if any(value < 0 for value in abi_value_ids):
            raise ValueError("retained matcher ABI identities must be non-negative")
        arguments = _retained_function_arguments(function)
        if arguments is not None and len(abi_value_ids) != len(arguments):
            raise ValueError("retained matcher ABI identity arity is inconsistent")
        if tuple(sorted(set(retained_scope.coalesced_axes))) != tuple(
            retained_scope.coalesced_axes
        ):
            raise ValueError("retained matcher coalesced axes are not canonical")
        if any(
            coordinate >= extent
            for coordinate, extent in zip(
                retained_scope.work_id, retained_scope.group_shape
            )
        ):
            raise ValueError("retained matcher work coordinate is outside its grid")
        records.append(
            {
                "function": function,
                "symbol": _retained_symbol_name(function),
                "scope": retained_scope,
                "group_fingerprint": _retained_string_attribute(
                    attributes, "spmw.group_fingerprint"
                ),
                "body_fingerprint": _retained_string_attribute(
                    attributes, "spmw.body_fingerprint"
                ),
                "abi_value_ids": abi_value_ids,
            }
        )

    if not saw_kernel:
        return ()

    groups = {}
    for record in records:
        scope = record["scope"]
        key = (scope.group_id, record["group_fingerprint"])
        groups.setdefault(key, []).append(record)

    for group_records in groups.values():
        first_scope = group_records[0]["scope"]
        for record in group_records[1:]:
            scope = record["scope"]
            if (
                scope.group_shape != first_scope.group_shape
                or scope.coalesced_axes != first_scope.coalesced_axes
            ):
                raise ValueError("retained matcher group has inconsistent topology")
        expected_size = math.prod(first_scope.group_shape)
        if len(group_records) != expected_size:
            raise ValueError("retained matcher group does not cover its complete grid")
        work_ids = tuple(record["scope"].work_id for record in group_records)
        if len(set(work_ids)) != len(work_ids):
            raise ValueError("retained matcher group repeats a work coordinate")
        expected_id, expected_group_fingerprint, expected_bodies = _spmw_group_contract(
            first_scope.group_shape,
            first_scope.coalesced_axes,
            [(record["scope"].work_id, record["function"]) for record in group_records],
        )
        if expected_id != first_scope.group_id:
            raise ValueError("retained matcher group identity is stale")
        if any(
            record["group_fingerprint"] != expected_group_fingerprint
            for record in group_records
        ):
            raise ValueError("retained matcher group fingerprint is stale")
        expected_bodies_by_work_id = dict(expected_bodies)
        if any(
            record["body_fingerprint"]
            != expected_bodies_by_work_id[record["scope"].work_id]
            for record in group_records
        ):
            raise ValueError("retained matcher body fingerprint is stale")
    return tuple(records)


def _module_has_retained_matcher_scopes(module):
    return bool(_validated_retained_matcher_contract(module))


def _effective_group_ids(records):
    fingerprints_by_group_id = {}
    for record in records:
        scope = record["scope"]
        fingerprints_by_group_id.setdefault(scope.group_id, set()).add(
            record["group_fingerprint"]
        )
    effective = {}
    for record in records:
        scope = record["scope"]
        key = (scope.group_id, record["group_fingerprint"])
        if len(fingerprints_by_group_id[scope.group_id]) == 1:
            effective[key] = scope.group_id
        else:
            effective[key] = int(record["group_fingerprint"], 16)
    if len(set(effective.values())) != len(effective):
        raise ValueError("retained matcher identities collide")
    return effective


def _stamp_matcher_work_scopes(
    target,
    module,
    trace,
):
    """Copy typed retained-IR scopes onto matcher records."""
    matches = list(getattr(trace, "matches", ()) or ())
    kernel_records = list(_validated_retained_matcher_contract(module))

    matches_by_symbol = {}
    for match in matches:
        matches_by_symbol.setdefault(match.func_name, []).append(match)

    body = getattr(module, "body", None)
    operations = getattr(body, "operations", ())
    functions_by_symbol = {}
    for function in operations:
        operation = getattr(function, "operation", None)
        if getattr(operation, "name", None) != "func.func":
            continue
        symbol = _retained_symbol_name(function)
        if symbol in functions_by_symbol:
            raise ValueError("retained module repeats a function symbol")
        functions_by_symbol[symbol] = function

    records_by_symbol = {record["symbol"]: dict(record) for record in kernel_records}
    if len(records_by_symbol) != len(kernel_records):
        raise ValueError("retained matcher kernels repeat a function symbol")
    for symbol, function_matches in matches_by_symbol.items():
        function = functions_by_symbol.get(symbol)
        if function is None:
            raise ValueError("matcher trace site has no retained function boundary")
        record = records_by_symbol.get(symbol)
        if record is None:
            group_id, group_fingerprint, body_fingerprints = _spmw_group_contract(
                (), (), [((), function)]
            )
            record = {
                "function": function,
                "symbol": symbol,
                "scope": MatcherWorkScope(group_id, (), (), ()),
                "group_fingerprint": group_fingerprint,
                "body_fingerprint": dict(body_fingerprints)[()],
                "abi_value_ids": (),
            }
            records_by_symbol[symbol] = record
        record["matches"] = function_matches

    if any("matches" not in record for record in records_by_symbol.values()):
        raise ValueError("retained matcher kernel has no matched implementation")

    records = list(records_by_symbol.values())
    non_kernel_keys = set()
    for record in records:
        attributes = getattr(record["function"], "attributes", {})
        if "df.kernel" in attributes:
            continue
        key = (record["scope"].group_id, record["group_fingerprint"])
        if key in non_kernel_keys:
            raise ValueError("non-kernel matcher boundaries are structurally ambiguous")
        non_kernel_keys.add(key)

    assigned = {id(match) for record in records for match in record["matches"]}
    if len(assigned) != len(matches):
        raise ValueError("matcher trace site has no retained function boundary")

    del target
    effective_group_ids = _effective_group_ids(records)
    for record in records:
        scope = record["scope"]
        effective_scope = MatcherWorkScope(
            effective_group_ids[(scope.group_id, record["group_fingerprint"])],
            scope.work_id,
            scope.group_shape,
            scope.coalesced_axes,
        )
        for match in record["matches"]:
            match.work_id = effective_scope.work_id
            _retain_matcher_work_scope(match, effective_scope)
            match.extra["spmw_body_fingerprint"] = record["body_fingerprint"]


class CompiledCallable:
    """An Allo-style callable backed by a compiled PIM artifact.

    Calls accept positional or keyword NumPy operands according to the original
    workload signature and return a :class:`RunResult`. For an executed run
    (``RunResult.kind`` in ``"device"``, ``"simulator"``, ``"functional"``) the
    output arguments are updated in place from ``RunResult.extra["outputs"]``,
    matching the mutation behavior of Allo's LLVM callable modules; a run that
    reports success without an output for a gathered or bound output argument
    raises RuntimeError. ``"virtual"`` and ``"simulator-timing-only"`` runs
    leave every argument untouched.
    """

    def __init__(
        self,
        workload,
        target,
        schedule,
        trace,
        compiled,
        cost=None,
        reduction_order=None,
        resources=None,
    ):
        self.workload = workload
        self.target = target
        self.schedule = schedule
        self.trace = trace
        self.compiled = compiled
        self.cost = cost
        self.reduction_order = normalize_reduction_order(reduction_order)
        # The resource budget the program was selected under (a search cap,
        # ``spmw_resources``), recorded on every result.
        self.resources = normalize_resource_budget(resources)
        # Candidates the enumerators pruned under the reduction-order policy:
        # ``(mode, reason)`` pairs recorded on the selected placements.
        self.reduction_policy_pruned = _reduction_policy_pruned(compiled)
        self.signature = inspect.signature(workload)
        self.__signature__ = self.signature
        self.__name__ = getattr(workload, "__name__", "compiled_workload")
        self.__doc__ = getattr(workload, "__doc__", None)
        self.placement_ranking = getattr(compiled, "placement_ranking", None)
        self._source_binding = _bind_carried_accumulator(
            source_backend_binding(target, trace, tuple(self.signature.parameters)),
            trace,
            tuple(self.signature.parameters),
        )
        self.last_result: RunResult | None = None
        # The realized backend program (a context's ``program``) may need the
        # artifact it belongs to (ranking views, repricing) and the source
        # workload (the APU v1 functional model executes it on the host).
        try:
            compiled.source_workload = workload
        except AttributeError:
            pass
        program = self.backend_program
        attach = getattr(program, "attach", None)
        if callable(attach):
            attach(compiled)

    @property
    def backend_program(self):
        """The backend's realized program object for the selected candidate
        (``Compiled.layout_ctx.program``), or None for backends whose
        executable is the command stream itself."""
        context = getattr(self.compiled, "layout_ctx", None)
        return getattr(context, "program", None)

    def __getattr__(self, name):
        # Explicit forwarding of the realized backend program's documented
        # introspection surface (``INTROSPECTION``): ``compiled.selected_plan``
        # on APU v1, ``compiled.plan`` on Gemini-II.  Nothing else is
        # forwarded; ordinary attribute errors stay attribute errors.
        if name.startswith("_") or "compiled" not in self.__dict__:
            raise AttributeError(name)
        program = self.backend_program
        if program is not None and name in getattr(type(program), "INTROSPECTION", ()):
            return getattr(program, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def __setattr__(self, name, value):
        # Writable introspection attributes of the backend program (Gemini-II
        # ``repetitions``) are forwarded too, so an assignment never lands
        # silently on the callable instead of the program.
        if not name.startswith("_") and "compiled" in self.__dict__:
            program = self.backend_program
            if program is not None and name in getattr(
                type(program), "INTROSPECTION", ()
            ):
                setattr(program, name, value)
                return
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs) -> RunResult:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = dict(bound.arguments)
        for backend_kwarg, parameter in self._source_binding.inputs.items():
            value = bound.arguments[parameter]
            if backend_kwarg in inputs and inputs[backend_kwarg] is not value:
                raise ValueError(
                    f"argument {backend_kwarg!r} collides with the Samsung runtime "
                    f"role {backend_kwarg!r}, which is bound to parameter "
                    f"{parameter!r}"
                )
            inputs[backend_kwarg] = value
        return self.run_backend(**inputs)

    def run_backend(self, **inputs) -> RunResult:
        """Invoke with explicit backend-role arrays.

        This compiler-testing escape hatch supports backend ABIs whose lowered
        operand roles differ from the source workload signature. Applications
        should normally call the object directly. The backend result must carry
        a valid ``RunResult.kind`` (see :class:`RunResult`).
        """
        result = self.compiled.run(**inputs)
        if not isinstance(result, RunResult):
            raise RuntimeError(
                f"backend returned {type(result).__name__}, expected RunResult"
            )
        result.validate()
        _stamp_reduction_order(result, self.reduction_order, self.trace)
        result.extra["resource_budget"] = (
            None if self.resources is None else dict(self.resources)
        )
        self._copy_outputs(result, inputs)
        self.last_result = result
        return result

    run = __call__

    @property
    def execution_graph(self):
        return self.compiled.execution_graph

    def estimate(self, candidate=None):
        """Evaluate the retained candidate graph without invoking a backend.

        ``candidate`` (a candidate name, family name or backend plan object)
        evaluates another scored candidate of the single-kernel ranking
        instead of the selected program; it must have been scored.
        """
        if self.cost is None:
            raise RuntimeError("compiled workload has no executable cost spec")
        if candidate is None:
            if self.execution_graph is None:
                raise RuntimeError("compiled workload has no retained execution graph")
            return self.cost.evaluate(self.execution_graph)
        from .spmw_autoschedule import candidate_matches
        from .spmw_plan import build_execution_graph

        ranking = self.placement_ranking
        if ranking is None or len(ranking.kernels) != 1:
            raise RuntimeError(
                "estimate(candidate) needs the ranking of a single-kernel program"
            )
        for _position, placement in ranking.kernels[0].candidates:
            if candidate_matches(placement, candidate):
                graph = build_execution_graph(
                    self.target,
                    self.trace,
                    placement,
                    self.cost,
                    host_moves=getattr(self.compiled, "host_moves", ()),
                )
                return self.cost.evaluate(graph)
        names = [
            placement.mode for _position, placement in ranking.kernels[0].candidates
        ]
        raise ValueError(
            f"{candidate!r} names no scored candidate; scored candidates: {names}"
        )

    @staticmethod
    def _bound_output_assignment(outputs, arguments, gather_roles, binding):
        """Destination for a single backend output under a source binding:
        the gather role, then the bound output parameter, then a name match
        that is not a bound input."""
        output_name, value = next(iter(outputs.items()))
        if len(gather_roles) == 1:
            return {gather_roles[0]: value}
        if binding.output is not None and binding.output in arguments:
            return {binding.output: value}
        bound_inputs = set(binding.inputs) | set(binding.inputs.values())
        if output_name in arguments and output_name not in bound_inputs:
            return {output_name: value}
        for name in arguments:
            if name not in bound_inputs and name.lower() == output_name.lower():
                return {name: value}
        return {}

    def _copy_outputs(self, result, arguments):
        """Write ``extra["outputs"]`` of an executed run into the arguments.

        Virtual and timing-only results write nothing. An executed result
        must cover every gathered output role (or, with none, the bound
        output parameter); a missing one raises RuntimeError.
        """
        if not result.executed:
            return
        outputs = result.outputs
        gather_roles = []
        for resolved in getattr(self.compiled, "host_moves", ()):
            verb = getattr(getattr(resolved, "verb", None), "name", None)
            if verb == "gather" and resolved.buffer_role in arguments:
                gather_roles.append(resolved.buffer_role)

        binding = getattr(self, "_source_binding", None)
        required = list(gather_roles)
        if not required:
            # Every externally visible destination of the source program
            # (matched stores and accepted copies into parameters), derived
            # from the trace's effects by retained identity, independently of
            # the binding; then the binding's own destinations.
            from .spmw_match_engine import external_effects

            names = list(arguments)
            ordinal_by_ref = {
                ref: int(ordinal)
                for ordinal, ref in (
                    getattr(self.trace, "source_value_refs", None) or {}
                ).items()
            }
            for _func, _memref, ref, _category in external_effects(
                self.trace, self.target
            ):
                ordinal = ordinal_by_ref.get(ref)
                if (
                    ordinal is not None
                    and ordinal < len(names)
                    and names[ordinal] not in required
                ):
                    required.append(names[ordinal])
            if binding is not None:
                for name in dict.fromkeys(
                    list(getattr(binding, "outputs", {}).values())
                    + ([binding.output] if binding.output is not None else [])
                ):
                    if name in arguments and name not in required:
                        required.append(name)
        if not outputs:
            if required:
                raise RuntimeError(
                    f"backend {result.backend!r} reported success "
                    f"(kind={result.kind!r}) but produced no output for role(s) "
                    f"{required}"
                )
            return

        bound_outputs = getattr(binding, "outputs", None) or {}
        if (
            binding is not None
            and len(outputs) > 1
            and all(key in bound_outputs or key not in arguments for key in outputs)
        ):
            # Several external destinations (a result and its copies), keyed
            # by the runtime's output names and mapped through the binding.
            assignments = {
                bound_outputs[key]: value
                for key, value in outputs.items()
                if key in bound_outputs and bound_outputs[key] in arguments
            }
        elif (
            binding is not None
            and (binding.inputs or binding.output is not None)
            and len(outputs) == 1
        ):
            assignments = self._bound_output_assignment(
                outputs, arguments, gather_roles, binding
            )
        else:
            assignments = {}
            lower_names = {name.lower(): name for name in arguments}
            for output_name, value in outputs.items():
                if output_name in arguments:
                    assignments[output_name] = value
                elif output_name.lower() in lower_names:
                    assignments[lower_names[output_name.lower()]] = value
            if len(outputs) == 1 and len(gather_roles) == 1:
                assignments.setdefault(gather_roles[0], next(iter(outputs.values())))

        missing = [name for name in required if name not in assignments]
        if missing:
            raise RuntimeError(
                f"backend {result.backend!r} reported success "
                f"(kind={result.kind!r}) but produced no output for role(s) "
                f"{missing}; it returned outputs {sorted(outputs)}"
            )

        for name, value in assignments.items():
            destination = arguments[name]
            if not isinstance(destination, np.ndarray):
                continue
            source = np.asarray(value)
            if source.shape != destination.shape:
                if source.size != destination.size:
                    raise ValueError(
                        f"backend output {name!r} has shape {source.shape}, "
                        f"but destination has shape {destination.shape}"
                    )
                source = source.reshape(destination.shape)
            np.copyto(destination, source, casting="same_kind")


def _bind_carried_accumulator(binding, trace, parameter_names):
    """Bind runtime role ``C`` to the parameter a carried accumulator reads.

    A ``y[i] += ...`` accumulator (``accumulator_init == "carried"``) starts
    from the caller's ``y``; a backend program that loads an initial
    accumulator (Samsung's ordered program) receives it as ``C`` and its
    result is written back to the same parameter.
    """
    if not binding.inputs:
        return binding
    refs = {
        operand.value_ref
        for match in getattr(trace, "matches", ())
        if match.extra.get("accumulator_init") == "carried"
        for operand in match.operands
        if operand.is_loop_carried and operand.value_ref is not None
    }
    if len(refs) != 1:
        return binding
    (ref,) = refs
    ordinals = [
        int(ordinal)
        for ordinal, source_ref in (
            getattr(trace, "source_value_refs", None) or {}
        ).items()
        if source_ref == ref and 0 <= int(ordinal) < len(parameter_names)
    ]
    if len(ordinals) != 1:
        return binding
    name = parameter_names[ordinals[0]]
    return type(binding)(
        inputs={**binding.inputs, "C": name}, output=binding.output or name
    )


def _reduction_policy_pruned(compiled):
    layouts = getattr(compiled, "layout", None)
    if not isinstance(layouts, (list, tuple)):
        layouts = [layouts]
    pruned = []
    for layout in layouts:
        for record in (getattr(layout, "extra", None) or {}).get(
            "reduction_policy_pruned", ()
        ):
            if record not in pruned:
                pruned.append(record)
    return tuple(pruned)


def _stamp_reduction_order(result, reduction_order, trace=None):
    """Record the reduction-order policy the executed artifact was compiled
    under, and the per-accumulator permission it implied."""
    result.extra["reduction_order"] = reduction_order
    policies = []
    for match in getattr(trace, "matches", ()) or ():
        policy = match.extra.get(POLICY_KEY)
        if policy is not None:
            entry = {
                "accumulator_type": policy.accumulator_type,
                "may_regroup": policy.may_regroup,
            }
            if entry not in policies:
                policies.append(entry)
    if policies:
        result.extra["reduction_policy"] = policies


class CompiledWorkload(typing.Protocol):
    """What every allo.compile result supports."""

    def __call__(self, *args, **kwargs) -> RunResult: ...

    def run(self, *args, **kwargs) -> RunResult: ...


def compile(
    workload,
    target,
    cost=None,
    *,
    backend=None,
    host_moves=None,
    layout=None,
    reduction_order=None,
    resources=None,
) -> CompiledWorkload:
    """Compile ``workload`` for ``target`` and return a NumPy-callable object.

    Stages (release gate G9, ``docs/release/design/shared-compilation-path.md``):

    1. **Typed program frontends.** ``APUG2TypedProgram``,
       ``APUG2ComposedContractionProgram`` and ``APUv1Program`` are program
       records without source: their semantics are stated in the shared
       program contract and their realization is one candidate of the
       common pipeline below (`allo.pim.apu_g2_record_frontend`,
       `allo.pim.apu_v1_record_frontend`).  A record fixes its own
       reduction grouping and rejects ``reduction_order``.
    2. **Shared front stages** (every ordinary source workload, every
       target; :func:`allo.spmw_pipeline.normalize_source`): the ``@allo``
       function is customized to MLIR, matched against the target's
       operation patterns (or the source-idiom view a backend declares as
       ``target.spmw_match_view``), its stores are classified and the
       coverage, accumulator-initialization, conversion, alias and copy
       contracts are enforced (``UnsupportedWorkloadError``), and the shared
       semantic facts are derived (:func:`allo.spmw_semantics.analyze_program`).
    3. **Common candidate pipeline** (``compile_for_target``): the target's
       enumerator produces the candidate placements of every kernel (Samsung,
       AiM, UPMEM and AttAcc placements; the APU v1 GVML vector plans of a
       plain-function contraction or the grouped program of a dataflow
       region; the Gemini-II composed-dot launch plans), the target-neutral
       legality pass filters them, the movement stage crosses layouts, every
       candidate is priced through its execution graph (a backend plan
       through its context's ``candidate_execution_graph``), ranked and
       materialized in cost order (`rank_matcher_placements`), and the
       selected program is lowered by the backend context and run by its
       runner.  ``layout=`` names a candidate, a candidate family or a
       backend plan object and restricts the ranking to it.

    A source workload on UPMEM lowers each matched kernel group to a
    calibrated physical plan and runs it on the provenance uPIMulator; a run
    reports DPU logic cycles. AiM and UPMEM have no typed route: both compile
    through the matcher.

    Parameters
    ----------
    workload : callable or object
        An Allo workload callable, a typed PIM program, or an object exposing
        ``build()``.
    target : Target or callable
        A built Tenon target or a zero-argument target builder.
    cost : CostSpec
        Executable cost program used by autoscheduling and virtual execution.
    backend : str or None
        ``None`` selects the target's normal simulator/device runner;
        ``"virtual"`` evaluates only the analytical performance graph.

    Returns
    -------
    A callable whose calls return a :class:`RunResult`. Check
    ``RunResult.kind`` before reading output arrays: only ``"device"``,
    ``"simulator"`` and ``"functional"`` runs write outputs. With
    ``backend="virtual"`` (kind ``"virtual"``) and on timing-only targets such
    as AiM (kind ``"simulator-timing-only"``) the output arrays are not
    written and keep whatever the caller put in them.
    host_moves : sequence or None
        Explicit host-transfer records. They are never discovered implicitly.
    layout : object
        Optional preselected placement or placement list.
    reduction_order : {"strict", "reassociate"} or None
        Whether partial sums of a reduction may be regrouped. ``"strict"``
        (the default, also selected by None) keeps the source accumulation
        order and rounding of every floating-point accumulator: schedules
        that split such a reduction (bank fibers, SIMD lane partials, tree
        reductions) are pruned before ranking, and a backend with no
        order-preserving program raises ``ReductionOrderError``. Integer
        accumulators wrap modulo 2^width, so ``"strict"`` still permits
        splitting them. ``"reassociate"`` permits any grouping. The choice is
        stamped on every accumulating match (``extra["reduction_policy"]``)
        and on every result (``RunResult.extra["reduction_order"]``).
        Typed program records fix their own grouping and reject this
        argument. See ``docs/release/design/reduction-policy.md``.
    resources : mapping or None
        A resource budget, ``{unit name: cap}`` over the target's structural
        units (``resources={"channel": 8}``): the enumerators bound their
        partition domains by it, the shared ranking selects among the
        candidates that fit, the realization is checked against it and it
        is recorded on every result (``extra["resource_budget"]``).  A cap
        above the device's unit count or an undeclared unit is refused
        (``allo.spmw_resources``).
    """
    workload = _materialize_workload(workload)
    target = _materialize_target(target)
    bound_cost = _resolve_cost(target, cost)
    requested_order = reduction_order
    reduction_order = normalize_reduction_order(reduction_order)

    if is_apu_v1_program(workload):
        # An APU v1 whole-function program record is a frontend: its lowering
        # (scalar ARC C or a proven native recipe) states its effects in the
        # shared program contract and is the record's candidate of the
        # common pipeline (`allo.pim.apu_v1_record_frontend`).
        program = normalize_apu_v1_program(
            workload, target, reduction_order=requested_order
        )
        workload = program.workload
    elif is_apu_g2_record(workload):
        # A typed Gemini-II program record is a frontend: its semantics are
        # stated in the shared program contract and its realization is a
        # candidate of the common pipeline (`allo.pim.apu_g2_record_frontend`).
        program = normalize_apu_g2_record(
            workload, target, reduction_order=reduction_order
        )
        workload = program.workload
    else:
        # Shared front stages: frontend, matcher, coverage/semantic contracts
        # and the semantic facts, identical for every target.
        program = normalize_source(workload, target, reduction_order=reduction_order)
        if program.trace.matches:
            # Derived once here and stamped on the trace: the backend
            # enumerators and contexts read these facts, never the IR text.
            program.program_facts()
    schedule, trace = program.schedule, program.trace
    module = program.module

    require_matcher_route_policy(target, trace)
    if module is not None:
        _stamp_matcher_work_scopes(target, module, trace)
    # The resource budget (``resources={"channel": 8}``): a cap on the units
    # a program may occupy, stamped before enumeration so the partition
    # domains, the ranking and the realization check all read it.
    budget = normalize_resource_budget(resources, target)
    stamp_resource_budget(trace, budget)
    launch_order = None
    if isinstance(host_moves, HostProgram):
        if host_moves.launches:
            launch_order = launch_schedule(host_moves, trace, module=module)
        records = list(host_moves.moves)
    else:
        records = list(host_moves or ())
    compiled = compile_for_target(
        target,
        trace,
        layout=layout,
        backend=backend,
        host_moves=records,
        buffer_metrics=_buffer_metric_manifest(workload, trace, records),
        cost=bound_cost,
        module=module,
        launch_schedule=launch_order,
    )
    return CompiledCallable(
        workload,
        target,
        schedule,
        trace,
        compiled,
        cost=bound_cost,
        reduction_order=reduction_order,
        resources=budget,
    )
