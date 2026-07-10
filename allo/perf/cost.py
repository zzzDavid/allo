# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable cost specifications.

A cost spec is ordinary Python code.  Binding it to a structural target runs
the builder and registers rules against target handle identity.  Executing a
rule emits cycle steps over concrete instances derived from the SPMW work
coordinate and the target unit tree.
"""

from __future__ import annotations

import contextlib
import dis
import hashlib
from importlib import metadata as importlib_metadata
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from enum import Enum
from types import CodeType, MappingProxyType, ModuleType

from .evaluator import Evaluator
from .graph import Activity, ExecutionGraph, HandleInstance, Occupancy


def _unit_path(unit):
    names = []
    while unit is not None:
        names.append(unit.name)
        unit = unit.parent
    return "/".join(reversed(names))


def handle_path(handle):
    """Return the stable structural path for a target handle."""
    from ..spmw_target import MemoryRef, Unit

    if isinstance(handle, Unit):
        return _unit_path(handle)
    if isinstance(handle, MemoryRef):
        return f"{handle_path(handle.memory)}[{handle.idx!r}]"
    owner = getattr(handle, "owner", None)
    name = getattr(handle, "name", None)
    if owner is None or not name:
        raise TypeError(f"{handle!r} is not a target unit/op/move/memory/register")
    return f"{_unit_path(owner)}/{name}"


def _handle_owner(handle):
    from ..spmw_target import MemoryRef, Unit

    if isinstance(handle, Unit):
        return handle
    if isinstance(handle, MemoryRef):
        return handle.memory.owner
    owner = getattr(handle, "owner", None)
    if owner is None:
        raise TypeError(f"{handle!r} has no target-tree owner")
    return owner


def _scope_coordinates(owner, work_id):
    units = []
    unit = owner
    while unit is not None and unit.parent is not None:
        if unit.mode not in ("device", "host") and unit.mapping:
            units.append(unit)
        unit = unit.parent
    units.reverse()
    values = tuple(work_id[: len(units)])
    if len(values) < len(units):
        values += (0,) * (len(units) - len(values))
    coordinates = []
    for coordinate, unit in zip(values, units):
        extent = 1
        for factor in unit.mapping:
            extent *= factor
        if coordinate < 0 or coordinate >= extent:
            raise ValueError(
                f"work coordinate {coordinate} is outside {unit.name!r} "
                f"extent {extent}"
            )
        axis_name = next(iter(unit.axes), unit.name)
        coordinates.append((axis_name, coordinate))
    return tuple(coordinates)


def _evaluate_index(value, work_id):
    """Substitute a work coordinate into a target ``SymExpr`` index."""
    from ..spmw_target import SymExpr, UnitId

    if isinstance(value, UnitId):
        if value.level >= len(work_id):
            raise ValueError(f"work coordinate {work_id!r} has no axis {value.level}")
        return int(work_id[value.level])
    if not isinstance(value, SymExpr):
        return value
    lhs = _evaluate_index(value.args[0], work_id)
    rhs = _evaluate_index(value.args[1], work_id)
    operations = {
        "add": lambda: lhs + rhs,
        "sub": lambda: lhs - rhs,
        "mul": lambda: lhs * rhs,
        "floordiv": lambda: lhs // rhs,
        "mod": lambda: lhs % rhs,
    }
    try:
        return operations[value.op]()
    except KeyError as exc:
        raise ValueError(f"unsupported target index expression {value.op!r}") from exc


def concrete_instance(handle, work_id):
    """Bind a structural target handle to one spatial instance."""
    from ..spmw_target import Memory, MemoryRef, Move, Op, Register, Unit

    owner = _handle_owner(handle)
    kind = "handle"
    for cls, label in (
        (Unit, "unit"),
        (Op, "op"),
        (Move, "move"),
        (MemoryRef, "memory"),
        (Memory, "memory"),
        (Register, "register"),
    ):
        if isinstance(handle, cls):
            kind = label
            break
    path = handle_path(handle)
    if isinstance(handle, MemoryRef):
        index = _evaluate_index(handle.idx, tuple(work_id))
        path = f"{handle_path(handle.memory)}[{index!r}]"
        capacity = handle.memory.capacity
    else:
        capacity = int(getattr(handle, "capacity", 1))
    return HandleInstance(
        path=path,
        coordinates=_scope_coordinates(owner, tuple(work_id)),
        capacity=capacity,
        kind=kind,
    )


@dataclass(frozen=True)
class CostEvent:
    """One target-bound operation or move presented to a cost rule."""

    id: str
    primitive: object
    work_id: tuple[int, ...] = ()
    metrics: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    attributes: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def create(cls, event_id, primitive, work_id=(), metrics=None, attributes=None):
        return cls(
            id=event_id,
            primitive=primitive,
            work_id=tuple(work_id),
            metrics=MappingProxyType(dict(metrics or {})),
            attributes=MappingProxyType(dict(attributes or {})),
        )

    def __getattr__(self, name):
        try:
            return self.metrics[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def instance(self, handle):
        return concrete_instance(handle, self.work_id)

    @property
    def operation_instance(self):
        return self.instance(self.primitive)


@dataclass(frozen=True)
class CostUse:
    """A cost-program request to occupy a structural or concrete handle."""

    handle: object
    cycles: int | None = None
    amount: int = 1


class CostContext:
    """Builder used while one cost rule emits activities."""

    def __init__(self, graph, event, dependencies):
        self.graph = graph
        self.event = event
        self._current = tuple(dependencies)
        self._counter = 0
        self._repeat = [1]
        self._parallel = []

    def use(self, handle, *, cycles=None, amount=1):
        return CostUse(handle, cycles, amount)

    def step(self, *, cycles=None, latency=None, occupy=(), name="step"):
        """Emit one cycle step; program order is sequential by default."""
        if latency is None:
            latency = cycles
        if latency is None:
            raise TypeError("cost.step requires cycles= or latency=")
        multiplier = self._repeat[-1]
        latency = int(latency) * multiplier
        if latency < 0:
            raise ValueError("cost step latency must be non-negative")

        occupancies = []
        for use in occupy:
            if not isinstance(use, CostUse):
                use = CostUse(use)
            handle = (
                use.handle
                if isinstance(use.handle, HandleInstance)
                else self.event.instance(use.handle)
            )
            duration = latency if use.cycles is None else int(use.cycles) * multiplier
            occupancies.append(Occupancy(handle, duration, use.amount))

        activity_id = f"{self.event.id}:cost:{self._counter}:{name}"
        self._counter += 1
        dependencies = self._parallel[-1]["entry"] if self._parallel else self._current
        self.graph.add(
            Activity(
                id=activity_id,
                primitive=handle_path(self.event.primitive),
                latency_cycles=latency,
                occupancy=tuple(occupancies),
                depends_on=tuple(dependencies),
                label=name,
                metadata={
                    **dict(self.event.attributes),
                    "work_id": self.event.work_id,
                },
            )
        )
        if self._parallel:
            self._parallel[-1]["terminals"].append(activity_id)
        else:
            self._current = (activity_id,)
        return activity_id

    @contextlib.contextmanager
    def repeat(self, count):
        """Summarize a sequentially repeated cost region without unrolling it."""
        count = int(count)
        if count < 0:
            raise ValueError("cost.repeat count must be non-negative")
        self._repeat.append(self._repeat[-1] * count)
        try:
            yield self
        finally:
            self._repeat.pop()

    @contextlib.contextmanager
    def parallel(self):
        """Emit enclosed steps from one common dependency frontier."""
        state = {"entry": self._current, "terminals": []}
        self._parallel.append(state)
        try:
            yield self
        finally:
            self._parallel.pop()
            self._current = tuple(state["terminals"]) or tuple(state["entry"])

    @property
    def terminals(self):
        return self._current


_BIND_STACK = []
_CODE_GLOBAL_NAMES_CACHE = {}
_CODE_GLOBAL_REFERENCES_CACHE = {}
_EMPTY_CLOSURE_CELL = object()
_FILE_DIGEST_CACHE = {}
_PACKAGE_VERSION_CACHE = {}
_DEPENDENCY_MANIFEST_METHOD = "__allo_fingerprint_manifest__"


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


class _FingerprintCanonicalizer:
    """Build address-free canonical data for one bound cost model."""

    _UNSUPPORTED = object()

    def __init__(self):
        self._active_containers = set()
        self._active_functions = set()
        self._active_objects = set()
        self._active_types = set()
        self._function_cache = {}
        self._object_cache = {}
        self._module_provenance_cache = {}
        self._type_provenance_cache = {}
        self._opaque_target_values = {}

    def explicit(self, value, path="fingerprint_data"):
        return self._value(value, path=path, dependency_module=None, explicit=True)

    def dependency(self, value, module, path):
        return self._value(value, path=path, dependency_module=module, explicit=False)

    def _value(self, value, *, path, dependency_module, explicit):
        scalar = self._scalar(value, path)
        if scalar is not self._UNSUPPORTED:
            return scalar

        value_type = type(value)
        if value_type in (list, tuple):
            with self._container(value, path):
                items = [
                    self._value(
                        item,
                        path=f"{path}[{index}]",
                        dependency_module=dependency_module,
                        explicit=explicit,
                    )
                    for index, item in enumerate(value)
                ]
            return [value_type.__name__, items]
        if value_type in (set, frozenset):
            with self._container(value, path):
                items = [
                    self._value(
                        item,
                        path=f"{path}[item]",
                        dependency_module=dependency_module,
                        explicit=explicit,
                    )
                    for item in value
                ]
            items.sort(key=_canonical_json)
            return [value_type.__name__, items]
        if value_type is dict or value_type is MappingProxyType:
            with self._container(value, path):
                items = []
                for index, (key, item) in enumerate(value.items()):
                    canonical_key = self._value(
                        key,
                        path=f"{path}[key {index}]",
                        dependency_module=dependency_module,
                        explicit=explicit,
                    )
                    canonical_value = self._value(
                        item,
                        path=f"{path}[value {index}]",
                        dependency_module=dependency_module,
                        explicit=explicit,
                    )
                    items.append([canonical_key, canonical_value])
            items.sort(key=lambda pair: _canonical_json(pair[0]))
            return ["dict", items]

        if explicit:
            type_name = f"{value_type.__module__}.{value_type.__qualname__}"
            raise TypeError(f"{path} contains unsupported value type {type_name}")
        return self._dependency_object(value, dependency_module, path)

    def _scalar(self, value, path):
        value_type = type(value)
        if value is None:
            return ["none"]
        if value_type is bool:
            return ["bool", value]
        if value_type is int:
            return ["int", str(value)]
        if value_type is float:
            if not math.isfinite(value):
                raise TypeError(f"{path} contains a non-finite float")
            return ["float", value.hex()]
        if value_type is complex:
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise TypeError(f"{path} contains a non-finite complex value")
            return ["complex", value.real.hex(), value.imag.hex()]
        if value_type is str:
            return ["str", value]
        if value_type is bytes:
            return ["bytes", value.hex()]
        if value is Ellipsis:
            return ["ellipsis"]
        if value is NotImplemented:
            return ["not_implemented"]
        return self._UNSUPPORTED

    def _dependency_object(self, value, dependency_module, path):
        structural = self._structural_reference(value)
        if structural is not None:
            return structural
        if inspect.isfunction(value):
            return ["function", self.function(value, dependency_module)]
        if inspect.ismethod(value):
            return [
                "bound_method",
                self.function(value.__func__, value.__func__.__module__),
                self.dependency(value.__self__, value.__func__.__module__, path),
            ]
        if (
            inspect.isbuiltin(value)
            or inspect.ismethoddescriptor(value)
            or inspect.ismethodwrapper(value)
        ):
            return self._native_callable(value, path)
        if isinstance(value, ModuleType):
            return ["module", self._module_provenance(value, path)]
        if isinstance(value, type):
            return ["type", self._type_provenance(value, path)]
        if isinstance(value, Enum):
            return [
                "enum",
                self._type_provenance(type(value), path),
                value.name,
                self.dependency(value.value, type(value).__module__, path),
            ]

        manifest_function = self._dependency_manifest_function(type(value), path)
        if manifest_function is not None:
            return self._manifest_dependency(value, manifest_function, path)
        if is_dataclass(value):
            return self._frozen_dataclass(value, path)
        self._unsupported_dependency(value, path)

    def _manifest_dependency(self, value, manifest_function, path):
        identity = id(value)
        if identity in self._object_cache:
            return self._object_cache[identity]
        if identity in self._active_objects:
            raise TypeError(f"{path} contains a recursive dependency manifest")
        self._active_objects.add(identity)
        try:
            manifest = manifest_function(value)
            canonical = [
                "typed_manifest",
                self._type_provenance(type(value), path),
                self.function(manifest_function, manifest_function.__module__),
                self.explicit(
                    manifest,
                    path=f"{path} {_DEPENDENCY_MANIFEST_METHOD} result",
                ),
            ]
        finally:
            self._active_objects.remove(identity)
        fingerprint = [
            "typed_manifest",
            type(value).__module__,
            type(value).__qualname__,
            hashlib.sha256(_canonical_json(canonical).encode()).hexdigest(),
        ]
        self._object_cache[identity] = fingerprint
        return fingerprint

    @staticmethod
    def _dependency_manifest_function(value_type, path):
        for owner in value_type.__mro__:
            if _DEPENDENCY_MANIFEST_METHOD not in vars(owner):
                continue
            manifest_function = vars(owner)[_DEPENDENCY_MANIFEST_METHOD]
            if not inspect.isfunction(manifest_function):
                raise TypeError(
                    f"{path} declares {_DEPENDENCY_MANIFEST_METHOD} as a "
                    "non-function"
                )
            return manifest_function
        return None

    def _frozen_dataclass(self, value, path):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if parameters is None or not parameters.frozen:
            self._unsupported_dependency(value, path)
        identity = id(value)
        if identity in self._object_cache:
            return self._object_cache[identity]
        if identity in self._active_objects:
            raise TypeError(f"{path} contains a recursive frozen dataclass")
        self._active_objects.add(identity)
        try:
            canonical = [
                "frozen_dataclass",
                self._type_provenance(type(value), path),
                [
                    [
                        descriptor.name,
                        self.dependency(
                            getattr(value, descriptor.name),
                            type(value).__module__,
                            f"{path}.{descriptor.name}",
                        ),
                    ]
                    for descriptor in dataclass_fields(value)
                ],
            ]
        finally:
            self._active_objects.remove(identity)
        fingerprint = [
            "frozen_dataclass",
            type(value).__module__,
            type(value).__qualname__,
            hashlib.sha256(_canonical_json(canonical).encode()).hexdigest(),
        ]
        self._object_cache[identity] = fingerprint
        return fingerprint

    def _native_callable(self, value, path):
        owner = getattr(value, "__objclass__", None)
        module_name = getattr(value, "__module__", None)
        if module_name is None and isinstance(owner, type):
            module_name = owner.__module__
        module = sys.modules.get(module_name)
        if not isinstance(module, ModuleType):
            self._unsupported_dependency(value, path)

        receiver = getattr(value, "__self__", None)
        if isinstance(receiver, ModuleType):
            receiver = None
        return [
            "native_callable",
            getattr(value, "__qualname__", getattr(value, "__name__", None)),
            self._module_provenance(module, path),
            (
                None
                if not isinstance(owner, type)
                else self._type_provenance(owner, path)
            ),
            (
                None
                if receiver is None
                else self.dependency(receiver, module_name, f"{path} receiver")
            ),
        ]

    def _module_provenance(self, module, path):
        identity = id(module)
        if identity in self._module_provenance_cache:
            return self._module_provenance_cache[identity]

        name = getattr(module, "__name__", None)
        if not isinstance(name, str) or not name:
            raise TypeError(f"{path} references a module without a stable name")
        module_values = vars(module)
        module_spec = module_values.get("__spec__")
        origin = getattr(module_spec, "origin", None)
        file_path = module_values.get("__file__")
        if file_path is not None:
            artifact = self._file_provenance(file_path, path)
        elif name == "builtins" or origin in ("built-in", "frozen"):
            artifact = {"kind": origin or "built-in", "sha256": None}
        else:
            raise TypeError(
                f"{path} references module {name!r} without source or binary "
                "provenance"
            )

        package_name = (module_values.get("__package__") or name).split(".", 1)[0]
        if package_name not in _PACKAGE_VERSION_CACHE:
            try:
                package_version = importlib_metadata.version(package_name)
            except importlib_metadata.PackageNotFoundError:
                package_version = None
            _PACKAGE_VERSION_CACHE[package_name] = package_version
        package_version = _PACKAGE_VERSION_CACHE[package_name]
        declared_version = module_values.get("__version__")
        if type(declared_version) not in (type(None), str, int, float):
            declared_version = None
        provenance = {
            "name": name,
            "package": package_name,
            "package_version": package_version,
            "declared_version": declared_version,
            "artifact": artifact,
            "python": self._python_provenance(),
        }
        self._module_provenance_cache[identity] = provenance
        return provenance

    @staticmethod
    def _file_provenance(file_path, path):
        try:
            normalized_path = os.fspath(file_path)
            stat = os.stat(normalized_path)
        except (OSError, TypeError) as error:
            raise TypeError(
                f"{path} references unreadable executable artifact {file_path!r}"
            ) from error
        cache_key = (
            normalized_path,
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        digest = _FILE_DIGEST_CACHE.get(cache_key)
        if digest is None:
            try:
                with open(normalized_path, "rb") as artifact_file:
                    digest = hashlib.sha256(artifact_file.read()).hexdigest()
            except OSError as error:
                raise TypeError(
                    f"{path} references unreadable executable artifact "
                    f"{file_path!r}"
                ) from error
            if len(_FILE_DIGEST_CACHE) >= 128:
                _FILE_DIGEST_CACHE.clear()
            _FILE_DIGEST_CACHE[cache_key] = digest

        basename = os.path.basename(normalized_path)
        if basename.endswith((".py", ".pyw")):
            kind = "python_source"
        elif basename.endswith((".pyc", ".pyo")):
            kind = "python_bytecode"
        else:
            kind = "native_binary"
        return {"kind": kind, "basename": basename, "sha256": digest}

    @staticmethod
    def _python_provenance():
        return {
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": list(sys.version_info),
            "abi_flags": getattr(sys, "abiflags", ""),
            "byteorder": sys.byteorder,
        }

    def _type_provenance(self, value_type, path):
        identity = id(value_type)
        if identity in self._type_provenance_cache:
            return self._type_provenance_cache[identity]
        if identity in self._active_types:
            return {
                "recursive": True,
                "module": value_type.__module__,
                "qualname": value_type.__qualname__,
            }

        module = sys.modules.get(value_type.__module__)
        if not isinstance(module, ModuleType):
            self._unsupported_dependency(value_type, path)
        module_provenance = self._module_provenance(module, path)

        self._active_types.add(identity)
        try:
            behavior = self._python_class_behavior(value_type, path)
            manifest = {
                "module": value_type.__module__,
                "qualname": value_type.__qualname__,
                "module_provenance": module_provenance,
                "behavior": behavior,
            }
        finally:
            self._active_types.remove(identity)
        provenance = {
            "module": value_type.__module__,
            "qualname": value_type.__qualname__,
            "sha256": hashlib.sha256(_canonical_json(manifest).encode()).hexdigest(),
        }
        self._type_provenance_cache[identity] = provenance
        return provenance

    def _python_class_behavior(self, value_type, path):
        if value_type.__module__ == "builtins":
            return []
        if issubclass(value_type, Enum):
            return [
                [name, self.dependency(member.value, value_type.__module__, path)]
                for name, member in value_type.__members__.items()
            ]

        behavior = []
        for name, member in sorted(vars(value_type).items()):
            functions = ()
            if inspect.isfunction(member):
                functions = (member,)
            elif isinstance(member, (staticmethod, classmethod)):
                functions = (member.__func__,)
            elif isinstance(member, property):
                functions = tuple(
                    function
                    for function in (member.fget, member.fset, member.fdel)
                    if function is not None
                )
            if functions:
                behavior.append(
                    [
                        name,
                        [
                            {
                                "module": function.__module__,
                                "qualname": function.__qualname__,
                                "code": self.code(function.__code__),
                                "attributes": self.dependency(
                                    function.__dict__,
                                    function.__module__,
                                    f"{path} class method {name} attributes",
                                ),
                            }
                            for function in functions
                        ],
                    ]
                )
                continue
            if name.startswith("_"):
                continue
            behavior.append(
                [
                    name,
                    self.dependency(
                        member,
                        value_type.__module__,
                        f"{path} class attribute {name}",
                    ),
                ]
            )
        return behavior

    @staticmethod
    def _unsupported_dependency(value, path):
        value_type = type(value)
        type_name = f"{value_type.__module__}.{value_type.__qualname__}"
        raise TypeError(
            f"{path} contains unsupported executable dependency type {type_name}; "
            "use canonical immutable data or implement "
            f"{_DEPENDENCY_MANIFEST_METHOD}"
        )

    @contextlib.contextmanager
    def _container(self, value, path):
        identity = id(value)
        if identity in self._active_containers:
            raise TypeError(f"{path} contains a recursive container")
        self._active_containers.add(identity)
        try:
            yield
        finally:
            self._active_containers.remove(identity)

    def code(self, code):
        constants = []
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                constants.append(["code", self.code(constant)])
            else:
                constants.append(
                    self._value(
                        constant,
                        path="function constant",
                        dependency_module=None,
                        explicit=True,
                    )
                )
        return {
            "argcount": code.co_argcount,
            "posonlyargcount": code.co_posonlyargcount,
            "kwonlyargcount": code.co_kwonlyargcount,
            "nlocals": code.co_nlocals,
            "flags": code.co_flags,
            "code": code.co_code.hex(),
            "exceptiontable": getattr(code, "co_exceptiontable", b"").hex(),
            "constants": constants,
            "names": list(code.co_names),
            "varnames": list(code.co_varnames),
            "freevars": list(code.co_freevars),
            "cellvars": list(code.co_cellvars),
        }

    def _code_global_names(self, code):
        cached = _CODE_GLOBAL_NAMES_CACHE.get(code)
        if cached is not None:
            return cached
        names = {
            instruction.argval
            for instruction in dis.get_instructions(code)
            if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME")
            and isinstance(instruction.argval, str)
        }
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                names.update(self._code_global_names(constant))
        result = tuple(sorted(names))
        if len(_CODE_GLOBAL_NAMES_CACHE) >= 4096:
            _CODE_GLOBAL_NAMES_CACHE.clear()
        _CODE_GLOBAL_NAMES_CACHE[code] = result
        return result

    def _code_global_references(self, code):
        cached = _CODE_GLOBAL_REFERENCES_CACHE.get(code)
        if cached is not None:
            return cached
        references = {}
        instructions = tuple(dis.get_instructions(code))
        for position, instruction in enumerate(instructions):
            if instruction.opname not in ("LOAD_GLOBAL", "LOAD_NAME"):
                continue
            if not isinstance(instruction.argval, str):
                continue
            attributes = []
            for successor in instructions[position + 1 :]:
                if successor.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
                    break
                attributes.append(successor.argval)
            references.setdefault(instruction.argval, set()).add(tuple(attributes))
        for constant in code.co_consts:
            if not isinstance(constant, CodeType):
                continue
            for name, paths in self._code_global_references(constant).items():
                references.setdefault(name, set()).update(paths)
        result = {
            name: tuple(sorted(paths)) for name, paths in sorted(references.items())
        }
        if len(_CODE_GLOBAL_REFERENCES_CACHE) >= 4096:
            _CODE_GLOBAL_REFERENCES_CACHE.clear()
        _CODE_GLOBAL_REFERENCES_CACHE[code] = result
        return result

    def _module_attribute_dependencies(self, module, paths, path):
        dependencies = []
        for attributes in paths:
            if not attributes:
                continue
            value = module
            for attribute in attributes:
                if isinstance(value, ModuleType):
                    values = vars(value)
                    if attribute not in values:
                        raise TypeError(
                            f"{path} references dynamic module attribute "
                            f"{value.__name__}.{attribute} without static provenance"
                        )
                    value = values[attribute]
                else:
                    value = inspect.getattr_static(value, attribute)
            dependencies.append(
                [
                    list(attributes),
                    self.dependency(
                        value,
                        getattr(value, "__module__", module.__name__),
                        f"{path}.{'.'.join(attributes)}",
                    ),
                ]
            )
        return dependencies

    def _function_dependencies(self, function, module):
        global_values = function.__globals__
        builtin_values = function.__builtins__
        if isinstance(builtin_values, ModuleType):
            builtin_values = vars(builtin_values)

        scoped_values = {"builtin": {}, "global": {}, "nonlocal": {}}
        unbound = []
        references = self._code_global_references(function.__code__)
        for name in self._code_global_names(function.__code__):
            if name in global_values:
                scoped_values["global"][name] = global_values[name]
            elif name in builtin_values:
                scoped_values["builtin"][name] = builtin_values[name]
            else:
                unbound.append(name)

        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__ or ()
        ):
            try:
                scoped_values["nonlocal"][name] = cell.cell_contents
            except ValueError:
                scoped_values["nonlocal"][name] = _EMPTY_CLOSURE_CELL

        dependencies = []
        for scope in ("builtin", "global", "nonlocal"):
            for name, value in sorted(scoped_values[scope].items()):
                if value is _EMPTY_CLOSURE_CELL:
                    canonical = ["empty_closure_cell"]
                else:
                    canonical = self.dependency(
                        value,
                        module,
                        f"{function.__qualname__} {scope} {name}",
                    )
                    if isinstance(value, ModuleType):
                        canonical = [
                            canonical,
                            self._module_attribute_dependencies(
                                value,
                                references.get(name, ()),
                                f"{function.__qualname__} {scope} {name}",
                            ),
                        ]
                dependencies.append([scope, name, canonical])
        return dependencies, unbound

    def function(self, function, dependency_module=None):
        module = dependency_module or function.__module__
        identity = id(function)
        if identity in self._function_cache:
            return self._function_cache[identity]
        if identity in self._active_functions:
            code_digest = hashlib.sha256(
                _canonical_json(self.code(function.__code__)).encode()
            ).hexdigest()
            return [
                "recursive",
                function.__module__,
                function.__qualname__,
                code_digest,
            ]

        self._active_functions.add(identity)
        try:
            dependencies, unbound = self._function_dependencies(function, module)
            canonical = {
                "module": function.__module__,
                "qualname": function.__qualname__,
                "name": function.__name__,
                "code": self.code(function.__code__),
                "defaults": self.dependency(
                    function.__defaults__, module, f"{function.__qualname__} defaults"
                ),
                "kwdefaults": self.dependency(
                    function.__kwdefaults__,
                    module,
                    f"{function.__qualname__} keyword defaults",
                ),
                "annotations": self.dependency(
                    function.__annotations__,
                    module,
                    f"{function.__qualname__} annotations",
                ),
                "attributes": self.dependency(
                    function.__dict__,
                    module,
                    f"{function.__qualname__} attributes",
                ),
                "dependencies": dependencies,
                "unbound": unbound,
            }
        finally:
            self._active_functions.remove(identity)
        fingerprint = {
            "module": function.__module__,
            "qualname": function.__qualname__,
            "sha256": hashlib.sha256(_canonical_json(canonical).encode()).hexdigest(),
        }
        self._function_cache[identity] = fingerprint
        return fingerprint

    def _structural_reference(self, value):
        from ..spmw_target import (
            DeviceScope,
            Memory,
            MemoryRef,
            Move,
            Op,
            Register,
            Target,
            Unit,
        )

        if isinstance(value, Target):
            return ["bound_target", value.name]
        if isinstance(value, DeviceScope):
            return ["device_scope", _unit_path(value._unit)]
        if isinstance(value, Unit):
            return ["unit", _unit_path(value)]
        if isinstance(value, MemoryRef):
            return [
                "memory_ref",
                self._structural_reference(value.memory),
                self.endpoint(value.idx),
            ]
        if isinstance(value, (Memory, Register, Move, Op)):
            name = getattr(value, "name", None)
            if name:
                return [value.__class__.__name__.lower(), handle_path(value)]
        return None

    def endpoint(self, value):
        from ..spmw_target import (
            AnyOf,
            Memory,
            OrOf,
            Register,
            SymExpr,
            Unit,
            UnitId,
        )

        structural = self._structural_reference(value)
        if structural is not None:
            return structural
        if isinstance(value, UnitId):
            return ["unit_id", value.level, _unit_path(value.unit)]
        if isinstance(value, SymExpr):
            return ["symbol", value.op, [self.endpoint(arg) for arg in value.args]]
        if isinstance(value, AnyOf):
            if isinstance(value.candidates, Memory):
                candidates = self.endpoint(value.candidates)
            else:
                candidates = [self.endpoint(item) for item in value.candidates]
                candidates.sort(key=_canonical_json)
            return ["any_of", candidates]
        if isinstance(value, OrOf):
            alternatives = [self.endpoint(item) for item in value.alternatives]
            alternatives.sort(key=_canonical_json)
            return ["or_of", alternatives]
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [self.endpoint(item) for item in value]]
        if isinstance(value, (Memory, Register, Unit)):
            identity = id(value)
            token = self._opaque_target_values.setdefault(
                identity, len(self._opaque_target_values)
            )
            return ["anonymous_handle", type(value).__name__, token]
        try:
            return self.explicit(value, path="target descriptor")
        except TypeError:
            identity = id(value)
            token = self._opaque_target_values.setdefault(
                identity, len(self._opaque_target_values)
            )
            value_type = type(value)
            return [
                "target_value",
                value_type.__module__,
                value_type.__qualname__,
                token,
            ]

    def target(self, target):
        def unit_descriptor(unit):
            axis_labels = list(unit.axes)
            mapping_extents = []
            for position, extent in enumerate(unit.mapping):
                if isinstance(extent, bool) or not isinstance(extent, int):
                    raise TypeError(
                        f"target unit {unit.name!r} mapping extent {position} "
                        "must be an integer"
                    )
                if extent <= 0:
                    raise ValueError(
                        f"target unit {unit.name!r} mapping extent {position} "
                        "must be positive"
                    )
                mapping_extents.append(extent)
            if axis_labels and len(axis_labels) != len(mapping_extents):
                raise ValueError(
                    f"target unit {unit.name!r} has {len(axis_labels)} axis "
                    f"labels but {len(mapping_extents)} mapping extents"
                )
            if any(not isinstance(label, str) or not label for label in axis_labels):
                raise TypeError(
                    f"target unit {unit.name!r} axis labels must be nonempty strings"
                )
            children = [unit_descriptor(child) for child in unit.children]
            children.sort(key=_canonical_json)
            memories = []
            for name, memory in sorted(unit.memories.items()):
                memories.append(
                    {
                        "name": name,
                        "geometry": self.explicit(
                            memory.geometry, path=f"target memory {name} geometry"
                        ),
                        "capacity": memory.capacity,
                    }
                )
            registers = []
            for name, register in sorted(unit.registers.items()):
                registers.append(
                    {
                        "name": name,
                        "lanes": register.lanes,
                        "width": register.width,
                        "slots": register.slots,
                        "axes": self.explicit(
                            register.axes, path=f"target register {name} axes"
                        ),
                        "capacity": register.capacity,
                    }
                )
            moves = []
            for name, move in sorted(unit.moves.items()):
                moves.append(
                    {
                        "name": name,
                        "src": self.endpoint(move.src),
                        "dst": self.endpoint(move.dst),
                        "verb": move.verb.name,
                        "capacity": move.capacity,
                        "emit": (
                            self.function(move.emit, move.emit.__module__)
                            if inspect.isfunction(move.emit)
                            else self.dependency(
                                move.emit, None, f"target move {name} emit"
                            )
                        ),
                    }
                )
            operations = []
            for name, operation in sorted(unit.ops.items()):
                operations.append(
                    {
                        "name": name,
                        "src": self.endpoint(operation.src),
                        "dst": self.endpoint(operation.dst),
                        "accumulates": operation.accumulates,
                        "capacity": operation.capacity,
                        "matchable": operation.matchable,
                        "fn": (
                            self.function(operation.fn, operation.fn.__module__)
                            if inspect.isfunction(operation.fn)
                            else self.dependency(
                                operation.fn, None, f"target operation {name} fn"
                            )
                        ),
                        "emit": (
                            self.function(operation.emit, operation.emit.__module__)
                            if inspect.isfunction(operation.emit)
                            else self.dependency(
                                operation.emit, None, f"target operation {name} emit"
                            )
                        ),
                    }
                )
            return {
                "name": unit.name,
                "mode": unit.mode,
                "axis_labels": self.explicit(
                    axis_labels, path=f"target unit {unit.name} axis labels"
                ),
                "mapping_extents": self.explicit(
                    mapping_extents,
                    path=f"target unit {unit.name} mapping extents",
                ),
                "capacity": unit.capacity,
                "memories": memories,
                "registers": registers,
                "moves": moves,
                "operations": operations,
                "children": children,
            }

        return {"name": target.name, "root": unit_descriptor(target.root)}


def _cost_fingerprint(spec, target, rules):
    canonicalizer = _FingerprintCanonicalizer()
    fingerprint_provider = None
    fingerprint_data = spec.fingerprint_data
    if callable(fingerprint_data):
        fingerprint_provider = canonicalizer.dependency(
            fingerprint_data,
            getattr(fingerprint_data, "__module__", None),
            "fingerprint_data provider",
        )
        fingerprint_data = fingerprint_data(target)
    snapshot = {
        "version": 3,
        "builder": canonicalizer.function(spec.builder),
        "rules": [
            [path, canonicalizer.function(function)]
            for path, function in sorted(rules.items())
        ],
        "materialization_scorer": (
            None
            if spec.materialization_scorer is None
            else canonicalizer.function(spec.materialization_scorer)
        ),
        "target": canonicalizer.target(target),
        "fingerprint_data_provider": fingerprint_provider,
        "fingerprint_data": canonicalizer.explicit(fingerprint_data),
    }
    return hashlib.sha256(_canonical_json(snapshot).encode()).hexdigest()[:16]


class BoundCostSpec:
    """A CostSpec whose rules are bound to one concrete target tree."""

    def __init__(self, spec, target):
        self.spec = spec
        self.target = target
        self.rules = {}
        _BIND_STACK.append(self)
        try:
            spec.builder(target)
        finally:
            popped = _BIND_STACK.pop()
            assert popped is self
        if not self.rules:
            raise ValueError(f"cost spec {spec.name!r} registered no rules")
        self._fingerprint = _cost_fingerprint(spec, target, self.rules)
        self._materialization_scorer = spec.materialization_scorer
        self.rules = MappingProxyType(dict(self.rules))

    @property
    def fingerprint(self):
        return self._fingerprint

    def add_rule(self, handle, function):
        path = handle_path(handle)
        if path in self.rules:
            raise ValueError(f"duplicate cost rule for {path!r}")
        self.rules[path] = function

    def _require_current_fingerprint(self):
        current = _cost_fingerprint(self.spec, self.target, self.rules)
        if current != self._fingerprint:
            raise RuntimeError(
                f"cost spec {self.spec.name!r} changed after it was bound"
            )

    def emit(self, graph, event, dependencies=()):
        path = handle_path(event.primitive)
        try:
            function = self.rules[path]
        except KeyError as exc:
            raise KeyError(
                f"cost spec {self.spec.name!r} implements no rule for {path!r}"
            ) from exc
        context = CostContext(graph, event, dependencies)
        function(event, context)
        if not context.terminals:
            raise ValueError(f"cost rule for {path!r} emitted no steps")
        return context.terminals

    def evaluate(self, graph):
        self._require_current_fingerprint()
        graph.metadata["cost_fingerprint"] = self.fingerprint
        return Evaluator().evaluate(graph)

    def score_materialization(self, materialization):
        """Score one exact backend materialization through this cost spec."""

        self._require_current_fingerprint()
        scorer = self._materialization_scorer
        if scorer is None:
            raise TypeError(
                f"cost spec {self.spec.name!r} has no materialization scorer"
            )
        result = scorer(self, materialization)
        self._require_current_fingerprint()
        return result


class CostSpec:
    """Reusable executable cost program."""

    def __init__(
        self,
        builder,
        *,
        target=None,
        name=None,
        fingerprint_data=None,
        materialization_scorer=None,
    ):
        self.builder = builder
        self.target_name = target
        self.name = name or builder.__name__
        self.fingerprint_data = fingerprint_data
        if materialization_scorer is not None and not inspect.isfunction(
            materialization_scorer
        ):
            raise TypeError("materialization_scorer must be a Python function or None")
        self.materialization_scorer = materialization_scorer

    def bind(self, target):
        target_name = getattr(target, "name", None)
        if self.target_name is not None and self.target_name != target_name:
            raise ValueError(
                f"cost spec {self.name!r} targets {self.target_name!r}, "
                f"not {target_name!r}"
            )
        return BoundCostSpec(self, target)

    def __repr__(self):
        return f"CostSpec({self.name!r}, target={self.target_name!r})"


def cost(
    function=None,
    *,
    target=None,
    name=None,
    fingerprint_data=None,
    materialization_scorer=None,
):
    """Decorate a cost builder, optionally with canonical provenance data."""

    def decorate(builder):
        return CostSpec(
            builder,
            target=target,
            name=name,
            fingerprint_data=fingerprint_data,
            materialization_scorer=materialization_scorer,
        )

    if function is None:
        return decorate
    return decorate(function)


def rule(handle):
    """Register a handle-specific rule while a CostSpec is binding."""
    if not _BIND_STACK:
        raise RuntimeError("@allo.rule must be declared inside an @allo.cost builder")

    def decorate(function):
        _BIND_STACK[-1].add_rule(handle, function)
        return function

    return decorate
