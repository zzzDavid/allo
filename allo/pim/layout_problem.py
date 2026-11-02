# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-neutral ownership contracts for minimum-movement layout synthesis.

The paper's layout search uses a nested flag of local iteration subspaces

``I = T_0 >= T_1 >= ... >= T_h``.

This module describes the target-fixed dimensions of that flag without
embedding backend names or movement costs.  Mapped unit axes and register
axes may be declared as owner coordinates.  Storage geometry is retained as
structural capability data, but never becomes an owner coordinate implicitly.
All spaces use :class:`allo.pim.gf2.GF2Subspace` directly.

:func:`owner_hierarchy_from_target` reads the owner hierarchy from a Tenon
``Target`` unit tree and register lane axes, so the paper's "any hierarchy
of compute axes" comes from the target description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .gf2 import GF2Subspace


class OwnerAxisOrigin(Enum):
    """Structural source of an owner axis.

    The source is capability data.  It does not select backend-specific
    behavior in the algebra or search.
    """

    MAPPED_UNIT = "mapped-unit"
    REGISTER_AXIS = "register-axis"


class LayoutValidationCode(Enum):
    """Stable, rename-insensitive validation reason codes."""

    INVALID_INTEGER = "invalid-integer"
    INVALID_ORDINAL = "invalid-ordinal"
    INVALID_DIAGNOSTIC_NAME = "invalid-diagnostic-name"
    INVALID_AXIS_ORIGIN = "invalid-axis-origin"
    INVALID_EXTENT = "invalid-extent"
    NON_POWER_OF_TWO_OWNER_EXTENT = "non-power-of-two-owner-extent"
    INVALID_ACTIVE_SUBGROUP = "invalid-active-subgroup"
    EMPTY_OWNER_LEVEL = "empty-owner-level"
    NONCONTIGUOUS_ORDINALS = "noncontiguous-ordinals"
    INVALID_RECORD_TYPE = "invalid-record-type"
    INVALID_LEVEL_INDEX = "invalid-level-index"
    INVALID_FLAG_LENGTH = "invalid-flag-length"
    AMBIENT_DIMENSION_MISMATCH = "ambient-dimension-mismatch"
    FLAG_ROOT_NOT_FULL = "flag-root-not-full"
    FLAG_DIMENSION_MISMATCH = "flag-dimension-mismatch"
    FLAG_NOT_NESTED = "flag-not-nested"
    INVALID_FORCED_LOCAL_LENGTH = "invalid-forced-local-length"
    FORCED_LOCAL_NOT_NESTED = "forced-local-not-nested"
    FORCED_LOCAL_OUTSIDE_FLAG = "forced-local-outside-flag"


class LayoutProblemValidationError(ValueError):
    """Typed failure for malformed owner geometry or local-space flags.

    ``location`` contains structural ordinals only.  Diagnostic names are
    deliberately excluded so alpha-renaming a capability description cannot
    change a failure's machine-readable identity.
    """

    def __init__(
        self,
        code: LayoutValidationCode,
        *,
        location: tuple[Any, ...] = (),
        details: tuple[Any, ...] = (),
    ) -> None:
        if not isinstance(code, LayoutValidationCode):
            raise TypeError("code must be a LayoutValidationCode")
        self.code = code
        self.location = tuple(location)
        self.details = tuple(details)
        location_text = "/".join(str(item) for item in self.location) or "root"
        details_text = "" if not self.details else f"; details={self.details!r}"
        super().__init__(f"{code.value} at {location_text}{details_text}")

    def diagnostic(self) -> dict[str, object]:
        """Return a stable JSON-compatible failure description."""

        return {
            "code": self.code.value,
            "location": list(self.location),
            "details": list(self.details),
        }


def _fail(
    code: LayoutValidationCode,
    *,
    location: tuple[Any, ...] = (),
    details: tuple[Any, ...] = (),
) -> None:
    raise LayoutProblemValidationError(code, location=location, details=details)


def _integer(
    value: object,
    *,
    location: tuple[Any, ...],
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(
            LayoutValidationCode.INVALID_INTEGER,
            location=location,
            details=(type(value).__name__,),
        )
    if value < minimum:
        _fail(
            (
                LayoutValidationCode.INVALID_EXTENT
                if minimum > 0
                else LayoutValidationCode.INVALID_ORDINAL
            ),
            location=location,
            details=(value, minimum),
        )
    return value


def _diagnostic_name(value: object, *, location: tuple[Any, ...]) -> str:
    if not isinstance(value, str):
        _fail(
            LayoutValidationCode.INVALID_DIAGNOSTIC_NAME,
            location=location,
            details=(type(value).__name__,),
        )
    return value


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _space_manifest(space: GF2Subspace) -> dict[str, object]:
    return {
        "dimension": space.dimension,
        "basis": list(space.basis),
    }


@dataclass(frozen=True, slots=True)
class OwnerAxis:
    """One power-of-two owner coordinate.

    ``physical_extent`` records the declared target extent.  A non-power-of-two
    physical extent is rejected unless an active subgroup explicitly selects a
    power-of-two subset.  ``active_subgroup_members`` maps logical packed owner
    values, in bit-pattern order, to physical ordinals.  For compatibility,
    supplying only ``active_subgroup_extent`` selects the canonical prefix
    ``range(active_subgroup_extent)``.  This is useful for an intentionally
    restricted execution mode such as 16 of UPMEM's 24 tasklets; the exact
    subset declaration must come from capability data, never from backend-name
    logic.
    """

    ordinal: int
    physical_extent: int
    origin: OwnerAxisOrigin
    active_subgroup_extent: int | None = None
    diagnostic_name: str = field(default="", compare=False, hash=False)
    active_subgroup_members: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        ordinal = _integer(self.ordinal, location=("owner-axis", "ordinal"), minimum=0)
        physical_extent = _integer(
            self.physical_extent,
            location=("owner-axis", ordinal, "physical-extent"),
            minimum=1,
        )
        if not isinstance(self.origin, OwnerAxisOrigin):
            _fail(
                LayoutValidationCode.INVALID_AXIS_ORIGIN,
                location=("owner-axis", ordinal, "origin"),
                details=(type(self.origin).__name__,),
            )
        diagnostic_name = _diagnostic_name(
            self.diagnostic_name,
            location=("owner-axis", ordinal, "diagnostic-name"),
        )

        subgroup = self.active_subgroup_extent
        if subgroup is not None:
            subgroup = _integer(
                subgroup,
                location=("owner-axis", ordinal, "active-subgroup-extent"),
                minimum=1,
            )
            if subgroup > physical_extent or not _is_power_of_two(subgroup):
                _fail(
                    LayoutValidationCode.INVALID_ACTIVE_SUBGROUP,
                    location=("owner-axis", ordinal),
                    details=(physical_extent, subgroup),
                )

        members_value = self.active_subgroup_members
        members = None
        if members_value is not None:
            if isinstance(members_value, (str, bytes)):
                _fail(
                    LayoutValidationCode.INVALID_ACTIVE_SUBGROUP,
                    location=("owner-axis", ordinal, "active-subgroup-members"),
                    details=(type(members_value).__name__,),
                )
            try:
                members = tuple(members_value)
            except TypeError:
                _fail(
                    LayoutValidationCode.INVALID_ACTIVE_SUBGROUP,
                    location=("owner-axis", ordinal, "active-subgroup-members"),
                    details=(type(members_value).__name__,),
                )
            members = tuple(
                _integer(
                    member,
                    location=(
                        "owner-axis",
                        ordinal,
                        "active-subgroup-member",
                        position,
                    ),
                    minimum=0,
                )
                for position, member in enumerate(members)
            )
            member_count = len(members)
            if (
                member_count < 2
                or not _is_power_of_two(member_count)
                or member_count > physical_extent
                or len(set(members)) != member_count
                or any(member >= physical_extent for member in members)
            ):
                _fail(
                    LayoutValidationCode.INVALID_ACTIVE_SUBGROUP,
                    location=("owner-axis", ordinal, "active-subgroup-members"),
                    details=(physical_extent, members),
                )
            if subgroup is not None and subgroup != member_count:
                _fail(
                    LayoutValidationCode.INVALID_ACTIVE_SUBGROUP,
                    location=("owner-axis", ordinal, "active-subgroup-members"),
                    details=(physical_extent, subgroup, member_count),
                )
            subgroup = member_count
        elif subgroup is not None:
            if subgroup == physical_extent:
                # The full prefix is exactly the implicit identity mapping.
                # Keep its canonical representation compact.
                subgroup = None
            else:
                members = tuple(range(subgroup))

        if subgroup == physical_extent and members == tuple(range(physical_extent)):
            subgroup = None
            members = None

        if subgroup is None and not _is_power_of_two(physical_extent):
            _fail(
                LayoutValidationCode.NON_POWER_OF_TWO_OWNER_EXTENT,
                location=("owner-axis", ordinal),
                details=(physical_extent,),
            )

        mapped_extent = physical_extent if subgroup is None else subgroup
        if mapped_extent == 1:
            _fail(
                LayoutValidationCode.INVALID_EXTENT,
                location=("owner-axis", ordinal, "mapped-extent"),
                details=(mapped_extent, 2),
            )

        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "physical_extent", physical_extent)
        object.__setattr__(self, "active_subgroup_extent", subgroup)
        object.__setattr__(self, "active_subgroup_members", members)
        object.__setattr__(self, "diagnostic_name", diagnostic_name)

    @property
    def mapped_extent(self) -> int:
        """Extent represented in the homogeneous GF(2) carrier."""

        return (
            self.physical_extent
            if self.active_subgroup_extent is None
            else self.active_subgroup_extent
        )

    @property
    def bit_count(self) -> int:
        """Number of carrier bits occupied by this owner coordinate."""

        return self.mapped_extent.bit_length() - 1

    def manifest(self) -> dict[str, object]:
        """Return the structural, rename-insensitive axis manifest."""

        manifest = {
            "ordinal": self.ordinal,
            "origin": self.origin.value,
            "physical_extent": self.physical_extent,
            "active_subgroup_extent": self.active_subgroup_extent,
            "mapped_extent": self.mapped_extent,
            "bit_count": self.bit_count,
        }
        if self.active_subgroup_members is not None:
            manifest["active_subgroup_members"] = list(self.active_subgroup_members)
        return manifest


@dataclass(frozen=True, slots=True)
class StorageAxis:
    """One memory/storage geometry axis that is not implicitly an owner."""

    ordinal: int
    extent: int
    diagnostic_name: str = field(default="", compare=False, hash=False)

    def __post_init__(self) -> None:
        ordinal = _integer(
            self.ordinal, location=("storage-axis", "ordinal"), minimum=0
        )
        extent = _integer(
            self.extent,
            location=("storage-axis", ordinal, "extent"),
            minimum=1,
        )
        diagnostic_name = _diagnostic_name(
            self.diagnostic_name,
            location=("storage-axis", ordinal, "diagnostic-name"),
        )
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "diagnostic_name", diagnostic_name)

    def manifest(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "extent": self.extent,
        }


@dataclass(frozen=True, slots=True)
class LocalAxis:
    """One explicitly local coordinate, expressed by its carrier bit count."""

    ordinal: int
    bit_count: int
    diagnostic_name: str = field(default="", compare=False, hash=False)

    def __post_init__(self) -> None:
        ordinal = _integer(self.ordinal, location=("local-axis", "ordinal"), minimum=0)
        bit_count = _integer(
            self.bit_count,
            location=("local-axis", ordinal, "bit-count"),
            minimum=1,
        )
        diagnostic_name = _diagnostic_name(
            self.diagnostic_name,
            location=("local-axis", ordinal, "diagnostic-name"),
        )
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "bit_count", bit_count)
        object.__setattr__(self, "diagnostic_name", diagnostic_name)

    @property
    def extent(self) -> int:
        return 1 << self.bit_count

    def manifest(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "bit_count": self.bit_count,
            "extent": self.extent,
        }


@dataclass(frozen=True, slots=True)
class OwnerLevel:
    """A labeled nesting level containing one or more structural owner axes."""

    ordinal: int
    axes: tuple[OwnerAxis, ...]
    diagnostic_name: str = field(default="", compare=False, hash=False)

    def __post_init__(self) -> None:
        ordinal = _integer(self.ordinal, location=("owner-level", "ordinal"), minimum=0)
        try:
            axes = tuple(self.axes)
        except TypeError:
            _fail(
                LayoutValidationCode.INVALID_RECORD_TYPE,
                location=("owner-level", ordinal, "axes"),
                details=(type(self.axes).__name__,),
            )
        if not axes:
            _fail(
                LayoutValidationCode.EMPTY_OWNER_LEVEL,
                location=("owner-level", ordinal),
            )
        for position, axis in enumerate(axes):
            if not isinstance(axis, OwnerAxis):
                _fail(
                    LayoutValidationCode.INVALID_RECORD_TYPE,
                    location=("owner-level", ordinal, "axis", position),
                    details=(type(axis).__name__,),
                )
        diagnostic_name = _diagnostic_name(
            self.diagnostic_name,
            location=("owner-level", ordinal, "diagnostic-name"),
        )
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "diagnostic_name", diagnostic_name)

    @property
    def bit_count(self) -> int:
        return sum(axis.bit_count for axis in self.axes)

    @property
    def mapped_extent(self) -> int:
        return 1 << self.bit_count

    def manifest(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "axes": [axis.manifest() for axis in self.axes],
            "mapped_extent": self.mapped_extent,
            "bit_count": self.bit_count,
        }


@dataclass(frozen=True, slots=True)
class OwnerHierarchy:
    """Ordered owner levels plus explicit local and storage geometry.

    Work-coordinate bits are ordered as all axes of level 0, then level 1,
    and so on, followed by local axes.  Storage axes do not consume carrier
    bits.  This order fixes the dimensions of every paper local space ``T_s``.
    """

    levels: tuple[OwnerLevel, ...]
    local_axes: tuple[LocalAxis, ...] = ()
    storage_axes: tuple[StorageAxis, ...] = ()

    def __post_init__(self) -> None:
        levels = self._records(self.levels, OwnerLevel, "owner-level")
        local_axes = self._records(self.local_axes, LocalAxis, "local-axis")
        storage_axes = self._records(self.storage_axes, StorageAxis, "storage-axis")

        self._require_ordinals(tuple(level.ordinal for level in levels), "owner-level")
        flattened_owner_axes = tuple(axis for level in levels for axis in level.axes)
        self._require_ordinals(
            tuple(axis.ordinal for axis in flattened_owner_axes), "owner-axis"
        )
        self._require_ordinals(tuple(axis.ordinal for axis in local_axes), "local-axis")
        self._require_ordinals(
            tuple(axis.ordinal for axis in storage_axes), "storage-axis"
        )

        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "local_axes", local_axes)
        object.__setattr__(self, "storage_axes", storage_axes)

    @staticmethod
    def _records(
        values: object,
        record_type: type,
        kind: str,
    ) -> tuple[Any, ...]:
        try:
            records = tuple(values)
        except TypeError:
            _fail(
                LayoutValidationCode.INVALID_RECORD_TYPE,
                location=(kind, "collection"),
                details=(type(values).__name__,),
            )
        for position, record in enumerate(records):
            if not isinstance(record, record_type):
                _fail(
                    LayoutValidationCode.INVALID_RECORD_TYPE,
                    location=(kind, position),
                    details=(type(record).__name__,),
                )
        return records

    @staticmethod
    def _require_ordinals(ordinals: tuple[int, ...], kind: str) -> None:
        expected = tuple(range(len(ordinals)))
        if ordinals != expected:
            _fail(
                LayoutValidationCode.NONCONTIGUOUS_ORDINALS,
                location=(kind,),
                details=(ordinals, expected),
            )

    @property
    def owner_bit_count(self) -> int:
        return sum(level.bit_count for level in self.levels)

    @property
    def local_bit_count(self) -> int:
        return sum(axis.bit_count for axis in self.local_axes)

    @property
    def carrier_bit_count(self) -> int:
        return self.owner_bit_count + self.local_bit_count

    @property
    def local_space_dimensions(self) -> tuple[int, ...]:
        """Target-fixed dimensions ``dim(T_0), ..., dim(T_h)``."""

        remaining = self.carrier_bit_count
        dimensions = [remaining]
        for level in self.levels:
            remaining -= level.bit_count
            dimensions.append(remaining)
        return tuple(dimensions)

    @property
    def owner_bit_ranges(self) -> tuple[tuple[int, int], ...]:
        """Half-open work-coordinate bit ranges for owner levels."""

        ranges = []
        offset = 0
        for level in self.levels:
            end = offset + level.bit_count
            ranges.append((offset, end))
            offset = end
        return tuple(ranges)

    def _checked_level(self, level_ordinal: object) -> int:
        if isinstance(level_ordinal, bool) or not isinstance(level_ordinal, int):
            _fail(
                LayoutValidationCode.INVALID_LEVEL_INDEX,
                location=("owner-level",),
                details=(type(level_ordinal).__name__, len(self.levels)),
            )
        if level_ordinal < 0 or level_ordinal >= len(self.levels):
            _fail(
                LayoutValidationCode.INVALID_LEVEL_INDEX,
                location=("owner-level",),
                details=(level_ordinal, len(self.levels)),
            )
        return level_ordinal

    def owner_coordinate_space_through(self, level_ordinal: int) -> GF2Subspace:
        """Row space of owner-coordinate projection through one level."""

        level = self._checked_level(level_ordinal)
        end = self.owner_bit_ranges[level][1]
        return GF2Subspace(
            self.carrier_bit_count, tuple(1 << bit for bit in range(end))
        )

    def local_coordinate_space_after(self, level_ordinal: int) -> GF2Subspace:
        """Kernel of owner-coordinate projection through one level."""

        level = self._checked_level(level_ordinal)
        start = self.owner_bit_ranges[level][1]
        return GF2Subspace(
            self.carrier_bit_count,
            tuple(1 << bit for bit in range(start, self.carrier_bit_count)),
        )

    def coordinate_flag(
        self,
        *,
        forced_local: tuple[GF2Subspace, ...] = (),
    ) -> LocalSpaceFlag:
        """Return the canonical work-coordinate flag for this hierarchy."""

        spaces = (GF2Subspace.full(self.carrier_bit_count),) + tuple(
            self.local_coordinate_space_after(level)
            for level in range(len(self.levels))
        )
        return LocalSpaceFlag(self, spaces, forced_local)

    def manifest(self) -> dict[str, object]:
        """Return the exact structural manifest, excluding diagnostic names."""

        return {
            "schema": "owner-hierarchy-v1",
            "levels": [level.manifest() for level in self.levels],
            "local_axes": [axis.manifest() for axis in self.local_axes],
            "storage_axes": [axis.manifest() for axis in self.storage_axes],
            "owner_bit_count": self.owner_bit_count,
            "local_bit_count": self.local_bit_count,
            "carrier_bit_count": self.carrier_bit_count,
            "owner_bit_ranges": [list(item) for item in self.owner_bit_ranges],
            "local_space_dimensions": list(self.local_space_dimensions),
        }


@dataclass(frozen=True, slots=True)
class LocalSpaceFlag:
    """A validated nested flag ``T_0 >= ... >= T_h`` in iteration space."""

    hierarchy: OwnerHierarchy
    spaces: tuple[GF2Subspace, ...]
    forced_local: tuple[GF2Subspace, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.hierarchy, OwnerHierarchy):
            _fail(
                LayoutValidationCode.INVALID_RECORD_TYPE,
                location=("local-space-flag", "hierarchy"),
                details=(type(self.hierarchy).__name__,),
            )
        spaces = self._spaces(self.spaces, "space")
        expected_count = len(self.hierarchy.levels) + 1
        if len(spaces) != expected_count:
            _fail(
                LayoutValidationCode.INVALID_FLAG_LENGTH,
                location=("local-space-flag",),
                details=(len(spaces), expected_count),
            )

        ambient = self.hierarchy.carrier_bit_count
        for ordinal, space in enumerate(spaces):
            if space.dimension != ambient:
                _fail(
                    LayoutValidationCode.AMBIENT_DIMENSION_MISMATCH,
                    location=("local-space-flag", "space", ordinal),
                    details=(space.dimension, ambient),
                )

        full = GF2Subspace.full(ambient)
        if spaces[0] != full:
            _fail(
                LayoutValidationCode.FLAG_ROOT_NOT_FULL,
                location=("local-space-flag", "space", 0),
                details=(spaces[0].rank, ambient),
            )

        expected_dimensions = self.hierarchy.local_space_dimensions
        for ordinal, (space, expected) in enumerate(zip(spaces, expected_dimensions)):
            if space.rank != expected:
                _fail(
                    LayoutValidationCode.FLAG_DIMENSION_MISMATCH,
                    location=("local-space-flag", "space", ordinal),
                    details=(space.rank, expected),
                )
            if ordinal and not space.is_subspace_of(spaces[ordinal - 1]):
                _fail(
                    LayoutValidationCode.FLAG_NOT_NESTED,
                    location=("local-space-flag", "space", ordinal),
                )

        forced_local = self._spaces(self.forced_local, "forced-local")
        owner_level_count = len(self.hierarchy.levels)
        if not forced_local:
            forced_local = tuple(
                GF2Subspace.zero(ambient) for _ in range(owner_level_count)
            )
        elif len(forced_local) != owner_level_count:
            _fail(
                LayoutValidationCode.INVALID_FORCED_LOCAL_LENGTH,
                location=("local-space-flag", "forced-local"),
                details=(len(forced_local), owner_level_count),
            )

        for ordinal, forced in enumerate(forced_local):
            if forced.dimension != ambient:
                _fail(
                    LayoutValidationCode.AMBIENT_DIMENSION_MISMATCH,
                    location=("local-space-flag", "forced-local", ordinal),
                    details=(forced.dimension, ambient),
                )
            if ordinal and not forced.is_subspace_of(forced_local[ordinal - 1]):
                _fail(
                    LayoutValidationCode.FORCED_LOCAL_NOT_NESTED,
                    location=("local-space-flag", "forced-local", ordinal),
                )
            if not forced.is_subspace_of(spaces[ordinal + 1]):
                _fail(
                    LayoutValidationCode.FORCED_LOCAL_OUTSIDE_FLAG,
                    location=("local-space-flag", "forced-local", ordinal),
                    details=(forced.rank, spaces[ordinal + 1].rank),
                )

        object.__setattr__(self, "spaces", spaces)
        object.__setattr__(self, "forced_local", forced_local)

    @staticmethod
    def _spaces(values: object, kind: str) -> tuple[GF2Subspace, ...]:
        try:
            spaces = tuple(values)
        except TypeError:
            _fail(
                LayoutValidationCode.INVALID_RECORD_TYPE,
                location=("local-space-flag", kind),
                details=(type(values).__name__,),
            )
        for ordinal, space in enumerate(spaces):
            if not isinstance(space, GF2Subspace):
                _fail(
                    LayoutValidationCode.INVALID_RECORD_TYPE,
                    location=("local-space-flag", kind, ordinal),
                    details=(type(space).__name__,),
                )
        return spaces

    def manifest(self) -> dict[str, object]:
        """Return a structural manifest suitable for search deduplication."""

        return {
            "schema": "local-space-flag-v1",
            "hierarchy": self.hierarchy.manifest(),
            "spaces": [_space_manifest(space) for space in self.spaces],
            "forced_local": [_space_manifest(space) for space in self.forced_local],
        }


class TargetHierarchyError(ValueError):
    """A target unit tree cannot be read as a power-of-two owner hierarchy."""


def _register_axis(target: Any, reference: object) -> tuple[str, str, int]:
    """Resolve ``"reg.axis"``, ``"reg"``, ``Register`` or ``(reg, axis)``."""

    from ..spmw_target import Register  # pylint: disable=import-outside-toplevel

    axis_name = None
    register = reference
    if isinstance(reference, tuple) and len(reference) == 2:
        register, axis_name = reference
    elif isinstance(reference, str) and "." in reference:
        register, axis_name = reference.split(".", 1)
    if isinstance(register, str):
        try:
            register = getattr(target, register)
        except AttributeError as error:
            raise TargetHierarchyError(
                f"target {target.name!r} has no unit or register named " f"{register!r}"
            ) from error
    if not isinstance(register, Register):
        raise TargetHierarchyError(
            f"owner level {reference!r} is neither a unit nor a register"
        )
    reg_name = register.name or "<anonymous register>"
    if not register.axes:
        raise TargetHierarchyError(
            f"register {reg_name!r} declares no lane axes; declare "
            "allo.reg(..., axes={'lane': N}) to use it as an owner level"
        )
    if axis_name is None:
        if len(register.axes) != 1:
            raise TargetHierarchyError(
                f"register {reg_name!r} has axes {sorted(register.axes)}; "
                "name one as 'register.axis'"
            )
        axis_name = next(iter(register.axes))
    if axis_name not in register.axes:
        raise TargetHierarchyError(
            f"register {reg_name!r} has no axis {axis_name!r} "
            f"(axes: {sorted(register.axes)})"
        )
    return reg_name, axis_name, int(register.axes[axis_name])


def owner_hierarchy_from_target(
    target: Any,
    level_names: tuple[object, ...] | list[object],
    *,
    local_bits: int = 0,
    active_extents: dict[str, int] | None = None,
) -> OwnerHierarchy:
    """Build the compute hierarchy ``U_1 x ... x U_h`` from a Tenon target.

    ``level_names`` lists owner levels from outermost to innermost.  Each
    entry names either a unit of the target's unit tree (every axis of its
    ``mapping`` becomes one owner axis of that level, origin
    ``MAPPED_UNIT``) or a register lane axis (``"vr0.lane"``, ``"vr0"`` when
    the register declares exactly one axis, a ``Register`` or a
    ``(register, axis)`` pair; origin ``REGISTER_AXIS``).  Units take
    precedence over registers of the same name.

    Every owner extent must be a power of two, because a layout is an
    invertible GF(2) map.  A non-power-of-two extent is rejected unless
    ``active_extents`` names a power-of-two active subgroup for that axis,
    keyed by ``"<unit or register>.<axis>"`` (for example 16 of UPMEM's 24
    tasklets).  ``local_bits`` appends one local (storage) axis of that
    many bits, so the carrier has ``owner bits + local_bits`` bits.
    """

    from ..spmw_target import Unit  # pylint: disable=import-outside-toplevel

    if isinstance(level_names, (str, bytes)):
        raise TypeError("level_names must be a sequence of level references")
    references = tuple(level_names)
    if not references:
        raise TargetHierarchyError("at least one owner level is required")
    if isinstance(local_bits, bool) or not isinstance(local_bits, int):
        raise TypeError("local_bits must be an integer")
    if local_bits < 0:
        raise TargetHierarchyError("local_bits must be non-negative")
    subgroups = dict(active_extents or {})
    used_subgroups = set()

    levels = []
    axis_ordinal = 0
    for level_ordinal, reference in enumerate(references):
        unit = reference if isinstance(reference, Unit) else None
        if unit is None and isinstance(reference, str):
            try:
                unit = target.unit(reference)
            except KeyError:
                unit = None
        if unit is not None:
            if unit.axes:
                axes_items = tuple(unit.axes.items())
            else:
                axes_items = tuple(
                    (str(position), extent)
                    for position, extent in enumerate(unit.mapping)
                )
            axes_items = tuple(
                (name, int(extent)) for name, extent in axes_items if int(extent) != 1
            )
            if not axes_items:
                raise TargetHierarchyError(
                    f"unit {unit.name!r} has no compute axis (mapping "
                    f"{unit.mapping!r}); it cannot be an owner level"
                )
            owner_name = unit.name
            origin = OwnerAxisOrigin.MAPPED_UNIT
        else:
            owner_name, axis_name, extent = _register_axis(target, reference)
            axes_items = ((axis_name, extent),)
            origin = OwnerAxisOrigin.REGISTER_AXIS

        owner_axes = []
        for axis_name, extent in axes_items:
            key = f"{owner_name}.{axis_name}"
            subgroup = subgroups.get(key)
            if subgroup is not None:
                used_subgroups.add(key)
            elif not _is_power_of_two(extent):
                raise TargetHierarchyError(
                    f"owner axis {key!r} has extent {extent}, which is not a "
                    "power of two; minimum-movement layouts are GF(2) maps "
                    "and need power-of-two owner extents (declare an active "
                    f"subgroup with active_extents={{{key!r}: <power of two>}})"
                )
            owner_axes.append(
                OwnerAxis(
                    axis_ordinal,
                    extent,
                    origin,
                    active_subgroup_extent=subgroup,
                    diagnostic_name=key,
                )
            )
            axis_ordinal += 1
        levels.append(OwnerLevel(level_ordinal, tuple(owner_axes), owner_name))

    unused = sorted(set(subgroups) - used_subgroups)
    if unused:
        raise TargetHierarchyError(
            f"active_extents names axes that are not owner axes: {unused}"
        )
    local_axes = (LocalAxis(0, local_bits, "local"),) if local_bits else ()
    return OwnerHierarchy(tuple(levels), local_axes)


__all__ = [
    "LayoutProblemValidationError",
    "LayoutValidationCode",
    "LocalAxis",
    "LocalSpaceFlag",
    "OwnerAxis",
    "OwnerAxisOrigin",
    "OwnerHierarchy",
    "OwnerLevel",
    "StorageAxis",
    "TargetHierarchyError",
    "owner_hierarchy_from_target",
]
