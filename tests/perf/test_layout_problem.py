# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-hierarchy contract tests.

The first two sections are ported from the snapshot (df35093)
test_layout_problem.py and test_layout_problem_adversarial.py; the last
section covers owner_hierarchy_from_target on the package's Tenon targets.
"""

import json
from dataclasses import FrozenInstanceError, dataclass

import pytest

import allo
from allo.pim.gf2 import GF2Subspace
from allo.pim.layout_problem import (
    LayoutProblemValidationError,
    LayoutValidationCode,
    LocalAxis,
    LocalSpaceFlag,
    OwnerAxis,
    OwnerAxisOrigin,
    OwnerHierarchy,
    OwnerLevel,
    StorageAxis,
    TargetHierarchyError,
    owner_hierarchy_from_target,
)
from allo.pim.targets import (
    build_apu_v1_target,
    build_samsung_target,
    build_upmem_target,
)


@dataclass(frozen=True)
class _CapabilityFixture:
    owners: tuple[tuple[str, int, int | None, str], ...]
    level_widths: tuple[int, ...]
    register_owners: tuple[tuple[str, int, int | None, str], ...]
    storage: tuple[int, ...]
    local_bits: tuple[int, ...]


# These are deliberately immutable data, not production backend-name switches.
# They adapt facts declared in pim/targets.py to target-neutral records.
CAPABILITY_FIXTURES = {
    "samsung": _CapabilityFixture(
        owners=(
            ("unit", 16, None, "pseudo_channel"),
            ("unit", 8, None, "pim"),
        ),
        level_widths=(1, 1),
        register_owners=(),
        storage=(16, 4, 16384, 128),
        local_bits=(1,),
    ),
    "aim": _CapabilityFixture(
        owners=(
            ("unit", 32, None, "channel"),
            ("unit", 4, None, "bank_group"),
            ("unit", 4, None, "bank"),
        ),
        level_widths=(1, 1, 1),
        register_owners=(),
        storage=(16, 16384, 1024),
        local_bits=(1,),
    ),
    "apu_v1": _CapabilityFixture(
        owners=(("unit", 4, None, "apuc"),),
        level_widths=(1, 1),
        register_owners=(("register", 32768, None, "lane"),),
        storage=((1 << 20), (64 << 10)),
        local_bits=(1,),
    ),
    "apu_v2": _CapabilityFixture(
        owners=(("unit", 16, None, "group"),),
        level_widths=(1,),
        register_owners=(),
        # L1 columns and MMB sets are storage geometry, not extra mapped PEs.
        storage=(65536, 4, 3072),
        local_bits=(2,),
    ),
    "upmem_power_of_two_subgroup": _CapabilityFixture(
        owners=(
            ("unit", 64, None, "dpu"),
            ("unit", 24, 16, "tasklet"),
        ),
        level_widths=(1, 1),
        register_owners=(),
        storage=(67108864, 65536, 24576),
        local_bits=(1,),
    ),
}


def _adapt_capability(data, *, rename_prefix=""):
    declarations = data.owners + data.register_owners
    owner_axes = []
    for ordinal, (kind, extent, subgroup, name) in enumerate(declarations):
        origin = (
            OwnerAxisOrigin.MAPPED_UNIT
            if kind == "unit"
            else OwnerAxisOrigin.REGISTER_AXIS
        )
        owner_axes.append(
            OwnerAxis(
                ordinal,
                extent,
                origin,
                subgroup,
                diagnostic_name=f"{rename_prefix}{name}",
            )
        )

    levels = []
    axis_offset = 0
    for level_ordinal, width in enumerate(data.level_widths):
        level_axes = tuple(owner_axes[axis_offset : axis_offset + width])
        axis_offset += width
        levels.append(
            OwnerLevel(
                level_ordinal,
                level_axes,
                diagnostic_name=f"{rename_prefix}level{level_ordinal}",
            )
        )
    assert axis_offset == len(owner_axes)

    local_axes = tuple(
        LocalAxis(
            ordinal,
            bits,
            diagnostic_name=f"{rename_prefix}local{ordinal}",
        )
        for ordinal, bits in enumerate(data.local_bits)
    )
    storage_axes = tuple(
        StorageAxis(
            ordinal,
            extent,
            diagnostic_name=f"{rename_prefix}storage{ordinal}",
        )
        for ordinal, extent in enumerate(data.storage)
    )
    return OwnerHierarchy(tuple(levels), local_axes, storage_axes)


@pytest.mark.parametrize(
    ("fixture", "origins", "level_bits", "local_dimensions"),
    (
        (
            "samsung",
            (
                OwnerAxisOrigin.MAPPED_UNIT,
                OwnerAxisOrigin.MAPPED_UNIT,
            ),
            (4, 3),
            (8, 4, 1),
        ),
        (
            "aim",
            (
                OwnerAxisOrigin.MAPPED_UNIT,
                OwnerAxisOrigin.MAPPED_UNIT,
                OwnerAxisOrigin.MAPPED_UNIT,
            ),
            (5, 2, 2),
            (10, 5, 3, 1),
        ),
        (
            "apu_v1",
            (
                OwnerAxisOrigin.MAPPED_UNIT,
                OwnerAxisOrigin.REGISTER_AXIS,
            ),
            (2, 15),
            (18, 16, 1),
        ),
        (
            "apu_v2",
            (OwnerAxisOrigin.MAPPED_UNIT,),
            (4,),
            (6, 2),
        ),
        (
            "upmem_power_of_two_subgroup",
            (
                OwnerAxisOrigin.MAPPED_UNIT,
                OwnerAxisOrigin.MAPPED_UNIT,
            ),
            (6, 4),
            (11, 5, 1),
        ),
    ),
)
def test_target_data_adapts_to_exact_owner_bits_and_nested_dimensions(
    fixture, origins, level_bits, local_dimensions
):
    hierarchy = _adapt_capability(CAPABILITY_FIXTURES[fixture])
    axes = tuple(axis for level in hierarchy.levels for axis in level.axes)

    assert tuple(axis.origin for axis in axes) == origins
    assert tuple(level.bit_count for level in hierarchy.levels) == level_bits
    assert hierarchy.local_space_dimensions == local_dimensions
    assert hierarchy.owner_bit_count == sum(level_bits)
    assert len(hierarchy.storage_axes) == len(CAPABILITY_FIXTURES[fixture].storage)

    flag = hierarchy.coordinate_flag()
    assert tuple(space.rank for space in flag.spaces) == local_dimensions
    assert flag.spaces[0] == GF2Subspace.full(hierarchy.carrier_bit_count)
    for level, (start, end) in enumerate(hierarchy.owner_bit_ranges):
        owner_projection_rows = hierarchy.owner_coordinate_space_through(level)
        projection_kernel = hierarchy.local_coordinate_space_after(level)
        assert owner_projection_rows.basis == tuple(1 << bit for bit in range(end))
        assert projection_kernel.basis == tuple(
            1 << bit for bit in range(end, hierarchy.carrier_bit_count)
        )
        assert owner_projection_rows.rank == end
        assert projection_kernel.rank == hierarchy.carrier_bit_count - end
        assert owner_projection_rows.is_direct_sum_with(projection_kernel)
        assert owner_projection_rows.sum(projection_kernel) == flag.spaces[0]
        assert start == (0 if level == 0 else hierarchy.owner_bit_ranges[level - 1][1])


def test_storage_geometry_does_not_silently_become_an_owner_axis():
    hierarchy = _adapt_capability(CAPABILITY_FIXTURES["apu_v2"])
    assert hierarchy.storage_axes[0].extent == 65536
    assert hierarchy.storage_axes[1].extent == 4
    assert hierarchy.owner_bit_count == 4
    assert hierarchy.carrier_bit_count == 6
    assert hierarchy.owner_bit_ranges == ((0, 4),)


def test_upmem_24_tasklets_fail_closed_without_declared_subgroup():
    with pytest.raises(LayoutProblemValidationError) as failure:
        OwnerAxis(
            0,
            24,
            OwnerAxisOrigin.MAPPED_UNIT,
            diagnostic_name="tasklet",
        )
    assert failure.value.diagnostic() == {
        "code": "non-power-of-two-owner-extent",
        "location": ["owner-axis", 0],
        "details": [24],
    }

    accepted = OwnerAxis(
        0,
        24,
        OwnerAxisOrigin.MAPPED_UNIT,
        active_subgroup_extent=16,
        diagnostic_name="tasklet",
    )
    assert accepted.mapped_extent == 16
    assert accepted.bit_count == 4


def test_owner_and_local_manifest_is_exact_and_rename_insensitive():
    original = _adapt_capability(CAPABILITY_FIXTURES["samsung"])
    renamed = _adapt_capability(
        CAPABILITY_FIXTURES["samsung"], rename_prefix="alpha_renamed_"
    )

    assert original == renamed
    assert original.manifest() == renamed.manifest()
    assert original.coordinate_flag().manifest() == renamed.coordinate_flag().manifest()
    assert json.loads(json.dumps(original.manifest(), sort_keys=True)) == {
        "carrier_bit_count": 8,
        "levels": [
            {
                "axes": [
                    {
                        "active_subgroup_extent": None,
                        "bit_count": 4,
                        "mapped_extent": 16,
                        "ordinal": 0,
                        "origin": "mapped-unit",
                        "physical_extent": 16,
                    }
                ],
                "bit_count": 4,
                "mapped_extent": 16,
                "ordinal": 0,
            },
            {
                "axes": [
                    {
                        "active_subgroup_extent": None,
                        "bit_count": 3,
                        "mapped_extent": 8,
                        "ordinal": 1,
                        "origin": "mapped-unit",
                        "physical_extent": 8,
                    }
                ],
                "bit_count": 3,
                "mapped_extent": 8,
                "ordinal": 1,
            },
        ],
        "local_axes": [{"bit_count": 1, "extent": 2, "ordinal": 0}],
        "local_bit_count": 1,
        "local_space_dimensions": [8, 4, 1],
        "owner_bit_count": 7,
        "owner_bit_ranges": [[0, 4], [4, 7]],
        "schema": "owner-hierarchy-v1",
        "storage_axes": [
            {"extent": 16, "ordinal": 0},
            {"extent": 4, "ordinal": 1},
            {"extent": 16384, "ordinal": 2},
            {"extent": 128, "ordinal": 3},
        ],
    }

    with pytest.raises(FrozenInstanceError):
        original.levels = ()
    with pytest.raises(FrozenInstanceError):
        original.levels[0].axes[0].physical_extent = 4


def _small_hierarchy():
    return OwnerHierarchy(
        (
            OwnerLevel(
                0,
                (
                    OwnerAxis(
                        0,
                        2,
                        OwnerAxisOrigin.MAPPED_UNIT,
                        diagnostic_name="outer",
                    ),
                ),
            ),
            OwnerLevel(
                1,
                (
                    OwnerAxis(
                        1,
                        2,
                        OwnerAxisOrigin.REGISTER_AXIS,
                        diagnostic_name="inner",
                    ),
                ),
            ),
        ),
        (LocalAxis(0, 1),),
    )


def test_noncoordinate_nested_flag_and_forced_local_spaces_are_valid():
    hierarchy = _small_hierarchy()
    # An invertible change of basis relative to the coordinate flag.
    t0 = GF2Subspace.full(3)
    t1 = GF2Subspace(3, (0b011, 0b100))
    t2 = GF2Subspace(3, (0b111,))
    forced = (
        GF2Subspace(3, (0b111,)),
        GF2Subspace(3, (0b111,)),
    )
    flag = LocalSpaceFlag(hierarchy, (t0, t1, t2), forced)

    assert flag.spaces == (t0, t1, t2)
    assert flag.forced_local == forced
    assert tuple(space.rank for space in flag.spaces) == (3, 2, 1)
    assert json.dumps(flag.manifest(), sort_keys=True)


@pytest.mark.parametrize(
    ("spaces", "forced", "code", "location"),
    (
        (
            (GF2Subspace(3, (0b001, 0b010)), GF2Subspace.full(3), GF2Subspace(3, (1,))),
            (),
            LayoutValidationCode.FLAG_ROOT_NOT_FULL,
            ("local-space-flag", "space", 0),
        ),
        (
            (GF2Subspace.full(3), GF2Subspace(3, (1,)), GF2Subspace(3, (1,))),
            (),
            LayoutValidationCode.FLAG_DIMENSION_MISMATCH,
            ("local-space-flag", "space", 1),
        ),
        (
            (
                GF2Subspace.full(3),
                GF2Subspace(3, (0b001, 0b010)),
                GF2Subspace(3, (0b100,)),
            ),
            (),
            LayoutValidationCode.FLAG_NOT_NESTED,
            ("local-space-flag", "space", 2),
        ),
        (
            (
                GF2Subspace.full(3),
                GF2Subspace(3, (0b001, 0b010)),
                GF2Subspace(3, (0b001,)),
            ),
            (
                GF2Subspace(3, (0b010,)),
                GF2Subspace(3, (0b010,)),
            ),
            LayoutValidationCode.FORCED_LOCAL_OUTSIDE_FLAG,
            ("local-space-flag", "forced-local", 1),
        ),
        (
            (
                GF2Subspace.full(3),
                GF2Subspace(3, (0b001, 0b010)),
                GF2Subspace(3, (0b001,)),
            ),
            (
                GF2Subspace.zero(3),
                GF2Subspace(3, (0b001,)),
            ),
            LayoutValidationCode.FORCED_LOCAL_NOT_NESTED,
            ("local-space-flag", "forced-local", 1),
        ),
    ),
)
def test_local_flag_failures_are_typed_and_structural(spaces, forced, code, location):
    with pytest.raises(LayoutProblemValidationError) as failure:
        LocalSpaceFlag(_small_hierarchy(), spaces, forced)
    assert failure.value.code is code
    assert failure.value.location == location


def test_forced_local_diagnostics_do_not_depend_on_names():
    def fail(prefix):
        hierarchy = _small_hierarchy()
        renamed = OwnerHierarchy(
            tuple(
                OwnerLevel(
                    level.ordinal,
                    tuple(
                        OwnerAxis(
                            axis.ordinal,
                            axis.physical_extent,
                            axis.origin,
                            axis.active_subgroup_extent,
                            diagnostic_name=f"{prefix}{axis.ordinal}",
                        )
                        for axis in level.axes
                    ),
                    diagnostic_name=f"{prefix}level{level.ordinal}",
                )
                for level in hierarchy.levels
            ),
            (LocalAxis(0, 1, diagnostic_name=f"{prefix}local"),),
        )
        with pytest.raises(LayoutProblemValidationError) as failure:
            LocalSpaceFlag(
                renamed,
                (
                    GF2Subspace.full(3),
                    GF2Subspace(3, (0b001, 0b010)),
                    GF2Subspace(3, (0b001,)),
                ),
                (
                    GF2Subspace(3, (0b100,)),
                    GF2Subspace.zero(3),
                ),
            )
        return failure.value.diagnostic()

    assert (
        fail("first_")
        == fail("renamed_")
        == {
            "code": "forced-local-outside-flag",
            "location": ["local-space-flag", "forced-local", 0],
            "details": [1, 2],
        }
    )


@pytest.mark.parametrize(
    ("constructor", "code"),
    (
        (
            lambda: OwnerAxis(0, 24, OwnerAxisOrigin.MAPPED_UNIT),
            LayoutValidationCode.NON_POWER_OF_TWO_OWNER_EXTENT,
        ),
        (
            lambda: OwnerAxis(
                0,
                24,
                OwnerAxisOrigin.MAPPED_UNIT,
                active_subgroup_extent=32,
            ),
            LayoutValidationCode.INVALID_ACTIVE_SUBGROUP,
        ),
        (
            lambda: OwnerLevel(0, ()),
            LayoutValidationCode.EMPTY_OWNER_LEVEL,
        ),
        (
            lambda: OwnerHierarchy(
                (
                    OwnerLevel(
                        1,
                        (OwnerAxis(0, 2, OwnerAxisOrigin.MAPPED_UNIT),),
                    ),
                )
            ),
            LayoutValidationCode.NONCONTIGUOUS_ORDINALS,
        ),
        (
            lambda: OwnerHierarchy(
                (
                    OwnerLevel(
                        0,
                        (OwnerAxis(1, 2, OwnerAxisOrigin.MAPPED_UNIT),),
                    ),
                )
            ),
            LayoutValidationCode.NONCONTIGUOUS_ORDINALS,
        ),
        (
            lambda: StorageAxis(0, 0),
            LayoutValidationCode.INVALID_EXTENT,
        ),
        (
            lambda: LocalAxis(0, 0),
            LayoutValidationCode.INVALID_EXTENT,
        ),
    ),
)
def test_negative_geometry_cases_fail_with_exact_reason(constructor, code):
    with pytest.raises(LayoutProblemValidationError) as failure:
        constructor()
    assert failure.value.code is code


def test_level_queries_fail_closed_with_typed_reason():
    hierarchy = _small_hierarchy()
    for level in (-1, 2, True, "0"):
        with pytest.raises(LayoutProblemValidationError) as failure:
            hierarchy.local_coordinate_space_after(level)
        assert failure.value.code is LayoutValidationCode.INVALID_LEVEL_INDEX


# --- adversarial (independent) ---


def _axis(ordinal, extent=2, *, subgroup=None, members=None, name=""):
    return OwnerAxis(
        ordinal,
        extent,
        OwnerAxisOrigin.MAPPED_UNIT,
        active_subgroup_extent=subgroup,
        diagnostic_name=name,
        active_subgroup_members=members,
    )


def test_empty_hierarchy_and_zero_dimensional_carrier_are_exact():
    hierarchy = OwnerHierarchy(())
    flag = hierarchy.coordinate_flag()

    assert hierarchy.owner_bit_count == 0
    assert hierarchy.local_bit_count == 0
    assert hierarchy.carrier_bit_count == 0
    assert hierarchy.owner_bit_ranges == ()
    assert hierarchy.local_space_dimensions == (0,)
    assert flag.spaces == (GF2Subspace.full(0),)
    assert flag.forced_local == ()
    assert hierarchy.manifest() == {
        "schema": "owner-hierarchy-v1",
        "levels": [],
        "local_axes": [],
        "storage_axes": [],
        "owner_bit_count": 0,
        "local_bit_count": 0,
        "carrier_bit_count": 0,
        "owner_bit_ranges": [],
        "local_space_dimensions": [0],
    }

    for query in (
        hierarchy.owner_coordinate_space_through,
        hierarchy.local_coordinate_space_after,
    ):
        with pytest.raises(LayoutProblemValidationError) as failure:
            query(0)
        assert failure.value.code is LayoutValidationCode.INVALID_LEVEL_INDEX
        assert json.loads(json.dumps(failure.value.diagnostic())) == {
            "code": "invalid-level-index",
            "location": ["owner-level"],
            "details": [0, 0],
        }


def test_ownerless_local_carrier_and_owner_only_zero_local_space_are_exact():
    ownerless = OwnerHierarchy((), (LocalAxis(0, 3),))
    assert ownerless.local_space_dimensions == (3,)
    assert ownerless.coordinate_flag().spaces == (GF2Subspace.full(3),)

    owner_only = OwnerHierarchy(
        (
            OwnerLevel(0, (_axis(0, 4),)),
            OwnerLevel(1, (_axis(1, 2),)),
        )
    )
    flag = owner_only.coordinate_flag()
    assert owner_only.carrier_bit_count == 3
    assert owner_only.local_space_dimensions == (3, 1, 0)
    assert tuple(space.rank for space in flag.spaces) == (3, 1, 0)
    assert flag.spaces[-1] == GF2Subspace.zero(3)


def test_multi_axis_levels_have_exact_truth_table_projection_kernels():
    hierarchy = OwnerHierarchy(
        (
            OwnerLevel(0, (_axis(0, 4), _axis(1, 2))),
            OwnerLevel(
                1,
                (
                    OwnerAxis(
                        2,
                        2,
                        OwnerAxisOrigin.REGISTER_AXIS,
                    ),
                ),
            ),
        ),
        (LocalAxis(0, 2),),
        (StorageAxis(0, 3),),
    )

    assert hierarchy.owner_bit_ranges == ((0, 3), (3, 4))
    assert hierarchy.local_space_dimensions == (6, 3, 2)
    assert hierarchy.owner_bit_count == 4
    assert hierarchy.local_bit_count == 2
    assert hierarchy.storage_axes[0].extent == 3

    for level, end in enumerate((3, 4)):
        row_space = hierarchy.owner_coordinate_space_through(level)
        kernel = hierarchy.local_coordinate_space_after(level)
        assert row_space.basis == tuple(1 << bit for bit in range(end))
        assert all(
            kernel.contains(vector) == ((vector & ((1 << end) - 1)) == 0)
            for vector in range(1 << hierarchy.carrier_bit_count)
        )


def test_collections_are_frozen_copies_not_mutable_aliases():
    axes = [_axis(0)]
    level = OwnerLevel(0, axes)
    levels = [level]
    locals_ = [LocalAxis(0, 1)]
    storage = [StorageAxis(0, 7)]
    hierarchy = OwnerHierarchy(levels, locals_, storage)
    spaces = [
        GF2Subspace.full(2),
        GF2Subspace(2, (0b10,)),
    ]
    flag = LocalSpaceFlag(hierarchy, spaces)

    axes.clear()
    levels.clear()
    locals_.clear()
    storage.clear()
    spaces.clear()

    assert len(level.axes) == 1
    assert len(hierarchy.levels) == 1
    assert len(hierarchy.local_axes) == 1
    assert len(hierarchy.storage_axes) == 1
    assert len(flag.spaces) == 2
    with pytest.raises(FrozenInstanceError):
        flag.spaces = ()


@pytest.mark.parametrize(
    ("constructor", "code", "location"),
    (
        (
            lambda: _axis(True),
            LayoutValidationCode.INVALID_INTEGER,
            ("owner-axis", "ordinal"),
        ),
        (
            lambda: _axis(-1),
            LayoutValidationCode.INVALID_ORDINAL,
            ("owner-axis", "ordinal"),
        ),
        (
            lambda: OwnerAxis(0, 2, "mapped-unit"),
            LayoutValidationCode.INVALID_AXIS_ORIGIN,
            ("owner-axis", 0, "origin"),
        ),
        (
            lambda: _axis(0, subgroup=True),
            LayoutValidationCode.INVALID_INTEGER,
            ("owner-axis", 0, "active-subgroup-extent"),
        ),
        (
            lambda: OwnerLevel(True, (_axis(0),)),
            LayoutValidationCode.INVALID_INTEGER,
            ("owner-level", "ordinal"),
        ),
        (
            lambda: OwnerLevel(0, None),
            LayoutValidationCode.INVALID_RECORD_TYPE,
            ("owner-level", 0, "axes"),
        ),
        (
            lambda: OwnerHierarchy(None),
            LayoutValidationCode.INVALID_RECORD_TYPE,
            ("owner-level", "collection"),
        ),
        (
            lambda: OwnerHierarchy((), None),
            LayoutValidationCode.INVALID_RECORD_TYPE,
            ("local-axis", "collection"),
        ),
        (
            lambda: OwnerHierarchy((), (), None),
            LayoutValidationCode.INVALID_RECORD_TYPE,
            ("storage-axis", "collection"),
        ),
        (
            lambda: LocalSpaceFlag("hierarchy", ()),
            LayoutValidationCode.INVALID_RECORD_TYPE,
            ("local-space-flag", "hierarchy"),
        ),
    ),
)
def test_malformed_types_fail_with_typed_structural_identity(
    constructor, code, location
):
    with pytest.raises(LayoutProblemValidationError) as failure:
        constructor()
    assert failure.value.code is code
    assert failure.value.location == location
    json.dumps(failure.value.diagnostic(), sort_keys=True)


@pytest.mark.parametrize(
    ("physical", "subgroup", "code"),
    (
        (24, 0, LayoutValidationCode.INVALID_EXTENT),
        (24, 1, LayoutValidationCode.INVALID_EXTENT),
        (24, 3, LayoutValidationCode.INVALID_ACTIVE_SUBGROUP),
        (24, 32, LayoutValidationCode.INVALID_ACTIVE_SUBGROUP),
    ),
)
def test_invalid_active_subgroups_fail_closed(physical, subgroup, code):
    with pytest.raises(LayoutProblemValidationError) as failure:
        _axis(0, physical, subgroup=subgroup)
    assert failure.value.code is code


def test_explicit_subgroup_is_structural_and_can_underutilize_power_of_two_axis():
    full = _axis(0, 32)
    subgroup = _axis(0, 32, subgroup=8)
    renamed = _axis(0, 32, subgroup=8, name="renamed")

    assert subgroup == renamed
    assert subgroup.manifest() == renamed.manifest()
    assert subgroup != full
    assert subgroup.physical_extent == 32
    assert subgroup.mapped_extent == 8
    assert subgroup.bit_count == 3
    assert subgroup.active_subgroup_members == tuple(range(8))
    assert full.bit_count == 5


def test_ordered_subgroup_members_preserve_exact_physical_mapping_identity():
    prefix = _axis(0, 24, subgroup=16)
    explicit_prefix = _axis(0, 24, members=tuple(range(16)))
    upper = _axis(0, 24, members=tuple(range(8, 24)))
    permuted = _axis(0, 24, members=tuple(reversed(range(16))))

    assert prefix == explicit_prefix
    assert len({prefix, upper, permuted}) == 3
    assert prefix.active_subgroup_extent == 16
    assert prefix.active_subgroup_members == tuple(range(16))
    assert upper.active_subgroup_extent == 16
    assert upper.active_subgroup_members == tuple(range(8, 24))
    assert prefix.manifest()["active_subgroup_members"] == list(range(16))
    assert upper.manifest()["active_subgroup_members"] == list(range(8, 24))
    assert prefix.manifest() != upper.manifest() != permuted.manifest()


def test_full_identity_subgroup_forms_canonicalize_to_one_compact_record():
    implicit = _axis(0, 32)
    extent_only = _axis(0, 32, subgroup=32)
    explicit = _axis(0, 32, members=tuple(range(32)))
    explicit_with_extent = _axis(
        0,
        32,
        subgroup=32,
        members=tuple(range(32)),
    )

    assert len({implicit, extent_only, explicit, explicit_with_extent}) == 1
    assert implicit.manifest() == extent_only.manifest() == explicit.manifest()
    assert implicit.active_subgroup_extent is None
    assert implicit.active_subgroup_members is None

    remapped_full_axis = _axis(0, 32, members=tuple(reversed(range(32))))
    assert remapped_full_axis != implicit
    assert remapped_full_axis.active_subgroup_extent == 32
    assert remapped_full_axis.active_subgroup_members == tuple(reversed(range(32)))


@pytest.mark.parametrize(
    ("members", "subgroup", "code", "location"),
    (
        ((), None, LayoutValidationCode.INVALID_ACTIVE_SUBGROUP, "members"),
        ((0,), None, LayoutValidationCode.INVALID_ACTIVE_SUBGROUP, "members"),
        ((0, 1, 2), None, LayoutValidationCode.INVALID_ACTIVE_SUBGROUP, "members"),
        ((0, 0), None, LayoutValidationCode.INVALID_ACTIVE_SUBGROUP, "members"),
        ((0, 24), None, LayoutValidationCode.INVALID_ACTIVE_SUBGROUP, "members"),
        ((0, True), None, LayoutValidationCode.INVALID_INTEGER, "member"),
        ((0, -1), None, LayoutValidationCode.INVALID_ORDINAL, "member"),
        ((0, 1), 4, LayoutValidationCode.INVALID_ACTIVE_SUBGROUP, "members"),
        (7, None, LayoutValidationCode.INVALID_ACTIVE_SUBGROUP, "members"),
        ("01", None, LayoutValidationCode.INVALID_ACTIVE_SUBGROUP, "members"),
    ),
)
def test_invalid_ordered_subgroup_mappings_fail_closed(
    members, subgroup, code, location
):
    with pytest.raises(LayoutProblemValidationError) as failure:
        _axis(0, 24, subgroup=subgroup, members=members)
    assert failure.value.code is code
    assert any(
        isinstance(part, str) and location in part for part in failure.value.location
    )
    json.dumps(failure.value.diagnostic(), sort_keys=True)


def test_noncontiguous_diagnostic_is_alpha_rename_invariant_and_json_stable():
    def fail(name):
        with pytest.raises(LayoutProblemValidationError) as failure:
            OwnerHierarchy(
                (
                    OwnerLevel(
                        0,
                        (
                            _axis(0, name=name),
                            _axis(2, name=f"{name}_other"),
                        ),
                        diagnostic_name=f"{name}_level",
                    ),
                )
            )
        return failure.value

    first = fail("first")
    renamed = fail("renamed")
    assert first.diagnostic() == renamed.diagnostic()
    assert json.loads(json.dumps(first.diagnostic())) == {
        "code": "noncontiguous-ordinals",
        "location": ["owner-axis"],
        "details": [[0, 2], [0, 1]],
    }


def test_flag_collection_and_ambient_failures_precede_algebra_exceptions():
    hierarchy = OwnerHierarchy((OwnerLevel(0, (_axis(0),)),))

    with pytest.raises(LayoutProblemValidationError) as failure:
        LocalSpaceFlag(hierarchy, None)
    assert failure.value.code is LayoutValidationCode.INVALID_RECORD_TYPE

    with pytest.raises(LayoutProblemValidationError) as failure:
        LocalSpaceFlag(hierarchy, (GF2Subspace.full(1),))
    assert failure.value.code is LayoutValidationCode.INVALID_FLAG_LENGTH

    with pytest.raises(LayoutProblemValidationError) as failure:
        LocalSpaceFlag(
            hierarchy,
            (GF2Subspace.full(1), GF2Subspace.zero(1)),
            (GF2Subspace.zero(2),),
        )
    assert failure.value.code is LayoutValidationCode.AMBIENT_DIMENSION_MISMATCH
    assert failure.value.location == ("local-space-flag", "forced-local", 0)


# --- owner_hierarchy_from_target ---


def _toy_apu_target():
    @allo.target("toy_apu")
    def toy():
        @allo.unit(mapping={"group": 4})
        def group():
            allo.reg(4, 16, axes={"lane": 4}, name="vr")
            allo.reg(4, 16, name="plain")

    return toy


def test_samsung_unit_tree_gives_channel_then_pim_levels():
    hierarchy = owner_hierarchy_from_target(
        build_samsung_target(), ("pseudo_channel", "pim"), local_bits=2
    )
    assert [level.bit_count for level in hierarchy.levels] == [4, 3]
    assert [level.diagnostic_name for level in hierarchy.levels] == [
        "pseudo_channel",
        "pim",
    ]
    axes = [axis for level in hierarchy.levels for axis in level.axes]
    assert [axis.diagnostic_name for axis in axes] == [
        "pseudo_channel.channel",
        "pim.pim",
    ]
    assert all(axis.origin is OwnerAxisOrigin.MAPPED_UNIT for axis in axes)
    assert hierarchy.carrier_bit_count == 9
    assert hierarchy.local_space_dimensions == (9, 5, 2)


def test_apu_v1_register_lane_axis_is_a_register_owner_level():
    hierarchy = owner_hierarchy_from_target(build_apu_v1_target(), ("apuc", "vr0.lane"))
    assert [level.bit_count for level in hierarchy.levels] == [2, 15]
    lane = hierarchy.levels[1].axes[0]
    assert lane.origin is OwnerAxisOrigin.REGISTER_AXIS
    assert lane.diagnostic_name == "vr0.lane"
    # A register with exactly one axis may be named without the axis.
    same = owner_hierarchy_from_target(build_apu_v1_target(), ("apuc", "vr0"))
    assert same == hierarchy


def test_toy_target_matches_the_paper_example_hierarchy():
    target = _toy_apu_target()
    one_level = owner_hierarchy_from_target(target, ("group",), local_bits=2)
    assert one_level.local_space_dimensions == (4, 2)
    two_level = owner_hierarchy_from_target(target, ("group", "vr.lane"))
    assert two_level.local_space_dimensions == (4, 2, 0)


def test_non_power_of_two_extent_is_rejected_with_a_clear_error():
    with pytest.raises(
        TargetHierarchyError, match="extent 24, which is not a power of two"
    ):
        owner_hierarchy_from_target(build_upmem_target(), ("dpu", "tasklet"))
    hierarchy = owner_hierarchy_from_target(
        build_upmem_target(),
        ("dpu", "tasklet"),
        active_extents={"tasklet.tasklet": 16},
    )
    tasklet = hierarchy.levels[1].axes[0]
    assert (tasklet.physical_extent, tasklet.mapped_extent) == (24, 16)
    assert hierarchy.owner_bit_count == 6 + 4


@pytest.mark.parametrize(
    "levels,message",
    (
        ((), "at least one owner level"),
        (("nonexistent",), "no unit or register"),
        (("group", "plain"), "declares no lane axes"),
        (("group", "vr.bank"), "has no axis 'bank'"),
    ),
)
def test_target_level_errors_are_explicit(levels, message):
    with pytest.raises(TargetHierarchyError, match=message):
        owner_hierarchy_from_target(_toy_apu_target(), levels)


def test_unit_without_compute_axis_and_unused_subgroup_are_rejected():
    with pytest.raises(TargetHierarchyError, match="no compute axis"):
        owner_hierarchy_from_target(build_apu_v1_target(), ("dma",))
    with pytest.raises(TargetHierarchyError, match="not owner axes"):
        owner_hierarchy_from_target(
            _toy_apu_target(), ("group",), active_extents={"x.y": 2}
        )
