"""Focused real-MLIR tests for workload-name-free structured plans."""

from dataclasses import FrozenInstanceError
from functools import lru_cache
from pathlib import Path
import re
import sys

import allo
import pytest

from allo.pim.apu_g2_structured_plans import (
    AmbiguousStructuredAPUG2Plan,
    CenteredGramStatisticsPlan,
    NoStructuredAPUG2Plan,
    NormalizedGramStatisticsPlan,
    RankTwoUpdateGemvChainPlan,
    SymmetricContractionPlan,
    UnitDiagonalTriangularContractionPlan,
    analyze_apu_g2_structured_plan,
)
from allo.pim.structured_program import StructuredProgram, analyze_structured_program


_PIM_ROOT = Path(__file__).resolve().parents[1] / "pim"
if str(_PIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIM_ROOT))

from lib import cell  # noqa: E402


@lru_cache(maxsize=None)
def _retained(name: str) -> str:
    workload = cell.load_workload(_PIM_ROOT / "apu_g2" / name, name)
    return str(allo.customize(workload.build(), enable_tensor=False).module)


def _effect_names(plan) -> tuple[str, ...]:
    return tuple(effect.operation for effect in plan.effects)


def test_real_rank_two_update_and_gemv_chain_plan_fields():
    plan = analyze_apu_g2_structured_plan(_retained("gemver"))

    assert isinstance(plan, RankTwoUpdateGemvChainPlan)
    assert (
        plan.matrix_argument,
        plan.first_row_factor_argument,
        plan.second_row_factor_argument,
        plan.first_column_factor_argument,
        plan.second_column_factor_argument,
        plan.state_argument,
        plan.projection_argument,
        plan.output_argument,
        plan.bias_argument,
    ) == tuple(range(9))
    assert plan.extent == 120
    assert plan.rank_update_coefficients == (1, 1)
    assert plan.accumulator_coefficients == (1, 1, 1, 1)
    assert _effect_names(plan) == (
        "rank_two_update",
        "transposed_matrix_vector_update",
        "vector_bias_update",
        "matrix_vector_update",
    )
    assert [effect.target_position for effect in plan.effects] == [0, 5, 5, 7]
    assert [effect.order for effect in plan.effects] == [0, 1, 2, 3]


def test_real_symmetric_contraction_plan_fields():
    plan = analyze_apu_g2_structured_plan(_retained("symm"))

    assert isinstance(plan, SymmetricContractionPlan)
    assert (
        plan.symmetric_matrix_argument,
        plan.rhs_argument,
        plan.output_argument,
    ) == (0, 1, 2)
    assert (plan.rows, plan.columns, plan.reduction_extent) == (60, 80, 60)
    assert (plan.alpha_coefficient, plan.beta_coefficient) == (5, 4)
    assert plan.predicate_operation == "sle"
    assert plan.branch_polarities == (True, False)
    assert _effect_names(plan) == (
        "output_prescale",
        "symmetric_lower_update",
        "symmetric_upper_update",
    )
    assert [effect.coefficient for effect in plan.effects] == [4, 5, 5]
    assert [effect.predicates for effect in plan.effects] == [
        (),
        (("sle", True),),
        (("sle", False),),
    ]
    with pytest.raises(FrozenInstanceError):
        plan.rows = 1


def test_real_unit_diagonal_triangular_contraction_plan_fields():
    plan = analyze_apu_g2_structured_plan(_retained("trmm"))

    assert isinstance(plan, UnitDiagonalTriangularContractionPlan)
    assert (plan.triangular_matrix_argument, plan.state_argument) == (0, 1)
    assert (plan.rows, plan.columns, plan.reduction_extent) == (60, 80, 60)
    assert plan.predicate_operation == "sgt"
    assert plan.implicit_diagonal_coefficient == 1
    assert plan.post_scale_coefficient == 1
    assert plan.reads_original_off_diagonal_state is True
    assert _effect_names(plan) == ("strict_triangular_update", "postscale")
    assert plan.effects[0].input_arguments == (1, 0, 1)
    assert plan.effects[0].predicates == (("sgt", True),)
    assert plan.effects[1].coefficient == 1


def test_real_centered_gram_statistics_plan_fields():
    plan = analyze_apu_g2_structured_plan(_retained("covariance"))

    assert isinstance(plan, CenteredGramStatisticsPlan)
    assert (plan.data_argument, plan.mean_argument, plan.gram_argument) == (0, 1, 2)
    assert (plan.sample_count_allocation, plan.gram_divisor_allocation) == (0, 1)
    assert (plan.sample_extent, plan.feature_extent) == (100, 80)
    assert (plan.mean_divisor, plan.gram_divisor) == (100, 99)
    assert _effect_names(plan) == (
        "sample_count_materialize",
        "gram_divisor_materialize",
        "mean_initialize",
        "mean_accumulate",
        "mean_divide",
        "center_in_place",
        "gram_initialize",
        "gram_accumulate",
        "gram_divide",
    )
    assert [effect.order for effect in plan.effects] == list(range(9))
    assert plan.effects[5].target_position == plan.data_argument
    assert plan.effects[8].divisor == 99


def test_real_normalized_gram_statistics_plan_fields():
    plan = analyze_apu_g2_structured_plan(_retained("correlation"))

    assert isinstance(plan, NormalizedGramStatisticsPlan)
    assert (
        plan.mean_input_argument,
        plan.variance_input_argument,
        plan.center_input_argument,
        plan.gram_argument,
    ) == (0, 1, 2, 3)
    assert (
        plan.sample_count_allocation,
        plan.root_sample_count_allocation,
        plan.unit_allocation,
        plan.mean_allocation,
        plan.variance_centered_allocation,
        plan.variance_allocation,
        plan.standard_deviation_allocation,
        plan.centered_allocation,
        plan.denominator_allocation,
        plan.normalized_allocation,
    ) == tuple(range(10))
    assert (plan.sample_extent, plan.feature_extent) == (100, 80)
    assert (plan.mean_divisor, plan.variance_divisor) == (100, 100)
    assert plan.root_sample_factor == 10
    assert (plan.zero_replacement, plan.diagonal_value) == (1, 1)
    assert len(plan.effects) == 23
    assert [effect.order for effect in plan.effects] == list(range(23))
    assert _effect_names(plan)[10:] == (
        "mean_accumulate",
        "mean_divide",
        "variance_center",
        "variance_accumulate",
        "variance_divide",
        "standard_deviation",
        "zero_clamp",
        "denominator_form",
        "center",
        "normalize",
        "gram_initialize",
        "gram_accumulate",
        "diagonal_override",
    )
    assert plan.effects[16].predicates == (("eq", True),)
    assert plan.effects[22].predicates == (("eq", True),)


def _alpha_rename(text: str) -> str:
    text = re.sub(r"@[-\w.$]+", "@renamed_function", text)
    ssa_names: dict[str, str] = {}

    def rename_ssa(match: re.Match[str]) -> str:
        source = match.group(0)
        if source not in ssa_names:
            ssa_names[source] = f"%renamed_{len(ssa_names)}"
        return ssa_names[source]

    text = re.sub(r"%[-\w.$]+", rename_ssa, text)
    label_index = 0

    def rename_label(match: re.Match[str]) -> str:
        nonlocal label_index
        key = match.group(1)
        replacement = f'{key} = "renamed_label_{label_index}"'
        label_index += 1
        return replacement

    return re.sub(
        r"\b(from|to|name|loop_name|op_name)\s*=\s*\"[^\"]*\"",
        rename_label,
        text,
    )


@pytest.mark.parametrize(
    "name",
    ["gemver", "symm", "trmm", "covariance", "correlation"],
)
def test_function_ssa_and_source_label_renaming_preserves_plan(name):
    original = analyze_apu_g2_structured_plan(_retained(name))
    renamed = analyze_apu_g2_structured_plan(_alpha_rename(_retained(name)))

    assert renamed == original


@pytest.mark.parametrize(
    "name,old,new",
    [
        ("gemver", "arith.addi", "arith.subi"),
        ("symm", "arith.cmpi sle", "arith.cmpi slt"),
        ("trmm", "arith.cmpi sgt", "arith.cmpi sge"),
        ("covariance", "arith.subi", "arith.addi"),
        ("correlation", "math.sqrt", "math.floor"),
    ],
)
def test_each_family_rejects_a_semantic_mutation(name, old, new):
    text = _retained(name)
    assert old in text
    mutated = text.replace(old, new, 1)

    with pytest.raises(NoStructuredAPUG2Plan):
        analyze_apu_g2_structured_plan(mutated)


def test_unsupported_cast_and_missing_or_extra_stores_fail_closed():
    cast_mutation = _retained("gemver").replace("arith.extui", "arith.extsi", 1)
    with pytest.raises(NoStructuredAPUG2Plan):
        analyze_apu_g2_structured_plan(cast_mutation)

    lines = _retained("gemver").splitlines()
    last_store = max(
        index for index, line in enumerate(lines) if "affine.store" in line
    )
    missing_store = "\n".join(lines[:last_store] + lines[last_store + 1 :])
    with pytest.raises(NoStructuredAPUG2Plan):
        analyze_apu_g2_structured_plan(missing_store)

    lines = _retained("trmm").splitlines()
    last_store = max(
        index for index, line in enumerate(lines) if "affine.store" in line
    )
    extra_store = "\n".join(
        lines[: last_store + 1] + [lines[last_store]] + lines[last_store + 1 :]
    )
    with pytest.raises(NoStructuredAPUG2Plan):
        analyze_apu_g2_structured_plan(extra_store)


def test_shape_only_lookalike_has_no_plan():
    text = r"""
    func.func @shape_only(%matrix: memref<120x120xi16>, %v0: memref<120xi16>, %v1: memref<120xi16>, %v2: memref<120xi16>, %v3: memref<120xi16>, %v4: memref<120xi16>, %v5: memref<120xi16>, %v6: memref<120xi16>, %v7: memref<120xi16>) attributes {itypes = "uuuuuuuuu", otypes = ""} {
      affine.for %axis = 0 to 120 {
        %value = affine.load %v0[%axis] {unsigned} : memref<120xi16>
        affine.store %value, %v6[%axis] : memref<120xi16>
      }
      return
    }
    """

    with pytest.raises(NoStructuredAPUG2Plan):
        analyze_apu_g2_structured_plan(text)


def test_program_input_and_ambiguous_program_are_explicit():
    program = analyze_structured_program(_retained("trmm"))
    assert analyze_apu_g2_structured_plan(program) == analyze_apu_g2_structured_plan(
        _retained("trmm")
    )

    ambiguous = StructuredProgram(program.functions + program.functions)
    with pytest.raises(AmbiguousStructuredAPUG2Plan):
        analyze_apu_g2_structured_plan(ambiguous)
