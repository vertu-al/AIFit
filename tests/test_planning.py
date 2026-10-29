"""Tests for preference-only and expected-free-energy policy planners."""

import pytest

from aifit.cue_task import CueTask, CueTaskConfig
from aifit.efe import expected_information_gain
from aifit.planning import (
    ExpectedFreeEnergyPlanner,
    Policy,
    PreferenceOnlyPlanner,
)


def _task(
    *,
    prior_left_good: float = 0.5,
    cue_reliability: float = 0.8,
    inspection_cost: float = 0.1,
) -> CueTask:
    return CueTask(
        CueTaskConfig(
            prior_left_good=prior_left_good,
            cue_reliability=cue_reliability,
            inspection_cost=inspection_cost,
        )
    )


def test_direct_commit_preference_values_match_the_prior() -> None:
    planner = PreferenceOnlyPlanner(_task(prior_left_good=0.8))

    left_evaluation = planner.evaluate(Policy.COMMIT_LEFT)
    right_evaluation = planner.evaluate(Policy.COMMIT_RIGHT)

    assert left_evaluation.expected_preference_value == pytest.approx(0.8)
    assert right_evaluation.expected_preference_value == pytest.approx(0.2)


def test_inspection_preference_value_uses_the_best_action_after_each_cue() -> None:
    planner = PreferenceOnlyPlanner(
        _task(
            prior_left_good=0.5,
            cue_reliability=0.8,
        )
    )

    evaluation = planner.evaluate(Policy.INSPECT_THEN_COMMIT)

    assert evaluation.expected_preference_value == pytest.approx(0.8)
    assert evaluation.inspection_cost == pytest.approx(0.1)
    assert evaluation.score == pytest.approx(0.7)


def test_direct_commit_has_no_expected_information_gain() -> None:
    planner = ExpectedFreeEnergyPlanner(
        _task(prior_left_good=0.5),
        epistemic_weight=1.0,
    )

    evaluation = planner.evaluate(Policy.COMMIT_LEFT)

    assert evaluation.expected_information_gain == pytest.approx(0.0)


def test_inspection_evaluation_contains_expected_information_gain() -> None:
    task = _task(
        prior_left_good=0.5,
        cue_reliability=0.8,
    )
    planner = ExpectedFreeEnergyPlanner(task, epistemic_weight=1.0)

    evaluation = planner.evaluate(Policy.INSPECT_THEN_COMMIT)

    expected_gain = expected_information_gain(
        prior_left_good=0.5,
        reliability=0.8,
    )

    assert evaluation.expected_information_gain == pytest.approx(expected_gain)
    assert evaluation.score == pytest.approx(
        0.8 + expected_gain - task.config.inspection_cost
    )


def test_preference_only_planner_commits_when_inspection_is_not_worth_its_cost() -> None:
    planner = PreferenceOnlyPlanner(
        _task(
            prior_left_good=0.5,
            cue_reliability=0.6,
            inspection_cost=0.11,
        )
    )

    assert planner.choose_policy() is Policy.COMMIT_LEFT


def test_efe_planner_can_inspect_for_epistemic_value() -> None:
    planner = ExpectedFreeEnergyPlanner(
        _task(
            prior_left_good=0.5,
            cue_reliability=0.6,
            inspection_cost=0.11,
        ),
        epistemic_weight=1.0,
    )

    assert planner.choose_policy() is Policy.INSPECT_THEN_COMMIT


def test_efe_planner_commits_when_the_prior_is_certain() -> None:
    planner = ExpectedFreeEnergyPlanner(
        _task(
            prior_left_good=1.0,
            cue_reliability=0.8,
            inspection_cost=0.0,
        ),
        epistemic_weight=1.0,
    )

    assert planner.choose_policy() is Policy.COMMIT_LEFT


def test_efe_planner_rejects_negative_epistemic_weight() -> None:
    with pytest.raises(ValueError, match="epistemic_weight"):
        ExpectedFreeEnergyPlanner(
            _task(),
            epistemic_weight=-0.1,
        )