"""Tests for the discrete hidden-state cue-and-commit environment."""

import random

import pytest

from aifit.cue_task import (
    CommitAction,
    Cue,
    CueTask,
    CueTaskConfig,
    HiddenState,
    Outcome,
)


def test_config_preserves_explicit_parameters() -> None:
    config = CueTaskConfig(
        prior_left_good=0.65,
        cue_reliability=0.8,
        inspection_cost=0.1,
    )

    assert config.prior_left_good == pytest.approx(0.65)
    assert config.cue_reliability == pytest.approx(0.8)
    assert config.inspection_cost == pytest.approx(0.1)


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {
            "prior_left_good": -0.1,
            "cue_reliability": 0.8,
            "inspection_cost": 0.1,
        },
        {
            "prior_left_good": 1.1,
            "cue_reliability": 0.8,
            "inspection_cost": 0.1,
        },
        {
            "prior_left_good": 0.5,
            "cue_reliability": -0.1,
            "inspection_cost": 0.1,
        },
        {
            "prior_left_good": 0.5,
            "cue_reliability": 1.1,
            "inspection_cost": 0.1,
        },
        {
            "prior_left_good": 0.5,
            "cue_reliability": 0.8,
            "inspection_cost": -0.1,
        },
    ],
)
def test_config_rejects_invalid_parameters(
    config_kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        CueTaskConfig(**config_kwargs)


def test_certain_left_prior_always_samples_left_good() -> None:
    task = CueTask(
        CueTaskConfig(
            prior_left_good=1.0,
            cue_reliability=0.8,
            inspection_cost=0.1,
        )
    )

    hidden_state = task.sample_hidden_state(random.Random(2026))

    assert hidden_state is HiddenState.LEFT_GOOD


def test_certain_right_prior_always_samples_right_good() -> None:
    task = CueTask(
        CueTaskConfig(
            prior_left_good=0.0,
            cue_reliability=0.8,
            inspection_cost=0.1,
        )
    )

    hidden_state = task.sample_hidden_state(random.Random(2026))

    assert hidden_state is HiddenState.RIGHT_GOOD


@pytest.mark.parametrize(
    ("hidden_state", "expected_cue"),
    [
        (HiddenState.LEFT_GOOD, Cue.LEFT),
        (HiddenState.RIGHT_GOOD, Cue.RIGHT),
    ],
)
def test_perfectly_reliable_cue_identifies_hidden_state(
    hidden_state: HiddenState,
    expected_cue: Cue,
) -> None:
    task = CueTask(
        CueTaskConfig(
            prior_left_good=0.5,
            cue_reliability=1.0,
            inspection_cost=0.1,
        )
    )

    cue = task.sample_cue(hidden_state, random.Random(2026))

    assert cue is expected_cue


@pytest.mark.parametrize(
    ("hidden_state", "expected_cue"),
    [
        (HiddenState.LEFT_GOOD, Cue.RIGHT),
        (HiddenState.RIGHT_GOOD, Cue.LEFT),
    ],
)
def test_zero_reliability_cue_is_always_opposite_to_hidden_state(
    hidden_state: HiddenState,
    expected_cue: Cue,
) -> None:
    task = CueTask(
        CueTaskConfig(
            prior_left_good=0.5,
            cue_reliability=0.0,
            inspection_cost=0.1,
        )
    )

    cue = task.sample_cue(hidden_state, random.Random(2026))

    assert cue is expected_cue


@pytest.mark.parametrize(
    ("hidden_state", "action", "expected_outcome"),
    [
        (
            HiddenState.LEFT_GOOD,
            CommitAction.LEFT,
            Outcome.SUCCESS,
        ),
        (
            HiddenState.LEFT_GOOD,
            CommitAction.RIGHT,
            Outcome.FAILURE,
        ),
        (
            HiddenState.RIGHT_GOOD,
            CommitAction.LEFT,
            Outcome.FAILURE,
        ),
        (
            HiddenState.RIGHT_GOOD,
            CommitAction.RIGHT,
            Outcome.SUCCESS,
        ),
    ],
)
def test_commit_outcome_matches_hidden_state(
    hidden_state: HiddenState,
    action: CommitAction,
    expected_outcome: Outcome,
) -> None:
    task = CueTask(
        CueTaskConfig(
            prior_left_good=0.5,
            cue_reliability=0.8,
            inspection_cost=0.1,
        )
    )

    outcome = task.outcome_for_commit(hidden_state, action)

    assert outcome is expected_outcome