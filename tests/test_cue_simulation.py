"""Tests for reproducible Chapter 8 cue-and-commit simulations."""

import random

import pytest

from aifit.cue_simulation import (
    run_episode,
    run_episodes,
    summarize_episodes,
)
from aifit.cue_task import CommitAction, Cue, CueTask, CueTaskConfig, Outcome
from aifit.planning import ExpectedFreeEnergyPlanner, Policy, PreferenceOnlyPlanner


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


def test_direct_commit_episode_records_no_cue_and_a_success() -> None:
    task = _task(prior_left_good=1.0)
    planner = PreferenceOnlyPlanner(task)

    result = run_episode(
        task,
        planner,
        rng=random.Random(2026),
    )

    assert result.selected_policy is Policy.COMMIT_LEFT
    assert result.cue is None
    assert result.posterior_left_good is None
    assert result.commit_action is CommitAction.LEFT
    assert result.outcome is Outcome.SUCCESS
    assert result.net_value == pytest.approx(1.0)


def test_inspection_episode_updates_beliefs_then_commits() -> None:
    task = _task(
        prior_left_good=0.5,
        cue_reliability=1.0,
        inspection_cost=0.1,
    )
    planner = PreferenceOnlyPlanner(task)

    result = run_episode(
        task,
        planner,
        rng=random.Random(2026),
    )

    assert result.selected_policy is Policy.INSPECT_THEN_COMMIT
    assert result.cue is not None
    assert result.posterior_left_good in {0.0, 1.0}
    assert result.outcome is Outcome.SUCCESS
    assert result.net_value == pytest.approx(0.9)

    if result.cue is Cue.LEFT:
        assert result.commit_action is CommitAction.LEFT
    else:
        assert result.commit_action is CommitAction.RIGHT


def test_seeded_episode_sequences_are_reproducible() -> None:
    task = _task()
    planner = ExpectedFreeEnergyPlanner(
        task,
        epistemic_weight=1.0,
    )

    first_run = run_episodes(
        task,
        planner,
        n_episodes=12,
        seed=2026,
    )
    second_run = run_episodes(
        task,
        planner,
        n_episodes=12,
        seed=2026,
    )

    assert first_run == second_run


def test_summary_reports_rates_and_mean_net_value() -> None:
    task = _task(prior_left_good=1.0)
    planner = PreferenceOnlyPlanner(task)

    episodes = run_episodes(
        task,
        planner,
        n_episodes=4,
        seed=2026,
    )
    summary = summarize_episodes(episodes)

    assert summary.n_episodes == 4
    assert summary.success_rate == pytest.approx(1.0)
    assert summary.inspection_rate == pytest.approx(0.0)
    assert summary.mean_net_value == pytest.approx(1.0)


def test_run_episodes_rejects_non_positive_episode_counts() -> None:
    task = _task()
    planner = PreferenceOnlyPlanner(task)

    with pytest.raises(ValueError, match="n_episodes"):
        run_episodes(
            task,
            planner,
            n_episodes=0,
            seed=2026,
        )