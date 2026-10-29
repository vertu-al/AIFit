from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from aifit.cue_task import CommitAction, Cue, CueTask, HiddenState, Outcome
from aifit.efe import bayes_update_left_good
from aifit.planning import Policy, PolicyEvaluation


class PolicyPlanner(Protocol):
    """Shared interface implemented by Chapter 8 policy planners."""

    def evaluate(self, policy: Policy) -> PolicyEvaluation:
        """Return transparent score components for a policy."""

    def choose_policy(self) -> Policy:
        """Return the planner's selected policy."""


@dataclass(frozen=True)
class EpisodeResult:
    """Trace of one sampled cue-and-commit episode."""

    selected_policy: Policy
    hidden_state: HiddenState
    cue: Cue | None
    posterior_left_good: float | None
    commit_action: CommitAction
    outcome: Outcome
    expected_preference_value: float
    expected_information_gain: float
    policy_score: float
    inspection_cost: float
    net_value: float


@dataclass(frozen=True)
class SimulationSummary:
    """Aggregate metrics for a sequence of Chapter 8 episodes."""

    n_episodes: int
    success_rate: float
    inspection_rate: float
    mean_net_value: float


def run_episode(
    task: CueTask,
    planner: PolicyPlanner,
    *,
    rng: random.Random,
) -> EpisodeResult:
    """Sample one episode after evaluating and selecting a policy."""
    selected_policy = planner.choose_policy()
    evaluation = planner.evaluate(selected_policy)
    hidden_state = task.sample_hidden_state(rng)

    cue: Cue | None = None
    posterior_left_good: float | None = None

    if selected_policy is Policy.COMMIT_LEFT:
        commit_action = CommitAction.LEFT
    elif selected_policy is Policy.COMMIT_RIGHT:
        commit_action = CommitAction.RIGHT
    else:
        cue = task.sample_cue(hidden_state, rng)
        posterior_left_good = bayes_update_left_good(
            prior_left_good=task.config.prior_left_good,
            cue_is_left=cue is Cue.LEFT,
            reliability=task.config.cue_reliability,
        )
        commit_action = (
            CommitAction.LEFT
            if posterior_left_good >= 0.5
            else CommitAction.RIGHT
        )

    outcome = task.outcome_for_commit(hidden_state, commit_action)
    outcome_value = 1.0 if outcome is Outcome.SUCCESS else 0.0
    net_value = outcome_value - evaluation.inspection_cost

    return EpisodeResult(
        selected_policy=selected_policy,
        hidden_state=hidden_state,
        cue=cue,
        posterior_left_good=posterior_left_good,
        commit_action=commit_action,
        outcome=outcome,
        expected_preference_value=evaluation.expected_preference_value,
        expected_information_gain=evaluation.expected_information_gain,
        policy_score=evaluation.score,
        inspection_cost=evaluation.inspection_cost,
        net_value=net_value,
    )


def run_episodes(
    task: CueTask,
    planner: PolicyPlanner,
    *,
    n_episodes: int,
    seed: int,
) -> tuple[EpisodeResult, ...]:
    """Run a reproducible sequence of independently sampled episodes."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be greater than or equal to one.")

    rng = random.Random(seed)

    return tuple(
        run_episode(task, planner, rng=rng)
        for _ in range(n_episodes)
    )


def summarize_episodes(
    episodes: Sequence[EpisodeResult],
) -> SimulationSummary:
    """Return success, inspection, and net-value metrics for episode traces."""
    if not episodes:
        raise ValueError("episodes must contain at least one episode result.")

    n_episodes = len(episodes)
    successes = sum(
        result.outcome is Outcome.SUCCESS
        for result in episodes
    )
    inspections = sum(
        result.selected_policy is Policy.INSPECT_THEN_COMMIT
        for result in episodes
    )
    mean_net_value = sum(result.net_value for result in episodes) / n_episodes

    return SimulationSummary(
        n_episodes=n_episodes,
        success_rate=successes / n_episodes,
        inspection_rate=inspections / n_episodes,
        mean_net_value=mean_net_value,
    )