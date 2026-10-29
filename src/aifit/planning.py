"""Policy evaluation for the Chapter 8 cue-and-commit task."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from aifit.cue_task import CueTask
from aifit.efe import bayes_update_left_good, expected_information_gain


class Policy(str, Enum):
    """Policies available at the start of a cue-and-commit episode."""

    COMMIT_LEFT = "commit_left"
    COMMIT_RIGHT = "commit_right"
    INSPECT_THEN_COMMIT = "inspect_then_commit"


@dataclass(frozen=True)
class PolicyEvaluation:
    """Transparent components of a policy score."""

    policy: Policy
    expected_preference_value: float
    expected_information_gain: float
    inspection_cost: float
    score: float


class PreferenceOnlyPlanner:
    """Choose the policy with the highest expected preference value."""

    def __init__(self, task: CueTask) -> None:
        self.task = task

    def evaluate(self, policy: Policy) -> PolicyEvaluation:
        """Evaluate a policy without assigning intrinsic value to information."""
        return _evaluate_policy(
            task=self.task,
            policy=policy,
            epistemic_weight=0.0,
        )

    def choose_policy(self) -> Policy:
        """Return the highest-scoring policy with deterministic tie-breaking."""
        return _choose_highest_scoring_policy(self.evaluate)


class ExpectedFreeEnergyPlanner:
    """Choose policies using preference value and expected information gain."""

    def __init__(
        self,
        task: CueTask,
        *,
        epistemic_weight: float,
    ) -> None:
        if epistemic_weight < 0.0:
            raise ValueError("epistemic_weight must be greater than or equal to zero.")

        self.task = task
        self.epistemic_weight = epistemic_weight

    def evaluate(self, policy: Policy) -> PolicyEvaluation:
        """Evaluate a policy using the Chapter 8 EFE-style score."""
        return _evaluate_policy(
            task=self.task,
            policy=policy,
            epistemic_weight=self.epistemic_weight,
        )

    def choose_policy(self) -> Policy:
        """Return the lowest-cost, highest-value policy under the EFE-style score."""
        return _choose_highest_scoring_policy(self.evaluate)


def _evaluate_policy(
    *,
    task: CueTask,
    policy: Policy,
    epistemic_weight: float,
) -> PolicyEvaluation:
    expected_preference_value = _expected_preference_value(task, policy)
    expected_gain = _expected_information_gain(task, policy)
    inspection_cost = _inspection_cost(task, policy)
    score = (
        expected_preference_value
        + epistemic_weight * expected_gain
        - inspection_cost
    )

    return PolicyEvaluation(
        policy=policy,
        expected_preference_value=expected_preference_value,
        expected_information_gain=expected_gain,
        inspection_cost=inspection_cost,
        score=score,
    )


def _choose_highest_scoring_policy(
    evaluate: callable,
) -> Policy:
    policies = (
        Policy.COMMIT_LEFT,
        Policy.COMMIT_RIGHT,
        Policy.INSPECT_THEN_COMMIT,
    )

    return max(
        policies,
        key=lambda policy: evaluate(policy).score,
    )


def _expected_preference_value(task: CueTask, policy: Policy) -> float:
    prior_left_good = task.config.prior_left_good

    if policy is Policy.COMMIT_LEFT:
        return prior_left_good

    if policy is Policy.COMMIT_RIGHT:
        return 1.0 - prior_left_good

    return _expected_success_after_inspection(task)


def _expected_information_gain(task: CueTask, policy: Policy) -> float:
    if policy is not Policy.INSPECT_THEN_COMMIT:
        return 0.0

    return expected_information_gain(
        prior_left_good=task.config.prior_left_good,
        reliability=task.config.cue_reliability,
    )


def _inspection_cost(task: CueTask, policy: Policy) -> float:
    if policy is Policy.INSPECT_THEN_COMMIT:
        return task.config.inspection_cost

    return 0.0


def _expected_success_after_inspection(task: CueTask) -> float:
    prior_left_good = task.config.prior_left_good
    reliability = task.config.cue_reliability
    prior_right_good = 1.0 - prior_left_good

    probability_left_cue = (
        reliability * prior_left_good
        + (1.0 - reliability) * prior_right_good
    )
    probability_right_cue = 1.0 - probability_left_cue

    posterior_after_left_cue = bayes_update_left_good(
        prior_left_good=prior_left_good,
        cue_is_left=True,
        reliability=reliability,
    )
    posterior_after_right_cue = bayes_update_left_good(
        prior_left_good=prior_left_good,
        cue_is_left=False,
        reliability=reliability,
    )

    success_after_left_cue = max(
        posterior_after_left_cue,
        1.0 - posterior_after_left_cue,
    )
    success_after_right_cue = max(
        posterior_after_right_cue,
        1.0 - posterior_after_right_cue,
    )

    return (
        probability_left_cue * success_after_left_cue
        + probability_right_cue * success_after_right_cue
    )