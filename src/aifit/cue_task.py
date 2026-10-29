"""Discrete hidden-state cue-and-commit environment for Chapter 8."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class HiddenState(str, Enum):
    """Latent state determining which commitment action succeeds."""

    LEFT_GOOD = "left_good"
    RIGHT_GOOD = "right_good"


class Cue(str, Enum):
    """Diagnostic observation returned by an inspection action."""

    LEFT = "left"
    RIGHT = "right"


class CommitAction(str, Enum):
    """Available direct commitment actions."""

    LEFT = "left"
    RIGHT = "right"


class Outcome(str, Enum):
    """Outcome following a commitment action."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class CueTaskConfig:
    """Configuration for a binary hidden-state cue-and-commit task."""

    prior_left_good: float
    cue_reliability: float
    inspection_cost: float

    def __post_init__(self) -> None:
        _validate_probability(self.prior_left_good, name="prior_left_good")
        _validate_probability(self.cue_reliability, name="cue_reliability")

        if self.inspection_cost < 0.0:
            raise ValueError("inspection_cost must be greater than or equal to zero.")


@dataclass(frozen=True)
class CueTask:
    """Sample hidden states and diagnostic cues for the Chapter 8 task."""

    config: CueTaskConfig

    def sample_hidden_state(self, rng: random.Random) -> HiddenState:
        """Sample the latent state according to the configured prior."""
        if rng.random() < self.config.prior_left_good:
            return HiddenState.LEFT_GOOD

        return HiddenState.RIGHT_GOOD

    def sample_cue(
        self,
        hidden_state: HiddenState,
        rng: random.Random,
    ) -> Cue:
        """Sample a noisy cue whose accuracy equals the configured reliability."""
        correct_cue = _cue_for_state(hidden_state)
        incorrect_cue = _opposite_cue(correct_cue)

        if rng.random() < self.config.cue_reliability:
            return correct_cue

        return incorrect_cue

    def outcome_for_commit(
        self,
        hidden_state: HiddenState,
        action: CommitAction,
    ) -> Outcome:
        """Return success exactly when the action matches the hidden state."""
        successful_action = _action_for_state(hidden_state)

        if action is successful_action:
            return Outcome.SUCCESS

        return Outcome.FAILURE


def _validate_probability(probability: float, *, name: str) -> None:
    """Raise ValueError unless a value lies in the closed unit interval."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must lie in the interval [0.0, 1.0].")


def _cue_for_state(hidden_state: HiddenState) -> Cue:
    """Return the cue consistent with a hidden state."""
    if hidden_state is HiddenState.LEFT_GOOD:
        return Cue.LEFT

    return Cue.RIGHT


def _opposite_cue(cue: Cue) -> Cue:
    """Return the other binary cue."""
    if cue is Cue.LEFT:
        return Cue.RIGHT

    return Cue.LEFT


def _action_for_state(hidden_state: HiddenState) -> CommitAction:
    """Return the commitment action that succeeds in a hidden state."""
    if hidden_state is HiddenState.LEFT_GOOD:
        return CommitAction.LEFT

    return CommitAction.RIGHT