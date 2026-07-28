"""Configuration objects for reproducible AIFit simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the Chapter 7 Bayesian thermostat.

    The frozen dataclass prevents simulation parameters from changing
    accidentally while a simulation is running.
    """

    steps: int = 500
    dt: float = 0.1
    seed: int = 7

    action_start_step: int = 100

    initial_position: float = 20.0
    initial_position_belief: float = 10.0
    preferred_temperature: float = 13.0

    def __post_init__(self) -> None:
        """Validate the simulation parameters."""
        if self.steps <= 0:
            raise ValueError("steps must be greater than zero")

        if self.dt <= 0:
            raise ValueError("dt must be greater than zero")

        if not 0 <= self.action_start_step < self.steps:
            raise ValueError(
                "action_start_step must be between zero and steps - 1"
            )

    @property
    def action_start_time(self) -> float:
        """Return the time at which action becomes active."""
        return self.action_start_step * self.dt