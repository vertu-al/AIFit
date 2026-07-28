"""Tests for the complete Chapter 7 simulation."""

import numpy as np
import pytest

from aifit.config import SimulationConfig
from aifit.simulation import run_simulation


def test_simulation_returns_requested_number_of_steps() -> None:
    config = SimulationConfig(
        steps=50,
        action_start_step=10,
    )

    trajectory = run_simulation(config)

    for values in trajectory.values():
        assert len(values) == 50


def test_action_is_zero_before_action_start() -> None:
    config = SimulationConfig(
        steps=50,
        action_start_step=10,
    )

    trajectory = run_simulation(config)

    assert np.allclose(
        trajectory["action"][:10],
        0.0,
    )


def test_action_becomes_active_after_action_start() -> None:
    config = SimulationConfig(
        steps=100,
        action_start_step=10,
    )

    trajectory = run_simulation(config)

    assert np.any(
        np.abs(trajectory["action"][10:]) > 0.0
    )


def test_temperature_approaches_preferred_temperature() -> None:
    config = SimulationConfig(
        steps=700,
        action_start_step=100,
    )

    trajectory = run_simulation(config)

    preferred = trajectory["preferred"][-1]

    initial_error = abs(
        trajectory["observation"][100] - preferred
    )

    final_error = abs(
        np.mean(trajectory["observation"][-30:]) - preferred
    )

    assert final_error < initial_error


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(
            steps=100,
            action_start_step=100,
        )