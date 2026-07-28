"""
AIFit

Chapter 7 — Active Generalized Filtering

Simulation of the Bayesian thermostat.

Perception updates beliefs.
Action changes the world.
Together they minimize prediction error.
"""

from __future__ import annotations

import numpy as np

from aifit.agent import ActiveInferenceAgent
from aifit.config import SimulationConfig
from aifit.environment import ThermalEnvironment
from aifit.plotting import plot_trajectory


def run_simulation(
    config: SimulationConfig | None = None,
) -> dict[str, np.ndarray]:
    """Run the Chapter 7 Active Inference simulation.

    Parameters
    ----------
    config:
        Reproducible simulation configuration. Default values are used
        when no configuration is supplied.

    Returns
    -------
    dict[str, np.ndarray]
        Time series of environmental, perceptual, and action variables.
    """

    config = config or SimulationConfig()
    rng = np.random.default_rng(config.seed)

    # Generative process: the actual environment
    environment = ThermalEnvironment(
        position=config.initial_position,
    )

    # Generative model: the agent's beliefs and preferences
    agent = ActiveInferenceAgent(
        position_belief=config.initial_position_belief,
        preferred_temperature=config.preferred_temperature,
    )

    trajectory: dict[str, list[float]] = {
        "time": [],
        "true_position": [],
        "belief": [],
        "observation": [],
        "prediction": [],
        "preferred": [],
        "sensory_error": [],
        "preference_error": [],
        "action": [],
        "free_energy": [],
    }

    for step in range(config.steps):
        # 1. Sensation
        observation = environment.observe(rng)

        # Errors before belief and action are updated
        sensory_error = agent.sensory_prediction_error(observation)
        preference_error = agent.preference_error(observation)

        # 2. Perception: change beliefs to explain sensory evidence
        agent.update_belief(
            observation,
            dt=config.dt,
        )

        # 3. Action: change the world toward preferred observations
        if step >= config.action_start_step:
            agent.update_action(
                observation,
                dt=config.dt,
            )
        else:
            agent.action = 0.0

        # This remains a pedagogical proxy for now
        free_energy = agent.free_energy_proxy(observation)

        # 4. Record the current trajectory
        trajectory["time"].append(step * config.dt)
        trajectory["true_position"].append(environment.position)
        trajectory["belief"].append(agent.position_belief)
        trajectory["observation"].append(observation)
        trajectory["prediction"].append(agent.predict_temperature())
        trajectory["preferred"].append(agent.preferred_temperature)
        trajectory["sensory_error"].append(sensory_error)
        trajectory["preference_error"].append(preference_error)
        trajectory["action"].append(agent.action)
        trajectory["free_energy"].append(free_energy)

        # 5. Apply action to the generative process
        environment.step(
            action=agent.action,
            rng=rng,
            dt=config.dt,
        )

    return {
        key: np.asarray(values)
        for key, values in trajectory.items()
    }


def main() -> None:
    """Run and plot the default Chapter 7 simulation."""
    config = SimulationConfig()
    trajectory = run_simulation(config)

    plot_trajectory(
        trajectory,
        action_start_time=config.action_start_time,
    )


if __name__ == "__main__":
    main()