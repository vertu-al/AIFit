"""
AIFit

Chapter 7 — Active Generalized Filtering

Simulation of the Bayesian Thermostat.

Perception updates beliefs.
Action changes the world.
Together they minimize prediction error.
"""

from __future__ import annotations

import numpy as np

from aifit.agent import ActiveInferenceAgent
from aifit.environment import ThermalEnvironment
from aifit.plotting import plot_trajectory


def run_simulation(
    steps: int = 500,
    dt: float = 0.1,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Run one Active Inference simulation.

    Returns
    -------
    dict[str, np.ndarray]
        Time series of all simulated variables.
    """

    rng = np.random.default_rng(seed)

    # Environment: the generative process
    environment = ThermalEnvironment(
        position=20.0,
    )

    # Agent: the generative model
    agent = ActiveInferenceAgent(
        position_belief=10.0,
        preferred_temperature=13.0,
    )

    trajectory: dict[str, list[float]] = {
        "time": [],
        "true_position": [],
        "belief": [],
        "observation": [],
        "prediction": [],
        "preferred": [],
        "action": [],
        "free_energy": [],
    }

    # Main action–perception loop
    for step in range(steps):
        # 1. Sense the environment
        observation = environment.observe(rng)

        # 2. Perception: update hidden-state beliefs
        agent.update_belief(
            observation,
            dt=dt,
        )

        # 3. Action begins after the initial perceptual phase
        if step > 100:
            agent.update_action(
                observation,
                dt=dt,
            )
        else:
            agent.action = 0.0

        # 4. Evaluate the current prediction-error objective
        free_energy = agent.free_energy_proxy(observation)

        # 5. Store the trajectory
        trajectory["time"].append(step * dt)
        trajectory["true_position"].append(environment.position)
        trajectory["belief"].append(agent.position_belief)
        trajectory["observation"].append(observation)
        trajectory["prediction"].append(agent.predict_temperature())
        trajectory["preferred"].append(agent.preferred_temperature)
        trajectory["action"].append(agent.action)
        trajectory["free_energy"].append(free_energy)

        # 6. Apply action to the environment
        environment.step(
            action=agent.action,
            rng=rng,
            dt=dt,
        )

    return {
        key: np.asarray(values)
        for key, values in trajectory.items()
    }


if __name__ == "__main__":
    trajectory = run_simulation()
    plot_trajectory(
        trajectory,
        action_start_time=10.0,
    )