"""
AIFit

Chapter 7 — Active Generalized Filtering

Simulation of the Bayesian Thermostat.

Perception updates beliefs.
Action changes the world.
Together they minimize prediction error.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from aifit.agent import ActiveInferenceAgent
from aifit.environment import ThermalEnvironment


def run_simulation(
    steps: int = 500,
    dt: float = 0.1,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """
    Run one Active Inference simulation.

    Returns
    -------
    dict
        Time series of all simulated variables.
    """

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Environment (the generative process)
    # ------------------------------------------------------------------

    environment = ThermalEnvironment(
        position=20.0,
    )

    # ------------------------------------------------------------------
    # Agent (the generative model)
    # ------------------------------------------------------------------

    agent = ActiveInferenceAgent(
        position_belief=10.0,
        preferred_temperature=13.0,
    )

    # ------------------------------------------------------------------
    # Store the trajectory
    # ------------------------------------------------------------------

    trajectory = {
        "time": [],
        "true_position": [],
        "belief": [],
        "observation": [],
        "prediction": [],
        "preferred": [],
        "action": [],
        "free_energy": [],
    }

    # ------------------------------------------------------------------
    # Main Action–Perception Loop
    # ------------------------------------------------------------------

    for step in range(steps):

        # --------------------------------------------------------------
        # 1. Sense the environment
        # --------------------------------------------------------------

        observation = environment.observe(rng)

        # --------------------------------------------------------------
        # 2. Perception
        # Update hidden-state beliefs
        # --------------------------------------------------------------

        agent.update_belief(
            observation,
            dt=dt,
        )

        # --------------------------------------------------------------
        # 3. Action
        # Turn action on after perception has converged
        # --------------------------------------------------------------

        if step > 100:

            agent.update_action(
                observation,
                dt=dt,
            )

        else:

            agent.action = 0.0

        # --------------------------------------------------------------
        # 4. Compute Free Energy
        # --------------------------------------------------------------

        free_energy = agent.free_energy_proxy(
            observation
        )

        # --------------------------------------------------------------
        # 5. Store trajectory
        # --------------------------------------------------------------

        trajectory["time"].append(step * dt)

        trajectory["true_position"].append(
            environment.position
        )

        trajectory["belief"].append(
            agent.position_belief
        )

        trajectory["observation"].append(
            observation
        )

        trajectory["prediction"].append(
            agent.predict_temperature()
        )

        trajectory["preferred"].append(
            agent.preferred_temperature
        )

        trajectory["action"].append(
            agent.action
        )

        trajectory["free_energy"].append(
            free_energy
        )

        # --------------------------------------------------------------
        # 6. Action changes the world
        # --------------------------------------------------------------

        environment.step(
            action=agent.action,
            rng=rng,
            dt=dt,
        )

    # ------------------------------------------------------------------
    # Convert lists to NumPy arrays
    # ------------------------------------------------------------------

    for key in trajectory:
        trajectory[key] = np.asarray(
            trajectory[key]
        )

    return trajectory


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def plot_trajectory(
    trajectory: dict[str, np.ndarray],
) -> None:
    """
    Visualize the Action–Perception cycle.
    """

    time = trajectory["time"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    # --------------------------------------------------------------
    # Hidden state
    # --------------------------------------------------------------

    axes[0].plot(
        time,
        trajectory["true_position"],
        label="True Position",
    )

    axes[0].plot(
        time,
        trajectory["belief"],
        label="Belief",
    )

    axes[0].set_ylabel("Position")
    axes[0].set_title("Hidden State")
    axes[0].legend()

    # --------------------------------------------------------------
    # Observations
    # --------------------------------------------------------------

    axes[1].plot(
        time,
        trajectory["observation"],
        label="Observation",
    )

    axes[1].plot(
        time,
        trajectory["prediction"],
        label="Prediction",
    )

    axes[1].plot(
        time,
        trajectory["preferred"],
        "--",
        label="Preferred",
    )

    axes[1].set_ylabel("Temperature")
    axes[1].set_title("Sensory Space")
    axes[1].legend()

    # --------------------------------------------------------------
    # Action
    # --------------------------------------------------------------

    axes[2].plot(
        time,
        trajectory["action"],
        label="Action",
    )

    axes[2].plot(
        time,
        trajectory["free_energy"],
        label="Free Energy",
    )

    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Value")
    axes[2].set_title("Action & Free Energy")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":

    trajectory = run_simulation()

    plot_trajectory(trajectory)