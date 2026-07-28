"""Plotting utilities for AIFit simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(
    trajectory: dict[str, np.ndarray],
    action_start_time: float = 10.0,
    save_figure: bool = True,
) -> None:
    """Visualize the Chapter 7 action–perception trajectory.

    Parameters
    ----------
    trajectory:
        Dictionary containing the simulated time series.
    action_start_time:
        Time at which action begins.
    save_figure:
        Whether the figure should be saved automatically.
    """

    time = trajectory["time"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 11),
        sharex=True,
    )

    # --------------------------------------------------------------
    # Shared phase markers
    # --------------------------------------------------------------

    for axis in axes:
        axis.axvspan(
            time[0],
            action_start_time,
            alpha=0.08,
            label="Perception only",
        )

        axis.axvspan(
            action_start_time,
            time[-1],
            alpha=0.04,
            label="Perception + action",
        )

        axis.axvline(
            action_start_time,
            linestyle="--",
            linewidth=1.2,
            label="Action on",
        )

    # --------------------------------------------------------------
    # Hidden state
    # --------------------------------------------------------------

    axes[0].plot(
        time,
        trajectory["true_position"],
        label=r"True state $x^*$",
    )

    axes[0].plot(
        time,
        trajectory["belief"],
        label=r"Belief $\mu_x$",
    )

    axes[0].set_ylabel("Position")
    axes[0].set_title(r"Hidden State: $x^*$ and $\mu_x$")
    axes[0].legend()

    # --------------------------------------------------------------
    # Sensory space
    # --------------------------------------------------------------

    axes[1].plot(
        time,
        trajectory["observation"],
        label=r"Observation $y$",
    )

    axes[1].plot(
        time,
        trajectory["prediction"],
        label=r"Prediction $g(\mu_x)$",
    )

    axes[1].plot(
        time,
        trajectory["preferred"],
        linestyle="--",
        label=r"Preference prior",
    )

    axes[1].set_ylabel("Temperature")
    axes[1].set_title(r"Sensory Prediction: $y$ and $g(\mu_x)$")
    axes[1].legend()

    # --------------------------------------------------------------
    # Action and free energy
    # --------------------------------------------------------------

    axes[2].plot(
        time,
        trajectory["action"],
        label=r"Action $a$",
    )

    axes[2].plot(
        time,
        trajectory["free_energy"],
        label=r"Free-energy proxy $\mathcal{F}$",
    )

    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Value")
    axes[2].set_title("Active Inference Control")
    axes[2].legend()

    fig.tight_layout()

    # --------------------------------------------------------------
    # Save the figure
    # --------------------------------------------------------------

    if save_figure:
        figures_dir = Path("figures/chapter07")
        figures_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            figures_dir
            / "figure_7_6_bayesian_thermostat.png"
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Figure saved to: {output_file.resolve()}")

    plt.show()