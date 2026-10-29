"""Create the tracked summary figure for the Chapter 8 planner benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CONDITION_ORDER = (
    "uncertain_modestly_informative",
    "uncertain_highly_informative",
    "uncertain_uninformative",
    "confident_prior_reliable_cue",
    "uncertain_expensive_inspection",
)

CONDITION_LABELS = {
    "uncertain_modestly_informative": "Uncertain\nmoderate cue",
    "uncertain_highly_informative": "Uncertain\nstrong cue",
    "uncertain_uninformative": "Uncertain\nchance cue",
    "confident_prior_reliable_cue": "Confident\nreliable cue",
    "uncertain_expensive_inspection": "Uncertain\nhigh cost",
}

PLANNER_ORDER = ("preference_only", "expected_free_energy")

PLANNER_LABELS = {
    "preference_only": "Preference-only",
    "expected_free_energy": "Expected free energy",
}

PLANNER_COLORS = {
    "preference_only": "#6C757D",
    "expected_free_energy": "#2A6F97",
}


@dataclass(frozen=True)
class BenchmarkRow:
    """One planner result for one controlled benchmark condition."""

    condition: str
    planner: str
    selected_policy: str
    mean_success_rate: float
    mean_inspection_rate: float
    mean_net_value: float


def parse_arguments() -> argparse.Namespace:
    """Parse paths for the tracked benchmark table and output figure."""
    parser = argparse.ArgumentParser(
        description="Plot the Chapter 8 cue-and-commit benchmark results."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/chapter08/summary_metrics.csv"),
        help="Path to the versioned Chapter 8 benchmark summary table.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/chapter08/figures/chapter08_benchmark_summary.png"),
        help="Path for the generated PNG figure.",
    )
    return parser.parse_args()


def load_benchmark_rows(input_path: Path) -> tuple[BenchmarkRow, ...]:
    """Load the benchmark rows needed for the summary figure."""
    with input_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = tuple(
            BenchmarkRow(
                condition=row["condition"],
                planner=row["planner"],
                selected_policy=row["selected_policy"],
                mean_success_rate=float(row["mean_success_rate"]),
                mean_inspection_rate=float(row["mean_inspection_rate"]),
                mean_net_value=float(row["mean_net_value"]),
            )
            for row in reader
        )

    expected_keys = {
        (condition, planner)
        for condition in CONDITION_ORDER
        for planner in PLANNER_ORDER
    }
    observed_keys = {(row.condition, row.planner) for row in rows}

    if observed_keys != expected_keys:
        raise ValueError(
            "The summary table must contain exactly one row for every "
            "configured condition and planner."
        )

    return rows


def row_for(
    rows: tuple[BenchmarkRow, ...],
    *,
    condition: str,
    planner: str,
) -> BenchmarkRow:
    """Return the unique row for one condition and planner."""
    return next(
        row
        for row in rows
        if row.condition == condition and row.planner == planner
    )


def plot_metric(
    axis: plt.Axes,
    rows: tuple[BenchmarkRow, ...],
    *,
    attribute: str,
    title: str,
    y_label: str,
) -> None:
    """Draw one grouped-bar comparison for a benchmark metric."""
    x_values = np.arange(len(CONDITION_ORDER))
    bar_width = 0.36

    for planner_index, planner in enumerate(PLANNER_ORDER):
        offset = (planner_index - 0.5) * bar_width
        values = [
            getattr(
                row_for(rows, condition=condition, planner=planner),
                attribute,
            )
            for condition in CONDITION_ORDER
        ]

        axis.bar(
            x_values + offset,
            values,
            width=bar_width,
            color=PLANNER_COLORS[planner],
            label=PLANNER_LABELS[planner],
        )

    axis.set_title(title, fontweight="bold")
    axis.set_ylabel(y_label)
    axis.set_xticks(x_values)
    axis.set_xticklabels(
        [CONDITION_LABELS[name] for name in CONDITION_ORDER],
        fontsize=8,
    )
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.3)
    axis.set_axisbelow(True)


def create_figure(rows: tuple[BenchmarkRow, ...], output_path: Path) -> None:
    """Create and save the Chapter 8 benchmark summary figure."""
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharex=True)

    plot_metric(
        axes[0],
        rows,
        attribute="mean_inspection_rate",
        title="Policy selection",
        y_label="Mean inspection rate",
    )
    plot_metric(
        axes[1],
        rows,
        attribute="mean_success_rate",
        title="Task outcome",
        y_label="Mean success rate",
    )
    plot_metric(
        axes[2],
        rows,
        attribute="mean_net_value",
        title="Outcome after inspection cost",
        y_label="Mean net value",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    figure.suptitle(
        "Chapter 8: Expected free energy in a cue-and-commit task",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    figure.text(
        0.5,
        0.01,
        "Each result aggregates 64 seeds x 128 episodes. "
        "An inspection rate of 0 means direct commitment; 1 means "
        "inspect then commit.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.88))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Load versioned results and write the benchmark summary figure."""
    arguments = parse_arguments()
    rows = load_benchmark_rows(arguments.input_csv)
    create_figure(rows, arguments.output_path)
    print(f"saved_to={arguments.output_path}")


if __name__ == "__main__":
    main()