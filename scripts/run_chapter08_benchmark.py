"""Run the versioned Chapter 8 benchmark and export summary metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from aifit.cue_benchmark import BenchmarkConfig, run_benchmark


def parse_args() -> argparse.Namespace:
    """Parse benchmark configuration and output-directory arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Chapter 8 cue-and-commit benchmark."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a benchmark JSON configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("results/chapter08"),
        type=Path,
        help="Directory in which benchmark artifacts are written.",
    )

    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    """Load one JSON benchmark configuration."""
    with config_path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def benchmark_rows(config: dict[str, Any]) -> list[dict[str, object]]:
    """Run both planners for every configured condition and return CSV rows."""
    seeds = tuple(
        range(
            config["seed_start"],
            config["seed_start"] + config["n_seeds"],
        )
    )
    rows: list[dict[str, object]] = []

    for condition in config["conditions"]:
        benchmark_config = BenchmarkConfig(
            prior_left_good=condition["prior_left_good"],
            cue_reliability=condition["cue_reliability"],
            inspection_cost=condition["inspection_cost"],
            epistemic_weight=config["epistemic_weight"],
            n_episodes_per_seed=config["n_episodes_per_seed"],
            seeds=seeds,
        )

        for result in run_benchmark(benchmark_config):
            rows.append(
                {
                    "condition": condition["name"],
                    "planner": result.planner_name,
                    "selected_policy": result.selected_policy.value,
                    "prior_left_good": condition["prior_left_good"],
                    "cue_reliability": condition["cue_reliability"],
                    "inspection_cost": condition["inspection_cost"],
                    "epistemic_weight": config["epistemic_weight"],
                    "n_seeds": result.n_seeds,
                    "n_episodes_per_seed": result.n_episodes_per_seed,
                    "expected_preference_value": result.expected_preference_value,
                    "expected_information_gain": result.expected_information_gain,
                    "policy_score": result.policy_score,
                    "mean_success_rate": result.mean_success_rate,
                    "mean_inspection_rate": result.mean_inspection_rate,
                    "mean_net_value": result.mean_net_value,
                }
            )

    return rows


def write_summary_metrics(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write benchmark rows to a stable CSV artifact."""
    fieldnames = [
        "condition",
        "planner",
        "selected_policy",
        "prior_left_good",
        "cue_reliability",
        "inspection_cost",
        "epistemic_weight",
        "n_seeds",
        "n_episodes_per_seed",
        "expected_preference_value",
        "expected_information_gain",
        "policy_score",
        "mean_success_rate",
        "mean_inspection_rate",
        "mean_net_value",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    *,
    config_path: Path,
    config: dict[str, Any],
    n_rows: int,
    output_path: Path,
) -> None:
    """Write metadata required to reproduce an exported benchmark result."""
    metadata = {
        "config_file": str(config_path),
        "config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "n_conditions": len(config["conditions"]),
        "n_rows": n_rows,
        "n_seeds": config["n_seeds"],
        "n_episodes_per_seed": config["n_episodes_per_seed"],
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def print_results(rows: list[dict[str, object]]) -> None:
    """Print one compact, readable line per planner-condition result."""
    for row in rows:
        print(
            f"condition={row['condition']} "
            f"planner={row['planner']} "
            f"policy={row['selected_policy']} "
            f"success={row['mean_success_rate']:.3f} "
            f"inspection={row['mean_inspection_rate']:.3f} "
            f"net_value={row['mean_net_value']:.3f}"
        )


def main() -> None:
    """Run all configured conditions and export their summary artifacts."""
    args = parse_args()
    config = load_config(args.config)
    rows = benchmark_rows(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "summary_metrics.csv"
    metadata_path = args.output_dir / "experiment_metadata.json"

    write_summary_metrics(rows, summary_path)
    write_metadata(
        config_path=args.config,
        config=config,
        n_rows=len(rows),
        output_path=metadata_path,
    )
    print_results(rows)
    print(f"saved_to={args.output_dir}")


if __name__ == "__main__":
    main()