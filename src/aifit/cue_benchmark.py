"""Aggregate benchmarks for Chapter 8 policy planners."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from aifit.cue_simulation import (
    PolicyPlanner,
    run_episodes,
    summarize_episodes,
)
from aifit.cue_task import CueTask, CueTaskConfig
from aifit.planning import (
    ExpectedFreeEnergyPlanner,
    Policy,
    PreferenceOnlyPlanner,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration shared by both Chapter 8 planners."""

    prior_left_good: float
    cue_reliability: float
    inspection_cost: float
    epistemic_weight: float
    n_episodes_per_seed: int
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        CueTaskConfig(
            prior_left_good=self.prior_left_good,
            cue_reliability=self.cue_reliability,
            inspection_cost=self.inspection_cost,
        )

        if self.epistemic_weight < 0.0:
            raise ValueError(
                "epistemic_weight must be greater than or equal to zero."
            )
        if self.n_episodes_per_seed < 1:
            raise ValueError(
                "n_episodes_per_seed must be greater than or equal to one."
            )
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed.")


@dataclass(frozen=True)
class PlannerBenchmarkResult:
    """Aggregate result for one planner under one task configuration."""

    planner_name: str
    selected_policy: Policy
    expected_preference_value: float
    expected_information_gain: float
    policy_score: float
    n_seeds: int
    n_episodes_per_seed: int
    mean_success_rate: float
    mean_inspection_rate: float
    mean_net_value: float


def run_benchmark(
    config: BenchmarkConfig,
) -> tuple[PlannerBenchmarkResult, PlannerBenchmarkResult]:
    """Evaluate preference-only and EFE planners under the same configuration."""
    task = CueTask(
        CueTaskConfig(
            prior_left_good=config.prior_left_good,
            cue_reliability=config.cue_reliability,
            inspection_cost=config.inspection_cost,
        )
    )
    preference_only = PreferenceOnlyPlanner(task)
    expected_free_energy = ExpectedFreeEnergyPlanner(
        task,
        epistemic_weight=config.epistemic_weight,
    )

    return (
        _run_planner_benchmark(
            planner_name="preference_only",
            task=task,
            planner=preference_only,
            config=config,
        ),
        _run_planner_benchmark(
            planner_name="expected_free_energy",
            task=task,
            planner=expected_free_energy,
            config=config,
        ),
    )


def _run_planner_benchmark(
    *,
    planner_name: str,
    task: CueTask,
    planner: PolicyPlanner,
    config: BenchmarkConfig,
) -> PlannerBenchmarkResult:
    selected_policy = planner.choose_policy()
    evaluation = planner.evaluate(selected_policy)

    summaries = tuple(
        summarize_episodes(
            run_episodes(
                task,
                planner,
                n_episodes=config.n_episodes_per_seed,
                seed=seed,
            )
        )
        for seed in config.seeds
    )

    return PlannerBenchmarkResult(
        planner_name=planner_name,
        selected_policy=selected_policy,
        expected_preference_value=evaluation.expected_preference_value,
        expected_information_gain=evaluation.expected_information_gain,
        policy_score=evaluation.score,
        n_seeds=len(config.seeds),
        n_episodes_per_seed=config.n_episodes_per_seed,
        mean_success_rate=fmean(
            summary.success_rate for summary in summaries
        ),
        mean_inspection_rate=fmean(
            summary.inspection_rate for summary in summaries
        ),
        mean_net_value=fmean(
            summary.mean_net_value for summary in summaries
        ),
    )