"""Tests for aggregated Chapter 8 planner benchmarks."""

import pytest

from aifit.cue_benchmark import BenchmarkConfig, run_benchmark
from aifit.planning import Policy


def _config(
    *,
    prior_left_good: float = 0.5,
    cue_reliability: float = 0.6,
    inspection_cost: float = 0.11,
    epistemic_weight: float = 1.0,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        prior_left_good=prior_left_good,
        cue_reliability=cue_reliability,
        inspection_cost=inspection_cost,
        epistemic_weight=epistemic_weight,
        n_episodes_per_seed=16,
        seeds=(2026, 2027),
    )


def test_benchmark_returns_one_result_for_each_planner() -> None:
    results = run_benchmark(_config())

    assert len(results) == 2
    assert [result.planner_name for result in results] == [
        "preference_only",
        "expected_free_energy",
    ]


def test_benchmark_exposes_epistemic_difference_in_policy_selection() -> None:
    preference_result, efe_result = run_benchmark(_config())

    assert preference_result.selected_policy is Policy.COMMIT_LEFT
    assert efe_result.selected_policy is Policy.INSPECT_THEN_COMMIT
    assert preference_result.mean_inspection_rate == pytest.approx(0.0)
    assert efe_result.mean_inspection_rate == pytest.approx(1.0)


def test_zero_epistemic_weight_matches_preference_only_selection() -> None:
    preference_result, efe_result = run_benchmark(
        _config(epistemic_weight=0.0)
    )

    assert preference_result.selected_policy is Policy.COMMIT_LEFT
    assert efe_result.selected_policy is Policy.COMMIT_LEFT


def test_benchmark_is_reproducible_for_identical_configurations() -> None:
    config = _config()

    assert run_benchmark(config) == run_benchmark(config)


def test_benchmark_reports_bounded_aggregate_metrics() -> None:
    results = run_benchmark(_config())

    for result in results:
        assert 0.0 <= result.mean_success_rate <= 1.0
        assert 0.0 <= result.mean_inspection_rate <= 1.0
        assert result.n_seeds == 2
        assert result.n_episodes_per_seed == 16


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"n_episodes_per_seed": 0},
        {"seeds": ()},
        {"epistemic_weight": -0.1},
    ],
)
def test_benchmark_config_rejects_invalid_control_parameters(
    config_kwargs: dict[str, object],
) -> None:
    baseline_kwargs: dict[str, object] = {
        "prior_left_good": 0.5,
        "cue_reliability": 0.8,
        "inspection_cost": 0.1,
        "epistemic_weight": 1.0,
        "n_episodes_per_seed": 16,
        "seeds": (2026,),
    }

    baseline_kwargs.update(config_kwargs)

    with pytest.raises(ValueError):
        BenchmarkConfig(**baseline_kwargs)