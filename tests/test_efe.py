"""Tests for discrete expected-free-energy probability primitives."""

import math

import pytest

from aifit.efe import (
    bayes_update_left_good,
    binary_entropy,
    expected_information_gain,
    kl_divergence_binary,
)


def test_binary_entropy_is_maximal_for_an_even_belief() -> None:
    assert binary_entropy(0.5) == pytest.approx(math.log(2.0))


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_binary_entropy_is_zero_for_a_deterministic_belief(
    probability: float,
) -> None:
    assert binary_entropy(probability) == pytest.approx(0.0)


def test_bayes_update_matches_hand_calculation_for_a_left_cue() -> None:
    posterior = bayes_update_left_good(
        prior_left_good=0.5,
        cue_is_left=True,
        reliability=0.8,
    )

    assert posterior == pytest.approx(0.8)


def test_bayes_update_matches_hand_calculation_for_a_right_cue() -> None:
    posterior = bayes_update_left_good(
        prior_left_good=0.5,
        cue_is_left=False,
        reliability=0.8,
    )

    assert posterior == pytest.approx(0.2)


def test_binary_kl_divergence_is_zero_for_identical_beliefs() -> None:
    assert kl_divergence_binary(0.5, 0.5) == pytest.approx(0.0)


def test_binary_kl_divergence_matches_hand_calculation() -> None:
    expected = 0.8 * math.log(0.8 / 0.5) + 0.2 * math.log(0.2 / 0.5)

    assert kl_divergence_binary(0.8, 0.5) == pytest.approx(expected)


def test_uninformative_cue_has_zero_expected_information_gain() -> None:
    information_gain = expected_information_gain(
        prior_left_good=0.5,
        reliability=0.5,
    )

    assert information_gain == pytest.approx(0.0)


def test_expected_information_gain_matches_symmetric_hand_calculation() -> None:
    expected = math.log(2.0) - binary_entropy(0.8)

    information_gain = expected_information_gain(
        prior_left_good=0.5,
        reliability=0.8,
    )

    assert information_gain == pytest.approx(expected)


def test_deterministic_prior_has_zero_expected_information_gain() -> None:
    information_gain = expected_information_gain(
        prior_left_good=1.0,
        reliability=0.8,
    )

    assert information_gain == pytest.approx(0.0)


def test_expected_information_gain_increases_with_cue_reliability() -> None:
    weak_cue_gain = expected_information_gain(
        prior_left_good=0.5,
        reliability=0.6,
    )
    strong_cue_gain = expected_information_gain(
        prior_left_good=0.5,
        reliability=0.8,
    )

    assert strong_cue_gain > weak_cue_gain


@pytest.mark.parametrize(
    ("prior_left_good", "reliability"),
    [
        (-0.1, 0.8),
        (1.1, 0.8),
        (0.5, -0.1),
        (0.5, 1.1),
    ],
)
def test_expected_information_gain_rejects_invalid_probabilities(
    prior_left_good: float,
    reliability: float,
) -> None:
    with pytest.raises(ValueError):
        expected_information_gain(
            prior_left_good=prior_left_good,
            reliability=reliability,
        )