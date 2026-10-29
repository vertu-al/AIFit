"""Probability primitives for discrete expected-free-energy demonstrations."""

from __future__ import annotations

import math


def binary_entropy(probability: float) -> float:
    """Return the Shannon entropy of a binary probability distribution in nats."""
    _validate_probability(probability, name="probability")

    if probability in {0.0, 1.0}:
        return 0.0

    return -probability * math.log(probability) - (
        1.0 - probability
    ) * math.log(1.0 - probability)


def bayes_update_left_good(
    *,
    prior_left_good: float,
    cue_is_left: bool,
    reliability: float,
) -> float:
    """Return P(left_good | cue) for a binary noisy diagnostic cue."""
    _validate_probability(prior_left_good, name="prior_left_good")
    _validate_probability(reliability, name="reliability")

    likelihood_given_left = reliability if cue_is_left else 1.0 - reliability
    likelihood_given_right = 1.0 - reliability if cue_is_left else reliability

    prior_right_good = 1.0 - prior_left_good
    cue_probability = (
        likelihood_given_left * prior_left_good
        + likelihood_given_right * prior_right_good
    )

    if cue_probability == 0.0:
        raise ValueError("The requested cue has zero probability under this model.")

    return likelihood_given_left * prior_left_good / cue_probability


def kl_divergence_binary(
    posterior_left_good: float,
    prior_left_good: float,
) -> float:
    """Return KL(posterior || prior) for binary belief distributions in nats."""
    _validate_probability(posterior_left_good, name="posterior_left_good")
    _validate_probability(prior_left_good, name="prior_left_good")

    posterior_right_good = 1.0 - posterior_left_good
    prior_right_good = 1.0 - prior_left_good

    return _x_log_ratio(posterior_left_good, prior_left_good) + _x_log_ratio(
        posterior_right_good,
        prior_right_good,
    )


def expected_information_gain(
    *,
    prior_left_good: float,
    reliability: float,
) -> float:
    """Return expected information gain from observing one binary cue in nats."""
    _validate_probability(prior_left_good, name="prior_left_good")
    _validate_probability(reliability, name="reliability")

    prior_right_good = 1.0 - prior_left_good
    probability_left_cue = (
        reliability * prior_left_good
        + (1.0 - reliability) * prior_right_good
    )
    probability_right_cue = 1.0 - probability_left_cue

    information_gain = 0.0

    for cue_is_left, cue_probability in (
        (True, probability_left_cue),
        (False, probability_right_cue),
    ):
        if cue_probability == 0.0:
            continue

        posterior_left_good = bayes_update_left_good(
            prior_left_good=prior_left_good,
            cue_is_left=cue_is_left,
            reliability=reliability,
        )
        information_gain += cue_probability * kl_divergence_binary(
            posterior_left_good,
            prior_left_good,
        )

    return information_gain


def _validate_probability(probability: float, *, name: str) -> None:
    """Raise ValueError unless a value is a valid closed-unit-interval probability."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must lie in the interval [0.0, 1.0].")


def _x_log_ratio(value: float, reference: float) -> float:
    """Return value * log(value / reference), handling boundary probabilities."""
    if value == 0.0:
        return 0.0
    if reference == 0.0:
        return math.inf

    return value * math.log(value / reference)