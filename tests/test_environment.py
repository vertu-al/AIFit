import numpy as np

from aifit.environment import ThermalEnvironment


def test_temperature_increases_with_position() -> None:
    environment = ThermalEnvironment(position=5.0)

    lower_temperature = environment.temperature()

    environment.position = 10.0
    higher_temperature = environment.temperature()

    assert higher_temperature > lower_temperature


def test_negative_action_reduces_position() -> None:
    environment = ThermalEnvironment(
        position=10.0,
        process_noise_std=0.0,
    )
    rng = np.random.default_rng(7)

    environment.step(
        action=-1.0,
        rng=rng,
        dt=0.1,
    )

    assert environment.position < 10.0