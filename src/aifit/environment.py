"""Continuous thermal environment for the Chapter 7 simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThermalEnvironment:
    """A one-dimensional world with a temperature gradient."""

    position: float = 20.0
    temperature_intercept: float = 10.0
    temperature_slope: float = 0.6
    process_noise_std: float = 0.02
    observation_noise_std: float = 0.15

    def temperature(self) -> float:
        """Return the noise-free temperature at the current position."""
        return (
            self.temperature_intercept
            + self.temperature_slope * self.position
        )

    def observe(self, rng: np.random.Generator) -> float:
        """Generate a noisy sensory observation."""
        noise = rng.normal(0.0, self.observation_noise_std)
        return self.temperature() + noise

    def step(
        self,
        action: float,
        rng: np.random.Generator,
        disturbance: float = 0.0,
        dt: float = 0.1,
    ) -> float:
        """Advance the hidden environmental state."""
        process_noise = rng.normal(0.0, self.process_noise_std)
        velocity = action + disturbance + process_noise
        self.position += dt * velocity

        return self.position