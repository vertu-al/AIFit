"""A minimal continuous Active Inference agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActiveInferenceAgent:
    """Infer hidden position and act toward a preferred temperature."""

    position_belief: float = 10.0
    preferred_temperature: float = 13.0

    temperature_intercept: float = 10.0
    temperature_slope: float = 0.6

    sensory_precision: float = 4.0
    belief_learning_rate: float = 0.08
    action_learning_rate: float = 0.45
    action_damping: float = 0.15

    action: float = 0.0

    def predict_temperature(self) -> float:
        """Predict sensory input from the current position belief."""
        return (
            self.temperature_intercept
            + self.temperature_slope * self.position_belief
        )

    def sensory_prediction_error(self, observation: float) -> float:
        """Return observed minus predicted temperature."""
        return observation - self.predict_temperature()

    def preference_error(self, observation: float) -> float:
        """Return preferred minus observed temperature."""
        return self.preferred_temperature - observation

    def update_belief(
        self,
        observation: float,
        dt: float = 0.1,
    ) -> float:
        """Update the hidden-state belief through prediction error."""
        error = self.sensory_prediction_error(observation)

        gradient = (
            self.temperature_slope
            * self.sensory_precision
            * error
        )

        self.position_belief += (
            dt
            * self.belief_learning_rate
            * gradient
        )

        return self.position_belief

    def update_action(
        self,
        observation: float,
        dt: float = 0.1,
    ) -> float:
        """Update action using the predicted sensory consequence of movement."""
        error = self.preference_error(observation)

        action_drive = (
            self.action_learning_rate
            * self.temperature_slope
            * self.sensory_precision
            * error
        )

        self.action += dt * (
            action_drive
            - self.action_damping * self.action
        )

        return self.action

    def free_energy_proxy(self, observation: float) -> float:
        """Return a pedagogical prediction-error objective."""
        sensory_error = self.sensory_prediction_error(observation)
        preference_error = self.preference_error(observation)

        return 0.5 * self.sensory_precision * (
            sensory_error**2
            + preference_error**2
        )