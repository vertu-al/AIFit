from aifit.agent import ActiveInferenceAgent


def test_belief_moves_toward_warmer_observation() -> None:
    agent = ActiveInferenceAgent(position_belief=0.0)

    previous_belief = agent.position_belief

    agent.update_belief(
        observation=20.0,
        dt=0.1,
    )

    assert agent.position_belief > previous_belief


def test_action_moves_toward_cooler_preference() -> None:
    agent = ActiveInferenceAgent(
        preferred_temperature=13.0,
    )

    agent.update_action(
        observation=18.0,
        dt=0.1,
    )

    assert agent.action < 0.0