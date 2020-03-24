import unittest
import pytest

from malib.environments import PBeautyGame
from malib.error import EnvironmentNotFound, RewardTypeNotFound, WrongActionInputLength

# The number of agent can be arbitrary
expected_config = [
    ("pbeauty", "abs", 7),
    ("pbeauty", "one", 48),
    ("pbeauty", "sqrt", 5),
    ("pbeauty", "square", 10),
    ("entry", "entry", 46),
]


def test_create_game():
    game = PBeautyGame(2)


def test_create_wrong_name_game():
    with pytest.raises(EnvironmentNotFound) as excinfo:
        game = PBeautyGame(2, "abc")

    assert "abc" in str(excinfo.value)


def test_create_wrong_name_reward():
    with pytest.raises(RewardTypeNotFound) as excinfo:
        game = PBeautyGame(2, game_name="pbeauty", reward_type="abc")

    assert "abc" in str(excinfo.value)


def test_create_wrong_game_name():
    with pytest.raises(EnvironmentNotFound) as excinfo:
        game = PBeautyGame(2, game_name="abc")

    assert "abc" in str(excinfo.value)


@pytest.mark.parametrize("game_name, reward_type, num_agent", expected_config)
def test_wrong_input_action_length(game_name, reward_type, num_agent):
    game = PBeautyGame(num_agent, game_name, reward_type=reward_type)
    with pytest.raises(WrongActionInputLength) as excinfo:
        game.step([10] * (num_agent + 1))

    assert f"Expected number of actions is {num_agent}" in str(excinfo.value)
