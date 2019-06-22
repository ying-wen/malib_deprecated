import pytest

from malib.environments import MatrixGame
from malib.error import EnvironmentNotFound, WrongNumberOfAgent, WrongNumberOfAction

expected_config = [
    ("coordination_0_0", 2, 2),
    ("coordination_same_action_with_preference", 2, 2),
    ("zero_sum_nash_0_1", 2, 2),
    ("matching_pennies", 2, 2),
    ("matching_pennies_3", 3, 2),
    ("prison_lola", 2, 2),
    ("prison", 2, 2),
    ("stag_hunt", 2, 2),
    ("chicken", 2, 2),
    ("harmony", 2, 2),
    ("wolf_05_05", 2, 2),
    ("climbing", 2, 3),
    ("penalty", 2, 3),
    ("rock_paper_scissors", 2, 3),
]

def test_create_game():
    game = MatrixGame("zero_sum_nash_0_1", 2, 2)

def test_create_wrong_name_game():
    with pytest.raises(EnvironmentNotFound) as excinfo:
        game = MatrixGame("abc", 2, 2)

    assert "abc" in str(excinfo.value)

@pytest.mark.parametrize("game_name, real_num_agent, real_num_action", expected_config)
def test_wrong_num_agent(game_name, real_num_agent, real_num_action):
    with pytest.raises(WrongNumberOfAgent) as excinfo:
        game = MatrixGame(game_name, 10, real_num_action)

    assert f"for {game_name} is {real_num_agent}" in str(excinfo.value)
    assert "agent" in str(excinfo.value)


@pytest.mark.parametrize("game_name, real_num_agent, real_num_action", expected_config)
def test_wrong_num_action(game_name, real_num_agent, real_num_action):
    with pytest.raises(WrongNumberOfAction) as excinfo:
        game = MatrixGame(game_name, real_num_agent, 10)

    assert f"for {game_name} is {real_num_action}" in str(excinfo.value)
    assert "action" in str(excinfo.value)
