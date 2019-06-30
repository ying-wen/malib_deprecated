import pytest

from malib.environments import StochasticMatrixGame
from malib.error import EnvironmentNotFound, WrongNumberOfAgent, \
    WrongNumberOfAction, WrongNumberOfState, WrongActionInputLength

expected_config = [
    ("PollutionTax", 2, 2, 2),
    ("three_matrix_games", 2, 2, 3),
]

def test_create_game():
    game = StochasticMatrixGame("PollutionTax", 2, 2, 2)

def test_create_wrong_name_game():
    with pytest.raises(EnvironmentNotFound) as excinfo:
        game = StochasticMatrixGame("abc", 2, 2, 2)

    assert "abc" in str(excinfo.value)

@pytest.mark.parametrize("game_name, real_num_agent, real_num_action, \
    real_num_state", expected_config)
def test_wrong_num_agent(game_name, real_num_agent, real_num_action, real_num_state):
    with pytest.raises(WrongNumberOfAgent) as excinfo:
        game = StochasticMatrixGame(game_name, 48, real_num_action, real_num_state)

    assert f"for {game_name} is {real_num_agent}" in str(excinfo.value)
    assert "agent" in str(excinfo.value)

@pytest.mark.parametrize("game_name, real_num_agent, real_num_action, \
    real_num_state", expected_config)
def test_wrong_num_action(game_name, real_num_agent, real_num_action, real_num_state):
    with pytest.raises(WrongNumberOfAction) as excinfo:
        game = StochasticMatrixGame(game_name, real_num_agent, 48, real_num_state)

    assert f"for {game_name} is {real_num_action}" in str(excinfo.value)
    assert "action" in str(excinfo.value)

@pytest.mark.parametrize("game_name, real_num_agent, real_num_action, \
    real_num_state", expected_config)
def test_wrong_num_state(game_name, real_num_agent, real_num_action, real_num_state):
    with pytest.raises(WrongNumberOfState) as excinfo:
        game = StochasticMatrixGame(game_name, real_num_agent, real_num_action, 48)

    assert f"for {game_name} is {real_num_state}" in str(excinfo.value)
    assert "state" in str(excinfo.value)

@pytest.mark.parametrize("game_name, real_num_agent, real_num_action, \
    real_num_state", expected_config)
def test_wrong_input_action_length(game_name, real_num_agent, real_num_action, real_num_state):
    game = StochasticMatrixGame(game_name, real_num_agent, real_num_action, real_num_state)

    with pytest.raises(WrongActionInputLength) as excinfo:
        game.step([10] * (real_num_agent + 1))

    assert f"Expected number of actions is {real_num_agent}" in str(excinfo.value)
