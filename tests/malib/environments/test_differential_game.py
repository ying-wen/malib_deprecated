import pytest

from malib.environments import DifferentialGame
from malib.error import EnvironmentNotFound, WrongNumberOfAgent

def test_create_game():
    game = DifferentialGame("zero_sum", 2)

def test_create_wrong_name_game():
    with pytest.raises(EnvironmentNotFound) as excinfo:
        game = DifferentialGame("abc", 2)

    assert "abc" in str(excinfo.value)

@pytest.mark.parametrize("game_name, real_num_agent", [
    ("zero_sum", 2),
    ("trigonometric", 2),
    ("mataching_pennies", 2),
    ("rotational", 2),
    ("wolf", 2),
    ("ma_softq", 2),
])
def test_wrong_num_agent(game_name, real_num_agent):
    with pytest.raises(WrongNumberOfAgent) as excinfo:
        game = DifferentialGame(game_name, 10)

    assert f"for {game_name} is {real_num_agent}" in str(excinfo.value)
    assert "agent" in str(excinfo.value)
