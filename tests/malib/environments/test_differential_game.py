import unittest
import pytest

from malib.environments import DifferentialGame
from malib.error import EnvironmentNotFound

class TestDifferentialGame(unittest.TestCase):
    def test_create_game(self):
        game = DifferentialGame("zero_sum", 2)

    def test_create_wrong_name_game(self):
        with pytest.raises(EnvironmentNotFound) as excinfo:
            game = DifferentialGame("abc", 2)

        assert "abc" in str(excinfo.value)
