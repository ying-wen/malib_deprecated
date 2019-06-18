import unittest
import pytest

from malib.environments import MatrixGame  
from malib.error import EnvironmentNotFound

class TestMatrixGame(unittest.TestCase):
    def test_create_game(self):
        game = MatrixGame("zero_sum_nash_0_1", 2, 2)

    def test_create_wrong_name_game(self):
        with pytest.raises(EnvironmentNotFound) as excinfo:
            game = MatrixGame("abc", 2, 2)

        assert "abc" in str(excinfo.value)
