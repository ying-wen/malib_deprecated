import unittest
import pytest

from malib.environments import StochasticMatrixGame   
from malib.error import EnvironmentNotFound

class TestStochasticMatrixGame(unittest.TestCase):
    def test_create_game(self):
        self.game = StochasticMatrixGame("PollutionTax", 2, 2, 2)

    def test_create_wrong_name_game(self):
        with pytest.raises(EnvironmentNotFound) as excinfo:
            game = StochasticMatrixGame("abc", 2, 2, 2)

        assert "abc" in str(excinfo.value)
