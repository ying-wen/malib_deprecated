import unittest
from malib.environments import StochasticMatrixGame

class TestStochasticMatrixGame(unittest.TestCase): 
    def test_create_game(self):
        self.game = StochasticMatrixGame("PollutionTax", 2, 2, 2)