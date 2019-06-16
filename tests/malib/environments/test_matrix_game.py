import unittest
from malib.environments import MatrixGame

class TestMatrixGame(unittest.TestCase):
    def test_create_game(self):
        self.game = MatrixGame("zero_sum_nash_0_1", 2, 2)