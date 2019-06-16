import unittest
from malib.environments import DifferentialGame

class TestDifferentialGame(unittest.TestCase): 
    def test_create_game(self):
        self.game = DifferentialGame("abc", 2)