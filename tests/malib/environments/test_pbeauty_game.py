import unittest
import pytest

from malib.environments import PBeautyGame

class TestPBeautyGame(unittest.TestCase):
    def test_create_game(self):
        self.game = PBeautyGame(2)
