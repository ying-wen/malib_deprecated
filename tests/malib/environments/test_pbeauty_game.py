import unittest
import pytest

from malib.environments import PBeautyGame
from malib.error import EnvironmentNotFound, RewardTypeNotFound

class TestPBeautyGame(unittest.TestCase):
    def test_create_game(self):
        self.game = PBeautyGame(2)

    def test_create_wrong_name_game(self):
        with pytest.raises(EnvironmentNotFound) as excinfo:
            game = PBeautyGame(2, 'abc')

        assert "abc" in str(excinfo.value)

    def test_create_wrong_name_reward(self):
        with pytest.raises(RewardTypeNotFound) as excinfo:
            game = PBeautyGame(2, game='pbeauty', reward_type='abc')

        assert "abc" in str(excinfo.value)
