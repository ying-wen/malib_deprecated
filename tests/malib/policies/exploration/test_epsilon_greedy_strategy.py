"""Tests for epsilon greedy strategy."""
import pickle
import unittest

import numpy as np

from malib.policies.explorations import EpsilonGreedyExploration
from tests.malib.environments.dummy import DummyDiscreteEnv


class SimplePolicy:
    """Simple policy for testing."""

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation):
        return self.action_space.sample()

    def get_actions(self, observations):
        return np.full(len(observations), self.action_space.sample())


class TestEpsilonGreedyStrategy(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.env = DummyDiscreteEnv()
        self.policy = SimplePolicy(action_space=self.env.action_space)
        self.epsilon_greedy_strategy = EpsilonGreedyExploration(
            action_space=self.env.action_space,
            total_timesteps=100,
            max_epsilon=1.0,
            min_epsilon=0.02,
            decay_ratio=0.1)

        self.env.reset()

    def test_epsilon_greedy_strategy(self):
        obs, _, _, _ = self.env.step(1)

        action = self.epsilon_greedy_strategy.get_action(0, obs, self.policy)
        assert self.env.action_space.contains(action)

        # epsilon decay by 1 step, new epsilon = 1 - 0.98 = 0.902
        random_rate = np.random.random(
            100000) < self.epsilon_greedy_strategy._epsilon
        assert np.isclose([0.902], [sum(random_rate) / 100000], atol=0.01)

        actions = self.epsilon_greedy_strategy.get_actions(
            0, [obs] * 5, self.policy)

        # epsilon decay by 6 steps in total, new epsilon = 1 - 6 * 0.98 = 0.412
        random_rate = np.random.random(
            100000) < self.epsilon_greedy_strategy._epsilon
        assert np.isclose([0.412], [sum(random_rate) / 100000], atol=0.01)

        for action in actions:
            assert self.env.action_space.contains(action)

    def test_epsilon_greedy_strategy_is_pickleable(self):
        obs, _, _, _ = self.env.step(1)
        for _ in range(5):
            self.epsilon_greedy_strategy.get_action(0, obs, self.policy)

        h_data = pickle.dumps(self.epsilon_greedy_strategy)
        strategy = pickle.loads(h_data)
        assert strategy._epsilon == self.epsilon_greedy_strategy._epsilon
