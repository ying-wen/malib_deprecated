import gym
from malib.spaces import Box, MASpace, MAEnvSpec
import numpy as np


class Wrapper:
    r"""Wraps the environment to allow a modular transformation.
    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.
    .. note::
        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.
    """

    def __init__(self, env, agent_num=None, action_space=None, observation_space=None):
        self.env = env
        self.action_space = action_space
        self.observation_space = observation_space
        self.agent_num = agent_num
        if self.agent_num is None:
            if hasattr(self.env, "agent_num"):
                self.agent_num = self.env.agent_num
            if hasattr(self.env, "n"):
                self.agent_num = self.env.n
            if hasattr(self.env, "n_agents"):
                self.agent_num = self.env.n_agents
        # print('malib', self.agent_num, env.get_obs_size(), env, env.action_space, env.observation_sapce)
        self.action_space = self.action_spaces = MASpace(
            tuple(self.action_space for _ in range(self.agent_num))
        )
        if self.observation_space is None:
            obs_dim = self.env.get_obs_size()
            self.observation_spaces = MASpace(
                tuple(
                    gym.spaces.Box(
                        low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32
                    )
                    for _ in range(self.agent_num)
                )
            )
        else:
            self.observation_spaces = MASpace(
                tuple(self.observation_space for _ in range(self.agent_num))
            )
        # if hasattr(self.env, "action_space") and hasattr(self.env, "observation_sapce"):

        #     self.observation_spaces = MASpace(tuple(gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32) for _ in range(self.agent_num)))
        # elif hasattr(self.env, "action_spaces") and hasattr(self.env, "observation_spaces"):
        #     self.action_spaces = self.env.action_spaces
        #     self.observation_spaces = self.env.observation_spaces

        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)
