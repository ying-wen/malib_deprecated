# Created by yingwen at 2019-03-16

import json
import os
from multiprocessing import Process

from malib.agents.agent_factory import *
from malib.environments import DifferentialGame
from malib.environments.particle import make_particle_env
from malib.logger.utils import set_logger
from malib.samplers.sampler import SingleSampler, MASampler
from malib.trainers import SATrainer, MATrainer
from malib.utils.random import set_seed
import numpy as np


def get_agent_by_type(type_name, i, env, hidden_layer_sizes, max_replay_buffer_size):
    if type_name == "SAC":
        return get_sac_agent(
            env,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
            policy_type="gumble",
        )
    elif type_name == "PR2":
        return get_pr2_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
            policy_type="gumble",
        )
    elif type_name == "PR2S":
        return get_pr2_soft_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
            policy_type="gumble",
        )
    elif type_name == "ROMMEO":
        return get_rommeo_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
            policy_type="gumble",
        )
    elif type_name == "DDPG":
        return get_ddpg_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
            policy_type="gumble",
        )


def train_fixed(seed, agent_setting, game_name="ma_softq"):
    set_seed(seed)
    suffix = f"fixed_play/{game_name}/{agent_setting}/{seed}"

    set_logger(suffix)

    batch_size = 1024
    training_steps = 25 * 60000
    exploration_steps = 2000
    max_replay_buffer_size = 1e5
    hidden_layer_sizes = (100, 100)
    max_path_length = 25

    agent_num = 3
    env = make_particle_env(game_name)
    agents = []
    agent_types = agent_setting.split("_")
    assert len(agent_types) == agent_num
    for i, agent_type in enumerate(agent_types):
        agents.append(
            get_agent_by_type(
                agent_type,
                i,
                env,
                hidden_layer_sizes=hidden_layer_sizes,
                max_replay_buffer_size=max_replay_buffer_size,
            )
        )

    sampler = MASampler(
        agent_num, batch_size=batch_size, max_path_length=max_path_length
    )
    sampler.initialize(env, agents)

    trainer = MATrainer(
        env=env,
        agents=agents,
        sampler=sampler,
        steps=training_steps,
        exploration_steps=exploration_steps,
        training_interval=10,
        extra_experiences=["annealing", "recent_experiences"],
        batch_size=batch_size,
    )

    trainer.run()


def main():
    #  PR2 - empirical estimation of opponent conditional policy
    #  PR2S - soft estimation of opponent conditional policy
    settings = [
        # 'ROMMEO_ROMMEO_ROMMEO',
        # 'PR2S_PR2S_PR2S',
        # 'PR2_PR2_PR2',
        "SAC_SAC_SAC",
        # 'DDPG_DDPG_DDPG',
    ]
    game = "simple_spread"
    for setting in settings:
        seed = 1 + int(23122134 / (3 + 1))
        train_fixed(seed, setting, game)


if __name__ == "__main__":
    main()
