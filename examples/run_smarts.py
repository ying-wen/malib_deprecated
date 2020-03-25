# Created by yingwen at 2019-03-16

import json
import os
from multiprocessing import Process
import gym
import argparse

from malib.agents.agent_factory import *

# from malib.environments import DifferentialGame
from malib.logger.utils import set_logger
from malib.samplers.sampler import SingleSampler, MASampler, SingleSampler
from malib.trainers import SATrainer, MATrainer
from malib.utils.random import set_seed
from gym_hiway.env.list_hiway_env import ListHiWayEnv
from malib.environments.wrappers import Wrapper
import numpy as np
from hiway.agent import Agent, AgentType

agent, observation_space, action_space = Agent.from_type(
    AgentType.Standard, max_episode_steps=5000
)


def get_agent_by_type(type_name, i, env, hidden_layer_sizes, max_replay_buffer_size):
    if type_name == "SAC":
        return get_sac_agent(
            env,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
        )
    elif type_name == "PR2":
        return get_pr2_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
        )
    elif type_name == "PR2S":
        return get_pr2_soft_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
        )
    elif type_name == "ROMMEO":
        return get_rommeo_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
        )
    elif type_name == "DDPG":
        return get_ddpg_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
        )
    elif type_name == "BiCNet":
        return get_bicnet_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
        )
    elif type_name == "CommNet":
        return get_commnet_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
        )
    elif type_name == "MADDPG":
        return get_maddpg_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
        )
    elif type_name == "MFAC":
        return get_mfac_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
        )


def train_fixed(seed, agent_setting, env_configs, fullly_centralized):
    set_seed(seed)
    scenario = env_configs["scenario"]
    suffix = f"fixed_play/{scenario}/{agent_setting}/{seed}"

    set_logger(suffix)

    batch_size = 50
    training_steps = 25 * 60000
    exploration_steps = 100
    max_replay_buffer_size = 1e5
    hidden_layer_sizes = (100, 100)
    max_path_length = 25

    agent_num = env_configs["n_agents"]
    raw_env = ListHiWayEnv(env_configs)
    env = Wrapper(env=raw_env, action_space=raw_env.action_sapce)
    
    if fullly_centralized:
        agent = get_agent_by_type(
                    agent_setting,
                    0,
                    env,
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_replay_buffer_size=max_replay_buffer_size,
                )

        sampler = SingleSampler(
            batch_size=batch_size, max_path_length=max_path_length
        )
        sampler.initialize(env, agent)
        extra_experiences = ["target_actions"]
        trainer = SATrainer(
            env=env,
            agent=agent,
            sampler=sampler,
            steps=training_steps,
            exploration_steps=exploration_steps,
            training_interval=10,
            extra_experiences=extra_experiences,
            batch_size=batch_size,
        )
    else:
        agents = []
        for i in range(agent_num):
            agents.append(
                get_agent_by_type(
                    agent_setting,
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
        extra_experiences = ["annealing", "recent_experiences", "target_actions"]
        trainer = MATrainer(
            env=env,
            agents=agents,
            sampler=sampler,
            steps=training_steps,
            exploration_steps=exploration_steps,
            training_interval=10,
            extra_experiences=extra_experiences,
            batch_size=batch_size,
        )

    trainer.run()

def action_fn(action):
    action[0:2] = (action[0:2] + 1.0) / 2.0
    return action

def main(args):
    env_configs = {
        "scenario": args.scenario,
        "n_agents": args.n_agents,
        "headless": args.headless,
        "episode_limit": args.episode_limit,
        "visdom": False,
        "timestep_sec": 0.1,
        "action_function": action_fn,
        "action_space": gym.spaces.Box(
            low=np.array([0, 0, -1]), high=np.array([1, 1, 1]), dtype=np.float32
        ),
        "agent_type": AgentType.Standard,
        "algo": args.algo
    }
    seed = 1 + int(23122134 / (3 + 1))
    fullly_centralized = False
    if args.algo in ['BiCNet','CommNet']:
        fullly_centralized = True
    print(env_configs)
    train_fixed(seed, args.algo, env_configs, fullly_centralized=fullly_centralized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hiway-malib-agent-example")
    parser.add_argument(
        "--scenario",
        default="./scenarios/loop",
        help="Either the scenario to run (see scenarios/ for some samples you can use) OR a directory of scenarios to sample from",
        type=str,
    )
    parser.add_argument(
        "--n_agents",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--headless",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--episode_limit",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--algo",
        default='MADDPG',
        help="MADDPG, DDPG, PR2, ROMMEO, BiCNet, ROMMEO, MFAC, SAC,  etc.",
        type=str,
    )
    args = parser.parse_args()
    main(args)
