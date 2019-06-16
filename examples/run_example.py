# Created by yingwen at 2019-03-16

from malib.samplers.sampler import MASampler
from malib.environments import DifferentialGame
from malib.policies import DeterministicMLPPolicy
from malib.value_functions import MLPValueFunction
from malib.replay_buffers import IndexedReplayBuffer
from malib.policies.explorations.ou_exploration import OUExploration
from malib.agents import MADDPGAgent

from copy import deepcopy
import random
import datetime
import pickle
import os
import numpy as np
import tensorflow as tf

from malib.logger import logger
from malib.logger import CsvOutput
from malib.logger import StdOutput
from malib.logger import TensorBoardOutput
from malib.logger import TextOutput


tf.random.set_seed(48)
np.random.seed(48)
random.seed(48)

checkpoint_directory = "/tmp/training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

agent_setting = 'MADDPG_MADDPG'
game_name = 'ma_softq'
agent_num = 2
batch_size = 64
exploration_step = 1000
env = DifferentialGame(game_name, agent_num)

suffix = game_name
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "log/{}/{}/{}/".format(suffix, agent_setting,current_time)

text_log_file = "{}debug.log".format(log_dir)
tabular_log_file = "{}progress.csv".format(log_dir)

logger.add_output(TextOutput(text_log_file))
logger.add_output(CsvOutput(tabular_log_file))
logger.add_output(TensorBoardOutput(log_dir))
logger.add_output(StdOutput())

agents = []


def get_maddpg_agent(env, agent_id, hidden_layer_sizes, max_replay_buffer_size):
    observation_space = env.env_specs.observation_space[agent_id]
    action_space = env.env_specs.action_space[agent_id]
    return MADDPGAgent(
        env_specs=env.env_specs,
        policy=DeterministicMLPPolicy(
            input_shapes=(observation_space.shape, ),
            output_shape=action_space.shape,
            hidden_layer_sizes=hidden_layer_sizes,
            name='policy_agent_{}'.format(agent_id)
        ),
        qf=MLPValueFunction(
            input_shapes=(observation_space.shape, (env.env_specs.action_space.flat_dim,)),
            output_shape=(1,),
            hidden_layer_sizes=hidden_layer_sizes,
            name='qf_agent_{}'.format(agent_id)
        ),
        replay_buffer=IndexedReplayBuffer(observation_dim=observation_space.shape[0],
                                          action_dim=action_space.shape[0],
                                          opponent_action_dim=env.env_specs.action_space.opponent_flat_dim(agent_id),
                                          max_replay_buffer_size=max_replay_buffer_size
                                          ),
        exploration_strategy=OUExploration(action_space),
        gradient_clipping=10.,
        agent_id=agent_id,
    )

hidden_layer_sizes = (10, 10)
max_replay_buffer_size = 1e5

for i in range(agent_num):
    if 'MADDPG' in agent_setting:
        agent = get_maddpg_agent(env, i, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
    agents.append(agent)


sampler = MASampler(agent_num)

sampler.initialize(env, agents)

for _ in range(1000):
    sampler.sample(explore=True)


def add_target_actions(batch_n, agents, batch_size):
    target_actions_n = []
    for i, agent in enumerate(agents):
        print(batch_n[i]['next_observations'].shape)
        target_actions_n.append(agent.act(batch_n[i]['next_observations'], use_target=True))

    for i in range(len(agents)):
        target_actions = target_actions_n[i]
        opponent_target_actions = np.reshape(np.delete(deepcopy(target_actions_n), i, 0), (batch_size, -1))
        target_actions = np.concatenate((target_actions, opponent_target_actions), 1)
        assert target_actions.shape[0] == batch_size
        batches[i]['target_actions'] = target_actions
    return batch_n


def add_recent_batches(batches, agents, batch_size):
    for batch, agent in zip(batches, agents):
        recent_batch = agent.replay_buffer.recent_batch(batch_size)
        batch['recent_observations'] = recent_batch['observations']
        batch['recent_actions'] = recent_batch['actions']
        batch['recent_opponent_actions'] = recent_batch['opponent_actions']
    return batches


def get_batches(agents, batch_size):
    assert len(agents) > 1
    batches = []
    indices = agents[0].replay_buffer.random_indices(batch_size)
    for agent in agents:
        batch = agent.replay_buffer.batch_by_indices(indices)
        batches.append(batch)
    return batches


for step in range(10000):
    sampler.sample()
    batches = get_batches(agents, batch_size)
    if 'MADDPG' in agent_setting:
        batches = add_target_actions(batches, agents, batch_size)
    for agent, batch in zip(agents, batches):
        loss = agent.train(batch)
        print(loss)


# deserialization example
serialized = pickle.dumps(agents)
deserialized = pickle.loads(serialized)