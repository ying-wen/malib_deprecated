# Created by yingwen at 2019-03-16

from malib.agents.agent_factory import *
from malib.samplers.sampler import MASampler
from malib.environments import DifferentialGame
from malib.logger.utils import set_logger
from malib.utils.random import set_seed
from malib.trainers import MATrainer

set_seed(0)

agent_setting = 'MADDPG'
game_name = 'ma_softq'
suffix = f'{game_name}/{agent_setting}'

set_logger(suffix)

agent_num = 2
batch_size = 128
training_steps = 10000
exploration_step = 1000
hidden_layer_sizes = (10, 10)
max_replay_buffer_size = 1e5

env = DifferentialGame(game_name, agent_num)
agents = []
for i in range(agent_num):
    agent = get_maddpg_agent(env, i, hidden_layer_sizes=hidden_layer_sizes, max_replay_buffer_size=max_replay_buffer_size)
    agents.append(agent)

sampler = MASampler(agent_num)
sampler.initialize(env, agents)

trainer = MATrainer(
    env=env, agents=agents, sampler=sampler,
    steps=training_steps, exploration_steps=exploration_step,
    extra_experiences=['target_actions'],
)

trainer.run()

