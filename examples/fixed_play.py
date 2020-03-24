# Created by yingwen at 2019-03-16
from multiprocessing import Process

from malib.agents.agent_factory import *
from malib.environments import DifferentialGame
from malib.logger.utils import set_logger
from malib.samplers.sampler import MASampler
from malib.trainers import MATrainer
from malib.utils.random import set_seed


def get_agent_by_type(type_name, i, env, hidden_layer_sizes, max_replay_buffer_size):
    if type_name == "SAC":
        return get_sac_agent(
            env,
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
    elif type_name == "ROMMEO-UNI":
        return get_rommeo_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
            uniform=True,
        )
    elif type_name == "DDPG-OM":
        return get_ddpgom_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
        )
    elif type_name == "DDPG-TOM":
        return get_ddpgtom_agent(
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
    elif type_name == "MADDPG":
        return get_maddpg_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
        )
    elif type_name == "MFAC":
        return get_maddpg_agent(
            env,
            agent_id=i,
            hidden_layer_sizes=hidden_layer_sizes,
            max_replay_buffer_size=max_replay_buffer_size,
        )


def train_fixed(seed, agent_setting, game_name="ma_softq"):
    set_seed(seed)

    suffix = f"fixed_play1/{game_name}/{agent_setting}/{seed}"

    set_logger(suffix)

    batch_size = 512
    training_steps = 2000
    exploration_steps = 100
    max_replay_buffer_size = 1e5
    hidden_layer_sizes = (128, 128)
    max_path_length = 1

    agent_num = 2
    env = DifferentialGame(game_name, agent_num)

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
        training_interval=1,
        extra_experiences=["annealing", "recent_experiences"],
        batch_size=batch_size,
    )

    trainer.run()


def main():
    settings = [
        "ROMMEO_ROMMEO",
    ]
    game = "ma_softq"
    for setting in settings:
        processes = []
        for e in range(1):
            seed = 1 + int(23122134 / (e + 1))

            def train_func():
                train_fixed(seed, setting, game)

            #   # # Awkward hacky process runs, because Tensorflow does not like
            p = Process(target=train_func, args=tuple())
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
