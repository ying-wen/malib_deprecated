"""
The trainer for multi-agent training.
"""
import pickle
from malib.trainers.utils import *


class MATrainer:
    """This class implements a multi-agent trainer.
    """
    def __init__(
            self, env, agents, sampler,
            batch_size=128,
            steps=10000,
            exploration_steps=100,
            training_interval=1,
            extra_experiences=['target_actions'],
            save_path=None,
            is_rommeo=False
    ):
        self.env = env
        self.agents = agents
        self.sampler = sampler
        self.batch_size = batch_size
        self.steps = steps
        self.exploration_steps = exploration_steps
        self.training_interval = training_interval
        self.extra_experiences = extra_experiences
        self.losses = []
        self.save_path = save_path

        self.rommeo = is_rommeo

    def setup(self, env, agents, sampler):
        self.env = env
        self.agents = agents
        self.sampler = sampler

    def sample_batches(self):
        assert len(self.agents) > 1
        batches = []
        indices = self.agents[0].replay_buffer.random_indices(self.batch_size)
        for agent in self.agents:
            batch = agent.replay_buffer.batch_by_indices(indices)
            batches.append(batch)
        return batches

    def do_communication(self):
        pass

    def individual_forward(self):
        pass

    def centralized_forward(self):
        pass

    def apply_gradient(self):
        pass

    def run(self):
        for step in range(self.steps):
            if step < self.exploration_steps:
                self.sampler.sample(explore=True)
                continue

            self.sampler.sample()
            if step % self.training_interval == 0:
                batches = self.sample_batches()
                for extra_experience in self.extra_experiences:
                    if extra_experience == 'annealing':
                        batches = add_annealing(batches, step - self.exploration_steps, annealing_scale=1.)
                        # print('annealing', batches[0]['annealing'])
                    elif extra_experience == 'target_actions':
                        batches = add_target_actions(batches, self.agents, self.batch_size)
                    elif extra_experience == 'recent_experiences':
                        batches = add_recent_batches(batches, self.agents, self.batch_size)
                agents_losses = []

                for i, (agent, batch) in enumerate(zip(self.agents, batches)):
                    agent_losses = agent.train(batch)
                    agents_losses.append(agent_losses)
                self.losses.append(agents_losses)
                print('agent 1', self.losses[-1][0])
                print('agent 2', self.losses[-1][1])

    def save(self):
        if self.save_path is None:
            self.save_path = '/tmp/agents.pickle'
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.agents, f, pickle.HIGHEST_PROTOCOL)

    def restore(self, restore_path):
        with open(restore_path, 'rb') as f:
            self.agents = pickle.load(f)

    def resume(self):
        pass

    def log_diagnostics(self):
        pass
