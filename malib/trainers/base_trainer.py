from abc import ABC, abstractmethod

"""
The trainer for multi-agent training.
"""


class Trainer(ABC):
    """This class implements a multi-agent trainer.
    """

    def __init__(self, env, agents, sampler):
        self.env = env
        self.agents = agents
        self.sampler = sampler
        self.has_setup = False

    @abstractmethod
    def setup(self):
        self.has_setup = True

    def obtain_samples(self, itr, batch_size):
        pass

    def do_communication(self):
        pass

    def individual_forward(self):
        pass

    def centralized_forward(self):
        pass

    def apply_gradient(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def restore(self):
        pass

    @abstractmethod
    def resume(self):
        pass

    def log_diagnostics(self):
        pass
