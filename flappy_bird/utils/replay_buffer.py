from collections import deque, namedtuple
import random
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self,
                 action_size: int,
                 buffer_size: int,
                 batch_size: int,
                 device,
                 seed: int = 123):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def get_tensor(self, experiences, extract_fn):
        return torch\
            .from_numpy(np.vstack([extract_fn(e) for e in experiences if e is not None]))\
            .float()\
            .to(self.device)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = self.get_tensor(experiences, lambda x: x.state)
        actions = self.get_tensor(experiences, lambda x: x.action)
        rewards = self.get_tensor(experiences, lambda x: x.reward)
        next_states = self.get_tensor(experiences, lambda x: x.next_state)
        dones = self.get_tensor(experiences, lambda x: x.done)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
