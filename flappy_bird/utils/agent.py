import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from flappy_bird.models.dqn import DQNModule
from flappy_bird.utils.replay_buffer import ReplayBuffer
from flappy_bird.game import wrapped_flappy_bird as flappy_bird

ACTION_SIZE = 2  # number of valid actions
OBSERVE = 100000.  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.0001  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of mini-batch
FRAME_PER_ACTION = 1

BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

IMG_WIDTH = 80
IMG_HEIGHT = 80
NBR_FRAMES = 4  # number of frames per pass to DQN

do_nothing = np.array([1, 0], dtype=np.float)
go_up = np.array([0, 1], dtype=np.float)
actions_list = [do_nothing, go_up]


class Agent:
    def __init__(self, seed: int = 123):
        self.game_state = None
        random.seed(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Q-networks
        self.local_dqn = DQNModule(IMG_WIDTH, IMG_HEIGHT, NBR_FRAMES)
        self.target_dqn = DQNModule(IMG_WIDTH, IMG_HEIGHT, NBR_FRAMES)
        self.optimizer = Adam(self.local_dqn.parameters(), lr=LR)

        self.buffer = ReplayBuffer(ACTION_SIZE, BUFFER_SIZE, BATCH_SIZE, self.device, seed)
        self.t_step = 0
        self.last_states = []

    def reset_game(self):
        self.game_state = flappy_bird.GameState()

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.buffer.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.buffer) > BATCH_SIZE:
            experiences = self.buffer.sample()
            self.learn(experiences)

    def update_last_steps(self, new_state):
        missing_state = NBR_FRAMES - len(self.last_states)
        if missing_state:
            self.last_states = self.last_states + [new_state] * missing_state
        else:
            self.last_states = self.last_states[1:] + [new_state]

    def act(self, state, eps=0.):
        return self.get_random_action() if random.random() < eps else self.get_policy_action(state)
        # new_state, reward, done = self.game_state.frame_step(actions_list[action_index])

    @staticmethod
    def get_random_action() -> int:
        return random.choice(np.arange(ACTION_SIZE))

    def get_policy_action(self, state) -> int:
        state = torch.from_numpy(state).float().to(self.device)
        self.local_dqn.eval()

        with torch.no_grad():
            output = self.local_dqn(state)
            action = np.argmax(output.cpu().data.numpy())

        self.local_dqn.train()
        return action

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_target_next = self.target_dqn(states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_target = rewards + (GAMMA * q_target_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.local_dqn(states).gather(1, actions)

        loss = F.mse_loss(q_target, q_expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.target_dqn.parameters(), self.local_dqn.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
