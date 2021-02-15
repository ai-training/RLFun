import random
import cv2
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
FRAME_PER_ACTION = 10

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
        self.game_state = None  # type: flappy_bird.GameState
        random.seed(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Q-networks
        self.local_dqn = DQNModule(IMG_WIDTH, IMG_HEIGHT, NBR_FRAMES).to(self.device)
        self.target_dqn = DQNModule(IMG_WIDTH, IMG_HEIGHT, NBR_FRAMES).to(self.device)
        self.optimizer = Adam(self.local_dqn.parameters(), lr=LR)

        self.buffer = ReplayBuffer(ACTION_SIZE, BUFFER_SIZE, BATCH_SIZE, self.device, seed)
        self.t_step = 0
        self.last_states = ()
        self.last_next_states = ()

        self.random_choice = .9
        self.random_choice_decay = 0.998

    def reset_game(self):
        self.game_state = flappy_bird.GameState()

    def step(self, action, reward, done):
        # Save experience in replay memory
        stacked_states = self.get_state_stack(self.last_states)
        stacked_next_states = self.get_state_stack(self.last_next_states)
        self.buffer.add(stacked_states, action, reward, stacked_next_states, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.buffer) > BATCH_SIZE:
            experiences = self.buffer.sample()
            self.learn(experiences)

    def act(self, action_index) -> tuple:
        action = actions_list[action_index]
        next_state, reward, done = self.game_state.frame_step(action)
        self.last_next_states = self.update_last_states(self.last_next_states, next_state)
        return next_state, reward, done

    def get_action(self, state, eps=0.):
        self.last_states = self.update_last_states(self.last_states, state)
        return self.get_random_action() if random.random() < eps else self.get_policy_action()

    def update_random_choice_weight(self):
        self.random_choice = max(.5, self.random_choice * self.random_choice_decay)

    def get_random_action(self) -> int:
        return np.random.choice(np.arange(ACTION_SIZE), p=[self.random_choice, 1 - self.random_choice])

    def get_policy_action(self) -> int:
        state_stack = self.get_state_stack(self.last_states)
        state_stack = np.array([state_stack])
        state_stack = torch.from_numpy(state_stack).float().to(self.device)
        self.local_dqn.eval()

        with torch.no_grad():
            output = self.local_dqn(state_stack)
            action = np.argmax(output.cpu().data.numpy())

        self.local_dqn.train()
        return action

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_target_next = self.target_dqn(states).detach().max(1)[0].unsqueeze(1)  # [64, 2] (DQN) -> [64, 1]
        # Compute Q targets for current states
        q_target = rewards + (GAMMA * q_target_next * (1 - dones))  # [64, 1]

        # Get expected Q values from local model
        q_expected = self.local_dqn(states).gather(1, actions)  # [64, 2] (DQN) -> [64, 1]

        loss = F.mse_loss(q_target, q_expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.target_dqn.parameters(), self.local_dqn.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    @staticmethod
    def process_state(state):
        state = cv2.resize(state, (80, 80))
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        _, state = cv2.threshold(state, 1, 255, cv2.THRESH_BINARY)
        return state / 255.

    @staticmethod
    def update_last_states(last_states: tuple, new_state) -> tuple:
        processed_state = Agent.process_state(new_state)
        missing_state = NBR_FRAMES - len(last_states)
        if missing_state:
            return last_states + (processed_state,) * missing_state
        else:
            return last_states[1:] + (processed_state,)

    @staticmethod
    def get_state_stack(states: tuple):
        return np.stack(states, axis=0)
