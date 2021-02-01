from game import wrapped_flappy_bird as game
from models.dqn import DQNModule
import numpy as np
import cv2
import os


print(os.getcwd())


GAME = 'bird'               # the name of the game being played for log files
ACTIONS = 2                 # number of valid actions
GAMMA = 0.99                # decay rate of past observations
OBSERVE = 100000.           # timesteps to observe before training
EXPLORE = 2000000.          # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001      # final value of epsilon
INITIAL_EPSILON = 0.0001    # starting value of epsilon
REPLAY_MEMORY = 50000       # number of previous transitions to remember
BATCH_SIZE = 32                  # size of mini-batch
FRAME_PER_ACTION = 1

IMG_WIDTH = 80
IMG_HEIGHT = 80
NBR_FRAMES = 4              # number of frames per pass to DQN


DQN = DQNModule(80, 80, 4)


# get the first state by doing nothing and preprocess the image to 80x80x4
game_state = game.GameState()
do_nothing = np.zeros(ACTIONS)
do_nothing[0] = 1

go_up = np.zeros(ACTIONS)
go_up[1] = 1

x_t, r_0, terminal = game_state.frame_step(do_nothing)
print(type(x_t))
x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

# saving and loading networks
# ...

# start training
epsilon = INITIAL_EPSILON
t = 0
while "flappy bird" != "angry bird":
    frequency = 15
    action = go_up if t % frequency == 0 else do_nothing
    t += 1
    print('up' if t % frequency == 0 else 'nothing')
    x_t, r_0, terminal = game_state.frame_step(action)

    # choose an action epsilon greedily
    # if t % FRAME_PER_ACTION == 0:
    #     choose action
    # else:
    #     action = do nothing
    # scale down epsilon
    # ...
    # run the selected action and observe next state and reward
    # ...
    # store the transition in D: replay buffer
    # ...
    # only train if done observing (t > OBSERVE)
    # ...
    # save progress every 10000 iterations (save model)
    # print info

    pass
