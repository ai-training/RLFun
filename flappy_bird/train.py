import torch
import numpy as np
from collections import deque
from flappy_bird.utils.agent import Agent

agent = Agent()
do_nothing = np.array([1, 0], dtype=np.float)


def train(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(n_episodes):
        agent.reset_game()
        if i_episode >= 100:
            agent.update_random_choice_weight()
        state, reward, done = agent.game_state.frame_step(do_nothing)
        score = 0
        for t in range(max_t):
            action = agent.get_action(state, eps)
            next_state, reward, done = agent.act(action)
            agent.step(action, reward, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps * eps_decay)
        print('\rEpisode {}\tAverage Score: {:.2f}\tRandom choice weight: {:.2f}\teps: {:.2f}'
              .format(i_episode, np.mean(scores_window), agent.random_choice, eps), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tRandom choice weight: {:.2f}\teps: {:.2f}'
                  .format(i_episode, np.mean(scores_window), agent.random_choice, eps))
        if np.mean(scores_window) >= 200.:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.local_dqn.state_dict(), 'checkpoint.pth')
            break

    return scores


scores = train(1000)

# plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()
