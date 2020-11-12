import gym
import torch
import torch.optim as optim

import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from network import Policy

env_name = 'CartPole-v1'
env = gym.make(env_name)
env.seed(0)
print(env_name)
print('observation space:', env.observation_space)
print('action space:', env.action_space)

device_num = 1
device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")

input_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = Policy(input_dim, action_dim, device_num=device_num).to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.02)

n_episodes = 10000
max_t = 1000
gamma = 0.999
print_every = 100

scores_deque = deque(maxlen=100)
scores = []
for i_episode in range(1, n_episodes+1):
    saved_log_probs = []
    rewards = []
    state = env.reset()
    for t in range(max_t):
        action, log_prob = policy.act(state)
        saved_log_probs.append(log_prob)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            print("Done", t, end='\r')
            break
    scores_deque.append(sum(rewards))
    scores.append(sum(rewards))
    
    discounts = [gamma**i for i in range(len(rewards)+1)]
    R = sum([a*b for a,b in zip(discounts, rewards)])

    # t=0 시점부터의 return을 모든 t시점의 log_prob에 곱한다. (Sutton 책은 이렇게 하지 않음. t시점의 return을 t시점의 log_prob에 곱함) 
    policy_loss = [-log_prob * R for log_prob in saved_log_probs]
    policy_loss = torch.cat(policy_loss).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    if i_episode % print_every == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    if np.mean(scores_deque)>=195.0:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
        break