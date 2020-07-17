import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.distributions import Categorical

device = "cuda" if torch.cuda.is_available() else "cpu"

GAMMA = 0.99
SEED = 123
RENDER = True
EPISODES = 1000

env = gym.make("CartPole-v0")
env.seed(SEED)
torch.manual_seed(SEED)

class PolicyNet(nn.Module):
    def __init__(self, n_state, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_state, 256)
        self.fc2 = nn.Linear(256, n_actions)

        self.save_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        act_score = F.softmax(x, dim=1)
        return act_score

policy = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = optim.Adam(policy.parameters())
eps = np.finfo(np.float32).eps.item()  # eps是一个很小的非负数

def select_action(state):
    probs = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
    m = Categorical(probs)
    action = m.sample()
    policy.save_log_probs.append(m.log_prob(action))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        returns.append(R)
    returns = torch.tensor(list(reversed(returns)))
    returns = (returns - returns.mean()) / (returns.std() + eps)  # 应该是对returns进行标准化？
    for log_prob, r in zip(policy.save_log_probs, returns):
        policy_loss.append(-log_prob * r)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    policy.save_log_probs, policy.rewards = [], []

for i_episode in range(EPISODES):
    state, ep_reward = env.reset(), 0
    for t in range(1000):
        action = select_action(state)
        state, reward, done, _ = env.step(action)
        if RENDER:
            env.render()
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break

    finish_episode()
    print('Episode {}\tLast reward: {:.2f}\t'.format(i_episode, ep_reward))

env.close()