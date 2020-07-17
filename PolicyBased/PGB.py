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
        self.action_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

        self.save_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_score = self.action_head(x)
        act_score = F.softmax(action_score, dim=1)
        value = self.value_head(x)
        return act_score, value

policy = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(device)
optimizer = optim.Adam(policy.parameters())
eps = np.finfo(np.float32).eps.item()  # eps是一个很小的非负数

def select_action(state):
    probs, value = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
    m = Categorical(probs)
    action = m.sample()
    policy.save_log_probs.append((m.log_prob(action), value))
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    value_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        returns.append(R)
    returns = torch.tensor(list(reversed(returns)))
    returns = (returns - returns.mean()) / (returns.std() + eps)  # 应该是对returns进行标准化？
    for (log_prob, v), r in zip(policy.save_log_probs, returns):
        advantage = r - v
        policy_loss.append(-log_prob * advantage)
        value_loss.append(F.smooth_l1_loss(r, v))
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    value_loss = torch.stack(value_loss).sum()
    loss = policy_loss + value_loss
    loss.backward()
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