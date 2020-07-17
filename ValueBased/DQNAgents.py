from utils import *
from itertools import count
from collections import deque
import torch.optim as optim

from NNModels import *

from Agents import *

class DQNAgent(Agent):
    def __init__(self, env, Net, device, ReplayMemory=DQNReplayMemory, capacity=1000, batch_size=128, net_sync=5, gamma=0.99, use_dueling=False, render=False):
        super(DQNAgent, self).__init__()
        self.env = env
        self.render = render
        self.device = device
        self.space_n = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        # DQN hyper parameters
        self.gamma = gamma
        self.net_sync = net_sync
        self.batch_size = batch_size

        self.memory = ReplayMemory(capacity)
        self.policy_net = Net(self.space_n, self.n_actions, use_dueling=use_dueling).to(device)
        self.target_net = Net(self.space_n, self.n_actions, use_dueling=use_dueling).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.action_selector = EGreedActionSelector()

    def select_action(self, state):
        return self.action_selector.select(self.policy_net, state, self.n_actions)

    def train(self, converge_line, print_log=True):
        while True:
            last_100_episodes_rewards = deque(maxlen=100)
            for e in count():
                state = self.env.reset()

                for t in count():
                    if self.render:
                        self.env.render()
                    state = torch.tensor(state, dtype=torch.float32).to(self.device)
                    action = self.select_action(state).to(self.device)
                    next_state, reward, done, _ = self.env.step(action.squeeze(0).item())
                    reward = torch.tensor([reward], device=self.device)
                    if done:
                        next_state = None
                    self.memory.push(state, action, next_state, reward)

                    state = next_state
                    self.learn()

                    if done:
                        last_100_episodes_rewards.append(t)
                        break

                if e % self.net_sync == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if print_log:
                    print(f"第 {e} 回合训练, 当前回合reward: {t}, 最后100回合平均reward: ", sum(last_100_episodes_rewards) / len(last_100_episodes_rewards))

                if sum(last_100_episodes_rewards) / len(last_100_episodes_rewards) > converge_line:
                    return


    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))


        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([torch.tensor(s, dtype=torch.float32).to(self.device)
                                           for s in batch.next_state if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = (state_action_values - expected_state_action_values.unsqueeze(1)) ** 2
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path="./saved_model.pkl"):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path="./saved_model.pkl"):
        self.policy_net.load_state_dict(torch.load(path))

class DDQNAgent(DQNAgent):
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32).to(self.device)
                                           for s in batch.next_state if s is not None]).reshape(-1, self.space_n)
        state_batch = torch.cat(batch.state).reshape(-1, self.space_n)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_action = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_action).squeeze(1).detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = (state_action_values - expected_state_action_values.unsqueeze(1)) ** 2
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()