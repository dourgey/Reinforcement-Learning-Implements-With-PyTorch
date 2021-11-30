from tqdm import tqdm
from random import random, randint
from environment import KArmsBandit
import matplotlib.pyplot as plt
import math

class EGreedyPolicy:
    def __init__(self, K, epsilon=0.1):
        self.K = K                      # 动作空间
        self.Q = [5 for _ in range(K)]  # 每个动作的预测动作值
        self.epsilon = epsilon
        self.N = [0 for _ in range(K)]  # 每个动作被选中的次数
        self.count = 0                  # 总计运行多少次
        self.total_reward = 0           # 累积回报
        self.attack_q_star = 0          # 命中最优动作次数

    def get_action(self, q_star):
        self.count += 1
        # 随机
        if random() < self.epsilon:
            action = randint(0, K - 1)
        # 贪心
        else:
            tmp = max(self.Q)
            idx = self.Q.index(tmp)
            action = idx
        self.N[action] += 1
        if action == q_star:
            self.attack_q_star += 1
        return action

    def update_Q(self, action, reward):
        self.total_reward += reward
        self.Q[action] = self.Q[action] + 1 / self.N[action] * (reward - self.Q[action])

class UCBPolicy:
    def __init__(self, K, c=2):
        self.K = K                      # 动作空间
        self.Q = [5 for _ in range(K)]  # 每个动作的预测动作值
        self.c = c
        self.N = [0 for _ in range(K)]  # 每个动作被选中的次数
        self.count = 0                  # 总计运行多少次
        self.total_reward = 0           # 累积回报
        self.attack_q_star = 0          # 命中最优动作次数

    def get_action(self, q_star):
        self.count += 1
        # UCB 算法
        tmp = [(self.Q[idx] + math.sqrt(math.log(self.count / (self.N[idx] + 1e-8))) * self.c) for idx in range(self.K)]
        action = tmp.index(max(tmp))
        self.N[action] += 1
        if action == q_star:
            self.attack_q_star += 1
        return action

    def update_Q(self, action, reward):
        self.total_reward += reward
        self.Q[action] = self.Q[action] + 1 / self.N[action] * (reward - self.Q[action])

K = 10
bandit = KArmsBandit(K)
policy0 = EGreedyPolicy(K, 0)
policy1 = EGreedyPolicy(K, 0.1)
policy2 = EGreedyPolicy(K, 0.01)
UCB_policy = UCBPolicy(K, 2)

mean_reward_list0 = []
best_action_rate0 = []
mean_reward_list1 = []
best_action_rate1 = []
mean_reward_list2 = []
best_action_rate2 = []
mean_reward_list_ucb = []
best_action_rate_ucb = []
for i in tqdm(range(100)):
    action = policy0.get_action(bandit.q_star)
    reward = bandit.get_reward(action)
    policy0.update_Q(action, reward)
    mean_reward_list0.append(policy0.total_reward / policy0.count)
    best_action_rate0.append(policy0.attack_q_star / policy0.count)

    action = policy1.get_action(bandit.q_star)
    reward = bandit.get_reward(action)
    policy1.update_Q(action, reward)
    mean_reward_list1.append(policy1.total_reward / policy1.count)
    best_action_rate1.append(policy1.attack_q_star / policy1.count)

    action = policy2.get_action(bandit.q_star)
    reward = bandit.get_reward(action)
    policy2.update_Q(action, reward)
    mean_reward_list2.append(policy2.total_reward / policy2.count)
    best_action_rate2.append(policy2.attack_q_star / policy2.count)

    action = UCB_policy.get_action(bandit.q_star)
    reward = bandit.get_reward(action)
    UCB_policy.update_Q(action, reward)
    mean_reward_list_ucb.append(UCB_policy.total_reward / UCB_policy.count)
    best_action_rate_ucb.append(UCB_policy.attack_q_star / UCB_policy.count)

plt.title('mean reward')
plt.plot(mean_reward_list0, label='e=0')
plt.plot(mean_reward_list1, label='e=0.1')
plt.plot(mean_reward_list2, label='e=0.01')
plt.plot(mean_reward_list_ucb, label='ucb c=2')
plt.legend()
plt.show()

plt.title('best action rate')
plt.plot(best_action_rate0, label='e=0')
plt.plot(best_action_rate1, label='e=0.1')
plt.plot(best_action_rate2, label='e=0.01')
plt.plot(best_action_rate_ucb, label='ucb c=2')
plt.legend()
plt.show()