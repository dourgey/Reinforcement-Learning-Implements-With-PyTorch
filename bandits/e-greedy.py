from tqdm import tqdm
from random import random, randint
from environment import KArmsBandit
import matplotlib.pyplot as plt

class EGreedyPolicy:
    def __init__(self, K, epsilon=0.1):
        self.K = K                      # 动作空间
        self.Q = [0 for _ in range(K)]  # 每个动作的预测动作值
        self.epsilon = epsilon
        self.N = [0 for _ in range(K)]  # 每个动作被选中的次数
        self.count = 0                  # 总计运行多少次
        self.total_reward = 0           # 累积回报
        self.attack_q_star = 0          # 命中最优动作次数

    def get_action(self, q_star):
        self.count += 1
        # 随机
        if random() > 1 - self.epsilon:
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

K = 10
bandit = KArmsBandit(K)
policy = EGreedyPolicy(K, 0.01)
mean_reward_list = []
best_action_rate = []
for i in tqdm(range(5000)):
    action = policy.get_action(bandit.q_star)
    reward = bandit.get_reward(action)
    policy.update_Q(action, reward)
    mean_reward_list.append(policy.total_reward / policy.count)
    best_action_rate.append(policy.attack_q_star / policy.count)

plt.plot(mean_reward_list)
plt.show()
plt.plot(best_action_rate)
plt.show()