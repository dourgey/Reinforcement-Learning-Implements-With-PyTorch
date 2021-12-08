import random
import numpy as np
import matplotlib.pyplot as plt

class KArmsBandit:
    """
    平稳的K臂赌博机
    """
    def __init__(self, k=10, epsilon, ucb):
        self.k = k
        self.real_values = np.random.randn(k)
        self.q_star = np.argmax(self.real_values)

    def reset(self):

    def act(self):
        pass

    def step(self):

    def get_reward(self, action):
        assert action >= 0 and action < self.action_space
        return self.real_values[action]# + np.random.randn()

if __name__ == '__main__':
    bandit = KArmsBandit(10)
    print(bandit.real_values, bandit.q_star)

    # plot figure 2.1
    q_true = [random.gauss(0, 1) for _ in range(10)]
    action_distribution = np.array([[q_true[i] + random.gauss(0, 1) for i in range(10)] for _ in range(100)])
    plt.violinplot(dataset=action_distribution)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.show()