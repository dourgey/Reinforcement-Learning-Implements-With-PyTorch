import random

class KArmsBandit:
    """
    平稳的K臂赌博机
    """
    def __init__(self, k):
        self.action_space = k
        self.real_values = [random.gauss(0, 1) for _ in range(k)]
        self.q_star = self.real_values.index(max(self.real_values))

    def get_reward(self, action):
        assert action >= 0 and action < 10
        return self.real_values[action]

if __name__ == '__main__':
    bandit = KArmsBandit(10)
    print(bandit.real_values, bandit.q_star)