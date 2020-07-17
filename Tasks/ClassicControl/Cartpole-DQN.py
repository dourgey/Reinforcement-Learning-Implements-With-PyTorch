import gym
from ValueBased.DQNAgents import *


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1").unwrapped
    agent = DDQNAgent(env, DQNNet, capacity=10000, device=device, render=False, use_dueling=True)
    agent.train(900)