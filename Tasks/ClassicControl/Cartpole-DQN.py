import gym
from ValueBased.DQNAgents import *


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v0").unwrapped
    agent = DDQNAgent(env, device=device, render=True, use_dueling=True)
    agent.train(190)