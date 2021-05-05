import numpy as np
import gym
from agent import Agent
from monitor import interact

env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)
