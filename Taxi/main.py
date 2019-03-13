# encoding: utf-8
import gym
from gym import envs
from gym.wrappers import Monitor


def taxi():
    env = gym.make("Taxi-v2")
    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)


if __name__ == "__main__":
    taxi()
