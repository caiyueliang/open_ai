# encoding: utf-8
import gym


# 手推车杆
def cart_pole():
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())     # take a random action


# 手推车杆
def cart_pole_1():
    env = gym.make('CartPole-v0')
    observation = env.reset()
    for t in range(1000):
        env.render()
        print('[observation] old:', observation)
        action = env.action_space.sample()
        print('[observation] action', action)
        observation, reward, done, info = env.step(action)
        print('[observation] new:', observation, 'reward, done, info', reward, done, info)
        # if done:
        #     print("Episode finished after {} timesteps".format(t + 1))
        #     break


if __name__ == "__main__":
    # cart_pole()
    cart_pole_1()
