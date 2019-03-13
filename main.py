# encoding: utf-8
import gym
from gym import envs
from gym.wrappers import Monitor


def environment():
    print('[environment]', envs.registry.all())


# 手推车杆
def cart_pole():
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())                     # take a random action


# 手推车杆
def cart_pole_1():

    env = gym.make('CartPole-v0')
    # print('[cart_pole_1]', env.action_space)                    # Discrete(2)
    # print('[cart_pole_1]', env.observation_space)               # Box(4,)
    # # action取非负整数0或1。Box表示一个n维的盒子，因此observation是一个4维的数组。我们可以试试box的上下限。
    # print('[cart_pole_1]', env.observation_space.high)          # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
    # print('[cart_pole_1]', env.observation_space.low)           # [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]

    env = Monitor(env=env, directory='./tmp/cartpole-experiment-0202', video_callable=False, write_upon_reset=True)

    observation = env.reset()                                   # 重置环境的状态，返回观察

    for t in range(100):
        env.render()                                            # 重绘环境的一帧
        print('[cart_pole_1] observation old:', observation)
        action = env.action_space.sample()
        print('[cart_pole_1] action', action)
        observation, reward, done, info = env.step(action)      # 推进一个时间步长，返回observation，reward，done，info
        print('[cart_pole_1] observation new:', observation, '[reward, done, info]:', reward, done, info)

        if done:
            print("[observation] Done after {} time steps".format(t + 1))
            break

    env.close()


if __name__ == "__main__":
    # environment()
    # cart_pole()
    cart_pole_1()
