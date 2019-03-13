# encoding: utf-8
import gym
from gym import envs
from gym.wrappers import Monitor
from DQN import DQN


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
        # action = t % 2
        print('[cart_pole_1] action', action)
        observation, reward, done, info = env.step(action)      # 推进一个时间步长，返回observation，reward，done，info
        print('[cart_pole_1] observation new:', observation, '[reward, done, info]:', reward, done, info)

        if done:
            print("[observation] Done after {} time steps".format(t + 1))
            break

    env.close()


# 手推车杆
def cart_pole_2():
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    # 以下可以显示这个环境的state 和 action
    print(env.action_space)
    print(env.observation_space.shape[0])
    print(env.observation_space.high)
    print(env.observation_space.low)

    # 初始化DQN的模型
    RL = DQN(n_actions=env.action_space.n,
             n_features=env.observation_space.shape[0],
             learning_rate=0.01,
             e_greedy=0.9,
             replace_target_iter=100,
             memory_size=2000,
             e_greedy_increment=0.001,
             use_e_greedy_increment=1000)

    steps = 0
    # 训练300个回合，这里环境模型，结束回合的标志是 倾斜程度和 X 的移动限度，你可以很容易从训练效果中看出来，当然了，也可以去看gym的底层代码，还是比较清晰的。
    for episode in range(300):

        observation = env.reset()
        ep_r = 0

        while True:  # 训练没有结束的时候循环
            env.render()                                                        # 刷新环境
            action = RL.choose_action(observation)                              # 根据状态选择行为
            observation_next, reward, done, info = env.step(action)             # 环境模型 采用行为，获得下个状态，和潜在的奖励

            x, x_dot, theta, theat_dot = observation_next                       # 这里拆分了 状态值 ，里面有四个参数
            print('x', x, 'x_dot', x_dot, 'theta', theta, 'theat_dot', theat_dot, 'reward', reward, 'done', done)
            # 这里用了，x 和theta的限度值 来判断奖励的幅度，当然也可以gym自带的 ，但是这个效率据说比较高
            reward1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            reward2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = reward1 + reward2
            print('env.x_threshold', env.x_threshold, 'env.theta_threshold_radians', env.theta_threshold_radians) # 将奖励综合
            print('reward1', reward1, 'reward2', reward2, 'reward', reward)
            RL.store_transition(observation, action, reward, observation_next)  # 先存储到记忆库

            ep_r += reward          # 这里只是为了观察奖励值是否依据实际情况变化，来方便判断模型的正确性
            if steps > 1000:        # 这里一开始先不学习，先积累奖励
                RL.learn()
            if done:                # 这里判断的是回合结束，显示奖励积累值，你可以看到每回合奖励的变化，来判定这样一连串行为的结果好不好
                print('episode :', episode,
                      'reward:', round(ep_r, 2),
                      "RL's epsilon", round(RL.epsilon, 3))
                break

            observation = observation_next          # 更新状态
            steps += 1
    RL.plot_cost()  # 训练结束后来观察我们的cost


if __name__ == "__main__":
    # environment()
    # cart_pole()
    # cart_pole_1()
    cart_pole_2()
