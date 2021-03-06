# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import gym

# 超参数
BATCH_SIZE = 32
LR = 0.001                   # learning rate
# 强化学习的参数
EPSILON = 1               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 1000
# 导入实验环境
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
print('N_ACTIONS', N_ACTIONS, 'N_STATES', N_STATES)


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)  # 初始化
        self.fc2 = nn.Linear(10, 10)
        self.fc2.weight.data.normal_(0, 0.1)  # 初始化
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # 初始化

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x =self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        # 记录学习到多少步
        self.learn_step_counter = 0     # for target update
        self.memory_counter = 0         # for storing memory
        # 初始化memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.loss_func = nn.L1Loss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(x)
            # print(torch.max(action_value, 1))
            # print(torch.max(action_value, 1)[1].data.numpy())
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    # s:当前状态， a:动作, r:reward奖励, s_:下一步状态
    def store_transaction(self, state, action, reward, state_next):
        transaction = np.hstack((state, [action, reward], state_next))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transaction
        self.memory_counter += 1

    def learn(self):
        # target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        # print('[learn] ============================================================')
        for i in range(10):
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
            b_memory = self.memory[sample_index, :]
            b_state = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
            b_action = Variable(torch.LongTensor(b_memory[:, N_STATES: N_STATES + 1].astype(int)))
            b_reward = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1: N_STATES + 2]))
            b_state_next = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))
            # print('b_state', b_state.size())
            # print('b_action', b_action.size())
            # print('b_reward', b_reward.size())
            # print('b_state_next', b_state_next.size())

            q_eval = self.eval_net(b_state_next).gather(1, b_action)
            # print('q_eval', q_eval.size())
            q_next = self.target_net(b_state_next).detach()
            # print('q_next', q_next.size())

            # print('q_next.max(1)[0]', q_next.max(1)[0].view(-1, 1).size())
            q_target = b_reward + GAMMA * q_next.max(1)[0].view(-1, 1)
            # print('q_target', q_target.size())

            loss = self.loss_func(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # def save(self):
    #
    # def load(self):


dqn = DQN()
print('\nCollecting experience...')
for i_episode in range(4000):
    observation = env.reset()
    total_reward = 0
    total_step = 0

    while True:
        env.render()

        action = dqn.choose_action(observation)
        # take action
        observation_next, r, done, info = env.step(action)

        # modify the reward
        x, x_dot, theta, theta_dot = observation_next
        reward1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        reward2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = reward1 + reward2
        dqn.store_transaction(observation, action, reward, observation_next)

        total_reward += reward
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if done:
            print('i_episode:', i_episode, 'reward:', round(total_reward, 2), 'step:', total_step)
            break

        observation = observation_next
        total_step += 1

