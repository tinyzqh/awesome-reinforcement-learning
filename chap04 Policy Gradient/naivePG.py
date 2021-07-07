import gym
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import collections

import torch.optim

Experience = collections.namedtuple(typename='Experience', field_names=['state', 'action', 'reward', 'done', 'nextState'])


class ExperienceBuffer:
    def __init__(self, args):
        self.buffer = collections.deque(maxlen=args.replay_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, done, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), done, np.array(next_states)

    def sample_trajectory(self):
        indices = np.arange(0, self.__len__())
        states, actions, rewards, done, next_states = zip(*[self.buffer[idx] for idx in indices])
        self.buffer.clear()
        return np.array(states), actions, np.array(rewards, dtype=np.float32), done, np.array(next_states)


class PG(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PG, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.softmax(self.fc2(x), dim=-1)
        return x


class Agent(nn.Module):
    def __init__(self, env, exp_buffer, args):
        super(Agent, self).__init__()
        self.env = env
        self.exp_buffer = exp_buffer
        self.args = args
        self.policy = None
        self.build_model()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr)

    def build_model(self):
        input_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.policy = PG(input_dim, action_dim)

    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)
        prob = self.policy(x)  # 得到神经网络输出的概率值
        c = Categorical(prob)  # 创建以参数prob为标准的类别分布，
        action = c.sample()  # 按照传入的prob中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引。
        return action

    def store_transition(self, state, action, r, done, state_next):
        exp = Experience(state, action, r, done, state_next)
        self.exp_buffer.append(exp)

    def learn(self):
        buffer = self.exp_buffer.sample_trajectory()
        states, actions, rewards, done, next_states = buffer
        for i in reversed(range(len(rewards))):
            if done[i]:
                rewards[i] = 0
            else:
                rewards[i] = self.args.gamma * rewards[i+1] + rewards[i]
        # Normalize reward
        r_mean = np.mean(rewards)
        r_std = np.mean(rewards)
        rewards = (rewards - r_mean) / r_std
        self.optimizer.zero_grad()
        for i in range(len(states)):
            state = torch.unsqueeze(torch.FloatTensor(states[i]), 0)
            action = torch.FloatTensor([actions[i]])
            prob = self.policy(state)
            c = Categorical(prob)
            loss = - c.log_prob(action) * rewards[i]
            loss.backward()
        self.optimizer.step()


def main():
    parser = argparse.ArgumentParser(description="the parameter of policy gradient")
    parser.add_argument('--replay_size', type=int, help="maximum capacity of the buffer", default=2000)
    parser.add_argument("--lr", type=float, help="learning rate used in the Adam optimizer", default=0.01)
    parser.add_argument("--gamma", type=float, help="gamma value used for Bellman approximation", default=0.99)
    arg = parser.parse_args()

    buffer = ExperienceBuffer(args=arg)
    env = gym.make('CartPole-v0')
    agent = Agent(env, buffer, arg)
    for epoch in range(10000):
        state, done = env.reset(), False  # 1. 获取初始状态信息
        episode_r = []
        while not done:
            # env.render()
            action = agent.choose_action(state)  # 2. 依据状态选择动作
            state_next, r, done, info = env.step(action.item())  # 3. 依据动作更新环境状态
            agent.store_transition(state, action.item(), r, done, state_next)

            if not done:
                state = state_next  # 4. 更新状态
            episode_r.append(r)
        agent.learn()  # 5. 智能体进行学习
        print("epoch: {} | len_ep_r: {} | avg_r: {}".format(epoch, len(episode_r), np.sum(episode_r) / len(episode_r)))
    env.close()


if __name__ == "__main__":

    main()

