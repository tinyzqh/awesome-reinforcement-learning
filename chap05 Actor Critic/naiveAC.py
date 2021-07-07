"""
# @Time    : 2021/7/3 8:04 上午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : naiveAC.py
"""

import argparse
import torch
import gym
import numpy as np
import collections
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


Experience = collections.namedtuple(typename="Experience", field_names=['state', 'action', 'reward', 'done', 'nextState'])


class ExperienceBuffer(object):
    def __init__(self, args):
        self.buffer = collections.deque(maxlen=args.replay_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample_trajectory(self):
        indices = np.arange(0, self.__len__())
        states, actions, rewards, done, next_states = zip(*[self.buffer[idx] for idx in indices])
        self.buffer.clear()
        return np.array(states), actions, np.array(rewards, dtype=np.float32), done, np.array(next_states)


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Scalar Value
        return x


class Agent(object):
    def __init__(self, env, exp_buffer, args):
        super(Agent, self).__init__()
        self.env = env
        self.exp_buffer = exp_buffer
        self.args = args
        self.actor = None
        self.critic = None
        self.build_model()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)

    def build_model(self):
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.actor = Actor(input_dim=obs_dim, output_dim=action_dim)
        self.critic = Critic(input_dim=obs_dim, output_dim=1)

    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)
        prob = self.actor(x)
        c = Categorical(prob)
        action = c.sample()
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
        r_std = np.std(rewards)
        rewards = (rewards - r_mean) / r_std

        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        state_v = torch.squeeze(self.critic(states_tensor), 1)
        prob = self.actor(states_tensor)
        c = Categorical(prob)

        self.actor_optimizer.zero_grad()
        adv = rewards_tensor - state_v.detach()
        actor_loss = torch.sum(-c.log_prob(actions_tensor) * adv)
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss = F.smooth_l1_loss(state_v, rewards_tensor)
        critic_loss.backward()
        self.critic_optimizer.step()


def main():
    parser = argparse.ArgumentParser(description="the parameter of actor critic")
    parser.add_argument('--replay_size', type=int, help="maximum capacity of the buffer", default=2000)
    parser.add_argument('--actor_lr', type=float, help='actor learning rate used in the Adam optimizer', default=0.01)
    parser.add_argument('--critic_lr', type=float, help='critic learning rate used in the Adam optimizer', default=0.01)
    parser.add_argument('--gamma', type=float, help="gamma value used for Bellman approximation", default=0.99)
    arg = parser.parse_args()

    buffer = ExperienceBuffer(args=arg)
    env = gym.make('CartPole-v0')
    agent = Agent(env, buffer, arg)
    for epoch in range(10000):
        state, done = env.reset(), False
        episode_r = []
        while not done:
            action = agent.choose_action(state)
            state_next, r, done, info = env.step(action.item())
            agent.store_transition(state, action.item(), r, done, state_next)
            if not done:
                state = state_next
            episode_r.append(r)
        agent.learn()
        print("epoch: {} | len_ep_r: {} | avg_r: {}".format(epoch, len(episode_r), np.sum(episode_r) / len(episode_r)))
        env.close()


if __name__ == "__main__":

    main()
