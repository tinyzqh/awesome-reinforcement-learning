"""
# @Time    : 2021/7/4 11:29 上午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : ppo_pendulum.py
"""

import argparse
import torch
import gym
import numpy as np
import collections
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

Experience = collections.namedtuple(typename="Experience",
                                    field_names=['state', 'action', 'reward', 'done', 'nextState', 'action_log_prob'])


class ExperienceBuffer(object):
    def __init__(self, args):
        self.buffer = collections.deque(maxlen=args.replay_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample_trajectory(self):
        indices = np.arange(0, self.__len__())
        states, actions, rewards, done, next_states, action_prob = zip(*[self.buffer[idx] for idx in indices])
        self.buffer.clear()
        return np.array(states), actions, np.array(rewards, dtype=np.float32), done, np.array(next_states), np.array(action_prob)


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.mu = nn.Linear(100, output_dim)
        self.sigma = nn.Linear(100, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x))
        return mu, sigma


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x


class Agent(object):
    def __init__(self, env, exp_buffer, args):
        super(Agent, self).__init__()
        self.env = env
        self.exp_buffer = exp_buffer
        self.args = args
        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.actor = None
        self.critic = None
        self.build_model()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)

    def build_model(self):
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.actor = Actor(input_dim=obs_dim, output_dim=action_dim)
        self.critic = Critic(input_dim=obs_dim, output_dim=1)

    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)
        mu, sigma = self.actor(x)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2.0, 2.0)
        return action, action_log_prob.item()

    def store_transition(self, state, action, r, done, state_next, action_prob):
        exp = Experience(state, action, r, done, state_next, action_prob)
        self.exp_buffer.append(exp)

    def learn(self):
        buffer = self.exp_buffer.sample_trajectory()
        states, actions, rewards, done, next_states, old_action_prob = buffer
        for i in reversed(range(len(rewards))):
            if done[i]:
                rewards[i] = 0
            else:
                rewards[i] = self.args.gamma * rewards[i + 1] + rewards[i]

        # Normalize reward
        r_mean = np.mean(rewards)
        r_std = np.mean(rewards)
        rewards = (rewards - r_mean) / (r_std + 1e-5)

        states_tensor = torch.FloatTensor(states)
        rewards_tensor = torch.unsqueeze(torch.FloatTensor(rewards), 1)
        actions_tensor = torch.unsqueeze(torch.tensor(actions, dtype=torch.int64), 1)
        old_log_action_prob = torch.unsqueeze(torch.FloatTensor(old_action_prob), 1)

        state_v = self.critic(states_tensor)
        delta = rewards_tensor - state_v
        advantage = delta.detach()

        mu, sigma = self.actor(states_tensor)
        dist = Normal(mu, sigma)
        new_action_log_prob = dist.log_prob(actions_tensor)
        ratio = torch.exp(new_action_log_prob - old_log_action_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

        # update actor network
        actor_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Compute critic loss
        critic_loss = F.smooth_l1_loss(rewards_tensor, state_v)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()


def main():
    parser = argparse.ArgumentParser(description="the parameter of actor critic")
    parser.add_argument('--replay_size', type=int, help="maximum capacity of the buffer", default=20000)
    parser.add_argument("--batch_size", type=int, help="batch size sampled from the replay buffer", default=128)
    parser.add_argument('--actor_lr', type=float, help='actor learning rate used in the Adam optimizer', default=1e-4)
    parser.add_argument('--critic_lr', type=float, help='critic learning rate used in the Adam optimizer', default=3e-4)
    parser.add_argument('--clip_param', help='clip_param for ppo algorithms', default=0.2, type=float)
    parser.add_argument('--max_grad_norm', help='max_grad_norm for ppo algorithms', default=0.5, type=float)
    parser.add_argument('--update_iteration', help='use data update models times', default=20, type=int)
    parser.add_argument('--gamma', type=float, help="gamma value used for Bellman approximation", default=0.90)
    arg = parser.parse_args()

    buffer = ExperienceBuffer(args=arg)
    env = gym.make('Pendulum-v0')
    agent = Agent(env, buffer, arg)
    for epoch in range(10000):
        state, done = env.reset(), False
        episode_r = []
        while not done:
            action, action_log_prob = agent.choose_action(state)
            state_next, r, done, info = env.step([action.item()])
            agent.store_transition(state, action.item(), r, done, state_next, action_log_prob)
            if not done:
                state = state_next
            episode_r.append(r)
        agent.learn()
        print("epoch: {} | avg_r: {} | ep_r: {} | len_ep {}".format(epoch, np.sum(episode_r) / len(episode_r),
                                                                    sum(episode_r), len(episode_r)))
        env.close()


if __name__ == "__main__":
    main()
