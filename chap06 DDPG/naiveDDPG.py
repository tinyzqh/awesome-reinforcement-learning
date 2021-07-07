"""
# @Time    : 2021/7/3 5:07 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : naiveDDPG.py
"""

import argparse
import torch
import gym
import numpy as np
import collections
import torch.nn as nn
import torch.nn.functional as F


Experience = collections.namedtuple(typename="Experience", field_names=['state', 'action', 'reward', 'done', 'nextState'])


class ExperienceBuffer(object):
    def __init__(self, args):
        self.buffer = collections.deque(maxlen=args.replay_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        states, actions, rewards, done, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), actions, np.array(rewards, dtype=np.float32), done, np.array(next_states)


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, action_scale):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, output_dim)
        self.action_scale = action_scale

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.action_scale * torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, output_dim)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], 1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent(object):
    def __init__(self, env, exp_buffer, args):
        super(Agent, self).__init__()
        self.env = env
        self.exp_buffer = exp_buffer
        self.args = args
        self.actor = None
        self.critic = None
        self.target_actor = None
        self.target_critic = None
        self.build_model()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)

    def build_model(self):
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_scale = self.env.action_space.high[0]

        self.actor = Actor(input_dim=obs_dim, output_dim=action_dim, action_scale=action_scale)
        self.target_actor = Actor(input_dim=obs_dim, output_dim=action_dim, action_scale=action_scale)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(input_dim=obs_dim+action_dim, output_dim=1)
        self.target_critic = Critic(input_dim=obs_dim+action_dim, output_dim=1)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)
        return self.actor(x)

    def store_transition(self, state, action, r, done, state_next):
        exp = Experience(state, action, r, done, state_next)
        self.exp_buffer.append(exp)

    def learn(self):
        for _ in range(0, self.args.update_iteration):

            buffer = self.exp_buffer.sample(self.args.batch_size)
            states, actions, rewards, done, next_states = buffer

            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.FloatTensor(actions)
            rewards_tensor = torch.unsqueeze(torch.FloatTensor(rewards), 1)
            done_tensor = torch.unsqueeze(1-torch.FloatTensor(done), 1)
            next_states_tensor = torch.FloatTensor(next_states)

            # Compute the target Q value
            target_q = self.target_critic(next_states_tensor, self.target_actor(next_states_tensor))
            target_q = rewards_tensor + (done_tensor * self.args.gamma * target_q).detach()

            # Get current Q estimate
            current_q = self.critic(states_tensor, actions_tensor)

            # Compute critic loss
            critic_loss = F.mse_loss(current_q, target_q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(states_tensor, self.actor(states_tensor)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)


def main():
    parser = argparse.ArgumentParser(description="the parameter of actor critic")
    parser.add_argument('--replay_size', type=int, help="maximum capacity of the buffer", default=20000)
    parser.add_argument("--batch_size", type=int, help="batch size sampled from the replay buffer", default=128)
    parser.add_argument('--actor_lr', type=float, help='actor learning rate used in the Adam optimizer', default=1e-4)
    parser.add_argument('--critic_lr', type=float, help='critic learning rate used in the Adam optimizer', default=1e-3)
    parser.add_argument('--tau', default=0.005, help='target smoothing coefficient', type=float)
    parser.add_argument('--exploration_noise', help='action noise for algorithms', default=0.1, type=float)
    parser.add_argument('--update_iteration', help='use data update models times', default=200, type=int)
    parser.add_argument('--gamma', type=float, help="gamma value used for Bellman approximation", default=0.99)
    arg = parser.parse_args()

    buffer = ExperienceBuffer(args=arg)
    env = gym.make('Pendulum-v0')
    agent = Agent(env, buffer, arg)
    for epoch in range(10000):
        state, done = env.reset(), False
        episode_r = []
        while not done:
            action = agent.choose_action(state).data.numpy().flatten()
            action = (action + np.random.normal(0, arg.exploration_noise, size=env.action_space.shape[0])).clip(
                env.action_space.low, env.action_space.high)
            state_next, r, done, info = env.step(action)
            agent.store_transition(state, action, r, done, state_next)
            if not done:
                state = state_next
            episode_r.append(r)
        agent.learn()
        print("epoch: {} | avg_r: {} | ep_r: {} | len_ep {}".format(epoch, np.sum(episode_r) / len(episode_r),
                                                                    sum(episode_r), len(episode_r)))
        env.close()


if __name__ == "__main__":

    main()
