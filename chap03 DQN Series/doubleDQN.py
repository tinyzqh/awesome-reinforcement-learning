import numpy as np
import argparse
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import collections
Experience = collections.namedtuple(typename='Experience', field_names=['state', 'action', 'reward', 'done', 'nextState'])


class ExperienceBuffer:
    def __init__(self, args):
        self.buffer = collections.deque(maxlen=args.replay_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        randomly sample the batch of transitions from the replay buffer.
        :param batch_size:
        :return:
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               dones, np.array(next_states)

    def sample_last(self):
        """
        sample the last transitions from the replay buffer.
        :return:
        """
        indices = [len(self.buffer)-1]
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return states[0], actions[0], rewards[0], dones[0], next_states[0]

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_q = self.fc2(x)
        return action_q


class Agent(object):
    def __init__(self, env, exp_buffer, args):
        self.env = env
        self.exp_buffer = exp_buffer
        self.args = args
        self.eval_net = None
        self.target_net = None
        self.build_model()  # 构建智能体模型
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.args.lr)
        self.loss_func = nn.MSELoss()

    def build_model(self):
        input_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.eval_net = DQN(input_dim, action_dim)
        self.target_net = DQN(input_dim, action_dim)

    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0) if state.shape.__len__() == 1 else state
        if np.random.uniform() > self.args.epsilon:
            action = self.env.action_space.sample()
            action = torch.tensor([action], dtype=torch.int64)
        else:
            action_value = self.eval_net.forward(x)
            action = torch.argmax(action_value, 1)
        return action

    def store_transition(self, state, action, r, done, state_next):
        """
        存储轨迹
        :param state:
        :param action:
        :param r:
        :param done:
        :param state_next:
        :return:
        """
        exp = Experience(state, action, r, done, state_next)
        self.exp_buffer.append(exp)

    def learn(self):
        """
        更新智能体模型
        :return:
        """
        buffer = self.exp_buffer.sample(self.args.batch_size)
        states, actions, rewards, done, next_states = buffer

        states = torch.FloatTensor(states)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.FloatTensor(rewards)
        done_mask = torch.BoolTensor(done)
        next_states = torch.FloatTensor(next_states)

        s_a_values = self.eval_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        next_s_actions = torch.argmax(self.eval_net.forward(next_states), 1)  # 评估网络选下一个状态的动作
        next_s_a_values = self.target_net(next_states).gather(1, next_s_actions.unsqueeze(-1)).squeeze(-1)

        next_s_a_values[done_mask] = 0.0  # last step in the episode doesn't have a discounted reward of the next state
        # detach() function to prevent gradients from flowing into the target network's graph
        next_s_a_values = next_s_a_values.detach()

        target = rewards + self.args.gamma * next_s_a_values
        loss = self.loss_func(target, s_a_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():
    parser = argparse.ArgumentParser(description="The parameter of DQN")
    parser.add_argument("--replay_size", type=int, help="maximum capacity of the buffer", default=2000)
    parser.add_argument("--lr", type=float, help="learning rate used in the Adam optimizer", default=1e-3)
    parser.add_argument("--batch_size", type=int, help="batch size sampled from the replay buffer", default=32)
    parser.add_argument("--gamma", type=float, help="gamma value used for Bellman approximation", default=0.99)
    parser.add_argument("--epsilon", type=float, help="epsilon for greedy", default=0.9)
    args = parser.parse_args()

    buffer = ExperienceBuffer(args=args)
    env = gym.make('CartPole-v0')
    agent = Agent(env, buffer, args)
    for epoch in range(10000):
        state, done = env.reset(), False  # 1. 获取初始状态信息
        episode_r = []
        while not done:
            # env.render()
            action = agent.choose_action(state)  # 2. 依据状态选择动作
            state_next, r, done, info = env.step(action.item())  # 3. 依据动作更新环境状态
            agent.store_transition(state, action, r, done, state_next)
            if len(agent.exp_buffer) == args.replay_size and epoch % 10 == 0:
                agent.learn()  # 4. 智能体进行学习
            if not done:
                state = state_next  # 5. 更新状态
            episode_r.append(r)
        if epoch % 20 == 0: # 更新target网络
            agent.target_net.load_state_dict(agent.eval_net.state_dict())
        print("epoch: {} | len_ep_r: {} | avg_r: {}".format(epoch, len(episode_r), np.sum(episode_r) / len(episode_r)))
    env.close()


if __name__ == "__main__":
    main()


