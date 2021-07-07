#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: value iteration.py
@time: 7/29/20 8:16 PM
@desc:
'''

import gym
import collections
from tensorboardX import SummaryWriter
import argparse


class Agent:
    def __init__(self, args):
        self.env = gym.make(args.env)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            # gather random experience from the environment
            action = self.env.action_space.sample()
            next_state, reward, is_done, _ = self.env.step(action)

            # update the reward and transition tables
            self.rewards[(self.state, action, next_state)] = reward
            self.transits[(self.state, action)][next_state] += 1

            self.state = self.env.reset() if is_done else next_state

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)

    def calc_action_value(self, state, action):
        next_state_all = self.transits[(state, action)]
        total_count = sum(next_state_all.values())
        action_value = 0
        for next_state, count in next_state_all.items():
            r = self.rewards[(state, action, next_state)]
            # Q(s,a) = \sum{P(s^{\prime}|s,a)(r + \gamma * V(s^{\prime}))}
            action_value += (count/total_count) * (r + args.gamma*self.values[next_state])
        return action_value

    def selection_action(self, state):
        """
        the best action to take from the given state
        :param state:
        :return:
        """

        # iterates over all possible actions in the environment and calculates the value for every action
        action_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]

        best_action = action_values.index(max(action_values))

        return best_action
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.selection_action(state=state)
            next_state, reward, is_done, _ = env.step(action)

            self.rewards[(state, action, next_state)] = reward
            self.transits[(state, action)][next_state] += 1

            total_reward += reward
            if is_done:
                break

            state = next_state
        return total_reward


def main(args):
    test_env = gym.make(args.env)
    agent = Agent(args)
    writer = SummaryWriter(comment="-v-iteration")
    iter = 0
    best_reward = 0
    while True:
        iter += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()
        reward = 0
        for _ in range(args.Test_Episodes):
            reward += agent.play_episode(test_env)

        reward /= args.Test_Episodes
        writer.add_scalar("reward", reward, iter)
        if reward > best_reward:
            print("Best  reward update {} -> {}".format(best_reward, reward))
            best_reward = reward

        if reward > 0.8:
            print("Solve in {} iterations".format(iter))
            break
    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="The Parameter of Value Iteration")

    parser.add_argument("--env", type=str, help="the name of environment", default="FrozenLake-v0")
    parser.add_argument("--gamma", type=float, help="The discount factor", default=0.9)
    parser.add_argument("--Test_Episodes", type=int, help="Test_Episodes decided whether stop training", default=20)

    args = parser.parse_args()
    print(args.env)

    main(args=args)