#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: cross entropy.py
@time: 7/28/20 10:36 AM
@desc:
'''

import argparse
import gym
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from tensorboardX import SummaryWriter

class Net(nn.Module):
    def __init__(self, args, obs_size, n_action):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, args.Hidden_Size),
            nn.ReLU(),
            nn.Linear(args.Hidden_Size, n_action)
        )
    def forward(self, x):
        return self.net(x)

def iterate_batches(env, net, args):
    Episode = namedtuple("Episode", field_names=["reward", "steps"])
    Single_Step = namedtuple("Single_Step", field_names=["observation", "action"])
    batch = []
    Episode_Steps = []
    episode_reward = 0.0

    obs = env.reset() # obtain the first observation
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor(obs)
        # get the probability of actions
        act_probs_v = sm(net(obs_v).unsqueeze(dim=0))
        # tensor.data convert the tensor into numpy array
        act_probs = act_probs_v.data.numpy()[0]
        # get the action by sampling the distribution
        action = np.random.choice(len(act_probs), p=act_probs)

        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward

        step = Single_Step(observation=obs, action=action) # [s,a]
        Episode_Steps.append(step) #

        if is_done:
            batch.append(Episode(reward=episode_reward, steps=Episode_Steps)) # saving the total reward and steps we have taken
            episode_reward = 0
            Episode_Steps = []
            next_obs = env.reset()


            if len(batch) == args.Batch_Size:
                yield batch
                batch = []
        obs = next_obs

def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)

    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps)) # add all obs of batch
        train_act.extend(map(lambda step: step.action, steps)) # add all actions of batch

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)

    return train_obs_v, train_act_v, reward_bound, float(np.mean(rewards))

def main(args):
    env = gym.make("CartPole-v0")

    net = Net(args, obs_size=env.observation_space.shape[0], n_action=env.action_space.n)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)

    # Use tensorboard --logdir runs
    writer = SummaryWriter(comment="-cartpole")

    for iter, batch in enumerate(iterate_batches(args=args, env=env, net=net)):
        obs_v, act_v, reward_bound, mean_reward = filter_batch(batch=batch, percentile=args.Percentule)
        optimizer.zero_grad()
        act_pre = net(obs_v) # shape=(batch, action_dim) but the shape of act_v
        loss = loss_func(act_pre, act_v)
        loss.backward()
        optimizer.step()
        print("Iter: {} | reward_mean {} | reward_bound {}".format(iter, mean_reward, reward_bound))

        writer.add_scalar("loss", loss.item(), iter)
        writer.add_scalar("reward_bound", reward_bound, iter)
        writer.add_scalar("reward_mean", mean_reward, iter)

        if mean_reward > 199:
            print("Solved!")
            break

    writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="The parameters of cross entropy RL")

    parser.add_argument("--Hidden_Size", type=int ,help="The Hidden Size of Neural Network", default=128)
    parser.add_argument("--Batch_Size", type=int, default=16)
    parser.add_argument("--Percentule", type=int, help="The percentage of good trajectory", default=70)

    args = parser.parse_args()
    main(args)