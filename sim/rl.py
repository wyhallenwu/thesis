import tianshou as ts
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from sim.env import SimEnv
import torch
import numpy as np
from torch import nn
from gymnasium.wrappers import FlattenObservation


def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_env = FlattenObservation(SimEnv())
    test_env = FlattenObservation(SimEnv())
    state_shape = train_env.observation_space.shape
    action_shape = train_env.action_space.shape
    net = Net(state_shape, hidden_sizes=[64, 64], device=device)
    actor = Actor(net, train_env.action_space.shape, device=device).to(device)
    critic = Critic(net, device=device).to(device)
    ac = ActorCritic(actor, critic)
    optim = torch.optim.Adam(ac.parameters(), lr=1e-3)
    dist = torch.distributions.Categorical
    policy = ts.policy.PPOPolicy(
        actor, critic, optim, dist)
    train_collector = ts.data.Collector(
        policy, train_env, ts.data.ReplayBuffer(200), exploration_noise=True)
    test_collector = ts.data.Collector(
        policy, test_env, exploration_noise=True)
    result = ts.trainer.onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=10000,
        repeat_per_collect=10, episode_per_test=10,
        batch_size=24, step_per_collect=100,
        stop_fn=lambda mean_rewards: mean_rewards >= 30)
    print(f'Finished training! Use {result["duration"]}')
