import tianshou as ts
from tianshou.data.batch import Batch
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from sim.env import SimEnv
import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from gymnasium.wrappers import FlattenObservation
HIDDEN_SIZE = 64
LAYER_NUM = 3


def ppo_run():
    """ppo algorithm."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_env = SimEnv("ppo")
    train_env = FlattenObservation(train_env)
    writer = SummaryWriter(
        'log/ppo/' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    logger = TensorboardLogger(writer, 1, 1, 1, 1)
    # test_env = SimEnv()
    print(train_env.observation_space.sample())
    state_shape = train_env.observation_space.shape
    action_shape = train_env.action_space.n
    net = Net(state_shape=train_env.observation_space.shape,
              action_shape=train_env.action_space.shape,
              hidden_sizes=[64, 64, 64], device=device)
    # net = ShareNet(train_env.observation_space.shape)
    actor = Actor(preprocess_net=net,
                  action_shape=train_env.action_space.n, device=device).to(device)
    critic = Critic(preprocess_net=net, device=device).to(device)
    ac = ActorCritic(actor=actor, critic=critic)
    optim = torch.optim.AdamW(ac.parameters(), lr=0.001)
    dist = torch.distributions.Categorical
    policy = ts.policy.PPOPolicy(
        actor=actor, critic=critic, optim=optim, dist_fn=dist, action_space=train_env.action_space)
    replay_buffer = ts.data.ReplayBuffer(1000)
    train_collector = ts.data.Collector(
        policy=policy, env=train_env, buffer=replay_buffer)
    result = ts.trainer.onpolicy_trainer(
        policy=policy, train_collector=train_collector, test_collector=None,
        max_epoch=30, step_per_epoch=1000,
        repeat_per_collect=4, episode_per_test=1,
        batch_size=128, step_per_collect=8,
        logger=logger)
    torch.save(policy.state_dict(
    ), f"model/ppo{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())}.pth")
    print(f'Finished training! Use {result["duration"]}')


def a2c_run():
    """a2c algorithm."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_env = SimEnv("a2c")
    train_env = FlattenObservation(train_env)
    writer = SummaryWriter(
        'log/a2c/' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    logger = TensorboardLogger(writer, 1, 1, 1, 1)
    # test_env = SimEnv()
    print(train_env.observation_space.sample())
    state_shape = train_env.observation_space.shape
    action_shape = train_env.action_space.n
    net = Net(state_shape=train_env.observation_space.shape,
              action_shape=train_env.action_space.shape,
              hidden_sizes=[64, 64, 64], device=device)
    # net = ShareNet(train_env.observation_space.shape)
    actor = Actor(preprocess_net=net,
                  action_shape=train_env.action_space.n, device=device).to(device)
    critic = Critic(preprocess_net=net, device=device).to(device)
    ac = ActorCritic(actor=actor, critic=critic)
    optim = torch.optim.Adam(ac.parameters(), lr=0.001)
    dist = torch.distributions.Categorical
    policy = ts.policy.A2CPolicy(
        actor=actor, critic=critic, optim=optim, dist_fn=dist, action_space=train_env.action_space)
    replay_buffer = ts.data.ReplayBuffer(1000)
    train_collector = ts.data.Collector(
        policy=policy, env=train_env, buffer=replay_buffer)
    result = ts.trainer.onpolicy_trainer(
        policy=policy, train_collector=train_collector, test_collector=None,
        max_epoch=30, step_per_epoch=1000,
        repeat_per_collect=4, episode_per_test=1,
        batch_size=128, step_per_collect=8,
        logger=logger)
    torch.save(policy.state_dict(
    ), f"model/a2c{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())}.pth")
    print(f'Finished training! Use {result["duration"]}')


def td3_run():
    # TODO: debug
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_env = SimEnv("td3")
    train_env = FlattenObservation(train_env)
    writer = SummaryWriter('log/td3/')
    logger = TensorboardLogger(writer, 1, 1, 1, 1)
    # test_env = SimEnv()
    print(train_env.observation_space.sample())
    state_shape = train_env.observation_space.shape
    action_shape = train_env.action_space.n
    actor_net = Net(state_shape=train_env.observation_space.shape, hidden_sizes=[64, 64],
                    action_shape=train_env.action_space.shape, device=device)
    critic_net = Net(state_shape=np.prod(train_env.observation_space.shape) +
                     np.prod(train_env.action_space.shape), hidden_sizes=[64, 64], device=device)
    # net = ShareNet(train_env.observation_space.shape)
    actor = Actor(preprocess_net=actor_net,
                  action_shape=train_env.action_space.n, device=device).to(device)
    critic1 = Critic(preprocess_net=critic_net, device=device).to(device)
    critic2 = Critic(preprocess_net=critic_net, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=0.0003)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=0.0003)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=0.0003)
    """
    actor(s -> logits)
    critic(s, a -> Q(s, a))
    """
    policy = ts.policy.TD3Policy(
        actor=actor, actor_optim=actor_optim, critic1=critic1,
        critic1_optim=critic1_optim, critic2=critic2, critic2_optim=critic2_optim, action_space=train_env.action_space)
    replay_buffer = ts.data.ReplayBuffer(2000)
    train_collector = ts.data.Collector(
        policy=policy, env=train_env, buffer=replay_buffer)
    result = ts.trainer.offpolicy_trainer(
        policy=policy, train_collector=train_collector, test_collector=None,
        max_epoch=20, step_per_epoch=5000, step_per_collect=16,
        episode_per_test=1,
        batch_size=64, update_per_step=1, logger=logger)
    torch.save(policy.state_dict(
    ), f"td3{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())}.pth")
    print(f'Finished training! Use {result["duration"]}')
