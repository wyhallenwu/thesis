import tianshou as ts
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from sim.env import SimEnv
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from gymnasium.wrappers import FlattenObservation


def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_env = SimEnv()
    train_env = FlattenObservation(train_env)
    writer = SummaryWriter('log/')
    logger = TensorboardLogger(writer, 1, 1, 1, 1)
    # test_env = SimEnv()
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
    result = ts.trainer.onpolicy_trainer(
        policy=policy, train_collector=train_collector, test_collector=None,
        max_epoch=10, step_per_epoch=1000,
        repeat_per_collect=2, episode_per_test=1,
        batch_size=24, step_per_collect=2,
        stop_fn=lambda mean_rewards: mean_rewards >= 30,
        logger=logger)
    print(f'Finished training! Use {result["duration"]}')
