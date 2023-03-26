import tianshou as ts
from env import SimEnv
import torch
import numpy as np
from torch import nn


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


def run():
    train_env = SimEnv()
    test_env = SimEnv()
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    policy = ts.policy.DQNPolicy(
        net, optim, 0.9, estimation_step=3, target_update_freq=100)
    train_collector = ts.data.Collector(
        policy, train_env, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = ts.data.Collector(
        policy, test_env, exploration_noise=True)
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=10000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        train_fn=lambda epoch, env_step: policy.set_eps(0.1),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= 30)
    print(f'Finished training! Use {result["duration"]}')
