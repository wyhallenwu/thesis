from sim.env import SimEnv
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces import Discrete
env = SimEnv()
env = FlattenObservation(env)
ac = Discrete(100)
print(env.observation_space.sample())
print(env.action_space.sample())
print(ac.shape)
print(env.observation_space.shape)
