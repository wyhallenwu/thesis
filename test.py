import gym
import sim


env = gym.make('simenv-v0')
env.reset()
env.step("test")
env.close()
