import gym
import sim


env = gym.make('simenv-v0')
env.step()
env.close()
