from gym.envs.registration import register

register(
    id="simenv-v0",
    entry_point="sim.env:SimEnv",
    max_episode_steps=800
)
