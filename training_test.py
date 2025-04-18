import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from create_env import create_env

env = create_env()

# Pass the wrapped env to SAC
model = SAC("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)

# Later, evaluation automatically uses the same fix
obs, info = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, trunc, info = env.step(action)
    env.render()
    if done or trunc:
        obs, info = env.reset()
