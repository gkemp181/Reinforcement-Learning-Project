import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from create_env import create_env

# render_mode = "human" allows to view the rendering as it is going
# This greatly reduces training speed
# Set to None to speed up training
# render_mode = None
env = create_env(render_mode="human")

# Pass the wrapped env to SAC
model = SAC("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)

# Later, evaluation automatically uses the same fix
obs, info = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, trunc, info = env.step(action)
    if done or trunc:
        obs, info = env.reset()
