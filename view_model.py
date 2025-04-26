import os
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from create_env import create_env

# File showing how to load a model and run it

env = create_env(render_mode="human")

# Path to model file
checkpoint_path = os.path.join("models", "recording_test2", "model.zip")

# Load model
model = SAC.load(checkpoint_path, env=env, verbose=1)

# Observe Model
obs, info = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, trunc, info = env.step(action)
    if done or trunc:
        obs, info = env.reset()