import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from create_env import create_env

# Initialize WandB project
run = wandb.init(project="pickup-and-place", name="test2")

# render_mode = "human" allows to view the rendering as it is going
# This greatly reduces training speed
# Set to None to speed up training
# render_mode = None
env = create_env(render_mode=None)

# Pass the wrapped env to SAC
model = SAC("MultiInputPolicy", env, verbose=1)

# Train the model with WandB logging
model.learn(
    total_timesteps=200_000,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"./models/{run.name}",
        verbose=2
    )
)

# Later, evaluation automatically uses the same fix
obs, info = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, trunc, info = env.step(action)
    if done or trunc:
        obs, info = env.reset()

env.close()

# Finish the WandB run
wandb.finish()