import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from sb3_contrib.common.wandb_callback import WandbCallback

import wandb

# Initialize WandB project
run = wandb.init(project="pickup-and-place", name="xarm7ELE392")

# Register environments
gym.register_envs(gymnasium_robotics)

# Create the environment
env = gym.make("FetchPickAndPlace-v3", render_mode="human")
obs, info = env.reset(seed=42)

# Initialize the model
model = PPO("MultiInputPolicy", env, verbose=1)

# Train the model with WandB logging
model.learn(
    total_timesteps=100_000,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"./models/{run.name}",
        verbose=2
    )
)

# Run the trained model
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

# Finish the WandB run
wandb.finish()