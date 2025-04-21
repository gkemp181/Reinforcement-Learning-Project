# import gymnasium as gym
# import numpy as np

# import gymnasium_robotics

# gym.register_envs(gymnasium_robotics)

# env = gym.make(
#     'FetchPickAndPlace-v3',
#     xml_file='./xarm7ELE392.xml',
#     forward_reward_weight=1,
#     ctrl_cost_weight=0.05,
#     contact_cost_weight=5e-4,
#     healthy_reward=1,
#     main_body=1,
#     healthy_z_range=(0.195, 0.75),
#     include_cfrc_ext_in_observation=True,
#     exclude_current_positions_from_observation=False,
#     reset_noise_scale=0.1,
#     frame_skip=25,
#     max_episode_steps=1000,
# )


import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO

import wandb
run = wandb.init(project="pickup-and-place")
run.name = "xarm7ELE392"
run.save()

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchPickAndPlace-v3", render_mode="human")
obs, info = env.reset(seed=42)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

for _ in range(1000):
   action, _ = model.predict(obs, deterministic=True)  # User-defined policy function
   obs, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      obs, info = env.reset()

env.close()