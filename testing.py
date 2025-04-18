import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO

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