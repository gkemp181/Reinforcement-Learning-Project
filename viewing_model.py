import gymnasium as gym
import time
import numpy as np
from stable_baselines3 import SAC  # or another algorithm like SAC or TD3

# Load the MuJoCo "Pick and Place" environment
env = gym.make("FetchPickAndPlace-v2", render_mode="human")  # "human" for real-time rendering

# Load the pretrained model
# Replace 'path_to_model.zip' with the actual path to your trained agent
model = SAC.load("C:\Users\defre\OneDrive\Desktop\ELE392Project\Reinforcement-Learning-Project\models\recording_test1.zip")

obs, _ = env.reset()

# Run the model in the environment
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()

    time.sleep(0.01)  # Slow down rendering to be watchable

env.close()

