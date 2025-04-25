import torch
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from your_wrapper_module import RandomizedFetchWrapper, create_env  # Adjust the import as needed

# Make environment with rendering
env = create_env(render_mode="human")
obs, info = env.reset()

# Get obs/action dimensions from the environment
obs_dim = obs["observation"].shape[0]
act_dim = env.action_space.shape[0]

# Define your model architecture (must match the saved model)
class PickPlacePolicy(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_dim),
            torch.nn.Tanh()  # Because action range is usually [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

# Load pretrained model
model = PickPlacePolicy(obs_dim, act_dim)
model.load_state_dict(torch.load("pickplace_model.pth"))  # Adjust path
model.eval()

# Run 1 episode
done = False
while not done:
    obs_tensor = torch.tensor(obs["observation"], dtype=torch.float32)
    with torch.no_grad():
        action = model(obs_tensor).numpy()

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()

