import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import wandb
from wandb.integration.sb3 import WandbCallback
from create_env import create_env

# 1. Init W&B with TensorBoard sync
run = wandb.init(
    project="pickup-and-place",
    name="sac_test3",
    sync_tensorboard=True,      # ⬅️ auto-upload SB3 logs
    save_code=True,
)

# 2. Create & wrap your environment for episode stats
raw_env = create_env(render_mode=None)
mon_env = Monitor(raw_env, info_keywords=("is_success",))  # records ep length, reward, and is_success
vec_env = DummyVecEnv([lambda: mon_env])
env    = VecMonitor(vec_env)  # gives you rollout/success_rate on top of ep_len/_rew

# 3. Build your SAC model (still logs to TB dir)
model = SAC(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard/",
)

# 4. Train with periodic model checkpoints
model.learn(
    total_timesteps=200_000,
    callback=WandbCallback(
        gradient_save_freq=100,              # log gradients every 100 updates
        model_save_path=f"./models/{run.name}",
        model_save_freq=300,              # save model every 10 000 timesteps
        verbose=2,
    ),
)

# Later, evaluation automatically uses the same fix
obs, info = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, trunc, info = env.step(action)
    if done or trunc:
        obs, info = env.reset()

env.close()

wandb.finish()
