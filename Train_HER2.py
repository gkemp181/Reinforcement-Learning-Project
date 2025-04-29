import os
os.environ["MUJOCO_GL"] = "egl"  # or "osmesa" if EGL unavailable

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import wandb
from wandb.integration.sb3 import WandbCallback
from create_env import create_env
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from gymnasium.wrappers import RecordEpisodeStatistics

# Variables
run_name = "Train_HER_4_28"
record_frequency = 2000  # record the agent's episode every 250
save_frequency = 500  # save the model every 100 episodes
show_progress_bar = True  # show progress bar during training
total_timesteps = 10_000_000  # total number of training timesteps

run = wandb.init(
    project="pickup-and-place",
    name=run_name,
    sync_tensorboard=True,      # ⬅️ auto-upload SB3 logs
    save_code=True,
    monitor_gym=False,
    config={
        "algo": "SAC",
        "env": "FetchPickAndPlace-v3",
        "total_timesteps": total_timesteps,
        "video_freq_steps": record_frequency,
        "save_freq_steps": save_frequency,
    }
)
# 1. Create Env
# Initialize Environment With Recording and WandB Logging
env = create_env(render_mode="rgb_array")
env = RecordVideo(env, video_folder=f"recordings/{run.name}/{run.id}", name_prefix="training",
                  video_length=200,
                episode_trigger=lambda x: x % run.config.video_freq_steps == 0)

env = DummyVecEnv([lambda: env])
env = VecMonitor(
    env,
    info_keywords=("is_success",),
)


# 2. Model Config
model = SAC(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=3e-4,            # You can tune this down if unstable
    buffer_size=int(1e6),          # Big replay buffer
    batch_size=256,                # High batch size
    tau=0.05,                      # Soft update
    gamma=0.95,                    # Slightly lower discount for faster reward propagation
    ent_coef="auto",
    train_freq=1,
    gradient_steps=1,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=8,           # Higher HER sampling
        goal_selection_strategy=GoalSelectionStrategy.FUTURE,
        online_sampling=True,
        max_episode_length=50,     # FetchPickAndPlace typically uses 50 step episodes
    ),
    verbose=1,
    tensorboard_log="./tensorboard/fetch_pick_place",
)

# 3. Train
model.learn(
    total_timesteps=10_000_000,
    log_interval=10,  # log every 10 episodes
    progress_bar=True,
    callback=WandbCallback(
        model_save_path=f"./models/{run.name}/{run.id}",
        model_save_freq=run.config.save_freq_steps,  
        verbose=2,
    )
)


# Evaluation loop
obs, info = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, trunc, info = env.step(action)
    if done or trunc:
        obs, info = env.reset()


run.log_artifact(model, "model")
env.close()
wandb.finish()
