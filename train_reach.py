import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import wandb
from wandb.integration.sb3 import WandbCallback
from reach_env import create_env

# Variables
run_name = "reach_model_4_28"
record_frequency = 1000  # record the agent's episode every 250
save_frequency = 250  # save the model every 100 episodes
show_progress_bar = True  # show progress bar during training
total_timesteps = 4_000_000  # total number of training timesteps

# Initialize WandB
run = wandb.init(
    project="pickup-and-place",
    name=run_name,
    sync_tensorboard=True,      # ⬅️ auto-upload SB3 logs
    save_code=True,
    monitor_gym=False,
    config={
        "algo": "SAC",
        "env": "FetchReach-v3",
        "total_timesteps": total_timesteps,
        "video_freq_steps": record_frequency,
        "save_freq_steps": save_frequency,
    }
)

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

# Initialize Model
model = SAC(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=f"tensorboard/{run.name}/{run.id}",
)

# Train with periodic model checkpoints
model.learn(
    total_timesteps=run.config.total_timesteps,
    progress_bar=show_progress_bar,
    callback=WandbCallback(
        model_save_path=f"./models/{run.name}/{run.id}",
        model_save_freq=run.config.save_freq_steps,  
        verbose=2,
    ),
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