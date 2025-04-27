import os
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import gymnasium_robotics
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import wandb
from wandb.integration.sb3 import WandbCallback
from create_env import create_env


run_name = "recording_test2"


training_period = 250  # record the agent's episode every 250
num_training_episodes = 10_000  # total number of training episodes
load_model = True  # whether to load a pre-trained model or not
checkpoint_path = f"./models/{run_name}/model.zip"

with wandb.init(name=run_name, project="pickup-and-place") as run:

    env = create_env(render_mode="rgb_array")
    env = RecordVideo(env, video_folder=f"sac_recording_test/{run_name}", name_prefix="training",
                    episode_trigger=lambda x: x % training_period == 0)
    env = RecordEpisodeStatistics(env)

    # 3. Build your SAC model (still logs to TB dir)
    if load_model and os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model = SAC.load(checkpoint_path, env=env, verbose=1)
    else:
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=1,
    )

    # 4. Train with periodic model checkpoints
    model.learn(
        total_timesteps=200_000,
        callback=WandbCallback(
            gradient_save_freq=training_period,              # log gradients every 100 updates
            model_save_path=f"./models/{run.name}",
            model_save_freq=training_period,              # save model every 10 000 timesteps
            verbose=2,
        ),
    )

    # Training loop
    for episode_num in range(num_training_episodes):
        obs, info = env.reset()

        episode_over = False
        while not episode_over:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_over = terminated or truncated

        run.log({f"episode-{episode_num}": info["episode"]})
        print(f"Episode {episode_num} finished with reward: {info['episode']['r']}, length: {info['episode']['l']}")
    
    run.log_artifact(model, "model")

    env.close()

    wandb.finish()




