import gradio as gr
import os
import numpy as np
import torch
import imageio
from stable_baselines3 import SAC
from custom_env import create_env

# Define the function that runs the model and outputs a video
def run_model_episode():
    # 1. Create environment with render_mode="rgb_array" (needed to capture frames)
    # e.g. user inputs:
    # Relative to center of table
    x_start, y_start = 0.0, 0.0
    x_targ, y_targ, z_targ = 0.1, 0.1, 0.1

    env = create_env(render_mode="rgb_array",
                    block_xy=(x_start, y_start),
                    goal_xyz=(x_targ, y_targ, z_targ))

    # 2. Load your trained model
    checkpoint_path = os.path.join("model", "model.zip")
    model = SAC.load(checkpoint_path, env=env, verbose=1)

    # 3. Rollout the episode
    frames = []
    obs, info = env.reset()

    for _ in range(200):  # Shorter rollout to avoid giant videos
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)

        frame = env.render()  # Get current frame as image (rgb_array)
        frames.append(frame)

        if done or trunc:
            obs, info = env.reset()

    env.close()

    # TODO This will probably need to save into a unique directory 
    # so it doesnt override when multiple people are running the app

    # 4. Save the frames into a video
    video_path = "run_video_2.mp4"
    imageio.mimsave(video_path, frames, fps=30)

    # 5. Return path to Gradio to display
    return video_path

# --------------------------------------
# Build the Gradio App
# --------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("Fetch Robot: Model Demo App")
    gr.Markdown("Click 'Run Model' to watch the SAC agent interact with the FetchPickAndPlace environment.")

    run_button = gr.Button("Run Model")
    output_video = gr.Video()

    run_button.click(fn=run_model_episode, inputs=[], outputs=output_video)

demo.launch(share=True)