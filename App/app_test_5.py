import gradio as gr
import os
import numpy as np
import torch
import imageio
from stable_baselines3 import SAC
from custom_env import create_env

# Define the function that runs the model and outputs a video
def run_model_episode(x_start, y_start, x_targ, y_targ, z_targ):
    # Create environment with user inputs
    env = create_env(render_mode="rgb_array",
                     block_xy=(x_start, y_start),
                     goal_xyz=(x_targ, y_targ, z_targ))

    # Load your trained model
    checkpoint_path = os.path.join("App", "model", "model.zip")
    model = SAC.load(checkpoint_path, env=env, verbose=1)

    # Rollout the episode
    frames = []
    obs, info = env.reset()

    for _ in range(200):  # Shorter rollout
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)

        frame = env.render()
        frames.append(frame)

        if done or trunc:
            obs, info = env.reset()

    env.close()

    # Save frames into a video
    video_path = "run_video.mp4"
    imageio.mimsave(video_path, frames, fps=30)

    return video_path

# --------------------------------------
# Build the Gradio App
# --------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("## Fetch Robot: Model Demo App")
    gr.Markdown("Enter start and target coordinates, then click 'Run Model' to watch the robot!")
    gr.Markdown("Coordinates are relative to the center of the table.")
    gr.Markdown("X and Y coordinates are in meters, Z coordinate is height in meters.") 
    gr.Markdown("0,0,0 is the center of the table.")

    with gr.Row():
        x_start = gr.Number(label="Start X", value=0.0)
        y_start = gr.Number(label="Start Y", value=0.0)

    with gr.Row():
        x_targ = gr.Number(label="Target X", value=0.1)
        y_targ = gr.Number(label="Target Y", value=0.1)
        z_targ = gr.Number(label="Target Z", value=0.1)

    run_button = gr.Button("Run Model")
    output_video = gr.Video()

    run_button.click(
        fn=run_model_episode,
        inputs=[x_start, y_start, x_targ, y_targ, z_targ],
        outputs=output_video
    )

demo.launch(share=True)

