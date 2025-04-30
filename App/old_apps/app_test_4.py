# <-- this must come first, before any mujoco / gym imports
import os
os.environ["MUJOCO_GL"] = "osmesa"


import gradio as gr
import numpy as np
import torch
import imageio
import time
from stable_baselines3 import SAC
from custom_env import create_env

def stream_frames():
    x_start, y_start = 0.0, 0.0
    x_targ, y_targ, z_targ = 0.1, 0.1, 0.1

    env = create_env(render_mode="rgb_array",
                     block_xy=(x_start, y_start),
                     goal_xyz=(x_targ, y_targ, z_targ))

    checkpoint_path = os.path.join("App", "model", "model.zip")
    model = SAC.load(checkpoint_path, env=env, verbose=1)

    obs, info = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)

        frame = env.render()  # Grab RGB frame
        yield frame  # Yield this frame to Gradio

        if done or trunc:
            obs, info = env.reset()

        time.sleep(0.033)  # ~30 FPS (1/30 seconds)

    env.close()

# Build Gradio app
with gr.Blocks() as demo:
    gr.Markdown("Fetch Robot: Live Model Demo App")
    frame_output = gr.Image()
    start_button = gr.Button("Start Streaming")

    start_button.click(fn=stream_frames, inputs=[], outputs=frame_output)

demo.queue()
demo.launch(share=True)