import os
import gradio as gr
import numpy as np
import torch
import imageio
from stable_baselines3 import SAC
from custom_env import create_env

# Update your run function to accept a model_name
def run_model_episode(x_start, y_start, x_targ, y_targ, z_targ, model_name, random_coords):
    
    # map the radio‐choice to the actual checkpoint on disk
    model_paths = {
        "Pick & Place (HER)": "App/model/pick_and_place_her.zip",
        "Pick & Place (Dense)":        "App/model/pick_and_place_dense.zip",
        "Push":         "App/model/push.zip",
        "Reach":         "App/model/reach.zip",
    }
    checkpoint_path = model_paths[model_name]

    # map the radio‐choice to the actual environment name
    environments = {
        "Pick & Place (HER)": "FetchPickAndPlace-v3",
        "Pick & Place (Dense)":        "FetchPickAndPlaceDense-v3",
        "Push":         "FetchPush-v3",
        "Reach":         "FetchReach-v3",
    }
    environment = environments[model_name]

    # Handle environment coordinates
    if(environment == "FetchPush-v3"):
        z_targ = 0.0

    block_xy=(x_start, y_start),
    goal_xyz=(x_targ, y_targ, z_targ)

    if random_coords:
        block_xy = None
        goal_xyz = None
        
    # create the env
    env = create_env(
        render_mode="rgb_array",
        block_xy=block_xy,
        goal_xyz=goal_xyz,
        environment=environment
    )

    # load the selected model
    model = SAC.load(checkpoint_path, env=env, verbose=0)

    frames = []
    obs, info = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        frames.append(env.render())
        if done or trunc:
            obs, info = env.reset()
    env.close()

    video_path = "run_video.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    return video_path


with gr.Blocks() as demo:
    gr.Markdown("## Fetch Robot: Model Demo App")
    gr.Markdown("Enter coordinates, pick a model, then click **Run Model**.")
    gr.Markdown("Coordinates are relative to the center of the table.")

    # 1) add a radio (or gr.Dropdown) for model selection
    model_selector = gr.Radio(
        choices=["Pick & Place (HER)", "Pick & Place (Dense)", "Push", "Reach"],
        value="Pick & Place (HER)",
        label="Select a model/environment"
    )

    # Randomize coordinates
    randomize = gr.Checkbox(
        label="Use randomized coordinates?",
        value=False
    )

    with gr.Row():
        x_start = gr.Number(label="Start X", value=0.0)
        y_start = gr.Number(label="Start Y", value=0.0)

    with gr.Row():
        x_targ = gr.Number(label="Target X", value=0.1)
        y_targ = gr.Number(label="Target Y", value=0.1)
        z_targ = gr.Number(label="Target Z", value=0.1)

    run_button   = gr.Button("Run Model")
    output_video = gr.Video()

    # 2) include the selector as an input to your click callback
    run_button.click(
        fn=run_model_episode,
        inputs=[x_start, y_start, x_targ, y_targ, z_targ, model_selector, randomize],
        outputs=output_video
    )

demo.launch(share=True)
