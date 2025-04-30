# <-- this must come first, before any mujoco / gym imports
import os
os.environ["MUJOCO_GL"] = "osmesa"


import gradio as gr
import wandb
import requests
from PIL import Image
from io import BytesIO

# Connect to W&B
api = wandb.Api()

# Replace this with your correct run path
ENTITY = "jarrett-defreitas-university-of-rhode-island"  # your wandb username or team
PROJECT = "pickup-and-place"
RUN_ID = "trr5oagz"  # NOT the display name; the ID like "3xi2sld8"

run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

# Collect all images and metrics
logged_images = []
logged_scalars = []

# Scan all rows of logged history
for row in run.scan_history():
    for key, val in row.items():
        # Handle images correctly
        if isinstance(val, list):
            for item in val:
                if isinstance(item, wandb.data_types.Image):
                    logged_images.append((key, item.url))
        elif isinstance(val, wandb.data_types.Image):
            logged_images.append((key, val.url))
        
        # Handle scalars (numbers like loss, accuracy)
        if isinstance(val, (int, float)):
            logged_scalars.append((key, val))

# Debug: show what was found
print("Logged Images:", logged_images)
print("Logged Scalars:", logged_scalars)

# --------------------------------------
# 3. Prepare Dropdown Choices
# --------------------------------------

image_keys = [key for key, _ in logged_images]
scalar_keys = [key for key, _ in logged_scalars]

# --------------------------------------
# 4. Define viewer functions
# --------------------------------------

# View image by selected key
def view_image(selected_key):
    for key, url in logged_images:
        if key == selected_key:
            response = requests.get(url)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            else:
                return None
    return None

# View scalar (number) by selected key
def view_scalar(selected_key):
    for key, value in logged_scalars:
        if key == selected_key:
            return f"{key}: {value}"
    return "Not found"

# --------------------------------------
# 5. Build the Gradio App
# --------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# 📈 WandB Run Viewer")
    gr.Markdown("View images and metrics logged to a specific W&B run.")

    with gr.Tab("Logged Images"):
        img_selector = gr.Dropdown(choices=image_keys, label="Select an Image Key")
        img_display = gr.Image()

        img_selector.change(fn=view_image, inputs=img_selector, outputs=img_display)

    with gr.Tab("Logged Scalars"):
        scalar_selector = gr.Dropdown(choices=scalar_keys, label="Select a Scalar Metric")
        scalar_display = gr.Textbox()

        scalar_selector.change(fn=view_scalar, inputs=scalar_selector, outputs=scalar_display)

demo.launch(share=True)
