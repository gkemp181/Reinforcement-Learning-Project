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
RUN_ID = "ll4bx47z"  # NOT the display name; the ID like "3xi2sld8"

run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

# Collect all images and metrics
logged_images = []
logged_scalars = []

for row in run.scan_history():
    for key, val in row.items():
        if isinstance(val, wandb.data_types.Image):
            logged_images.append((key, val.url))
        if isinstance(val, (int, float)):  # Scalars like loss, accuracy
            logged_scalars.append((key, val))

# Build lists for dropdowns
image_keys = [key for key, _ in logged_images]
scalar_keys = [key for key, _ in logged_scalars]

# Function to fetch and return an image
def view_image(selected_key):
    for key, url in logged_images:
        if key == selected_key:
            response = requests.get(url)
            return Image.open(BytesIO(response.content))
    return None

# Function to view a logged scalar
def view_scalar(selected_key):
    for key, value in logged_scalars:
        if key == selected_key:
            return f"{key}: {value}"
    return "Not found"

# Gradio tabs: one for images, one for scalars
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
