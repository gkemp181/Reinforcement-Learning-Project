import gradio as gr
import wandb
import requests
from PIL import Image
from io import BytesIO

api = wandb.Api()

run = api.run("jarrett-defreitas-university-of-rhode-island/pickup-and-place/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]


# Extract image URLs
image_urls = [item[0]["path"] for item in media_items if isinstance(item, list)]

def get_image(index):
    if 0 <= index < len(image_urls):
        url = "https://api.wandb.ai" + image_urls[index]
        response = requests.get(url)
        return Image.open(BytesIO(response.content))
    return None

gr.Interface(fn=get_image, inputs=gr.inputs.Slider(0, len(image_urls)-1, step=1), outputs="image").launch()