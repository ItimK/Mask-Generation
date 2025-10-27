import gradio as gr
from rembg import remove
from PIL import Image
import io
import numpy as np
from datetime import datetime

def process_image(input_image):
    """Removes background and returns both alpha and mask images."""
    if input_image is None:
        return None, None

    # Remove background
    output_image = remove(input_image)

    # Extract alpha channel for mask
    output_array = np.array(output_image)
    if output_array.shape[2] == 4:
        alpha_channel = output_array[:, :, 3]
    else:
        alpha_channel = np.ones((output_array.shape[0], output_array.shape[1]), dtype=np.uint8) * 255

    mask_image = Image.fromarray(alpha_channel, mode='L')

    return output_image, mask_image

# Gradio UI
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Image(type="pil", label="Transparent Image"),
        gr.Image(type="pil", label="Mask (B&W)"),
    ],
    title="🖼️ Background Remover (REMBG)",
    description="Upload an image to remove its background and generate a mask.",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
