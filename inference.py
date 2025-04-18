import modal

# Create and attach the volume
volume = modal.Volume.from_name("inference-results", create_if_missing=True)

# Define the Modal app
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "diffusers",
        "torch",
        "torchvision",
        "transformers",
        "controlnet_aux",
        "safetensors",
        "Pillow",
        "xformers",
        "peft",  
        "mediapipe", 
        "accelerate"
    )
    .add_local_dir("./manga_images", remote_path="/data")
    .add_local_dir("./results", remote_path="/local")
)

app = modal.App("flux-inference", image=image)

@app.function(gpu="H100", timeout=86400, volumes={"/inference": volume})
def run():
    from diffusers import FluxControlPipeline,FluxTransformer2DModel
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np
    import torch
    import torchvision.transforms as transforms

    from huggingface_hub import login

    # Login to Hugging Face
    login("**")

    edit_transformer = FluxTransformer2DModel.from_pretrained("sayakpaul/FLUX.1-dev-edit-v0", torch_dtype=torch.bfloat16)
    # Load the base model first
    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        transformer = edit_transformer
    ).to("cuda")

    # Now load the LoRA weights
    pipe.load_lora_weights(
        "/local/pytorch_lora_weights.safetensors",
        adapter_name="finetuned_flux"
    )

    # Define the path to your input image
    image_path = "/data/bw_image_1.png"

    # Load and transform the input image to match expected resolution
    image = load_image(image_path)
    
    # Resize while maintaining aspect ratio
    
    prompt = "Give me a colourized version of this image"

    gen_images = pipe(
        prompt=prompt,
        control_image=image,
        num_inference_steps=50,
        guidance_scale=7.0,
        height = image.height,
        width = image.width,
    ).images[0]
    
    gen_images.save("/inference/output.png")