#IMPORTS
import modal


#MODAL CONFIGURATION
cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

volume = modal.Volume.from_name("manga-training-results", create_if_missing=True)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "diffusers",
        "transformers",
        "torch",
        "torchvision",
        "datasets",
        "wandb",
        "bitsandbytes",
        "peft",
        "sentencepiece",
        "git+https://github.com/huggingface/diffusers.git"
    )
    .add_local_python_source("train_control_lora_flux")
    .add_local_dir("./manga_images", remote_path="/data")
)

app = modal.App("manga-lora-training",image=image)


@app.function(gpu="H100",timeout=86400, volumes={"/results": volume})
def run():
    #IMPORTS
    import torch

    # print(torch.cuda.is_available()) # check done

    from huggingface_hub import login

    login("**")

    import argparse
    import copy
    import logging
    import math
    import os
    import random
    import shutil
    from contextlib import nullcontext
    from pathlib import Path

    import accelerate
    import numpy as np
    import torch
    import transformers
    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
    from huggingface_hub import create_repo, upload_folder
    from packaging import version
    from peft import LoraConfig, set_peft_model_state_dict
    from peft.utils import get_peft_model_state_dict
    from PIL import Image
    from torchvision import transforms
    from tqdm.auto import tqdm

    import diffusers
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxControlPipeline, FluxTransformer2DModel
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import (
        cast_training_params,
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
        free_memory,
    )
    from diffusers.utils import check_min_version, is_wandb_available, load_image, make_image_grid
    from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
    from diffusers.utils.torch_utils import is_compiled_module
    
    from datasets import Dataset

    if is_wandb_available():
        import wandb
    
    check_min_version("0.33.0.dev0")

    logger = get_logger(__name__)
    
    wandb.login(key="**")

    NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]
    
    from train_control_lora_flux import main, parse_args
    
    args_list = [
        "--pretrained_model_name_or_path", "black-forest-labs/FLUX.1-dev",
        "--manga_dataset_dir", "/data",
        "--num_train_epochs", "1",
        "--bw_prefix", "bw_image_",
        "--color_prefix", "color_image_",
        "--output_dir", "/results",
        "--mixed_precision", "bf16",
        "--train_batch_size", "1",  # Reduced from 2 for stability with 10 images
        "--rank", "32",  # Lower rank for faster testing (original 64 is fine for full training)
        "--gradient_accumulation_steps", "2",  # Reduced from 4
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--learning_rate", "1e-4",  # Keep this if using AdamW8bit
        "--report_to", "wandb",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", "20",  # Drastically reduced from 3000 for quick test
        "--validation_steps", "5",  # Validate every 5 steps
        "--validation_image", "/data/bw_image_9.png",
        "--validation_prompt", "colorize the following image",  
        "--offload",
        "--seed", "0",
    ]
    
    args = parse_args(args_list)
    main(args)


if __name__ == "__main__":
    with app.run():
        train.call()


    