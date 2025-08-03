# Manga Colorization Project

This project focuses on using deep learning models to colorize black-and-white manga images. It leverages technologies like PyTorch, Hugging Face Transformers, and Modal for cloud-based execution.

## Project Structure

- **`test.py`**: Sets up the cloud-based training environment using Modal, instantiates libraries and dependencies, and triggers the training process.

- **`inference.py`**: Loads pre-trained models and performs colorization using pre-trained LoRA weights on selected black-and-white manga images.

- **`train_control_lora_flux.py`**: Contains the core logic for training the Flux model on a dataset of paired black-and-white and colored manga images.

- **`manga_images/`**: Contains the training dataset with paired black-and-white (`bw_image_*.png`) and colored (`color_image_*.png`) manga images.

- **`results/`**: Directory containing the trained model weights (`pytorch_lora_weights.safetensors`).

- **`inference-results/`**: Output directory containing the colorized manga images generated during inference.

## Dataset

The dataset consists of 50 paired manga images:
- **Black-and-white images**: `bw_image_1.png` to `bw_image_50.png`
- **Colored reference images**: `color_image_1.png` to `color_image_50.png`

## Training Process

1. **Data Preparation**: Setup a dataset of black-and-white and colored manga images.
2. **Model Setup**: Use a Flux transformer with LoRA configuration for personalized weight updates.
3. **Training Execution**: Train on cloud with customized configurations like learning rate, batch size, etc.
4. **Validation**: Validate model performance at regular intervals.
5. **Result Storage**: Save trained weights for inference and push results to designated repository if required.

### Training Configuration
- **Model**: FLUX.1-dev transformer with LoRA fine-tuning
- **Training Steps**: 200 (reduced for quick testing)
- **Batch Size**: 4
- **Learning Rate**: 1e-4
- **LoRA Rank**: 16
- **Mixed Precision**: bf16
- **Cloud Platform**: Modal with A100-40GB GPU

### How to Run Training

```bash
python test.py
```

Make sure to configure paths to your dataset and any preferred model configurations before starting the script.

## Inference Process

1. **Setup**: Create and attach a storage volume for results.
2. **Model Loading**: Load pre-trained FLUX model and trained LoRA weights.
3. **Inference**: Colorize images using prompts for enhanced visual output and save the results.

### How to Run Inference

```bash
python inference.py
```

The script will process images from `bw_image_15.png` to `bw_image_25.png` and generate colorized outputs in the `inference-results/` directory.

## Requirements

- **Python 3.11**
- **PyTorch**
- **Transformers**
- **Diffusers**
- **Modal**
- **PEFT** (Parameter-Efficient Fine-Tuning)
- **Accelerate**
- **PIL (Pillow)**
- **NumPy**
- **Safetensors**

### Installation

Install the necessary dependencies:

```bash
pip install torch torchvision transformers diffusers peft accelerate pillow numpy safetensors modal
```

## Project Description for Resume

**Manga Colorization using FLUX Diffusion Models**
Developed a deep learning pipeline to automatically colorize black-and-white manga images using FLUX transformer models with LoRA fine-tuning, achieving high-quality color generation through flow-matching diffusion techniques and cloud-based training infrastructure. Implemented custom dataset preprocessing for paired BW-color manga images and deployed the training workflow on Modal's cloud platform with GPU acceleration. Successfully trained and validated the model to generate vibrant, contextually-appropriate colorizations while preserving original manga artwork structure and style.

## Key Domains & Fields
- **Computer Vision** - Image-to-image translation, colorization
- **Deep Learning** - Transformer architectures, diffusion models
- **Generative AI** - Flow-matching models, conditional image generation
- **Machine Learning Engineering** - Model fine-tuning, LoRA adaptation
- **Cloud Computing** - Modal platform, distributed training
- **Neural Networks** - FLUX architecture, attention mechanisms

## Results

The trained model successfully generates colorized manga images that:
- Maintain the original artwork structure and layout
- Apply contextually appropriate colors
- Preserve fine details and artistic style
- Generate vibrant and visually appealing results

Sample outputs can be found in the `inference-results/` directory.

## License

Please refer to the licensing terms [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
