# Flux_manga-colorization_LoRA

LoRA fine-tuning flux for the task of colorizing manga panels and pages

train_control_lora_flux contains the main modules needed for training.
<br/>
The modules can be used by calling test.py (which is the lora fine_tuning code).
<br/>
Download the safetensors file from modal after training is complete and load that file as fine-tuned weights to the flux model in inference.py for model inferencing.
<br/>
Model was trained for 1 epoch with 10 images due to limited resources. Results are shown in inference-results. Results can be improved with further fine-tuning with better resource.
