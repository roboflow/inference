# Cosmos 3 Edge - Physical AI Vision-Language Model

Cosmos 3 Edge is NVIDIA's compact physical-AI reasoner: a 4B vision-language model that answers questions about images (and video) with an emphasis on physical scenes, and the base of the Cosmos 3 fine-tunes trained on Roboflow. Only the autoregressive reasoner tower is served here; the diffusion world model has its own entry.

## Overview

- **Visual Question Answering** - Answer questions about what is happening in a scene
- **Image Captioning** - Describe images, with a physical-world bias
- **Reasoning** - Optional `<think>` block, returned on request
- **Fine-tuning** - Roboflow multimodal projects fine-tune it as `cosmos3-edge-vlm`

!!! warning "GPU Required"
    Cosmos 3 Edge loads in bf16 and needs a CUDA GPU; CPU inference is not practical.

!!! info "License & Attribution"
    **License**: NVIDIA Open Model License (OpenMDW 1.1)<br>**Source**: [HuggingFace](https://huggingface.co/nvidia/Cosmos3-Edge)

## Pre-trained Model IDs

| Model ID | Description |
|----------|-------------|
| `nvidia/cosmos-3-edge` | The Cosmos 3 Edge reasoner (4B) |

You can also use fine-tuned models from Roboflow by specifying the model id of a `cosmos3-edge-vlm` training as the model ID (requires API key).

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128` |

`transformers>=5.15` is required (the `cosmos3_edge` model type); the loader raises a clear error on older versions.

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ LoRA fine-tuning on multimodal projects (`cosmos3-edge-vlm`); video (`cosmos3-edge`) fine-tunes are not servable yet |
| **Upload Weights** | ❌ Not supported |
| **Serverless API (v2)** | ⚠️ Limited support (not yet fully stable) |
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via the Cosmos 3 Edge block |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU required) |

## Usage Examples

### Visual Question Answering

```python
import cv2
from inference_models import AutoModel

model = AutoModel.from_pretrained("nvidia/cosmos-3-edge")
image = cv2.imread("path/to/image.jpg")

answers = model.prompt(
    images=image,
    prompt="Is the walkway free of obstacles?",
    max_new_tokens=256,
)
print(f"Answer: {answers[0]}")
```

### System Prompt

Append a system prompt after the `<system_prompt>` sentinel:

```python
answers = model.prompt(
    images=image,
    prompt="What could go wrong here?<system_prompt>You are a safety inspector.",
)
```

### Returning the Reasoning

```python
result = model.prompt(images=image, prompt="Why is the floor wet?", return_thinking=True)
print(result[0]["thinking"])
print(result[0]["answer"])
```

### Using Fine-tuned Models

A Roboflow fine-tune is a LoRA adapter over the base checkpoint; the loader applies and merges it for you.

```python
import cv2
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "your-workspace/your-model-id",
    api_key="your_roboflow_api_key",
)

image = cv2.imread("path/to/image.jpg")
answers = model.prompt(images=image, prompt="your custom question", max_new_tokens=512)
print(f"Answer: {answers[0]}")
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS` | `512` | Tokens generated when `max_new_tokens` is not given |
| `INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE` | `INFERENCE_MODELS_DEFAULT_DO_SAMPLE` | Sample instead of greedy decoding |
