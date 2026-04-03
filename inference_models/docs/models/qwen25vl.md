# Qwen2.5-VL - Vision Language Model

Qwen2.5-VL is a state-of-the-art vision-language model developed by Alibaba Cloud that excels at understanding images and answering questions about visual content.

## Overview

Qwen2.5-VL is a powerful multimodal model capable of:

- **Visual Question Answering** - Answer complex questions about image content
- **Image Captioning** - Generate detailed descriptions of images
- **Multi-image Understanding** - Reason across multiple images simultaneously
- **Fine-grained Recognition** - Identify specific objects, text, and details
- **Spatial Reasoning** - Understand spatial relationships and layouts

!!! warning "GPU Recommended"
    Qwen2.5-VL works best with GPU acceleration. CPU inference may be very slow or may not work properly.

!!! info "License & Attribution"
    **License**: Apache 2.0<br>**Source**: [Qwen Team](https://github.com/QwenLM/Qwen2-VL)<br>**Paper**: [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)

## Pre-trained Model IDs

Qwen2.5-VL pre-trained models are available and do **not** require a Roboflow API key.

| Model ID | Description |
|----------|-------------|
| `qwen25-vl-7b` | 7B parameter model - balanced performance and speed |

You can also use fine-tuned models from Roboflow by specifying `project/version` as the model ID (requires API key).

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ LoRA fine-tuning only |
| **Upload Weights** | ✅ Upload fine-tuned models |
| **Serverless API (v2)** | ⚠️ Limited support (not yet fully stable) |
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via Qwen2.5-VL block |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU recommended) |

## Usage Examples

### Visual Question Answering

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
image = cv2.imread("path/to/image.jpg")

# Ask a question
answers = model.prompt(
    images=image,
    prompt="What objects are visible in this image?",
    max_new_tokens=512
)
print(f"Answer: {answers[0]}")
```

### Image Captioning

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
image = cv2.imread("path/to/image.jpg")

# Generate detailed caption
captions = model.prompt(
    images=image,
    prompt="Describe this image in detail.",
    max_new_tokens=512
)
print(f"Caption: {captions[0]}")
```

### Multi-image Understanding

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Load multiple images
image1 = cv2.imread("path/to/image1.jpg")
image2 = cv2.imread("path/to/image2.jpg")

# Compare images
answers = model.prompt(
    images=[image1, image2],
    prompt="What are the differences between these two images?",
    max_new_tokens=512
)
print(f"Answer: {answers[0]}")
```

### Using Fine-tuned Models

```python
import cv2
from inference_models import AutoModel

# Load your fine-tuned model from Roboflow
model = AutoModel.from_pretrained(
    "your-project/version",
    api_key="your_roboflow_api_key"
)

image = cv2.imread("path/to/image.jpg")

# Use with custom prompt
answers = model.prompt(
    images=image,
    prompt="your custom question",
    max_new_tokens=512
)
print(f"Answer: {answers[0]}")
```

## Workflows Integration

Qwen2.5-VL can be used in Roboflow Workflows for complex computer vision pipelines.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Use GPU** - Qwen2.5-VL requires GPU for acceptable performance
2. **Optimize prompts** - Clear, specific prompts yield better results
3. **Adjust max_new_tokens** - Increase for longer responses, decrease for faster inference

