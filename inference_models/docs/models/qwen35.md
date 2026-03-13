# Qwen3.5 - Vision Language Model

Qwen3.5 is a native vision-language model from Alibaba Cloud's Qwen team, built on a hybrid architecture that fuses linear attention with a sparse mixture-of-experts. It excels at multimodal reasoning, coding, agent capabilities, and visual understanding.

## Overview

Qwen3.5 is a multimodal model capable of:

- **Visual Question Answering** - Answer complex questions about image content
- **Image Captioning** - Generate detailed descriptions of images
- **Visual Reasoning** - Multi-step logical reasoning over images, including scientific problems and puzzles
- **Document Understanding** - Parse and analyze document content, OCR, and chart reading
- **Spatial Intelligence** - Object counting, relative positioning, and spatial relationship understanding
- **Fine-grained Recognition** - Identify specific objects, text, and details

!!! warning "GPU Recommended"
    Qwen3.5 works best with GPU acceleration. CPU inference may be very slow or may not work properly.

!!! info "License & Attribution"
    **License**: Apache 2.0<br>**Source**: [Qwen Team](https://github.com/QwenLM/Qwen3.5)

## Pre-trained Model IDs

Qwen3.5 pre-trained models are available and do **not** require a Roboflow API key.

| Model ID | Description |
|----------|-------------|
| `qwen3_5-0.8b` | 0.8B parameter model - compact and efficient |
| `qwen3_5-2b` | 2B parameter model - better accuracy |

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
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via Qwen3.5 block |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU recommended) |

## Usage Examples

Qwen3.5 is a general-purpose vision-language model — you can prompt it with any vision or language task you have in mind. The examples below are just common starting points.

### Visual Question Answering

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("qwen3_5-0.8b")
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
model = AutoModel.from_pretrained("qwen3_5-0.8b")
image = cv2.imread("path/to/image.jpg")

# Generate detailed caption
captions = model.prompt(
    images=image,
    prompt="Describe this image in detail.",
    max_new_tokens=512
)
print(f"Caption: {captions[0]}")
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

Qwen3.5 can be used in Roboflow Workflows for complex computer vision pipelines.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Use GPU** - Qwen3.5 requires GPU for acceptable performance
2. **Optimize prompts** - Clear, specific prompts yield better results
3. **Adjust max_new_tokens** - Increase for longer responses, decrease for faster inference
