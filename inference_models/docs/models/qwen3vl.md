# Qwen3-VL - Vision Language Model

Qwen3-VL is the latest vision-language model from Alibaba Cloud's Qwen team, offering enhanced visual understanding and reasoning capabilities.

## Overview

Qwen3-VL is an advanced multimodal model capable of:

- **Visual Question Answering** - Answer complex questions about image content
- **Image Captioning** - Generate detailed descriptions of images
- **Multi-image Understanding** - Reason across multiple images simultaneously
- **Fine-grained Recognition** - Identify specific objects, text, and details
- **Spatial Reasoning** - Understand spatial relationships and layouts
- **Document Understanding** - Parse and analyze document content

!!! warning "GPU Recommended"
    Qwen3-VL works best with GPU acceleration. CPU inference may be very slow or may not work properly.

!!! info "License & Attribution"
    **License**: Apache 2.0<br>**Source**: [Qwen Team](https://github.com/QwenLM/Qwen3-VL)

## Pre-trained Model IDs

Qwen3-VL pre-trained models are available and do **not** require a Roboflow API key.

| Model ID | Description |
|----------|-------------|
| `qwen3vl-2b-instruct` | 2B parameter instruction-tuned model - optimized for following instructions |

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
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via Qwen3-VL block |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU recommended) |

## Usage Examples

### Visual Question Answering

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("qwen3vl-2b-instruct")
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
model = AutoModel.from_pretrained("qwen3vl-2b-instruct")
image = cv2.imread("path/to/image.jpg")

# Generate detailed caption
captions = model.prompt(
    images=image,
    prompt="Describe this image in detail.",
    max_new_tokens=512
)
print(f"Caption: {captions[0]}")
```

### Document Analysis

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("qwen3vl-2b-instruct")
image = cv2.imread("path/to/document.jpg")

# Extract information from document
answers = model.prompt(
    images=image,
    prompt="Extract all the key information from this document.",
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

Qwen3-VL can be used in Roboflow Workflows for complex computer vision pipelines. The Qwen3-VL block supports all task types and can be combined with other blocks for advanced processing.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Use GPU** - Qwen3-VL requires GPU for acceptable performance
2. **Optimize prompts** - Clear, specific prompts yield better results
3. **Adjust max_new_tokens** - Increase for longer responses, decrease for faster inference
4. **Fine-tune for your domain** - LoRA fine-tuning significantly improves task-specific performance

