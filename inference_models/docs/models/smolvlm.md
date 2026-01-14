# SmolVLM2 - Vision Language Model

SmolVLM2 is a compact and efficient vision-language model developed by HuggingFace that provides strong multimodal understanding capabilities in a small package.

## Overview

SmolVLM2 is a lightweight VLM designed for efficiency while maintaining strong performance:

- **Visual Question Answering** - Answer questions about image content
- **Image Captioning** - Generate descriptive captions for images
- **Efficient Inference** - Smaller model size enables faster inference
- **Multi-image Support** - Process multiple images in a single prompt
- **General Understanding** - Handle diverse vision-language tasks

!!! warning "GPU Recommended"
    SmolVLM2 works best with GPU acceleration. CPU inference may be slow.

!!! info "License & Attribution"
    **License**: Apache 2.0<br>**Source**: [HuggingFace](https://huggingface.co/HuggingFaceTB/SmolVLM2-Instruct)

## Pre-trained Model IDs

SmolVLM2 pre-trained models are available and do **not** require a Roboflow API key.

| Model ID | Description |
|----------|-------------|
| `smolvlm2` | 2B parameter instruction-tuned model for general vision-language tasks |

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
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via SmolVLM2 block |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU recommended) |

## Usage Examples

### Visual Question Answering

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("smolvlm2")
image = cv2.imread("path/to/image.jpg")

# Ask a question
answers = model.prompt(
    images=image,
    prompt="What is in this image?",
    max_new_tokens=400
)
print(f"Answer: {answers[0]}")
```

### Image Captioning

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("smolvlm2")
image = cv2.imread("path/to/image.jpg")

# Generate caption (uses default prompt if none provided)
captions = model.prompt(
    images=image,
    prompt="Describe what's in this image.",
    max_new_tokens=400
)
print(f"Caption: {captions[0]}")
```

### Multi-image Understanding

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("smolvlm2")

# Load multiple images
images = [
    cv2.imread("path/to/image1.jpg"),
    cv2.imread("path/to/image2.jpg"),
    cv2.imread("path/to/image3.jpg")
]

# Analyze multiple images together in a single prompt
answers = model.prompt(
    images=images,
    prompt="What are the differences between these images?",
    images_to_single_prompt=True,
    max_new_tokens=400
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
    max_new_tokens=400
)
print(f"Answer: {answers[0]}")
```

## Workflows Integration

SmolVLM2 can be used in Roboflow Workflows for complex computer vision pipelines.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Use GPU** - SmolVLM2 benefits from GPU acceleration
2. **Optimize prompts** - Clear, specific prompts yield better results
3. **Adjust max_new_tokens** - Default is 400; increase for longer responses
4. **Multi-image prompts** - Use `images_to_single_prompt=True` to analyze multiple images together in a single prompt

