# Qwen3.8 - Vision Language Model

Qwen3.8 is a native vision-language model from Alibaba Cloud's Qwen team. The open-weights 27B dense model shares the Qwen3.5 hybrid architecture (Gated DeltaNet linear attention interleaved with full attention) and delivers substantially stronger multimodal reasoning, document understanding, and visual grounding than the smaller Qwen3.5 checkpoints.

## Overview

Qwen3.8 is a multimodal model capable of:

- **Visual Question Answering** - Answer complex questions about image content
- **Image Captioning** - Generate detailed descriptions of images
- **Visual Reasoning** - Multi-step logical reasoning over images, including scientific problems and puzzles
- **Document Understanding** - Parse and analyze document content, OCR, and chart reading
- **Spatial Intelligence** - Object counting, relative positioning, and spatial relationship understanding
- **Fine-grained Recognition** - Identify specific objects, text, and details

!!! warning "Large-GPU Required"
    The 27B model needs roughly 56 GB of VRAM in bf16 (80GB-class GPU recommended). CPU inference is not practical.

!!! info "License & Attribution"
    **License**: Apache 2.0<br>**Source**: [Qwen Team](https://github.com/QwenLM)

!!! info "Transformers version"
    Qwen3.8 requires `transformers>=5.8.0` (it reuses the `qwen3_5` architecture class, but ships a new tokenizer and chat template).

## Pre-trained Model IDs

| Model ID | Description |
|----------|-------------|
| `qwen3_8-27b` | 27B parameter dense vision-language model |

You can also use fine-tuned models from Roboflow by specifying `project/version` as the model ID (requires API key).

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

## Usage Examples

### Visual Question Answering

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("qwen3_8-27b")
image = cv2.imread("path/to/image.jpg")

# Ask a question
answers = model.prompt(
    images=image,
    prompt="What objects are visible in this image?",
    max_new_tokens=512
)
print(f"Answer: {answers[0]}")
```

## Performance Tips

1. **Use a large GPU** - the 27B model needs an 80GB-class GPU in bf16
2. **Optimize prompts** - Clear, specific prompts yield better results
3. **Adjust max_new_tokens** - Increase for longer responses, decrease for faster inference
