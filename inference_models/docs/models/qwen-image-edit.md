# Qwen-Image-Edit - Instruction-Based Image Editing

Qwen-Image-Edit is a diffusion-based image editing model developed by Alibaba's Qwen team.
Given a source image and a natural-language instruction (e.g. *"change the sky to a sunset"*,
*"remove the background"*), it returns an edited image.

## Overview

**Resources**: [Hugging Face Model](https://huggingface.co/Qwen/Qwen-Image-Edit), [Qwen-Image-Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-Lightning)

Key features include:

- **Instruction-following edits** - Describe the change in plain language; no masks required
- **Diffusion pipeline** - Built on `diffusers.QwenImageEditPipeline`
- **Lightning LoRA path** - Optional lightx2v step-distillation LoRA for ~4-step,
  low-VRAM inference on consumer GPUs
- **Configurable device placement** - Model-level or sequential CPU offload for
  limited-VRAM cards (see `INFERENCE_MODELS_QWEN_IMAGE_EDIT_CPU_OFFLOAD`)

## License

**Apache 2.0**

!!! info "Open Source License"
    Qwen-Image-Edit is licensed under Apache 2.0 by Alibaba's Qwen team, making it free
    for both commercial and non-commercial use.

    The lightx2v Qwen-Image-Lightning LoRA is likewise released under Apache 2.0.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Pre-trained Model IDs

| Model ID | Description |
|----------|-------------|
| `qwen-image-edit` | Qwen-Image-Edit instruction-based image editing model |

!!! warning "GPU Required"
    Qwen-Image-Edit requires GPU acceleration. CPU inference is not supported, and the
    model is not available on Roboflow Hosted Inference.

    The full base model needs a high-VRAM GPU (or `sequential` CPU offload). For consumer
    GPUs, enable the Lightning LoRA (`use_lightning_lora=True`), which combines ~4-step
    inference with sequential offload and input downscaling.

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `hugging-face` | `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` + `image-editing` |

!!! note "Additional dependency: `diffusers`"
    The pipeline is imported lazily from the `diffusers` package, which is not part of the
    core `inference-models` dependencies. Install it via the `image-editing` extra:
    `pip install "inference-models[image-editing]"`.

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ Not available for training |
| **Upload Weights** | ❌ Not supported |
| **Serverless API (v2)** | ❌ Not available (GPU-only, no hosted execution) |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via the Qwen-Image-Edit block (local GPU server) |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU required) |

## Usage Examples

### Editing an image (Lightning LoRA, direct from Hugging Face)

The Lightning path pulls the base model and LoRA straight from Hugging Face — no Roboflow
API key required — and runs in ~4 diffusion steps:

```python
import cv2
from inference_models.models.qwen_image_edit.qwen_image_edit_hf import QwenImageEditHF

model = QwenImageEditHF.from_pretrained(
    local_files_only=False,
    use_lightning_lora=True,
)

image = cv2.imread("path/to/image.jpg")

edited = model.edit(
    image=image,
    prompt="change the sky to a dramatic sunset",
    seed=42,
)
edited.save("edited.png")  # PIL Image
```

### Editing an image (full model, Roboflow registry)

```python
import cv2
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "qwen-image-edit",
    api_key="your_roboflow_api_key",
)

image = cv2.imread("path/to/image.jpg")

edited = model.edit(image=image, prompt="remove the background")
edited.save("edited.png")
```

### Generation parameters

`edit()` accepts the following optional parameters (unset values auto-select based on
whether the Lightning LoRA is active):

| Parameter | Default (base / Lightning) | Notes |
|-----------|----------------------------|-------|
| `num_inference_steps` | 28 / 4 | Diffusion denoising steps |
| `guidance_scale` | 5.0 / 1.0 | Classifier-free guidance scale |
| `seed` | None | Set for reproducible outputs |
| `scale_megapixels` | None / 0.35 | Downscale cap applied before inference (never upscales) |
| `negative_prompt` | `" "` | Standard diffusion negative prompt |
