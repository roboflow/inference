# Gemma 4 - Vision Language Model

Gemma 4 is Google’s multimodal instruction-tuned model family (vision + text), loaded in `inference-models` through Hugging Face `transformers` via the `Gemma4HF` class in `inference_models/models/gemma4/gemma4_hf.py`.

## Overview

The `Gemma4HF` implementation supports:

- **Visual question answering and captioning** — natural-language prompts over one or more images
- **Hugging Face checkpoints** — `AutoModelForMultimodalLM` and `AutoProcessor` with optional 4-bit quantization on CUDA
- **Roboflow-style packages** — optional `inference_config.json` for resize / letterbox preprocessing, and PEFT layouts with a `base/` directory plus `adapter_config.json`
- **Configurable visual token budget** — optional `gemma_image_seq_length` when loading (values `70`, `140`, `280`, `560`, `1120`)

!!! warning "GPU recommended"

    Gemma 4 is intended for GPU inference. CPU runs may be impractically slow for larger checkpoints.

!!! info "License & attribution"

    **License**: Gemma Terms of Use<br>**Terms**: By using Gemma 4 you agree to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

## Pre-trained model IDs

Hosted Gemma 4 checkpoints use the Hugging Face backend (`hugging-face`). Pass one of these **model IDs** to `AutoModel.from_pretrained` (they select which checkpoint package is downloaded):

| Model ID | Notes |
|----------|--------|
| `gemma-4-e2b-it` | Smaller instruction-tuned variant |
| `gemma-4-e4b-it` | Mid-size instruction-tuned variant |
| `gemma-4-31b-it` | Large instruction-tuned variant |
| `gemma-4-26b-a4b-it` | Alternate large configuration |

**Registry architecture:** `inference-models` registers a single implementation under `model_architecture` **`gemma-4`** (vision-language task, `hugging-face` backend), the same way families like YOLOv8 use one architecture key with variant-specific **model IDs**. The Roboflow weights API should return `modelArchitecture: "gemma-4"` for every hosted Gemma 4 checkpoint while keeping the variant-specific `modelId` above; optional `modelVariant` can describe the checkpoint for UIs.

For **local** packages, use `"model_architecture": "gemma-4"` in `model_config.json` (see `examples/gemma4/model_config.example.json` and [Load Models Locally](../how-to/local-packages.md)); Roboflow `project/version` paths follow the same layout rules.

## Supported backends

| Backend | Extras required |
|---------|-----------------|
| `hugging-face` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` (match your CUDA stack) |

When calling `AutoModel.from_pretrained`, pass `backend="hugging-face"` if you want to force the Hugging Face package path.

## Roboflow platform compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ LoRA-style adapters (layout with `adapter_config.json`; base weights under `base/`) |
| **Upload weights** | ✅ Compatible with standard model packages |
| **Serverless API (v2)** | ⚠️ Limited support (not yet fully stable) |
| **Workflows** | ⚠️ Platform support may evolve; self-hosted `inference-models` is the primary path |
| **Edge deployment (Jetson)** | ❌ Not targeted for this stack |
| **Self-hosting** | ✅ Use `inference-models` on GPU |

## Usage examples

### Basic prompt

```python
import cv2
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "gemma-4-e2b-it",
    backend="hugging-face",
)
image = cv2.imread("path/to/image.jpg")

answers = model.prompt(
    images=image,
    prompt="What is the main subject in this image?",
    max_new_tokens=256,
    do_sample=False,
)
print(answers[0])
```

### Custom system prompt

`Gemma4HF` accepts a single user-facing `prompt` string. Append `<system_prompt>` followed by the system text to override the default system message (see `pre_process_generation` in `gemma4_hf.py`).

```python
user = "How many backpacks are clearly visible?"
system = "Answer with a number first, then one short sentence."
prompt = f"{user}<system_prompt>{system}"

answers = model.prompt(images=image, prompt=prompt, input_color_format="bgr")
```

### Loading options (`Gemma4HF.from_pretrained`)

Extra keyword arguments forwarded from `AutoModel.from_pretrained` include:

| Argument | Purpose |
|----------|---------|
| `gemma_image_seq_length` | Sets `processor.image_seq_length` to one of `70`, `140`, `280`, `560`, `1120` (visual token budget; higher = more detail, more compute) |
| `disable_quantization` | On CUDA, default is 4-bit NF4 unless this is `True` |
| `quantization_config` | Optional `BitsAndBytesConfig` to override default quantization |
| `trust_remote_code` | Passed to Hugging Face loaders |
| `local_files_only` | Default `True` in `Gemma4HF` — set `False` if the loader must reach the network |

## Defaults and environment variables

Generation defaults (`max_new_tokens`, `do_sample`, `enable_thinking`, sampling hyperparameters, `skip_special_tokens`) are driven by `inference_models.configuration` and can be tuned with `INFERENCE_MODELS_GEMMA4_*` variables. See [Configure environment variables](../how-to/environment-variables.md#gemma-4).

## Example script

The repository includes a small end-to-end sample: `inference_models/examples/gemma4/count_backpacks.py` (run from the package / monorepo root with `uv run` as described in that file’s docstring).

## Performance tips

1. **Prefer GPU** with a recent PyTorch + `transformers` stack aligned to the checkpoint card.
2. **Lower `gemma_image_seq_length`** for faster runs when fine visual detail is not required; raise it for OCR, small text, or dense scenes.
3. **Set `do_sample=False`** for deterministic answers when temperature sampling is not needed.
4. **Match `input_color_format`** to your buffer layout (`bgr` for OpenCV `cv2.imread` by default, `rgb` for RGB numpy arrays).
