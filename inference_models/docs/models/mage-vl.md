# Mage-VL - Codec-Native Vision Language Model

Mage-VL is Microsoft's codec-native, proactive-streaming multimodal model: a Mage-ViT visual
encoder trained from scratch, paired with a Qwen3-4B-Instruct language backbone.

!!! warning "Not available yet - weights registration pending"
    The `mage-vl` model id is not registered on the Roboflow platform yet, so
    `AutoModel.from_pretrained("mage-vl")` does not resolve. This page goes live with
    the registration. Until then the examples below will fail with a model-not-found
    error.

## Overview

What distinguishes Mage-VL from the other video-capable VLMs here is how it reads a video. Instead
of sampling frames uniformly and hoping the interesting moments land on a sampled frame, it reads
the **codec's own bitcost** — the bits the encoder spent per macroblock. Regions the codec spent
bits on are the regions that changed; predictable background is cheap and gets dropped. The
selected patches are packed into a small set of canvases, and those canvases are what the model
sees.

- **Video Question Answering** - Answer questions about the content of a video file
- **Visual Question Answering** - The usual single-image prompting
- **Spatial Reasoning** - Understand spatial relationships and layouts
- **On-screen Text** - Read scoreboards, chyrons, and captions out of video

!!! warning "GPU Recommended"
    Mage-VL works best with GPU acceleration. CPU inference may be very slow.

!!! info "License & Attribution"
    **License**: Apache 2.0<br>**Source**: [microsoft/Mage](https://github.com/microsoft/Mage)

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128` |

Video prompting additionally needs `codec-video-prep`, installed **with `--no-deps`** — it
carries a stale `numpy<2.0` pin its compiled extension does not actually need, and an
`opencv-python-headless` pin that would collide with the `opencv-python` already installed:

```bash
pip install --no-deps "codec-video-prep>=0.2.5,<0.3.0"
```

`codec-video-prep` provides the `cv-preinfer` console script the default codec engine shells
out to. **The environment's `bin` directory must be on `PATH`** — the engine resolves the binary
by name, not by import — or set `CV_PREINFER_BIN` to its full path. `ffmpeg` and `ffprobe` must
also be on `PATH`.

## Codec Engines

| Engine | Where it runs | Prep time, 30s 960x540 h264 clip → 16 canvases |
|--------|---------------|-----------------------------------------------|
| `hevc` (default) | CPU, via `cv-preinfer` | ~1.8s |
| `dcvc-rt` | GPU, via the `neural_codec/` bundled in the model package | ~25s |

`dcvc-rt` is the neural codec from the paper. It is much slower in practice unless its CUDA
kernels are compiled — without them it falls back to pytorch — and it decodes every frame up to
the last sampled one to keep temporal references valid, so cost grows with clip length rather
than with the number of canvases requested. It also needs `scipy` (an undeclared requirement
of the DCVC source bundled in the model package), which is not installed by default — install
it yourself if you opt in.

Codec results are cached on disk under `$HF_HOME/online_codec`, or under `ONLINE_CODEC_CACHE_DIR`
if set. The cache is scoped by a fingerprint of the video file's content (size, mtime, and a hash
of the first and last MiB), so overwriting a file in place is a cache miss, not a stale hit.

## Usage Examples

### Video Question Answering

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("mage-vl")

answers = model.prompt(
    video="path/to/clip.mp4",
    prompt="Describe this video in detail, including any text on screen.",
    max_new_tokens=300,
)
print(answers[0])
```

### Choosing the codec engine and canvas budget

```python
answers = model.prompt(
    video="path/to/clip.mp4",
    prompt="What happens in this video?",
    codec_engine="dcvc-rt",  # default: "hevc"
    target_canvas=24,        # canvases packed out of the video
)
```

### Visual Question Answering

```python
import cv2
from inference_models import AutoModel

model = AutoModel.from_pretrained("mage-vl")
image = cv2.imread("path/to/image.jpg")

answers = model.prompt(
    images=image,
    prompt="Describe this image in detail.",
)
print(answers[0])
```

Images are accepted as `[0, 255]` numpy arrays or torch tensors (or lists of either), matching
the other models in this package. OpenCV hands back BGR, which is assumed by default; pass
`input_color_format="rgb"` for RGB input.

## Configuration

| Environment Variable | Default | Meaning |
|---------------------|---------|---------|
| `INFERENCE_MODELS_MAGE_VL_DEFAULT_MAX_NEW_TOKENS` | `512` | Generation length cap |
| `INFERENCE_MODELS_MAGE_VL_DEFAULT_DO_SAMPLE` | `False` | Sampling vs. greedy decoding |
| `INFERENCE_MODELS_MAGE_VL_DEFAULT_CODEC_ENGINE` | `hevc` | Codec engine for video prompting |
| `INFERENCE_MODELS_MAGE_VL_DEFAULT_TARGET_CANVAS` | `16` | Canvases packed out of a video |
| `INFERENCE_MODELS_MAGE_VL_DEFAULT_MAX_PIXELS` | `153664` | Pixel budget per canvas |
| `ONLINE_CODEC_CACHE_DIR` | `$HF_HOME/online_codec` | Codec result cache |

## Notes

- The checkpoint ships its architecture as package-local code, loaded through transformers'
  dynamic module machinery from the local package directory only.
- Streaming — the model's proactive-streaming cognition gate — is not wired up. Video prompting
  takes a complete file.
