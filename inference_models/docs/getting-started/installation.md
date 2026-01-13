# Installation Guide

This guide covers all installation options for the `inference-models` package.

## Prerequisites

- Python 3.9 - 3.12
- pip or uv package manager
- For GPU support: CUDA-compatible GPU with appropriate drivers

## Recommended: Using uv

We recommend using `uv` for faster and more reliable installations:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Learn more about uv at [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/).

## Basic Installation

### CPU Installation

For CPU-only environments:

```bash
# Using uv (recommended)
uv pip install inference-models

# Using pip
pip install inference-models
```

This installs the base package with PyTorch CPU support.

### CPU with ONNX Backend

For running models trained on Roboflow platform (recommended for CPU):

```bash
# Using uv
uv pip install "inference-models[onnx-cpu]"

# Using pip
pip install "inference-models[onnx-cpu]"
```

## GPU Installation

### CUDA 12.8

For the latest CUDA version with full backend support:

```bash
# Using uv (recommended)
uv pip install "inference-models[torch-cu128,onnx-cu12,trt10]" "tensorrt==10.12.0.36"

# Using pip
pip install "inference-models[torch-cu128,onnx-cu12,trt10]" "tensorrt==10.12.0.36"
```

### CUDA 12.4

```bash
uv pip install "inference-models[torch-cu124,onnx-cu12,trt10]" "tensorrt==10.12.0.36"
```

### CUDA 12.6

```bash
uv pip install "inference-models[torch-cu126,onnx-cu12,trt10]" "tensorrt==10.12.0.36"
```

### CUDA 11.8

```bash
uv pip install "inference-models[torch-cu118,onnx-cu118,trt10]" "tensorrt==10.12.0.36"
```

!!! note "TensorRT Version"
    We recommend pinning `tensorrt==10.12.0.36` to match the version used to compile TRT engines. 
    TRT engines require an exact match between compilation and runtime environments.

## Jetson Installation

### Jetson with JetPack 6 (CUDA 12.6)

```bash
uv pip install "inference-models[torch-jp6-cu126,onnx-jp6-cu126]"
```

!!! info "Jetson TensorRT"
    Jetson installations should use the pre-compiled TensorRT package shipped with JetPack.
    Do not install the `trt10` extra on Jetson devices.

## Additional Models & Features

### MediaPipe Models

Enables MediaPipe-based models including Face Detection:

```bash
uv pip install "inference-models[mediapipe]"
```

### Grounding DINO

Enables the Grounding DINO open-vocabulary object detection model:

```bash
uv pip install "inference-models[grounding-dino]"
```

### Flash Attention (Experimental)

For faster LLM/VLM inference (requires compilation):

```bash
uv pip install "inference-models[flash-attn]"
```

!!! warning "Compilation Required"
    Flash Attention requires extensive compilation and may take significant time to install.

### SAM2 Real-Time

SAM2 Real-Time requires manual installation from GitHub:

```bash
# First install inference-models
pip install "inference-models[torch-cu124]"

# Then install SAM2 Real-Time
pip install git+https://github.com/Gy920/segment-anything-2-real-time.git
```

!!! note "PyPI Restriction"
    Due to PyPI restrictions on Git dependencies, SAM2 Real-Time must be installed separately.

## Combining Extras

You can combine multiple extras in a single installation:

```bash
# GPU with multiple backends and additional models
uv pip install "inference-models[torch-cu128,onnx-cu12,trt10,mediapipe,grounding-dino]" "tensorrt==10.12.0.36"
```

!!! important "Conflicting Extras"
    Some extras cannot be installed together:
    
    - Only one `torch-*` extra at a time
    - Only one `onnx-*` extra at a time
    
    The library will prevent conflicting installations when using `uv`.

## Reproducible Installations

For production deployments requiring strict dependency control, use the `uv.lock` file:

```bash
# Clone the repository
git clone https://github.com/roboflow/inference.git
cd inference/inference_models

# Install from lock file
uv sync --frozen
```

See the [official Docker builds](https://github.com/roboflow/inference/tree/main/inference_models/dockerfiles) for examples.

## Verifying Installation

Test your installation:

```python
from inference_models import AutoModel

# This will show available backends
AutoModel.describe_runtime()

# Try loading a model
model = AutoModel.from_pretrained("rfdetr-base")
print("Installation successful!")
```

## Troubleshooting

### Missing Dependencies Error

If you see an error about missing dependencies when loading a model:

1. Check which backend the model requires
2. Install the appropriate extra (e.g., `onnx-cpu`, `trt10`)
3. See [Backend Selection](../auto-loading/backend-selection.md) for details

### CUDA Version Mismatch

Ensure your CUDA version matches the installed extras:

```bash
# Check CUDA version
nvidia-smi

# Install matching extras
uv pip install "inference-models[torch-cu128,onnx-cu12]"  # for CUDA 12.8
```

### TensorRT Engine Errors

TensorRT engines must match the compilation environment exactly:

- Same TensorRT version
- Same CUDA version
- Same GPU architecture

Use `tensorrt==10.12.0.36` for compatibility with Roboflow-provided engines.

## Next Steps

- [Quick Overview](overview.md) - Learn basic usage and concepts
- [Principles & Architecture](principles.md) - Understand the design
- [Models Overview](../models/index.md) - Explore available models

