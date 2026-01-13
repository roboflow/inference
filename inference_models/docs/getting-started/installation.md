# 📦 Installation Guide

This guide covers all installation options for the `inference-models` package.

## ✅ Prerequisites

- **Python 3.9 - 3.12**
- **pip** or **uv** package manager
- For GPU support: **CUDA-compatible GPU** with appropriate drivers

## 🚀 Recommended: Using uv

We recommend using **`uv`** for faster and more reliable installations:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Learn more about uv at [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/).

## 📋 What Gets Installed

### Base Installation

The base `inference-models` package includes:

- **PyTorch** (CPU) - Deep learning framework
- **Hugging Face Transformers** - Transformer models support
- **OpenCV** - Computer vision utilities
- **Supervision** - Vision utilities and annotations

For information about which extras are required for specific model architectures, see the [Supported Models](../models/index.md) documentation.

### Optional Extras

Install additional backends and specialized models using extras:

| Extra | What It Provides | When to Use |
|-------|------------------|-------------|
| **Backend Extras** | | |
| `torch-cpu` | PyTorch CPU-only | CPU-only environments, development |
| `torch-cu118` | PyTorch + CUDA 11.8 | NVIDIA GPUs with CUDA 11.8 (legacy) |
| `torch-cu124` | PyTorch + CUDA 12.4 | NVIDIA GPUs with CUDA 12.4 |
| `torch-cu126` | PyTorch + CUDA 12.6 | NVIDIA GPUs with CUDA 12.6 |
| `torch-cu128` | PyTorch + CUDA 12.8 | NVIDIA GPUs with CUDA 12.8 |
| `torch-jp6-cu126` | PyTorch for Jetson JetPack 6 | NVIDIA Jetson devices (see [Hardware Compatibility](hardware-compatibility.md)) |
| `onnx-cpu` | ONNX Runtime CPU | CPU inference, Roboflow models |
| `onnx-cu118` | ONNX Runtime + CUDA 11.8 | GPU inference with CUDA 11.8 |
| `onnx-cu12` | ONNX Runtime + CUDA 12.x | GPU inference with CUDA 12.x |
| `onnx-jp6-cu126` | ONNX Runtime for Jetson | NVIDIA Jetson devices (see [Hardware Compatibility](hardware-compatibility.md)) |
| `trt10` | TensorRT 10 | Maximum GPU performance, production |
| **Model Extras** | | |
| `mediapipe` | MediaPipe models | Face detection, pose estimation |

## 💻 Basic Installation

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

## 🎮 GPU Installation

!!! warning "TensorRT Version Compatibility"

    **TensorRT engines are sensitive to version compatibility.** A TensorRT engine compiled with a specific TensorRT version may not work with a different runtime version.

    - **Roboflow platform** provides TensorRT packages compiled with **TensorRT 10.12.0.36** and maintains forward compatibility within the 10.x series
    - **Custom compiled engines** are not guaranteed to be forward compatible - match the exact TensorRT version used during compilation
    - **Best practice**: Match your TensorRT version with other dependencies in your environment

    When installing the `trt10` extra, we recommend pinning to `tensorrt==10.12.0.36` for compatibility with Roboflow-provided engines.

### CUDA 12.8

```bash
# Using uv (recommended)
uv pip install "inference-models[torch-cu128,onnx-cu12,trt10]" "tensorrt==10.12.0.36"

# Using pip
pip install "inference-models[torch-cu128,onnx-cu12,trt10]" "tensorrt==10.12.0.36"
```

### CUDA 12.6

```bash
uv pip install "inference-models[torch-cu126,onnx-cu12,trt10]" "tensorrt==10.12.0.36"
```

### CUDA 12.4

```bash
uv pip install "inference-models[torch-cu124,onnx-cu12,trt10]" "tensorrt==10.12.0.36"
```

### CUDA 11.8 (Legacy)

```bash
uv pip install "inference-models[torch-cu118,onnx-cu118,trt10]" "tensorrt==10.12.0.36"
```

## 🤖 Jetson Installation

For NVIDIA Jetson devices, see the [Hardware Compatibility](hardware-compatibility.md) guide for detailed installation instructions and platform-specific requirements.

### Jetson with JetPack 6 (CUDA 12.6)

```bash
uv pip install "inference-models[torch-jp6-cu126,onnx-jp6-cu126]"
```

!!! info "Jetson TensorRT"
    Jetson installations should use the pre-compiled TensorRT package shipped with JetPack.
    Do not install the `trt10` extra on Jetson devices.

## 🔧 Additional Features

### MediaPipe Models

Enables MediaPipe-based models including Face Detection:

```bash
uv pip install "inference-models[mediapipe]"
```

### SAM2 Real-Time

SAM2 Real-Time requires manual installation from GitHub:

```bash
# First install inference-models with any CUDA backend
pip install "inference-models[torch-cu128]"  # or torch-cu126, torch-cu124, etc.

# Then install SAM2 Real-Time
pip install git+https://github.com/Gy920/segment-anything-2-real-time.git
```

!!! note "PyPI Restriction"
    Due to PyPI restrictions on Git dependencies, SAM2 Real-Time must be installed separately.

## 🔗 Combining Extras

You can combine multiple extras in a single installation:

```bash
# GPU with multiple backends and additional models
uv pip install "inference-models[torch-cu128,onnx-cu12,trt10,mediapipe]" "tensorrt==10.12.0.36"
```

!!! important "Conflicting Extras"
    Some extras cannot be installed together:
    
    - Only one `torch-*` extra at a time
    - Only one `onnx-*` extra at a time
    
    The library will prevent conflicting installations when using `uv`.

## 🔒 Reproducible Installations

For production deployments requiring strict dependency control, use the `uv.lock` file:

```bash
# Clone the repository
git clone https://github.com/roboflow/inference.git
cd inference/inference_models

# Install from lock file
uv sync --frozen
```

See the [official Docker builds](https://github.com/roboflow/inference/tree/main/inference_models/dockerfiles) for examples.

## ✅ Verifying Installation

Test your installation:

```python
from inference_models import AutoModel

# This will show available backends
AutoModel.describe_compute_environment()

# Try loading a model
model = AutoModel.from_pretrained("rfdetr-base")
print("Installation successful!")
```

## 🔧 Troubleshooting

### Missing Dependencies Error

If you see an error about missing dependencies when loading a model:

1. Check which backend the model requires
2. Install the appropriate extra (e.g., `onnx-cpu`, `trt10`)

### CUDA Version Mismatch

**Rule of thumb:** Match the **major CUDA version** between your system and the installed extras. Do not install packages built for a newer CUDA version than what's installed on your system, as they may require CUDA symbols from `*.so` libraries that aren't available in older installations.

**Check your CUDA version:**

```bash
# Check CUDA compiler version (most reliable)
nvcc --version

# Check where CUDA is installed
ls -la /usr/local/cuda
```

!!! warning "nvidia-smi Can Be Misleading"
    `nvidia-smi` shows the **driver version** and maximum supported CUDA version, not the actual CUDA toolkit version installed. Always verify with `nvcc --version` or check the `/usr/local/cuda` symlink.

**Install matching extras:**

```bash
# For CUDA 12.x
uv pip install "inference-models[torch-cu128,onnx-cu12]"

# For CUDA 11.8
uv pip install "inference-models[torch-cu118,onnx-cu118]"
```

## 🚀 Next Steps

- [Quick Overview](overview.md) - Learn basic usage and concepts
- [Principles & Architecture](principles.md) - Understand the design
- [Models Overview](../models/index.md) - Explore available models

