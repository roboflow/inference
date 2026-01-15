# Choose the Right Backend

This guide helps you select the optimal backend for your use case based on performance, compatibility, and deployment requirements.

## Quick Decision Guide

**For NVIDIA GPU:**

- ✅ Use TensorRT (production) or PyTorch (development)
- ⚠️ ONNX only if others unavailable

**For CPU:**

- ✅ Use PyTorch (best performance)
- ⚠️ ONNX only for Roboflow-trained models or if PyTorch unavailable

**For Development:**

- ✅ Use PyTorch (easy debugging, dynamic graphs, best developer experience)

## Backend Comparison

| Backend | Performance | Compatibility | Best For | When to Use |
|---------|-------------|---------------|----------|-------------|
| **TensorRT** | ⭐⭐⭐⭐⭐ Fastest | NVIDIA GPUs only | Production GPU deployment, real-time inference | Always use on NVIDIA GPUs for production |
| **PyTorch** | ⭐⭐⭐⭐ Fast | Widest model support | Development, CPU inference, non-NVIDIA GPUs | Default choice for development and CPU |
| **ONNX** | ⭐⭐⭐ Slower | Cross-platform CPU/GPU | Older Roboflow-trained models | Only when TensorRT/PyTorch unavailable |

## Backend Details

### TensorRT - Maximum Performance

**When to use:**

- Production deployment on NVIDIA GPUs
- Real-time applications
- Maximum throughput
- Batch processing on GPU

**Pros:**

- ✅ Fastest inference on NVIDIA GPUs
- ✅ Optimized kernels and graph optimizations
- ✅ Low latency and high throughput
- ✅ Efficient memory usage

**Cons:**

- ❌ NVIDIA GPUs only
- ❌ Requires exact CUDA/TensorRT version match
- ❌ Less flexible for debugging

**Installation:**
```bash
# TensorRT 10 (CUDA 12.x)
pip install "inference-models[trt10]"
```

**Example:**
```python
from inference_models import AutoModel

# Force TensorRT backend
model = AutoModel.from_pretrained("yolov8n-640", backend="trt")
```

### PyTorch - Best for Development and CPU

**When to use:**

- Development and prototyping
- CPU inference (best CPU performance)
- Debugging model behavior
- Non-NVIDIA GPUs
- Research and experimentation

**Pros:**

- ✅ Excellent performance on CPU and GPU
- ✅ Widest model support
- ✅ Easy debugging with dynamic graphs
- ✅ Full PyTorch ecosystem access
- ✅ Flexible and well-documented
- ✅ Better than ONNX for most use cases

**Cons:**

- ❌ Not as fast as TensorRT on NVIDIA GPUs
- ❌ Larger memory footprint than TensorRT

**Installation:**
```bash
# CPU
pip install "inference-models[torch-cpu]"

# GPU with CUDA 12.8
pip install "inference-models[torch-cu128]"
```

**Example:**
```python
from inference_models import AutoModel

# Force PyTorch backend
model = AutoModel.from_pretrained("yolov8n-640", backend="torch")
```

### ONNX - Fallback Option

**When to use:**

- Older Roboflow-trained models (exported in ONNX format)
- When TensorRT and PyTorch are not available
- Legacy systems requiring ONNX

**Pros:**

- ✅ Cross-platform (Windows, Linux, macOS)
- ✅ Required for older Roboflow-trained models
- ✅ Smaller package size

**Cons:**

- ❌ Slower than PyTorch and TensorRT
- ❌ Limited to static computation graphs
- ❌ Less flexible than PyTorch

!!! note "Future Direction"
    Roboflow is moving away from ONNX and will primarily deliver TensorRT engines and PyTorch-based models for better performance.

**Installation:**
```bash
# CPU
pip install "inference-models[onnx-cpu]"

# GPU with CUDA 12.x
pip install "inference-models[onnx-cu12]"
```

**Example:**
```python
from inference_models import AutoModel

# Force ONNX backend
model = AutoModel.from_pretrained("yolov8n-640", backend="onnx")
```

## Automatic Backend Selection

By default, `AutoModel` automatically selects the best available backend:

**Priority order:** TensorRT > PyTorch > Hugging Face > ONNX

```python
from inference_models import AutoModel

# Automatic selection - uses best available
model = AutoModel.from_pretrained("yolov8n-640")
# On NVIDIA GPU with TensorRT: uses TensorRT (fastest)
# On NVIDIA GPU without TensorRT: uses PyTorch
# On CPU: uses PyTorch (best CPU performance)
# Falls back to ONNX only if others unavailable
```

The auto-selection considers:

- Installed dependencies
- Hardware availability (CPU vs GPU)
- CUDA version compatibility
- Model architecture support

!!! tip "Recommendation"
    For best performance, install TensorRT for NVIDIA GPUs or PyTorch for CPU/development. ONNX is primarily needed for older Roboflow-trained models. Moving forward, Roboflow will deliver TensorRT engines and PyTorch-based models whenever possible.

## Checking Available Backends

See what backends are installed in your environment:

```python
from inference_models import AutoModel

AutoModel.describe_runtime()
```

This displays:

- Installed backends
- CUDA version and availability
- TensorRT version
- Available extras
- Hardware information

## Next Steps

- [Installation Guide](../getting-started/installation.md) - Install specific backends
- [Backends and Installation Options](../getting-started/backends.md) - Detailed backend information
- [Hardware Compatibility](../getting-started/hardware-compatibility.md) - Check hardware requirements


