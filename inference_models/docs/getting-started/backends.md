# Backends and Installation Options

The `inference-models` package uses a **composable extras system** - install only the backends and models you need. This page explains all available backends, their use cases, and installation options.

## Philosophy

Rather than forcing you to install all dependencies upfront, `inference-models` lets you compose your installation based on:

- **Your hardware** - CPU vs GPU, CUDA version
- **Your models** - Which model architectures you'll use
- **Your deployment target** - Development vs production, edge vs cloud

This approach:

- ✅ Reduces installation size and time
- ✅ Avoids dependency conflicts
- ✅ Gives you full control over your environment
- ✅ Enables reproducible builds

See [Understand Core Concepts](../how-to/understand-core-concepts.md) for more details on this design philosophy.

## Available Backends

### PyTorch (Torch)

**Default backend for maximum flexibility and development.**

- ✅ **Pros**: Widest model support, easy debugging, dynamic graphs
- ⚠️ **Cons**: Slower than optimized backends, larger memory footprint
- 🎯 **Best for**: Development, prototyping, models without ONNX/TRT support

**Installation:**

=== "uv"
    ```bash
    # CPU only
    uv pip install inference-models
    
    # GPU with CUDA 12.8
    uv pip install "inference-models[torch-cu128]"
    
    # GPU with CUDA 12.6
    uv pip install "inference-models[torch-cu126]"
    
    # GPU with CUDA 12.4
    uv pip install "inference-models[torch-cu124]"
    
    # GPU with CUDA 11.8
    uv pip install "inference-models[torch-cu118]"
    
    # Jetson (JetPack 6, CUDA 12.6)
    uv pip install "inference-models[torch-jp6-cu126]"
    ```

=== "pip"
    ```bash
    # CPU only
    pip install inference-models
    
    # GPU with CUDA 12.8
    pip install "inference-models[torch-cu128]"
    
    # GPU with CUDA 12.6
    pip install "inference-models[torch-cu126]"
    
    # GPU with CUDA 12.4
    pip install "inference-models[torch-cu124]"
    
    # GPU with CUDA 11.8
    pip install "inference-models[torch-cu118]"
    
    # Jetson (JetPack 6, CUDA 12.6)
    pip install "inference-models[torch-jp6-cu126]"
    ```

**Supported Models**: All PyTorch-based models (YOLOv8, RFDetr, SAM, Florence-2, etc.)

### ONNX Runtime

**Cross-platform compatibility with good performance.**

- ✅ **Pros**: Good CPU/GPU performance, cross-platform, required for Roboflow-trained models
- ⚠️ **Cons**: Not as fast as TensorRT on GPU, limited to static graphs
- 🎯 **Best for**: Production CPU deployments, Roboflow-trained models, cross-platform compatibility

**Installation:**

=== "uv"
    ```bash
    # CPU
    uv pip install "inference-models[onnx-cpu]"
    
    # GPU with CUDA 12.x
    uv pip install "inference-models[onnx-cu12]"
    
    # GPU with CUDA 11.x
    uv pip install "inference-models[onnx-cu11]"
    ```

=== "pip"
    ```bash
    # CPU
    pip install "inference-models[onnx-cpu]"
    
    # GPU with CUDA 12.x
    pip install "inference-models[onnx-cu12]"
    
    # GPU with CUDA 11.x
    pip install "inference-models[onnx-cu11]"
    ```

**Supported Models**: YOLO family (v8-v12), YOLO-NAS, Roboflow-trained models

!!! warning "Required for Roboflow Models"
    Models trained on the Roboflow platform are exported to ONNX format. You **must** install the ONNX backend to use them.

### TensorRT (TRT)

**Maximum GPU performance for NVIDIA hardware.**

- ✅ **Pros**: Fastest inference on NVIDIA GPUs, optimized kernels, low latency
- ⚠️ **Cons**: NVIDIA-only, requires exact environment match, longer first load
- 🎯 **Best for**: Production GPU deployments, real-time applications, maximum throughput

**Installation:**

=== "uv"
    ```bash
    # TensorRT 10 (CUDA 12.x)
    uv pip install "inference-models[trt10]" tensorrt

    # TensorRT 8 (CUDA 11.x)
    uv pip install "inference-models[trt8]" tensorrt
    ```

=== "pip"
    ```bash
    # TensorRT 10 (CUDA 12.x)
    pip install "inference-models[trt10]" tensorrt

    # TensorRT 8 (CUDA 11.x)
    pip install "inference-models[trt8]" tensorrt
    ```

**Supported Models**: YOLO family (v8-v12), YOLO-NAS, RFDetr

!!! danger "Environment Matching Required"
    TensorRT engines are compiled for specific environments. The runtime environment must **exactly match**:

    - TensorRT version
    - CUDA version
    - GPU architecture (compute capability)

    Mismatches will cause loading failures. Install the TensorRT version compatible with your target environment by specifying the exact version: `tensorrt==x.y.z`.

### Hugging Face Transformers

**Access to transformer-based models from Hugging Face Hub.**

- ✅ **Pros**: Huge model ecosystem, latest research models, easy fine-tuning
- ⚠️ **Cons**: Larger models, slower than specialized backends
- 🎯 **Best for**: Vision-language models, transformers, Hugging Face ecosystem

**Installation:**

Included in base installation - no extra required.

```bash
uv pip install inference-models
```

**Supported Models**: OWLv2, TrOCR, and other Hugging Face models

### MediaPipe

**Optimized for mobile and edge devices.**

- ✅ **Pros**: Highly optimized, mobile-friendly, efficient
- ⚠️ **Cons**: Limited model selection, specific use cases
- 🎯 **Best for**: Face detection, mobile deployment, edge devices

**Installation:**

=== "uv"
    ```bash
    uv pip install "inference-models[mediapipe]"
    ```

=== "pip"
    ```bash
    pip install "inference-models[mediapipe]"
    ```

**Supported Models**: MediaPipe Face Detection

## Model-Specific Extras

Some models require additional dependencies beyond backends:

### SAM (Segment Anything)

```bash
uv pip install "inference-models[sam]"
```

### SAM2 (Segment Anything 2)

```bash
uv pip install "inference-models[sam2]"
```

### CLIP

```bash
uv pip install "inference-models[clip]"
```

### DocTR (Document Text Recognition)

```bash
uv pip install "inference-models[doctr]"
```

### Grounding DINO

```bash
uv pip install "inference-models[grounding-dino]"
```

### YOLO-World

```bash
uv pip install "inference-models[yolo-world]"
```

### CogVLM

```bash
uv pip install "inference-models[cogvlm]"
```

## Combining Extras

You can combine multiple extras in a single installation:

```bash
# GPU setup with multiple backends and models
uv pip install "inference-models[torch-cu128,onnx-cu12,trt10,sam,clip]" tensorrt

# CPU setup with ONNX and specific models
uv pip install "inference-models[onnx-cpu,sam2,doctr]"
```

## Recommended Installations

### Development (CPU)

```bash
uv pip install "inference-models[onnx-cpu,sam,clip]"
```

Includes ONNX for Roboflow models and popular model extras.

### Development (GPU)

```bash
uv pip install "inference-models[torch-cu128,onnx-cu12,sam,clip]"
```

Includes PyTorch and ONNX backends with popular models.

### Production (CPU)

```bash
uv pip install "inference-models[onnx-cpu]"
```

Minimal installation with ONNX for best CPU performance.

### Production (GPU - Maximum Performance)

```bash
uv pip install "inference-models[torch-cu128,onnx-cu12,trt10]" tensorrt
```

All GPU backends for maximum flexibility and performance.

### Edge/Embedded (Jetson)

```bash
uv pip install "inference-models[torch-jp6-cu126,onnx-cu12,mediapipe]"
```

Optimized for NVIDIA Jetson devices with JetPack 6.

## Backend Selection Priority

When multiple backends are installed, `AutoModel` selects backends in this order:

1. **TensorRT** (if GPU available and model supports it)
2. **PyTorch** (default, widest compatibility)
3. **ONNX** (good performance, cross-platform)
4. **Hugging Face** (for transformer models)
5. **MediaPipe** (for specific models)

You can override this by specifying `backend_type`:

```python
from inference_models import AutoModel

# Force ONNX backend
model = AutoModel.from_pretrained("yolov8n-640", backend="onnx")

# Force TensorRT backend
model = AutoModel.from_pretrained("yolov8n-640", backend="trt")
```

## Checking Installed Backends

See what backends are available in your environment:

```python
from inference_models import AutoModel

AutoModel.describe_runtime()
```

This shows:
- Installed backends
- CUDA version and availability
- TensorRT version
- Available extras

## Next Steps

- [Installation Guide](installation.md) - Detailed installation instructions
- [Understand Core Concepts](../how-to/understand-core-concepts.md) - Design philosophy
- [Quick Overview](overview.md) - Get started with your first model
- [Supported Models](../models/index.md) - Browse available models

