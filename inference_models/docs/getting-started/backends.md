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

### Core ML (Apple Silicon)

**Native Core ML packages for Macs with Apple Silicon.**

- ✅ **Pros**: Runs on the Apple GPU through Core ML directly, loads in under a second, fastest option on a Mac
- ⚠️ **Cons**: Apple Silicon with macOS 13+ only, needs a Core ML package registered for the model
- 🎯 **Best for**: Local development and edge deployments on Macs

**Installation:**

=== "uv"
    ```bash
    uv pip install "inference-models[coreml]"
    ```

=== "pip"
    ```bash
    pip install "inference-models[coreml]"
    ```

**Supported Models**: RF-DETR object detection and instance segmentation

Packages run on the compute units set by `INFERENCE_MODELS_COREML_COMPUTE_UNITS` (default `CPUAndGPU`).
A zipped `.mlpackage` is extracted once into the package's `coreml_cache/` directory. Models without a
Core ML package keep using ONNX Runtime's `CoreMLExecutionProvider`.

Core ML is used only when it can run: macOS 13+ on Apple Silicon, with a coremltools build that includes its Core ML
runtime. Elsewhere the `coreml` extra installs nothing, Core ML packages are not requested from the Roboflow API, and models
load through their other backends. Installing or removing coremltools changes which package a cached model resolves to
on its next load.

### ExecuTorch

**A `.pte` program exported with `torch.export`, no ONNX step.**

- ✅ **Pros**: Exported straight from PyTorch; the same backend runs programs lowered to TensorRT on NVIDIA GPUs or to XNNPACK on the CPU, and parts TensorRT cannot convert can run on ExecuTorch's CUDA backend in the same file
- ⚠️ **Cons**: Local packages only for now, fixed batch size per program, a TensorRT program is tied to the GPU and TensorRT version it was built for
- 🎯 **Best for**: Running RF-DETR models exported with `rfdetr`'s ExecuTorch export or with Torch-TensorRT (`torch_tensorrt.save(..., output_format="executorch")`)

**Installation:**

ExecuTorch is not installed by any extra. Install the package that matches how the program was exported:

=== "uv"
    ```bash
    # programs that use TensorRT, alone or with ExecuTorch's CUDA backend (NVIDIA GPUs)
    uv pip install "torch_tensorrt[executorch]"
    # programs that use XNNPACK (CPU)
    uv pip install "executorch>=1.6"
    ```

=== "pip"
    ```bash
    # programs that use TensorRT, alone or with ExecuTorch's CUDA backend (NVIDIA GPUs)
    pip install "torch_tensorrt[executorch]"
    # programs that use XNNPACK (CPU)
    pip install "executorch>=1.6"
    ```

ExecuTorch 1.6 requires Python 3.11+ and `torch==2.15.0`, so install it before `inference-models`, which then keeps
that torch and picks a matching `torchvision`. Torch-TensorRT 2.15 also requires `filelock>=4.2.1`, while
`inference-models` pins `filelock<4`: pip cannot install both in one command, and in this order it reports the
`filelock` conflict, but both packages work. Torch-TensorRT brings TensorRT 11 under the same `tensorrt` module as the
`trt10` extras, so keep them in separate environments.

**Supported Models**: RF-DETR object detection

A package is a directory. The program takes the same normalized NCHW tensor as the ONNX export and returns raw boxes
and class logits, so pre- and post-processing are shared with the ONNX backend. Programs exported with device-resident
inputs get their input on the GPU they expect; others get it in host memory and copy it themselves. For an XNNPACK
program, pass `device=torch.device("cpu")` to keep pre- and post-processing on the CPU too.

```
rfdetr-executorch-package/
├── weights.pte             # the ExecuTorch program, for example from rfdetr's ExecuTorch export
├── weights.ptd             # only if the program keeps data outside the .pte (the CUDA backend can)
├── class_names.txt         # one class name per line, in the model's output order
├── inference_config.json   # input size, resize mode and normalization used in training
└── model_config.json       # tells AutoModel which backend runs the package
```

For a model trained with `rfdetr` at 384x384 on RGB images, `inference_config.json` is:

```json
{
  "network_input": {
    "training_input_size": {"height": 384, "width": 384},
    "dynamic_spatial_size_supported": false,
    "color_mode": "rgb",
    "resize_mode": "stretch",
    "input_channels": 3,
    "scaling_factor": 255,
    "normalization": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
  }
}
```

and `model_config.json` is:

```json
{"model_architecture": "rfdetr", "task_type": "object-detection", "backend_type": "executorch"}
```

Then load it by path:

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("/path/to/rfdetr-executorch-package")
```

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
uv pip install "inference-models[torch-jp6-cu126,onnx-cu12]"
```

Optimized for NVIDIA Jetson devices with JetPack 6.

## Backend Selection Priority

When multiple backends are installed, `AutoModel` selects backends in this order:

1. **Core ML** (on Apple Silicon with macOS 13+ and the `coreml` extra, when the model has a Core ML package)
2. **TensorRT** (if GPU available and model supports it)
3. **ONNX** (good performance, cross-platform)
4. **PyTorch** (widest compatibility)
5. **Hugging Face** (for transformer models)

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

