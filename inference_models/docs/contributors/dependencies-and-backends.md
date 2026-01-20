# Dependencies and Backends 🔧

This guide explains how `inference-models` manages dependencies and supports multiple inference backends across different hardware environments.

## Core Philosophy: Broad Baseline, Modular Extensions 🎯

The dependency strategy of `inference-models` is designed around a key insight: **most users want to run many models in development, but only specific models in production**.

### The Baseline Approach

The base installation includes a carefully selected set of dependencies that enable the majority of models and workloads to run out of the box. This includes:

- **PyTorch** - The primary backend for most models
- **Transformers** - For vision-language models and embeddings
- **OpenCV** - Image processing utilities
- **NumPy** - Array operations

This baseline is intentionally broad. The goal is to **maximize the number of models available in a single environment** while keeping the installation manageable. With just `uv pip install inference-models`, you get access to the vast majority of models in the library - YOLOv8, SAM, CLIP, Florence-2, and many others.

### When Extras Are Needed

Optional extras are introduced only when:

1. **Alternative backends are needed** - ONNX Runtime or TensorRT for specific deployment scenarios
2. **Models require specialized dependencies** - Certain models need packages that would conflict with the baseline or are too heavy to include by default

We deliberately minimize the number of extras while maximizing model coverage. The goal is to avoid a situation where every model requires its own extra - that would fragment the ecosystem and create a poor developer experience.

### Deployment-Time Decisions

This strategy defers the decision of "what capabilities do I need?" to installation time. In development, you might install the full baseline to experiment with many models. In production, you might install a targeted set of extras to optimize for your specific use case.

For example:

- **Development**: `uv pip install inference-models` - Get broad model coverage

- **Production (GPU, TensorRT)**: `uv pip install "inference-models[trt10]"` - Optimize for maximum GPU performance
 
- **Production (CPU, ONNX)**: `uv pip install "inference-models[onnx-cpu]"` - Optimize for CPU deployment
 
- **Specialized workload**: `uv pip install "inference-models[onnx-cu12,sam,ocr]"` - Custom blend for specific needs

This approach requires some learning to master, but we believe it's manageable with proper documentation and provides significantly better flexibility and developer experience compared to the old `inference` library.

### Implications for Contributors

This philosophy shapes how the library is structured:

1. **Lazy loading** - Components are imported only when used, not at module import time
2. **Graceful error handling** - Missing optional dependencies produce clear, actionable error messages
3. **Core stability** - The core library must work without optional dependencies; optional deps cannot break core functionality
4. **Minimal extras** - New extras should only be added when absolutely necessary

When contributing, you must keep this in mind. The baseline should remain stable and functional, and optional components should fail gracefully with helpful guidance.

## Backends and Hardware Compatibility 🖥️

`inference-models` supports multiple inference backends, each designed to work with different hardware environments. The choice of backend is primarily driven by your deployment hardware and the model packages available for that hardware.

### PyTorch Backend

PyTorch is included in the baseline installation and serves as the primary backend for most models. It supports both CPU and CUDA-enabled GPUs.

**Hardware compatibility:**
- CPU (all platforms)
- NVIDIA GPUs with CUDA support


**Installation:**
```bash
uv pip install inference-models                    # Includes PyTorch CPU
uv pip install "inference-models[torch-cu121]"     # PyTorch with CUDA 12.1
```

### ONNX Runtime Backend

ONNX Runtime is an optional backend that provides cross-platform model execution. It uses "execution providers" to adapt to different hardware.

**Hardware compatibility:**
- CPU (all platforms) - via `CPUExecutionProvider`

- NVIDIA GPUs - via `CUDAExecutionProvider` or `TensorrtExecutionProvider`

- Other accelerators (DirectML, CoreML, etc.)

**Model format:** ONNX (`.onnx` files)

**When it's used:**

- When ONNX model packages are selected (automatically or manually)

- Cross-platform deployments

- CPU-optimized inference

**Installation:**
```bash
uv pip install "inference-models[onnx-cpu]"        # CPU execution provider
uv pip install "inference-models[onnx-cu12]"       # CUDA execution provider (CUDA 12.x)
```

### TensorRT Backend

TensorRT is NVIDIA's inference optimization engine, available as an optional backend for NVIDIA GPU deployments.

**Hardware compatibility:**

- NVIDIA GPUs only (requires CUDA)

- Engines are compiled for specific GPU architectures

**Model format:** TensorRT engines (`.engine` or `.plan` files)

**When it's used:**

- Automatically chosen when TRT package available and compatible with TRT installation

- GPU deployments targeting maximum throughput


**Important considerations:**

- TensorRT engines are hardware-specific and must be compiled for your target GPU (Roboflow provides convenient 
ways of compilation)

- Engines are not always portable across different GPU architectures

- Requires TensorRT installation (not included in baseline)

**Installation:**
```bash
uv pip install "inference-models[trt10]"           # TensorRT 10.x
```

## Graceful Handling of Missing Dependencies 🛡️

A critical aspect of the modular architecture is graceful degradation when optional dependencies are missing. The library must never crash at import time due to missing optional dependencies.

### Example: TensorRT Components

TensorRT is an optional backend. Here's how missing dependencies are handled:

```python
from inference_models.errors import MissingDependencyError

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not TRT tools required to run models with TRT backend - this error means that some additional "
        f"dependencies are not installed in the environment. If you run the `inference-models` library directly in your "
        f"Python program, make sure the following extras of the package are installed: `trt10` - installation can only "
        f"succeed for Linux and Windows machines with Cuda 12 installed. Jetson devices, should have TRT 10.x "
        f"installed for all builds with Jetpack 6. "
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="XXX",
    ) from import_error

```

### Key Principles

1. **Import-time safety** - Optional imports are wrapped in try/except blocks
2. **Clear error messages** - Users get actionable guidance on what to install
3. **Documentation links** - Every error points to detailed documentation
4. **Core stability** - Missing optional dependencies never break core functionality

### Contributor Guidelines

When adding code that uses optional dependencies:

1. **Check availability** before using the dependency
2. **Provide clear errors** with installation instructions
3. **Use lazy imports** to avoid import-time failures
4. **Test without the dependency** to ensure graceful degradation
5. **Document the requirement** in the model or feature documentation

**Critical rule:** The core library must remain functional without any optional dependencies. Optional components can fail gracefully, but they cannot bring down the core.

## Adding Dependencies: Guidelines for Contributors 🛠️

When contributing to `inference-models`, carefully consider whether a new dependency is necessary and where it belongs.

### 1. Evaluate Necessity

Ask yourself:
 
- Can we implement this ourselves with reasonable effort?

- Is there a lighter alternative that provides the same functionality?

- Can this be optional rather than part of the baseline?

- Does this dependency conflict with existing baseline dependencies?

**Prefer baseline inclusion when:**
 
- The dependency enables a broad class of models

- It's lightweight and stable

- It doesn't conflict with existing dependencies

- Most users would need it

**Prefer optional extra when:**

- The dependency is heavy or specialized

- It conflicts with baseline dependencies

- It's only needed for specific models or backends

- It's hardware-specific (like TensorRT)

### 2. Choose the Right Category

- **Baseline dependency** - Add to core `dependencies` in pyproject.toml (rare, requires strong justification)
- **Backend extra** - For alternative inference engines (onnx, tensorrt)
- **Model extra** - For models requiring specialized dependencies
- **Development dependency** - Only for testing/development

### 3. Add to pyproject.toml

```toml
[project.optional-dependencies]
your-feature = [
    "new-package>=1.0.0,<2.0.0",  # Pin major version to avoid breaking changes
]
```

### 4. Use Lazy Imports

```python
from inference_models.utils.imports import LazyClass

YourModel = LazyClass(
    module_name="inference_models.models.your_model",
    class_name="YourModel",
)
```

### 5. Handle Missing Dependencies Gracefully

```python
from inference_models.errors import MissingDependencyError

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Clear error message",
        help_url="XXX",
    ) from import_error
```

### 6. Version Pinning Strategy

Pin major versions to avoid breaking changes:

```toml
# ✅ Good
"torch>=2.0.0,<3.0.0"
"onnxruntime>=1.16.0,<2.0.0"

# ❌ Avoid
"torch>=2.0.0"      # Could break on torch 3.0
"torch==2.0.0"      # Too restrictive
```

## Next Steps 🎓

- [Core Architecture](core-architecture.md) - Understand the overall system design
- [Adding a Model](adding-model.md) - Learn how to add multi-backend support
- [Writing Tests](writing-tests.md) - Test with different dependency configurations


