# Core Concepts

Understanding the core principles and design philosophy behind `inference-models`.

## 🎯 Core Principles

### 📚 1. Clear Public Interface

The library maintains a **strict separation between public and internal APIs** to ensure stability and clarity for users.

**For end users** - Import only from the main package:

```python
from inference_models import AutoModel, ObjectDetectionModel, Detections
```

The public interface (`inference_models/__init__.py`) exposes only the essential symbols needed for typical usage:

- `AutoModel` - Main entry point for loading models
- `AutoModelPipeline` - For loading model pipelines
- Base model classes (`ObjectDetectionModel`, `ClassificationModel`, etc.)
- Prediction types (`Detections`, `InstanceDetections`, `ClassificationPrediction`, etc.)
- Common enums (`BackendType`, `ColorFormat`, `Quantization`)

**For custom model developers** - Import from the developer tools module:

```python
from inference_models.developer_tools import (
    get_model_package_contents,
    register_model_provider,
    ModelMetadata,
    ONNXPackageDetails,
)
```

The developer tools interface (`inference_models/developer_tools.py`) exposes utilities for creating custom models:

- Model package loading utilities
- Weight provider registration
- Metadata and configuration classes
- Backend-specific helpers (ONNX, TensorRT, PyTorch)
- Runtime environment introspection

**Why this matters:**

- **Stability** - Public API changes are carefully managed and documented
- **Clarity** - Clear guidance on what's safe to use vs. internal implementation details
- **Maintainability** - Internal refactoring doesn't break user code
- **Discoverability** - Small, focused API surface makes it easier to learn

!!! warning "Import Guidelines"
    **Always import from `inference_models` or `inference_models.developer_tools`**. Never import from internal modules like `inference_models.models.yolo.yolov8` or `inference_models.utils.*` - these are implementation details that may change without notice.

### 🔄 2. Multi-Backend by Design

We define a **model** as weights trained on a dataset, which can be exported or compiled into multiple equivalent **model packages**, each optimized for specific environments.

```
Model (trained weights)
  ├── PyTorch Package (flexibility, development)
  ├── ONNX Package (cross-platform, good performance)
  ├── TensorRT Package (maximum GPU performance)
  └── Hugging Face Package (transformer models)
```

The library automatically selects the best available backend based on:

- Installed dependencies
- Hardware capabilities
- Model availability
- User preferences

### 📦 3. Minimal Dependencies

We aim to keep extra dependencies minimal while covering as broad a range of models as possible.

**Base Installation** includes:
- PyTorch (CPU)
- Hugging Face Transformers
- Common CV utilities (OpenCV, Supervision)

**Optional Extras** add:
- ONNX Runtime (CPU or GPU)
- TensorRT (GPU only)
- Specialized models (MediaPipe, Grounding DINO)

### ⚡ 4. Runtime Backend Selection

Backend selection happens **dynamically at runtime** based on:

1. **Model metadata**: What backends are available for this model?
2. **Environment checks**: What backends are installed?
3. **Hardware detection**: What devices are available?
4. **User override**: Explicit backend specification

**Default preference order**: TensorRT → ONNX → PyTorch → Hugging Face → others

### 🎯 5. Behavior-Based Interfaces

The library follows a **minimalist interface design philosophy**. Models that exhibit similar behavior share slim, functionally-justified interfaces:

- **Detection models** (YOLO, RFDetr, etc.) share methods for bounding box predictions
- **Segmentation models** share methods that return masks alongside boxes
- **Classification models** share methods for class predictions
- **Vision-Language models** share prompting and text generation methods
- **Embedding models** share feature extraction methods
- **Depth estimation models** share depth map output methods
- **OCR models** share text extraction methods

```python
# Models with similar behavior share interfaces
yolo_model = AutoModel.from_pretrained("yolov8n-640")
rfdetr_model = AutoModel.from_pretrained("rfdetr-base")
predictions_yolo = yolo_model(image)    # Same interface
predictions_rfdetr = rfdetr_model(image)  # Same interface

# Different behaviors have different interfaces
vlm_model = AutoModel.from_pretrained("florence2-base")
result = vlm_model.prompt(image, "Describe this")  # Different interface for VLMs
```

**Key principle**: Interfaces are defined by behavior, not by forcing unification. Models can implement completely custom interfaces when their behavior doesn't fit existing patterns - the library supports loading and running fully custom model implementations without requiring conformance to predefined interfaces.

## Architecture Overview

### Component Layers

```
┌─────────────────────────────────────────┐
│         User Application                │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         AutoModel / AutoPipeline        │  ← High-level API
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│      Model Resolution & Loading         │  ← Auto-loading system
│  - Backend selection                    │
│  - Package resolution                   │
│  - Dependency management                │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         Weights Providers               │  ← Model retrieval
│  - Roboflow API                         │
│  - Local filesystem                     │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│         Model Implementations           │  ← Backend-specific code
│  - PyTorch models                       │
│  - ONNX models                          │
│  - TensorRT models                      │
│  - Hugging Face models                  │
└─────────────────────────────────────────┘
```

### 🧩 Key Components

#### AutoModel

The main entry point for loading models. When you call `AutoModel.from_pretrained(...)`, it orchestrates the entire model loading process:

- Retrieves model metadata from the appropriate source (Roboflow API or local filesystem)
- Negotiates the best backend based on your environment and installed dependencies
- Selects the optimal model package variant
- Downloads and caches model files if needed
- Instantiates the model with the selected backend

This abstraction allows you to load any model with a single line of code, regardless of its architecture or backend.

#### Weights Providers

Weights providers are responsible for retrieving model metadata and downloading model packages from different sources. The library is designed to support multiple providers, allowing you to add custom sources for model packages.

**Currently implemented providers:**

- **Roboflow Provider** - Connects to the Roboflow API to fetch models trained on the Roboflow platform. It handles authentication, version resolution, and downloading model artifacts. This provider enables seamless access to both public models and your private custom models.

- **Local Provider** - Loads models from your local filesystem. This is useful for development, offline deployment, or when working with models that aren't hosted on Roboflow. It supports both standard model packages and custom model implementations.



## 📦 Model Package Structure

A model package is a collection of files that together define everything needed to run inference with a specific model variant. The exact structure varies depending on the model architecture and backend, but typically includes:

- **Model weights** - The trained parameters in backend-specific format (`.pt` for PyTorch, `.onnx` for ONNX Runtime, `.engine` for TensorRT)
- **Metadata files** - Information about the model architecture, task type, and backend
- **Task-specific artifacts** - Class labels for detection/classification, vocabulary files for text models, etc.
- **Roboflow inference config** (when applicable) - A standardized metadata format used by the Roboflow platform to define the interface between training and inference time. This config specifies preprocessing requirements (image resizing, normalization), postprocessing parameters (NMS thresholds, confidence filtering), and other runtime behavior needed to properly instrument the model.

**Example: Object Detection Model Package**
```
yolov8n-640-onnx/
├── model.onnx              # ONNX weights
├── class_names.txt         # Object class labels
└── inference_config.json   # Roboflow preprocessing/postprocessing config
```

**Example: Vision-Language Model Package**
```
florence2-base-hf/
├── model.safetensors       # Hugging Face weights
├── config.json             # Model architecture config
├── tokenizer.json          # Text tokenizer
└── preprocessor_config.json # Image preprocessing config
```

**Example: Custom Model Package**

For custom models with architectures not in the main `inference-models` package, the package also includes the model implementation code:

```
my_custom_model/
├── model_config.json       # Model metadata (architecture, task type, backend)
├── model.py                # Model implementation (custom architecture code)
└── weights.pt              # Model weights (optional, can be downloaded separately)
```

Custom model packages allow you to use `inference-models` as a deployment tool for proprietary or experimental architectures. The library handles model loading, backend management, and integration with the Roboflow ecosystem, while you provide the model implementation. See [Load Models from Local Packages](../how-to/local-packages.md) for details on creating custom model packages.

## 🔍 Backend Selection Process

When you call `AutoModel.from_pretrained()`, the library goes through a sophisticated backend selection process to find the optimal model package for your environment:

1. **Retrieve model metadata** - Contacts the weights provider (Roboflow API or local filesystem) to get information about all available model packages, including their backends, quantization levels, batch size support, and dependencies.

2. **Filter by installed backends** - Examines your Python environment to detect which backends are available (PyTorch, ONNX Runtime, TensorRT, Hugging Face) and filters out packages that require backends you don't have installed.

3. **Filter by compatibility** - Checks hardware and software compatibility for each remaining package. This includes GPU availability, CUDA version compatibility, ONNX opset support, TensorRT version matching, and platform-specific requirements.

4. **Apply user preferences** - If you specified preferences for backend, quantization, or batch size, further filters packages to match your requirements.

5. **Rank by priority** - Sorts remaining packages according to the preference order (TensorRT > PyTorch > Hugging Face > ONNX) and other optimization criteria.

6. **Select best package** - Chooses the highest-ranked package. If no compatible package is found, provides a detailed error message explaining why each package was filtered out.

7. **Download and cache** - Downloads the selected package files to local cache if not already present, verifying file hashes for integrity.

8. **Instantiate model** - Loads the model using the selected backend and returns a ready-to-use model instance.

## 💾 Cache

The library uses two types of caching to improve performance:

- **Auto-Resolution Cache** - Stores backend selection decisions to avoid repeated API calls
- **Model Package Cache** - Stores downloaded model files to avoid re-downloading

For detailed information about cache configuration, locations, and management, see the [Cache Management Guide](../how-to/cache-management.md).

## 🚀 Next Steps

- [Cache Management Guide](../how-to/cache-management.md) - Learn about cache management and configuration
- [Supported Models](../models/index.md) - Browse available models
- [Contributors Guide](../contributors/architecture.md) - Implementation details

