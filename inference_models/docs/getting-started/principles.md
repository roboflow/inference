# Principles & Architecture

Understanding the core principles and design philosophy behind `inference-models`.

## Core Principles

### 1. Multi-Backend by Design

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

### 2. Minimal Dependencies

We aim to keep extra dependencies minimal while covering as broad a range of models as possible.

**Base Installation** includes:
- PyTorch (CPU)
- Hugging Face Transformers
- Common CV utilities (OpenCV, Supervision)

**Optional Extras** add:
- ONNX Runtime (CPU or GPU)
- TensorRT (GPU only)
- Specialized models (MediaPipe, Grounding DINO)

### 3. Runtime Backend Selection

Backend selection happens **dynamically at runtime** based on:

1. **Model metadata**: What backends are available for this model?
2. **Environment checks**: What backends are installed?
3. **Hardware detection**: What devices are available?
4. **User override**: Explicit backend specification

**Default preference order**: TensorRT → PyTorch → ONNX

### 4. Unified Interface

All models expose a consistent interface regardless of backend:

```python
# Same API for all models
model = AutoModel.from_pretrained(model_id)
predictions = model(images)
```

Model-specific interfaces inherit from base classes:
- `ObjectDetectionModel`
- `InstanceSegmentationModel`
- `ClassificationModel`
- `TextImageEmbeddingModel`
- And more...

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

### Key Components

#### AutoModel

The main entry point for loading models. Handles:
- Model metadata retrieval
- Backend negotiation
- Package selection
- Model instantiation

#### Weights Providers

Retrieve model metadata and download model packages:
- **Roboflow Provider**: Fetches models from Roboflow API
- **Local Provider**: Loads models from local directories

#### Model Registry

Maps `(architecture, task_type, backend)` tuples to model classes:

```python
REGISTERED_MODELS = {
    ("yolov8", "object-detection", "onnx"): YOLOv8ForObjectDetectionOnnx,
    ("yolov8", "object-detection", "trt"): YOLOv8ForObjectDetectionTRT,
    ("rfdetr", "object-detection", "torch"): RFDetrForObjectDetectionTorch,
    # ... many more
}
```

#### Base Model Classes

Abstract base classes defining model interfaces:

- `ObjectDetectionModel`: Bounding box predictions
- `InstanceSegmentationModel`: Boxes + masks
- `ClassificationModel`: Class predictions
- `KeyPointsDetectionModel`: Keypoint predictions
- `TextImageEmbeddingModel`: Embeddings
- `DepthEstimationModel`: Depth maps
- `StructuredOCRModel`: OCR with structure
- And more...

## Model Package Structure

A model package contains:

```
model_package/
├── model_config.json          # Model metadata
├── weights.{pt,onnx,engine}   # Model weights
├── class_names.txt            # Class labels (if applicable)
└── inference_config.json      # Preprocessing config (if applicable)
```

### Model Config

```json
{
  "model_id": "yolov8n-640",
  "model_architecture": "yolov8",
  "task_type": "object-detection",
  "backend": "onnx",
  "quantization": "fp32",
  "input_shape": [1, 3, 640, 640]
}
```

## Backend Selection Process

1. **Retrieve model metadata** from weights provider
2. **Filter available packages** by installed backends
3. **Rank packages** by preference order
4. **Check environment compatibility** (GPU, CUDA version, etc.)
5. **Select best package** or fail with helpful error
6. **Download and cache** model files
7. **Instantiate model** with selected backend

## Dependency Management

### Composable Extras

Dependencies are organized into composable extras:

```toml
[project.optional-dependencies]
torch-cu128 = ["torch>=2.0.0", "torchvision", "pycuda"]
onnx-cpu = ["onnxruntime>=1.15.1"]
onnx-cu12 = ["onnxruntime-gpu>=1.17.0", "pycuda"]
trt10 = ["tensorrt-cu12>=10.0.0", "pycuda"]
mediapipe = ["rf-mediapipe>=0.9"]
```

### Conflict Resolution

The library prevents conflicting installations:

```toml
[tool.uv]
conflicts = [
  [
    { extra = "torch-cpu" },
    { extra = "torch-cu118" },
    { extra = "torch-cu124" },
    { extra = "torch-cu128" },
  ],
  # ...
]
```

## Caching Strategy

### Auto-Resolution Cache

Caches backend selection decisions to avoid repeated API calls:

```python
cache_key = hash(model_id, backend_preferences, environment)
if cache_key in cache:
    return cached_model_class
```

### Model Package Cache

Downloaded model files are cached locally:

```
~/.cache/inference-models/
├── yolov8n-640/
│   ├── onnx-fp32/
│   │   ├── model.onnx
│   │   └── class_names.txt
│   └── trt-fp16/
│       └── model.engine
└── rfdetr-base/
    └── torch-fp32/
        └── model.pt
```

## Design Goals

1. **Ease of Use**: Simple API for common cases
2. **Flexibility**: Advanced options for power users
3. **Performance**: Optimal backend selection
4. **Reliability**: Robust error handling and validation
5. **Extensibility**: Easy to add new models and backends
6. **Transparency**: Clear feedback about what's happening

## Next Steps

- [Auto-Loading Overview](../auto-loading/overview.md) - Deep dive into model loading
- [Backend Selection](../auto-loading/backend-selection.md) - How backends are chosen
- [Contributors Guide](../contributors/architecture.md) - Implementation details

