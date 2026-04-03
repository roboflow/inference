# Core Architecture

This guide provides an in-depth look at the `inference-models` architecture, covering intermediate and low-level implementation details.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Code                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      AutoModel API                          │
│  - from_pretrained()                                        │
│  - __call__() for inference                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Weights Providers                         │
│  - Roboflow Platform                                        │
│  - Pre-trained Models Registry                              │
│  - Custom Providers                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Packages                            │
│  - Metadata (task, input size, classes)                     │
│  - Backend-specific implementations                         │
│  - Weight files                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend Engines                           │
│  - PyTorch                                                  │
│  - ONNX Runtime                                             │
│  - TensorRT                                                 │
└─────────────────────────────────────────────────────────────┘
```

## Library Codebase Structure

```
inference_models/
├── inference_models/          # Main package
│   ├── auto_model.py         # AutoModel entry point
│   ├── auto_pipeline.py      # AutoModelPipeline entry point
│   ├── configuration.py      # Global configuration
│   ├── developer_tools.py    # Public API for custom models
│   ├── models/               # Model implementations
│   │   ├── auto_loaders/     # AutoModel loading logic
│   │   │   ├── core.py       # AutoModel class
│   │   │   ├── dependency_models.py  # Dependent model handling
│   │   │   ├── models_registry.py    # Model class registry
│   │   │   ├── auto_negotiation.py   # Package selection logic
│   │   │   └── ...
│   │   ├── backends/         # Backend-specific base classes
│   │   │   ├── pytorch/      # PyTorch backend
│   │   │   ├── onnx/         # ONNX backend
│   │   │   └── tensorrt/     # TensorRT backend
│   │   ├── common/           # Shared utilities
│   │   │   ├── cuda.py       # CUDA context management
│   │   │   ├── onnx.py       # ONNX utilities
│   │   │   ├── torch.py      # PyTorch utilities
│   │   │   └── trt.py        # TensorRT utilities
│   │   ├── yolov8/           # YOLOv8 family
│   │   ├── yolov11/          # YOLOv11 family
│   │   └── ...               # Other model families
│   ├── model_pipelines/      # Multi-model pipelines
│   │   ├── auto_loaders/     # AutoModelPipeline loading logic
│   │   │   ├── core.py       # AutoModelPipeline class
│   │   │   └── pipelines_registry.py  # Pipeline registry
│   │   └── ...               # Pipeline implementations
│   ├── weights_providers/    # Weight source management
│   │   ├── core.py           # Provider registry
│   │   ├── entities.py       # Data structures
│   │   └── roboflow.py       # Roboflow platform integration
│   ├── utils/                # General utilities
│   │   ├── download.py       # File download with retry
│   │   ├── imports.py        # Lazy imports
│   │   └── onnx_introspection.py
│   └── runtime_introspection/ # Environment detection
└── tests/                    # Test suite
```

## Key Components

### 1. AutoModel 🎯

**Location:** `inference_models/models/auto_loaders/core.py`

AutoModel is the primary interface for loading and using computer vision models. When you call `AutoModel.from_pretrained()`, the library handles the entire process of finding, downloading, and initializing the right model for your environment.

The loading process starts by parsing the model identifier you provide - this can be either a model ID (like "yolov8n-640") or a local filesystem path. For model IDs, AutoModel queries the configured weights provider to discover what model packages are available. Each model can have multiple packages representing different backend implementations (PyTorch, ONNX, TensorRT) with varying quantization levels and batch size configurations.

Once the available packages are known, AutoModel performs auto-negotiation to select the best match. This negotiation considers your explicit preferences (like requesting a specific backend), your hardware capabilities (CPU vs CUDA), and the characteristics of each package. The goal is to select the package that will deliver optimal performance while meeting your constraints.

After selecting a package, AutoModel downloads the necessary files to a local cache (if not already cached) and instantiates the appropriate model class. For models that depend on other models (see Dependent Models section), AutoModel recursively loads those dependencies first. The result is a ready-to-use model instance that you can immediately call with your images.

**Key Methods:**

- `from_pretrained(model_id, **kwargs)` - Load and initialize a model
- `describe_model(model_id)` - Display model metadata and available packages
- `describe_model_package(model_id, package_id)` - Show detailed package information
- `describe_compute_environment()` - Inspect runtime environment and available backends

**Example:**

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("yolov8n-640")
predictions = model(image)
```

### 2. Weights Providers 📦

**Location:** `inference_models/weights_providers/`

Weights providers are the abstraction layer between AutoModel and the actual sources of model weights and metadata. When AutoModel needs to load a model, it doesn't know upfront where to find it or what packages are available - that's the job of weights providers.

The library supports a pluggable provider system. The default provider is Roboflow, which connects to the Roboflow platform to serve both custom-trained models (models you've trained on your own datasets) and a curated registry of pre-trained models (like YOLOv8, SAM, CLIP, and others). When you call `AutoModel.from_pretrained("yolov8n-640")`, the Roboflow provider is queried to return metadata about that model.

The metadata returned by a provider includes essential information like the model's task type (object detection, classification, etc.), its architecture, and critically, the list of available model packages. Each package represents a different way to run the same model - different backends, quantization levels, or optimizations. The provider also supplies download URLs and checksums for all the files needed by each package.

You can implement custom providers to serve models from your own infrastructure, S3 buckets, or any other source. The provider interface is designed to be simple: given a model ID, return the metadata and package information. This extensibility allows `inference-models` to work in diverse deployment scenarios while maintaining a consistent user experience.

### 3. Model Packages 📦

A model package is a specific implementation of a model, bundled with all the files needed to run it. The key insight is that a single model (like "yolov8n-640") can have multiple packages, each optimized for different scenarios.

Consider this: you have one model architecture trained on one dataset, but you might want to run it in different ways depending on your deployment environment. You might need a PyTorch version for development, an ONNX version for cross-platform deployment, or a TensorRT version for maximum GPU performance. You might want FP32 precision for accuracy or INT8 quantization for speed. You might need to process single images or large batches.

Each combination of these choices represents a different model package. The weights provider returns metadata for all available packages, and AutoModel's auto-negotiation logic selects the best one for your situation. This selection considers both your explicit preferences (like `backend="onnx"`) and your environment's capabilities (like whether CUDA is available).

The auto-negotiation process ranks packages based on expected performance characteristics. For example, on a CUDA-enabled system, a TensorRT INT8 package would typically rank higher than an ONNX FP32 package because it offers better throughput. However, if you explicitly request ONNX, the negotiation respects that constraint and picks the best ONNX package available.

This design means you write the same code (`AutoModel.from_pretrained("model-id")`) regardless of where or how you're deploying, and the library automatically adapts to your environment. The package abstraction handles the complexity of supporting multiple backends and optimization strategies without exposing that complexity to users.

### 4. AutoModelPipeline 🔗

**Location:** `inference_models/model_pipelines/auto_loaders/core.py`

Some computer vision tasks require multiple models working together in sequence or parallel. For example, detecting faces and then estimating where each person is looking requires two separate models: one for face detection and another for gaze estimation. AutoModelPipeline provides a unified interface for these multi-model workflows.

When you load a pipeline using `AutoModelPipeline.from_pretrained()`, you specify a pipeline identifier (like "face-and-gaze-detection"). The library maintains a registry of known pipelines, each with a default configuration that specifies which models to use. For the face-and-gaze pipeline, the defaults might be a MediaPipe face detector and an L2CS gaze estimation model.

The pipeline loader uses AutoModel internally to load each constituent model, which means all the same auto-negotiation and caching logic applies. You can override the default models by providing custom parameters - for instance, you might want to run one model on CPU and another on GPU, or use different model variants.

Once loaded, the pipeline presents a single callable interface. You pass in your image, and the pipeline coordinates the execution of all its models, handling data flow between stages and returning a unified result. This abstraction hides the complexity of managing multiple models and lets you treat sophisticated multi-stage workflows as simple, single-call operations.

Pipelines are registered in a central registry that maps pipeline identifiers to their implementation classes and default model configurations. This registry-based approach makes it easy to add new pipelines without changing the AutoModelPipeline interface.

**Example:**

```python
from inference_models import AutoModelPipeline

# Load pipeline with default models
pipeline = AutoModelPipeline.from_pretrained("face-and-gaze-detection")
results = pipeline(image)

# Load pipeline with custom model parameters
pipeline = AutoModelPipeline.from_pretrained(
    "face-and-gaze-detection",
    models_parameters=[
        "mediapipe/face-detector",
        {"model_id_or_path": "l2cs-net/rn50", "device": "cuda"}
    ]
)
```

### 5. Dependent Models 🤝

**Location:** `inference_models/models/auto_loaders/dependency_models.py`

Certain models cannot function on their own - they require other models to be loaded and available in memory. The primary example in the library is Roboflow Instant models, which are specialized detectors that depend on a base model to provide feature extraction or embeddings.

When a model declares dependencies in its metadata, AutoModel handles the recursive loading automatically. Before initializing the main model, AutoModel first loads each dependency using the same `from_pretrained()` process. This means dependencies benefit from the same auto-negotiation, caching, and backend selection as any other model.

The dependency loading process is carefully controlled to prevent infinite recursion - dependencies themselves cannot have dependencies. Once all dependencies are loaded, they're injected into the main model's initialization parameters, making them available for the model to use during inference.

You can control how dependencies are loaded using the `dependency_models_params` parameter. This lets you specify different backends, devices, or quantization levels for each dependency. For example, you might run the main model on GPU with FP16 precision while running a dependency on CPU with FP32 precision.

The library also supports forwarding certain parameters from the main model to its dependencies. This is useful when you want to apply the same configuration (like cache settings or compilation flags) across the entire model stack without having to specify it separately for each dependency.

**Example:**

```python
from inference_models import AutoModel

# Load model with dependencies (automatic)
model = AutoModel.from_pretrained("roboflow-instant-model-id")

# Load with custom dependency configuration
model = AutoModel.from_pretrained(
    "roboflow-instant-model-id",
    dependency_models_params={
        "base_model": {
            "model_id_or_path": "base-model-id",
            "backend": "onnx",
            "device": "cuda"
        }
    }
)
```

## Error Handling 🚨

**Location:** `inference_models/errors.py`

The library uses a structured exception hierarchy to provide clear, actionable error messages. Every error in `inference-models` inherits from a base `InferenceModelsError` class, making it easy to catch all library-specific exceptions if needed.

The most important aspect of our error handling philosophy is that **every error should have documentation**. When something goes wrong, we don't just tell you what failed - we tell you why it failed and how to fix it. Each exception includes a `help_url` field that points to documentation explaining the error in detail and providing solutions.

The exception hierarchy is organized by the type of problem:

- **Model loading errors** - Issues finding, downloading, or initializing models
- **Dependency errors** - Missing Python packages or incompatible versions
- **Access errors** - Authentication or authorization failures
- **Package errors** - Corrupted downloads or invalid model packages
- **Parameter errors** - Invalid configuration or arguments

When you encounter an error, the exception message will guide you toward the solution. For example, if you try to load a model that requires TensorRT but don't have it installed, you'll get a `MissingDependencyError` with instructions on how to install the TensorRT extra. If a model download fails checksum validation, you'll get a `CorruptedModelPackageError` with steps to clear your cache and retry.

## Next Steps

- [Dependencies and Backends](dependencies-and-backends.md) - Deep dive into backends and dependency management
- [Adding a Model](adding-model.md) - Contribute a new model
- [Writing Tests](writing-tests.md) - Test your contributions

