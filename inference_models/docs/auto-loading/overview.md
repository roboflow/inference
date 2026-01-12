# Auto-Loading Overview

The auto-loading system is the core feature of `inference-models`, automatically handling model resolution, backend selection, and instantiation.

## What is Auto-Loading?

Auto-loading eliminates the complexity of manually selecting and configuring models. Instead of:

```python
# Manual approach (old way)
from inference_models.models.yolov8.yolov8_object_detection_onnx import YOLOv8ForObjectDetectionOnnx

model = YOLOv8ForObjectDetectionOnnx.from_pretrained(
    model_path="/path/to/model",
    device="cuda",
    # ... many configuration options
)
```

You simply write:

```python
# Auto-loading approach (new way)
from inference_models import AutoModel

model = AutoModel.from_pretrained("yolov8n-640")
```

The system automatically:
1. Retrieves model metadata
2. Selects the best backend
3. Downloads model files
4. Instantiates the correct model class
5. Configures the model appropriately

## The AutoModel Class

`AutoModel` is the main entry point for loading models.

### Basic Usage

```python
from inference_models import AutoModel

# Load a model
model = AutoModel.from_pretrained("rfdetr-base")

# Run inference
predictions = model(image)
```

### Key Methods

#### `from_pretrained()`

Load a model from a model ID or path:

```python
model = AutoModel.from_pretrained(
    model_name_or_path="yolov8n-640",
    weights_provider="roboflow",  # or "local"
    api_key=None,  # for private models
    backend_type=None,  # auto-select or specify
    device=None,  # auto-detect or specify
    verbose=False,  # show loading details
    **kwargs  # model-specific options
)
```

#### `describe_model()`

Get information about a model without loading it:

```python
AutoModel.describe_model(
    model_id="rfdetr-base",
    weights_provider="roboflow",
    api_key=None,
    pull_artefacts_size=False  # fetch file sizes
)
```

Output includes:
- Model architecture and task type
- Available model packages
- Backend requirements
- Package sizes (if requested)

#### `describe_runtime()`

Show information about your runtime environment:

```python
AutoModel.describe_runtime()
```

Output includes:
- Available backends
- CUDA information
- TensorRT version
- Installed extras

#### `list_available_models()`

List all registered models:

```python
AutoModel.list_available_models()
```

## Auto-Loading Process

### Step 1: Model Metadata Retrieval

The system retrieves model metadata from the weights provider:

```python
metadata = get_model_from_provider(
    provider="roboflow",
    model_id="yolov8n-640",
    api_key=api_key
)
```

Metadata includes:
- Model architecture (e.g., "yolov8")
- Task type (e.g., "object-detection")
- Available model packages
- Dependencies

### Step 2: Backend Selection

The system filters and ranks available packages:

1. **Filter by installed backends**
   - Check if PyTorch is available
   - Check if ONNX Runtime is available
   - Check if TensorRT is available

2. **Filter by hardware compatibility**
   - GPU availability
   - CUDA version
   - TensorRT version
   - Device compute capability

3. **Rank by preference**
   - Default order: TRT → Torch → ONNX
   - User can override with `backend_type` parameter

4. **Select best package**
   - Choose highest-ranked compatible package
   - Fail with helpful error if none available

### Step 3: Package Download

Download model files to local cache:

```python
download_files_to_directory(
    files=package.package_artefacts,
    target_directory=cache_dir,
    verify_hash=True
)
```

Files are cached at:
```
~/.cache/inference-models/{model_id}/{package_id}/
```

### Step 4: Model Instantiation

Resolve the model class and instantiate:

```python
model_class = resolve_model_class(
    model_architecture="yolov8",
    task_type="object-detection",
    backend="onnx"
)

model = model_class.from_pretrained(
    model_name_or_path=cache_dir,
    **init_kwargs
)
```

## Caching

### Auto-Resolution Cache

Caches backend selection decisions to avoid repeated API calls:

```python
# First call: retrieves metadata and selects backend
model1 = AutoModel.from_pretrained("yolov8n-640")

# Second call: uses cached decision
model2 = AutoModel.from_pretrained("yolov8n-640")
```

Disable caching:

```python
model = AutoModel.from_pretrained(
    "yolov8n-640",
    use_auto_resolution_cache=False
)
```

### Model Package Cache

Downloaded files are cached locally and reused:

```python
# First call: downloads files
model1 = AutoModel.from_pretrained("yolov8n-640")

# Second call: uses cached files
model2 = AutoModel.from_pretrained("yolov8n-640")
```

## Error Handling

The auto-loading system provides helpful error messages:

### Missing Backend

```
ModelLoadingError: Could not load model 'yolov8n-640'.
No compatible model packages found for your environment.

Available packages require:
  - ONNX Runtime (install with: pip install "inference-models[onnx-cpu]")
  - TensorRT (install with: pip install "inference-models[trt10]")

Help: https://docs.inference-models.com/installation
```

### Incompatible Hardware

```
ModelLoadingError: Model package requires CUDA 12.4 but found CUDA 11.8.

Please install a compatible package or upgrade CUDA.

Help: https://docs.inference-models.com/installation
```

### Model Not Found

```
ModelRetrievalError: Could not retrieve model 'invalid-model-id' from Roboflow API.

Please check:
  - Model ID is correct
  - You have access to the model (use api_key if private)
  - Model is published on Roboflow

Help: https://docs.inference-models.com/models
```

## Advanced Features

### Custom Weights Provider

Load models from custom sources:

```python
from inference_models.weights_providers.core import WEIGHTS_PROVIDERS

def my_provider(model_id, api_key):
    # Custom logic to retrieve model metadata
    return ModelMetadata(...)

WEIGHTS_PROVIDERS["custom"] = my_provider

model = AutoModel.from_pretrained(
    "my-model",
    weights_provider="custom"
)
```

### Model Dependencies

Some models depend on other models (e.g., pipelines):

```python
# Automatically loads dependency models
model = AutoModel.from_pretrained("face-and-gaze-pipeline")
```

The system:
1. Detects dependencies in metadata
2. Loads dependency models first
3. Passes them to the main model

### Direct Local Loading

Load from a local directory without metadata:

```python
model = AutoModel.from_pretrained(
    "/path/to/model/directory",
    model_type="yolov8",
    task_type="object-detection",
    backend_type="onnx"
)
```

## Next Steps

- [AutoModel API](automodel.md) - Complete API reference
- [Backend Selection](backend-selection.md) - How backends are chosen
- [Model Resolution](model-resolution.md) - Deep dive into resolution logic

