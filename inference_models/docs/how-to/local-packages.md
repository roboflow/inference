# Load Models from Local Packages

This guide shows how to load models from local directories and custom code packages.

## Overview

The `inference-models` library supports loading models from:

1. **Local model packages** - Pre-downloaded model files
2. **Local code packages** - Custom model implementations
3. **Checkpoint files** - Direct loading from model checkpoints

## Loading from Local Model Packages

### Basic Usage

If you have a model package directory with the standard structure:

```
my_model/
├── model_config.json
├── weights.pt (or .onnx, .engine)
├── class_names.txt
└── inference_config.json
```

Load it with:

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "/path/to/my_model",
    weights_provider="local"
)
```

### Model Package Structure

A valid model package must contain:

#### Required Files

**`model_config.json`** - Model metadata:
```json
{
  "model_id": "my-custom-model",
  "model_architecture": "yolov8",
  "task_type": "object-detection",
  "backend": "torch",
  "quantization": "fp32"
}
```

**Weights file** - One of:
- `weights.pt` (PyTorch)
- `model.onnx` (ONNX)
- `model.engine` (TensorRT)
- `model.pth` (PyTorch state dict)

#### Optional Files

**`class_names.txt`** - Class labels (one per line):
```
person
car
dog
cat
```

**`inference_config.json`** - Preprocessing configuration:
```json
{
  "resize_mode": "letterbox",
  "input_size": [640, 640],
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225]
}
```

### Specifying Backend

If the model package supports multiple backends, specify which to use:

```python
model = AutoModel.from_pretrained(
    "/path/to/my_model",
    weights_provider="local",
    backend_type="onnx"  # or "torch", "trt"
)
```

## Loading from Checkpoints

Load directly from a checkpoint file without a full package:

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "/path/to/checkpoint.pt",
    model_type="yolov8",
    task_type="object-detection",
    backend_type="torch"
)
```

This is useful for:
- Loading models mid-training
- Testing custom checkpoints
- Quick experimentation

## Loading Custom Code Packages

For models with custom code (not in the standard registry):

### Enable Local Code Packages

```python
model = AutoModel.from_pretrained(
    "/path/to/custom_model",
    allow_local_code_packages=True,
    weights_provider="local"
)
```

!!! warning "Security Warning"
    Only enable `allow_local_code_packages` for trusted sources. 
    This allows execution of arbitrary Python code.

### Custom Package Structure

```
custom_model/
├── model_config.json
├── weights.pt
├── model.py              # Custom model implementation
└── __init__.py
```

**`model.py`**:
```python
from inference_models.models.base.object_detection import ObjectDetectionModel

class CustomDetector(ObjectDetectionModel):
    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        # Load model
        return cls(...)
    
    def pre_process(self, images, **kwargs):
        # Custom preprocessing
        pass
    
    def forward(self, pre_processed_images, **kwargs):
        # Custom inference
        pass
    
    def post_process(self, model_results, **kwargs):
        # Custom postprocessing
        pass
```

**`model_config.json`**:
```json
{
  "model_id": "custom-detector",
  "model_architecture": "custom",
  "task_type": "object-detection",
  "backend": "custom",
  "entry_point": "model.CustomDetector"
}
```

## Offline Model Distribution

### Prepare Model Package

1. Download model from Roboflow:
```python
from inference_models import AutoModel

# This downloads and caches the model
model = AutoModel.from_pretrained("yolov8n-640")

# Find cache location
import os
cache_dir = os.path.expanduser("~/.cache/inference-models/yolov8n-640")
print(f"Model cached at: {cache_dir}")
```

2. Copy the cache directory to your offline environment

3. Load from local path:
```python
model = AutoModel.from_pretrained(
    "/path/to/copied/model",
    weights_provider="local"
)
```

### Package for Distribution

Create a distributable package:

```bash
# Create package directory
mkdir my_model_package
cd my_model_package

# Copy model files
cp ~/.cache/inference-models/yolov8n-640/* .

# Create archive
tar -czf my_model_package.tar.gz .
```

Distribute and extract:

```bash
tar -xzf my_model_package.tar.gz -C /path/to/models/
```

## Environment Variables

Control cache and loading behavior:

```python
import os

# Set custom cache directory
os.environ["INFERENCE_MODELS_CACHE_DIR"] = "/custom/cache/path"

# Disable hash verification (not recommended)
os.environ["INFERENCE_MODELS_VERIFY_HASH"] = "false"

# Set download timeout
os.environ["INFERENCE_MODELS_DOWNLOAD_TIMEOUT"] = "300"
```

## Advanced Options

### Disable Auto-Resolution Cache

```python
model = AutoModel.from_pretrained(
    "/path/to/model",
    use_auto_resolution_cache=False
)
```

### Custom Model Access Manager

Track model file access:

```python
from inference_models.models.auto_loaders.model_access_manager import ModelAccessManager

access_manager = ModelAccessManager()

model = AutoModel.from_pretrained(
    "/path/to/model",
    model_access_manager=access_manager
)

# Query accessed files
files = access_manager.get_accessed_files()
```

### Untrusted Packages

By default, only trusted packages are loaded. To allow untrusted:

```python
model = AutoModel.from_pretrained(
    "/path/to/model",
    allow_untrusted_packages=True
)
```

!!! danger "Security Risk"
    Untrusted packages may contain malicious code. Only use with packages from trusted sources.

## Troubleshooting

### Missing model_config.json

If loading fails due to missing config:

```python
# Specify manually
model = AutoModel.from_pretrained(
    "/path/to/weights.pt",
    model_type="yolov8",
    task_type="object-detection",
    backend_type="torch"
)
```

### Incompatible Backend

Ensure the backend is installed:

```bash
# For ONNX
pip install "inference-models[onnx-cpu]"

# For TensorRT
pip install "inference-models[trt10]" "tensorrt==10.12.0.36"
```

### Hash Verification Failures

If hash verification fails but you trust the source:

```python
model = AutoModel.from_pretrained(
    "/path/to/model",
    verify_hash_while_download=False
)
```

## Next Steps

- [Custom Model Development](custom-models.md) - Build your own models
- [Add a New Model](add-model.md) - Contribute to the library
- [Model Pipelines](model-pipelines.md) - Chain multiple models

