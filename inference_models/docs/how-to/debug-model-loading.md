# Debug Model Loading Issues

This guide helps you troubleshoot common errors when loading models with `inference-models`.

## Common Error Types

### Missing Dependencies

**Error:**
```
MissingDependencyError: Required dependency 'onnxruntime' is not installed
```

**Cause:** The backend required for the model is not installed.

**Solution:**
```bash
# Install the missing backend
pip install "inference-models[onnx-cpu]"

# Or for GPU
pip install "inference-models[onnx-cu12]"
```

**Check what's installed:**
```python
from inference_models import AutoModel

AutoModel.describe_runtime()
```

### No Model Packages Available

**Error:**
```
NoModelPackagesAvailableError: Could not find any model package announced by weights provider
```

**Cause:** No model package is registered for the requested model.

**Solutions:**

1. **Verify model ID:**
```python
# Correct format for Roboflow models
model = AutoModel.from_pretrained("project-id/version", api_key="your_key")

# Correct format for pre-trained models
model = AutoModel.from_pretrained("yolov8n-640")
```

2. **Check if model is ready:**

   - For Roboflow models, ensure training is complete
   - Check the Roboflow dashboard for model status

3. **Install required backends:**
```bash
# Install multiple backends for better compatibility
pip install "inference-models[torch-cpu,onnx-cpu]"
```

### Unauthorized Access

**Error:**
```
UnauthorizedModelAccessError: Access to model denied
```

**Cause:** Missing or invalid API key for private Roboflow models.

**Solution:**
```python
from inference_models import AutoModel

# Provide API key
model = AutoModel.from_pretrained(
    "your-project/1",
    api_key="your_roboflow_api_key"
)
```

**Get your API key:** See [how to find your Roboflow API key](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key)

### Corrupted Model Package

**Error:**
```
CorruptedModelPackageError: Model package is corrupted or incomplete
```

**Cause:** Downloaded model files are corrupted or incomplete, or bug in the code.

**Solution:**

1. **Clear the cache:**
```python
import shutil
from inference_models.configuration import INFERENCE_HOME

# Remove cache directory
cache_dir = INFERENCE_HOME / "cache"
if cache_dir.exists():
    shutil.rmtree(cache_dir)
```

2. **Re-download the model:**
```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("yolov8n-640")
```

3. **If the issue persists, report it:**

After verifying the cache is cleared and the problem continues, [open a GitHub issue](https://github.com/roboflow/inference/issues/new) with details about the model and error.

### Backend Compatibility Issues

**Error:**
```
Could not load any of model package candidate
```

**Cause:** No compatible backend for your environment.

**Solution:**

1. **Check your environment:**
```python
from inference_models import AutoModel

AutoModel.describe_runtime()
```

2. **Install compatible backends:**
```bash
# For CPU
pip install "inference-models[torch-cpu,onnx-cpu]"

# For NVIDIA GPU
pip install "inference-models[torch-cu128,onnx-cu12,trt10]" tensorrt
```

3. **Force a specific backend:**
```python
# Try different backends
model = AutoModel.from_pretrained("yolov8n-640", backend="torch")
# or
model = AutoModel.from_pretrained("yolov8n-640", backend="onnx")
```



## Common Problems

### "Model not found" on Roboflow Platform

**Problem:** Model ID is correct but still getting errors.

**Checklist:**

- ✅ Model training is complete
- ✅ Model version exists (check dashboard)
- ✅ API key is provided and valid
- ✅ Project ID format is correct: `project-id/version`

**Example:**
```python
# Correct format
model = AutoModel.from_pretrained(
    "my-project-abc123/2",  # project-id/version
    api_key="your_api_key"
)
```

### CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Use smaller batch size:**
```python
from inference_models.developer_tools import generate_batch_chunks

# For list of images
batch_size = 4
for i in range(0, len(images), batch_size):
    batch = images[i:i + batch_size]
    predictions = model(batch)

# For 4D tensor (batch, channels, height, width)
import torch
images_tensor = torch.stack([...])  # Your images as tensor
for chunk, padding_size in generate_batch_chunks(images_tensor, chunk_size=4):
    predictions = model(chunk)
    # Remove padding from results if needed
    if padding_size > 0:
        predictions = predictions[:-padding_size]
```

2. **Use CPU instead:**
```python
model = AutoModel.from_pretrained("yolov8n-640", device="cpu")
```

3. **Clear CUDA cache:**
```python
import torch

torch.cuda.empty_cache()
```

### Scenario 3: TensorRT Version Mismatch

**Error:**
```
TensorRT version mismatch: model requires X.X but found Y.Y
```

**Solution:**

1. **Check TensorRT version:**
```python
import tensorrt as trt
print(f"TensorRT version: {trt.__version__}")
```

2. **Install matching version:**
```bash
# For TensorRT 10 (CUDA 12.x)
pip install "inference-models[trt10]" tensorrt

# For TensorRT 8 (CUDA 11.x)
pip install "inference-models[trt8]" tensorrt
```

3. **Use different backend:**
```python
# Fall back to ONNX or PyTorch
model = AutoModel.from_pretrained("yolov8n-640", backend="onnx")
```

### Scenario 4: Network/Download Issues

**Error:**
```
ConnectionError: Failed to download model
```

**Solutions:**

1. **Check internet connection**

2. **Retry with timeout:**
```python
# The library automatically retries failed downloads
model = AutoModel.from_pretrained("yolov8n-640")
```

3. **Use offline mode (if model is cached):**
```python
# Models are cached after first download
# Subsequent loads work offline
model = AutoModel.from_pretrained("yolov8n-640")
```

4. **Manual download:**
```python
from inference_models.developer_tools import download_files_to_directory

# Download to specific directory
download_files_to_directory(
    model_id="yolov8n-640",
    target_dir="./models"
)
```

### Scenario 5: Import Errors

**Error:**
```
ImportError: cannot import name 'AutoModel'
```

**Solution:**

1. **Verify installation:**
```bash
pip list | grep inference-models
```

2. **Reinstall:**
```bash
pip uninstall inference-models
pip install "inference-models[torch-cpu]"
```

3. **Check Python version:**
```bash
python --version  # Should be 3.8+
```

## Advanced Debugging

### Inspect Model Package

```python
from inference_models.developer_tools import get_model_package_contents

# Get model package details
contents = get_model_package_contents("path/to/model/package")

print(f"Config: {contents.config}")
print(f"Files: {contents.files}")
```

### Test Specific Backend

```python
from inference_models import AutoModel

# Test each backend individually
backends = ["torch", "onnx", "trt"]

for backend in backends:
    try:
        model = AutoModel.from_pretrained(
            "yolov8n-640",
            backend=backend,
            verbose=True
        )
        print(f"✅ {backend} backend works")
    except Exception as e:
        print(f"❌ {backend} backend failed: {e}")
```

### Check Model Metadata

```python
from inference_models.developer_tools import get_model_from_provider

# Get model metadata without loading
metadata = get_model_from_provider(
    model_id="yolov8n-640",
    api_key=None  # Not needed for public models
)

print(f"Architecture: {metadata.model_architecture}")
print(f"Task: {metadata.task_type}")
print(f"Available packages: {len(metadata.model_packages)}")
```

## Getting Help

If you're still experiencing issues:

1. **Check GitHub Issues:** [roboflow/inference/issues](https://github.com/roboflow/inference/issues)

2. **Provide debug information:**
```python
from inference_models import AutoModel
import sys

print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

AutoModel.describe_runtime()
```

3. **Include error traceback:**
```python
import traceback

try:
    model = AutoModel.from_pretrained("your-model-id")
except Exception as e:
    print(traceback.format_exc())
```

4. **Contact Roboflow Support:** For Roboflow-hosted models, contact [support@roboflow.com](mailto:support@roboflow.com)

## Next Steps

- [Choose the Right Backend](choose-backend.md) - Select optimal backend for your use case
- [Installation Guide](../getting-started/installation.md) - Detailed installation instructions
- [Hardware Compatibility](../getting-started/hardware-compatibility.md) - Check hardware requirements

