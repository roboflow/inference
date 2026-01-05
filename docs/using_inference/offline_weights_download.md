# Model Weights Download

When deploying Roboflow Inference, model weights are downloaded to your device where inference runs locally. This guide covers how to download and cache model weights for different deployment methods.

## Overview

Model weights are downloaded automatically the first time you run inference with a given model. The weights are cached locally on your device, and all inference is performed on-device (not in the cloud).

**How It Works:**

1. Download model weights to your device while connected to the internet
2. Weights are cached locally on your machine
3. Run inference on-device using the cached weights

This approach works across all Roboflow deployment methods and ensures fast, local inference.

!!! warning "Important: Default Cache Location"
    By default, model weights are stored in `/tmp/cache`, which is **cleared on system reboot**. For production deployments or scenarios where you need weights to persist across reboots, you must configure a persistent cache directory using the `MODEL_CACHE_DIR` environment variable (see [Cache Location](#cache-location) section below).

!!! info "Enterprise Offline Mode"
    For enterprise deployments requiring completely disconnected operation, see our [Enterprise Offline Mode documentation](https://docs.roboflow.com/deploy/enterprise-deployment/offline-mode). This guide focuses on model weights download and caching, while maintaining connectivity for usage tracking, billing, and workflow updates.

## InferencePipeline (Video Streaming)

The `InferencePipeline` is designed for real-time video processing and streaming applications. You can pre-download model weights to ensure they're cached before running inference.

### Pre-downloading Weights

The simplest way to download weights for an `InferencePipeline` is to initialize the pipeline once while connected to the internet. This will automatically download and cache the model weights.

```python
from inference import InferencePipeline

api_key = "YOUR_ROBOFLOW_API_KEY"

# Initialize the pipeline (this downloads the model weights)
pipeline = InferencePipeline.init(
    model_id="rfdetr-base",
    video_reference=0,  # Use any valid video source
    on_prediction=lambda predictions, video_frame: None,  # Dummy callback
    api_key=api_key,
)

# Start and immediately stop to trigger weight download
pipeline.start()
pipeline.terminate()

print("Model weights downloaded successfully!")
```

### Alternative: Using `get_model()`

You can also pre-download weights by loading the model directly with `get_model()`:

```python
from inference import get_model

get_model("rfdetr-base")

print("Model weights cached successfully!")
```

### Running Inference

Once the weights are cached, you can run inference on your device:

```python
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

api_key = "YOUR_ROBOFLOW_API_KEY"

# This will use cached weights for on-device inference
pipeline = InferencePipeline.init(
    model_id="rfdetr-base",
    video_reference="path/to/video.mp4",
    on_prediction=render_boxes,
    api_key=api_key,
)

pipeline.start()
pipeline.join()
```

### Cache Location

By default, model weights are cached in `/tmp/cache`. **This directory is cleared on system reboot**, which means you'll need to re-download model weights after each restart.

For production deployments or any scenario where you need weights to persist across reboots, you **must** configure a persistent cache directory using the `MODEL_CACHE_DIR` environment variable:

```python
import os
# Set to a persistent directory (not /tmp)
os.environ["MODEL_CACHE_DIR"] = "/home/user/.roboflow/cache"

from inference import InferencePipeline
# ... rest of your code
```

Alternatively, set it system-wide:

```bash
export MODEL_CACHE_DIR="/home/user/.roboflow/cache"
```

Make sure the directory exists and has appropriate permissions:

```bash
mkdir -p /home/user/.roboflow/cache
chmod 755 /home/user/.roboflow/cache
```

## Client SDK (Image-Based Inference)

The `InferenceHTTPClient` is used for image-based inference and can run models or workflows. When using a self-hosted Inference server, you can pre-load models to ensure weights are cached for on-device inference.

### Option 1: Pre-loading Models with `load_model()`

Use the `load_model()` method to download and cache model weights on your self-hosted server:

```python
from inference_sdk import InferenceHTTPClient

# Connect to your self-hosted Inference server
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="YOUR_ROBOFLOW_API_KEY"
)

# Pre-load the model (downloads weights to server cache)
client.load_model(model_id="rfdetr-base")
print("Model loaded on server!")
```

### Option 2: Trigger Download via First Inference

Alternatively, run a single inference to trigger the weight download:

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="YOUR_ROBOFLOW_API_KEY"
)

# First inference downloads the weights
result = client.infer(
    "path/to/image.jpg",
    model_id="rfdetr-base"
)
print("Model weights cached after first inference!")
```

### Running Inference

Once weights are cached on the server, all subsequent inference requests will use the cached weights for fast, on-device inference:

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="YOUR_ROBOFLOW_API_KEY"
)

# Uses cached weights for on-device inference
result = client.infer(
    "path/to/image.jpg",
    model_id="rfdetr-base"
)
```

### Running Workflows

Workflows can also be pre-cached. To ensure workflows use cached weights, you need to:

1. Download model weights for all models used in the workflow
2. Cache the workflow definition

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="YOUR_ROBOFLOW_API_KEY"
)

# Pre-load models used in your workflow
client.load_model(model_id="your-model/1")

# Run workflow once to cache definition
result = client.run_workflow(
    workspace_name="your-workspace",
    workflow_id="your-workflow",
    images={"image": "path/to/image.jpg"}
)
print("Workflow cached and ready!")
```

### Docker Configuration

When running Inference in Docker, it's **critical** to mount a persistent cache volume to preserve downloaded weights across container restarts and system reboots:

```bash
docker run -d \
  -p 9001:9001 \
  -v /path/to/persistent/cache:/tmp/cache \
  -e MODEL_CACHE_DIR=/tmp/cache \
  roboflow/roboflow-inference-server-cpu:latest
```

**Important considerations:**

- Without a mounted volume, weights are stored inside the container and will be **lost on container restart or system reboot**
- The host path (`/path/to/persistent/cache`) should be on persistent storage, not in `/tmp`
- Ensure the mounted directory has appropriate permissions for the container user
- This allows you to pre-populate weights before deployment and ensures they persist across updates

Example with a persistent host directory:

```bash
# Create persistent cache directory on host
mkdir -p /var/lib/roboflow/cache

# Run container with persistent cache
docker run -d \
  -p 9001:9001 \
  -v /var/lib/roboflow/cache:/tmp/cache \
  -e MODEL_CACHE_DIR=/tmp/cache \
  roboflow/roboflow-inference-server-cpu:latest
```

## Native Python API

The native Python API automatically downloads and caches weights when you load a model with `get_model()`.

### Pre-downloading Weights

```python
from inference import get_model

# Load model (downloads and caches weights)
model = get_model(
    model_id="rfdetr-base",
    api_key="YOUR_ROBOFLOW_API_KEY"
)
print("Model weights cached!")
```

### Running Inference

```python
from inference import get_model

# Uses cached weights for on-device inference
model = get_model(
    model_id="rfdetr-base",
    api_key="YOUR_ROBOFLOW_API_KEY"
)

results = model.infer("path/to/image.jpg")
```

## Verifying Cached Models

### For InferencePipeline and Native Python API

Check the cache directory for downloaded model files:

```bash
ls -lh /tmp/cache
```

You should see directories for each cached model, typically named with the model ID.

### For Client SDK (Self-Hosted Server)

Use the `list_loaded_models()` method to see which models are currently loaded on the server:

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="YOUR_ROBOFLOW_API_KEY"
)

loaded_models = client.list_loaded_models()
print(f"Loaded models: {loaded_models}")
```

## Best Practices

1. **Configure Persistent Cache First**: Before downloading any weights, configure `MODEL_CACHE_DIR` to point to a persistent directory (not `/tmp`). This is **essential** for production deployments to avoid losing cached weights on reboot.

2. **Pre-download During Setup**: Download all required model weights during your deployment setup phase to ensure they're cached and ready.

3. **Use Persistent Cache in Docker**: Always mount a persistent volume when running in Docker containers. Weights stored in the container filesystem will be lost on restart.

4. **Verify Before Deployment**: Always verify that models are properly cached and that the cache directory persists across reboots before deploying to production environments.

5. **Document Model IDs**: Keep a list of all model IDs and versions your application requires for easier pre-caching and troubleshooting.

6. **Consider Storage**: Model weights can be large (100MB - 1GB+ per model). Ensure sufficient disk space is available in your persistent cache directory.

7. **Test Reboot Behavior**: After caching weights, test that they persist after a system reboot to ensure your cache configuration is correct.

## Troubleshooting

### Weights Disappear After Reboot

**Symptom**: Model weights need to be re-downloaded after every system reboot.

**Cause**: The default cache location (`/tmp/cache`) is cleared on reboot.

**Solution**: Configure a persistent cache directory:

```bash
# Set persistent cache location
export MODEL_CACHE_DIR="/home/user/.roboflow/cache"

# Create the directory
mkdir -p /home/user/.roboflow/cache

# For Docker, use a persistent volume mount
docker run -d \
  -p 9001:9001 \
  -v /var/lib/roboflow/cache:/tmp/cache \
  -e MODEL_CACHE_DIR=/tmp/cache \
  roboflow/roboflow-inference-server-cpu:latest
```

### Model Not Found Error

If you get a "model not found" error:

- **Check if cache was cleared**: If using default `/tmp/cache`, weights are lost on reboot. Configure a persistent cache directory.
- Verify the model was actually downloaded (check cache directory with `ls -lh $MODEL_CACHE_DIR`)
- Ensure you're using the exact same `model_id` as when downloading
- Check that `MODEL_CACHE_DIR` is set correctly if using a custom location

### Workflow Definition Missing

For workflows, ensure you've run the workflow at least once while connected to download and cache the workflow definition. The workflow definition is separate from model weights.

### Permission Issues

Ensure the application has read/write permissions to the cache directory:

```bash
chmod -R 755 /path/to/cache
```

For Docker containers, ensure the mounted directory has appropriate permissions for the container user (typically UID 1000 or root depending on the image).

## Related Resources

- [InferencePipeline Documentation](inference_pipeline.md)
- [Client SDK Documentation](../inference_helpers/inference_sdk.md)
- [Native Python API Documentation](native_python_api.md)
- [Docker Configuration Options](../quickstart/docker_configuration_options.md)
