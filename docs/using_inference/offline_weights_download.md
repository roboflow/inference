# Offline Weights Download

When deploying Roboflow Inference in environments with limited or no internet connectivity, you can pre-download model weights to enable offline operation. This guide covers how to download weights for different deployment methods.

## Overview

Model weights are downloaded automatically the first time you run inference. However, in offline scenarios such as edge devices, air-gapped networks, or production environments with restricted internet access, you'll want to pre-cache these weights.

**How It Works:**

1. Download model weights while connected to the internet
2. Cache them locally on your machine
3. Run inference offline using the cached weights

This approach works across all Roboflow deployment methods.

## InferencePipeline (Video Streaming)

The `InferencePipeline` is designed for real-time video processing and streaming applications. To use it offline, you need to pre-download the model weights before disconnecting from the internet.

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

model = get_model(model_id="rfdetr-base", api_key="YOUR_ROBOFLOW_API_KEY")
print("Model loaded and cached!")
```

### Using Offline

Once the weights are cached, you can use the `InferencePipeline` without an internet connection:

```python
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

api_key = "YOUR_ROBOFLOW_API_KEY"

# This will use cached weights - no internet required
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

By default, model weights are cached in `/tmp/cache`. You can customize this location using the `MODEL_CACHE_DIR` environment variable:

```python
import os
os.environ["MODEL_CACHE_DIR"] = "/path/to/your/cache"

from inference import InferencePipeline
# ... rest of your code
```

## Client SDK (Image-Based Inference)

The `InferenceHTTPClient` is used for image-based inference and can run models or workflows. When using a self-hosted Inference server, you can pre-load models to enable offline operation.

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

### Using Offline

Once weights are cached on the server, all subsequent inference requests will work offline:

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="YOUR_ROBOFLOW_API_KEY"
)

# No internet required - uses cached weights
result = client.infer(
    "path/to/image.jpg",
    model_id="rfdetr-base"
)
```

### Running Workflows Offline

Workflows can also be pre-cached. To run workflows offline, you need to:

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
print("Workflow cached and ready for offline use!")
```

### Docker Configuration for Offline Use

When running Inference in Docker, you can mount a persistent cache volume to preserve downloaded weights:

```bash
docker run -d \
  -p 9001:9001 \
  -v /path/to/cache:/tmp/cache \
  -e MODEL_CACHE_DIR=/tmp/cache \
  roboflow/roboflow-inference-server-cpu:latest
```

This ensures that weights persist across container restarts and can be pre-populated before deployment.

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

### Using Offline

```python
from inference import get_model

# Uses cached weights - no internet required
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

1. **Pre-download During Setup**: Download all required model weights during your deployment setup phase while internet is available

2. **Use Persistent Cache**: Configure `MODEL_CACHE_DIR` to a persistent location, especially in Docker environments

3. **Verify Before Deployment**: Always verify that models work offline before deploying to production environments

4. **Document Model IDs**: Keep a list of all model IDs and versions your application requires for easier pre-caching

5. **Consider Storage**: Model weights can be large (100MB - 1GB+ per model). Ensure sufficient disk space is available

## Troubleshooting

### Model Not Found Error

If you get a "model not found" error when running offline:

- Verify the model was actually downloaded (check cache directory)
- Ensure you're using the exact same `model_id` as when downloading
- Check that `MODEL_CACHE_DIR` is set correctly if using a custom location

### Workflow Definition Missing

For workflows, ensure you've run the workflow at least once while online to cache the definition. The workflow definition is separate from model weights.

### Permission Issues

Ensure the application has read/write permissions to the cache directory:

```bash
chmod -R 755 /path/to/cache
```

## Related Resources

- [InferencePipeline Documentation](inference_pipeline.md)
- [Client SDK Documentation](../inference_helpers/inference_sdk.md)
- [Native Python API Documentation](native_python_api.md)
- [Docker Configuration Options](../quickstart/docker_configuration_options.md)
