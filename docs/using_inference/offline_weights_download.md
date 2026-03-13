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

## Cache Location

By default, model weights are cached in `/tmp/cache`. **This directory is cleared on system reboot**, which means you'll need to re-download model weights after each restart.

For production deployments or any scenario where you need weights to persist across reboots, you **must** configure a persistent cache directory using the `MODEL_CACHE_DIR` environment variable:

```python
import os
# Set to a persistent directory (not /tmp)
os.environ["MODEL_CACHE_DIR"] = "/home/user/.roboflow/cache"

from inference import get_model
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

!!! tip "Docker Deployments"

    When running Inference in Docker, mount a persistent cache volume to preserve weights across container restarts. See [Docker Configuration Options — Persistent Model Cache](../quickstart/docker_configuration_options.md#persistent-model-cache) for details.

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

## Best Practices

1. **Configure Persistent Cache First**: Before downloading any weights, configure `MODEL_CACHE_DIR` to point to a persistent directory (not `/tmp`). This is **essential** for production deployments to avoid losing cached weights on reboot.

2. **Pre-download During Setup**: Download all required model weights during your deployment setup phase to ensure they're cached and ready.

3. **Use Persistent Cache in Docker**: Always [mount a persistent volume](../quickstart/docker_configuration_options.md#persistent-model-cache) when running in Docker containers. Weights stored in the container filesystem will be lost on restart.

4. **Verify Before Deployment**: Always verify that models are properly cached and that the cache directory persists across reboots before deploying to production environments.

5. **Document Model IDs**: Keep a list of all model IDs and versions your application requires for easier pre-caching and troubleshooting.

6. **Consider Storage**: Model weights can be large (100MB - 1GB+ per model). Ensure sufficient disk space is available in your persistent cache directory.

7. **Test Reboot Behavior**: After caching weights, test that they persist after a system reboot to ensure your cache configuration is correct.

## Troubleshooting

### Weights Disappear After Reboot

The default cache location (`/tmp/cache`) is cleared on reboot. Configure a persistent cache directory as described in the [Cache Location](#cache-location) section, or use a [persistent volume mount for Docker](../quickstart/docker_configuration_options.md#persistent-model-cache).

### Model Not Found Error

- Verify the model was actually downloaded (check cache directory with `ls -lh $MODEL_CACHE_DIR`)
- Ensure you're using the exact same `model_id` as when downloading
- Check that `MODEL_CACHE_DIR` is set correctly if using a custom location

### Permission Issues

Ensure the application has read/write permissions to the cache directory:

```bash
chmod -R 755 /path/to/cache
```
