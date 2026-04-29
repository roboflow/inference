# TensorRT Compilation

TensorRT (TRT) is NVIDIA's high-performance deep learning inference optimizer and runtime library. It dramatically accelerates model inference on NVIDIA GPUs through layer fusion, precision calibration, and kernel auto-tuning.

## Why TensorRT?

TensorRT can provide **2-10x faster inference** compared to standard frameworks like PyTorch or ONNX Runtime, with lower latency and higher throughput. However, TensorRT compilation is complex:

- **Hardware-specific**: Engines must be compiled for a specific GPU architecture
- **Time-consuming**: Compilation can take 10-60 minutes per model
- **Technical expertise**: Requires understanding of precision modes, batch sizes, and optimization profiles
- **Version-sensitive**: TensorRT and CUDA versions must match between compilation and runtime

## How Roboflow Helps

Roboflow's platform and `inference-models` ecosystem simplify TensorRT deployment with three compilation options:

| Feature | Automatic Compilation (RF Cloud) | On-Demand Compilation (RF Cloud) | Local Compilation |
|---------|----------------------------------|----------------------------------|-------------------|
| **Availability** | Paid plans (new models) | Paid plans (existing models) | Early access program |
| **When to Use** | New models trained on the platform | Models created before auto-compilation | Any model, any GPU architecture |
| **GPU Support** | Roboflow platform GPUs | L4, T4, L40S | Any NVIDIA GPU |
| **Setup Required** | None (automatic) | CLI + workspace whitelisting | Early access enrollment |
| **Compilation Location** | Roboflow cloud (automatic) | Roboflow cloud (on-demand) | Your own hardware |
| **Best For** | New training workflows | Retroactive compilation | Custom GPU architectures |

### 1. Automatic Compilation After Training (Paid Plans)

For **new models on paid Roboflow plans**, TensorRT compilation happens automatically after training completes:

- ✅ **Automatic optimization** - Models are compiled immediately after training
- ✅ **GPU-specific engines** - Compiled for the GPU devices available on the Roboflow platform
- ✅ **Zero configuration** - No manual setup or compilation required
- ✅ **Production-ready** - Optimized engines ready for deployment

!!! note "New Models Only"
    Automatic compilation is enabled for **new models** trained on paid plans. For older models, use the on-demand compilation option below.

All models with TensorRT backend implementation are supported. See the [Models Overview](../models/index.md) page for the complete list of models with TRT backend support.

### 2. On-Demand Compilation on Roboflow Platform (Experimental)

For models created before automatic compilation was available, the `inference-cli` provides an on-demand compilation command that triggers TensorRT compilation jobs on Roboflow's cloud infrastructure.

- 🔄 **Compile existing models** - Retroactively compile models trained before auto-compilation
- ☁️ **Cloud-based** - Runs on Roboflow's infrastructure
- ⚠️ **Limited GPU support** - Restricted to GPU types available in Roboflow's cloud (L4, T4, L40S)

!!! warning "Limited GPU Support"
    On-demand compilation is limited to the GPU types available in Roboflow's cloud infrastructure (currently NVIDIA L4, T4, and L40S).

    This feature requires:

    - ✅ **Paid Roboflow account**
    - ✅ **Workspace whitelisting** - Contact Roboflow support to enable this feature for your workspace

See the [On-Demand Compilation](#on-demand-compilation-on-roboflow-platform) section below for detailed usage instructions.

### 3. Local Compilation CLI (Early Access)

For customers who need to compile models for **any NVIDIA GPU or Jetson device**, Roboflow offers a **local compilation CLI** that enables:

- 🔧 **Compile on any NVIDIA GPU** - Use your own hardware for TensorRT compilation
- 🎯 **Any GPU architecture** - Not limited to cloud-available devices
- ☁️ **Automatic artifact registration** - Compiled engines are uploaded to the Roboflow platform
- 🚀 **Seamless deployment** - `inference-models` automatically downloads and uses the compiled engines
- 🔒 **Compile once, deploy everywhere** - Share compiled models across your infrastructure

```bash
pip install inference-cli
inference enterprise inference-compiler compile-model \
    --model-id <project-id>/<version> \
    --api-key <your_api_key>
```

---

## On-Demand Compilation on Roboflow Platform

Detailed instructions for using the on-demand cloud compilation feature.

### Installation

Install the Roboflow CLI:

=== "uv"
    ```bash
    uv pip install inference-cli
    ```

=== "pip"
    ```bash
    pip install inference-cli
    ```

!!! tip "CLI included with inference"
    If you have `inference` installed, the CLI is already available.

### Basic Usage

Compile a model for specific GPU devices:

```bash
inference rf-cloud batch-processing trt-compile \
    --model-id <project-id>/<version> \
    --device nvidia-l4 \
    --device nvidia-t4
```

### Command Options

| Option | Description | Required |
|--------|-------------|----------|
| `--model-id`, `-m` | Model ID to compile (format: `workspace/project/version`) | ✅ Yes |
| `--device`, `-d` | Target GPU device(s) for compilation | ✅ Yes |
| `--job-id`, `-j` | Custom job identifier (auto-generated if not provided) | ❌ No |
| `--notifications-url` | Webhook URL for job completion notifications | ❌ No |
| `--api-key` | Roboflow API key (uses `ROBOFLOW_API_KEY` env var if not provided) | ❌ No |

### Supported Devices

Currently supported compilation targets:

- `nvidia-l4` - NVIDIA L4 GPU
- `nvidia-t4` - NVIDIA T4 GPU
- `nvidia-l40s` - NVIDIA L40S GPU

### Example: Compile for Multiple Devices

```bash
# Set your API key
export ROBOFLOW_API_KEY="your_api_key_here"

# Compile model for L4, T4, and L40S GPUs
inference rf-cloud batch-processing trt-compile \
    --model-id <project-id>/<version> \
    --device nvidia-l4 \
    --device nvidia-t4 \
    --device nvidia-l40s \
    --job-id my-trt-compilation-job
```

### Monitoring Compilation Jobs

Check the status of your compilation job:

```bash
# List all batch jobs
inference rf-cloud batch-processing list-jobs

# Get details of a specific job
inference rf-cloud batch-processing job-details --job-id my-trt-compilation-job

# View job logs
inference rf-cloud batch-processing logs --job-id my-trt-compilation-job
```

### Using Compiled Models

Once compilation completes, the TensorRT engine is automatically available when loading your model:

```python
from inference_models import AutoModel

# TRT backend is used automatically when a compiled engine is available
model = AutoModel.from_pretrained(
    "<project-id>/<version>",
    api_key="your_api_key"
)

# Runs on the optimized TensorRT engine
results = model(image)
```

---

## Getting Access

To use TRT compilation via the CLI (options 2 and 3):

1. **Upgrade to a paid plan** - Visit [Roboflow Pricing](https://roboflow.com/pricing)
2. **Contact support** - Email support@roboflow.com to request access for your workspace
3. **Provide your workspace ID** - Include your workspace name in the request

## Best Practices

1. **Compile on your deployment hardware** - TRT engines are not portable across GPU architectures, so compile on the same hardware (or same compute capability) you will use in production
2. **Validate accuracy** - Verify that the TRT model's output matches your ONNX or PyTorch baseline before deploying
3. **Plan for compilation time** - Large models can take 30-60 minutes to compile

## Learn More

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Roboflow Inference CLI](../../inference_helpers/inference_cli.md)
- [Models Overview](../models/index.md)

