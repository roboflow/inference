# TensorRT Compilation

TensorRT (TRT) is NVIDIA's high-performance deep learning inference optimizer and runtime library. It dramatically accelerates model inference on NVIDIA GPUs through layer fusion, precision calibration, and kernel auto-tuning.

## Why TensorRT?

TensorRT can provide **2-10x faster inference** compared to standard frameworks like PyTorch or ONNX Runtime, with lower latency and higher throughput. However, TensorRT compilation is complex:

- **Hardware-specific**: Engines must be compiled for specific GPU architectures
- **Time-consuming**: Compilation can take 10-60 minutes per model
- **Technical expertise**: Requires understanding of precision modes, batch sizes, and optimization profiles
- **Version compatibility**: TensorRT versions and CUDA versions must align

## How Roboflow Helps

Roboflow's platform and `inference-models` ecosystem simplify TensorRT deployment with three compilation options:

| Feature | Automatic Compilation (RF Cloud) | On-Demand Compilation (RF Cloud) | Local Compilation |
|---------|----------------------------------|----------------------------------|-------------------|
| **Availability** | Paid plans (new models) | Paid plans (existing models) | Early access program |
| **When to Use** | New models trained on platform | Models created before auto-compilation | Any model, any GPU architecture |
| **GPU Support** | Roboflow platform GPUs | Limited (L4, T4 only) | Any NVIDIA GPU |
| **Setup Required** | None (automatic) | CLI + workspace whitelisting | Early access enrollment |
| **Compilation Location** | Roboflow cloud (automatic) | Roboflow cloud (on-demand) | Your own hardware |
| **Best For** | New training workflows | Retroactive compilation | Custom GPU architectures |

### 1. Automatic Compilation After Training (Paid Plans)

For **new models on paid Roboflow plans**, TensorRT compilation happens automatically after training completes:

- ✅ **Automatic optimization** - Models are compiled immediately after training
- ✅ **GPU-specific engines** - Compiled for GPU devices used on the Roboflow platform
- ✅ **Zero configuration** - No manual setup or compilation required
- ✅ **Production-ready** - Optimized engines ready for deployment

!!! note "New Models Only"
    Automatic compilation is enabled for **new models** trained on paid plans. For older models, use the on-demand compilation option below.

All models with TensorRT backend implementation are supported. See the [Models Overview](../models/index.md) page for the complete list of models with TRT backend support.

### 2. On-Demand Compilation on Roboflow Platform (Experimental)

For **models created before automatic compilation** was enabled, the `inference-cli` provides an on-demand compilation command that triggers TensorRT compilation jobs on Roboflow's cloud infrastructure.

- 🔄 **Compile existing models** - Retroactively compile models trained before auto-compilation
- ☁️ **Cloud-based** - Runs on Roboflow's infrastructure
- ⚠️ **Limited GPU support** - Only for GPU devices available in Roboflow's cloud (L4, T4)

!!! warning "Limited GPU Support"
    On-demand compilation only works for **limited types of GPU devices** available in Roboflow's cloud infrastructure (currently NVIDIA L4 and T4).

    This feature requires:

    - ✅ **Paid Roboflow account**
    - ✅ **Workspace whitelisting** - Contact Roboflow support to enable this feature for your workspace

See the [On-Demand Compilation](#on-demand-compilation-on-roboflow-platform) section below for detailed usage instructions.

### 3. Local Compilation CLI (Early Access)

For customers who need to compile models for **any NVIDIA GPU**, Roboflow offers a **local compilation CLI** that enables:

- 🔧 **Compile on any NVIDIA GPU** - Use your own hardware for TensorRT compilation
- 🎯 **Any GPU architecture** - Not limited to cloud-available devices
- ☁️ **Automatic artifact registration** - Compiled engines are uploaded to Roboflow platform
- 🚀 **Seamless deployment** - `inference-models` automatically downloads and uses registered engines
- 🔒 **Compile once, use everywhere** - Share compiled models across your infrastructure

!!! info "Early Access Program"
    The local compilation CLI is currently **closed-source** and available through our early access program.

    **Interested?** Contact support@roboflow.com to join the early access program and get access to local compilation tools.

---

## On-Demand Compilation on Roboflow Platform

This section provides detailed instructions for using the on-demand compilation feature described in option 2 above.

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

### Example: Compile for Multiple Devices

```bash
# Set your API key
export ROBOFLOW_API_KEY="your_api_key_here"

# Compile model for both L4 and T4 GPUs
inference rf-cloud batch-processing trt-compile \
    --model-id <project-id>/<version> \
    --device nvidia-l4 \
    --device nvidia-t4 \
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

Once compilation completes, the TensorRT engines are automatically available when deploying your model:

```python
from inference_models import AutoModel

# Load model - TRT backend will be used automatically if available
model = AutoModel.from_pretrained(
    "<project-id>/<version>",
    api_key="your_api_key"
)

# Inference runs on optimized TensorRT engine
results = model(image)
```

---

## Getting Access

To use TRT compilation via CLI (options 2 and 3):

1. **Upgrade to a paid plan** - Visit [Roboflow Pricing](https://roboflow.com/pricing)
2. **Contact support** - Email support@roboflow.com to request workspace whitelisting
3. **Provide workspace ID** - Include your workspace name in the request

## Best Practices

1. **Compile for your deployment GPU** - Ensure you compile for the same GPU architecture you'll use in production
2. **Test before production** - Validate TRT model accuracy matches your ONNX/PyTorch baseline
3. **Monitor compilation time** - Large models can take 30-60 minutes to compile

## Learn More

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Roboflow Inference CLI](../../inference_helpers/inference_cli.md)
- [Models Overview](../models/index.md)

