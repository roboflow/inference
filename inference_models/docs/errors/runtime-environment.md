# Runtime & Environment Errors

**Base Classes:** `EnvironmentConfigurationError`, `InvalidEnvVariable`, `MissingDependencyError`

Runtime and environment errors occur when the system detects issues with the runtime environment, configuration, or dependencies. These errors can happen at various stages and are typically caused by missing dependencies, invalid environment variables, or incorrect environment setup.

---

## MissingDependencyError

**Required dependency is not installed.**

### Overview

This error occurs when a required Python package or system library is not installed in the environment. Different model backends (ONNX, TensorRT, PyTorch) require different dependencies.

### When It Occurs

**Scenario 1: ONNX backend dependencies missing**

- `onnxruntime` not installed

- Wrong ONNX Runtime variant for your hardware (CPU vs GPU)

- Missing CUDA-specific ONNX Runtime for GPU

**Scenario 2: TensorRT backend dependencies missing**

- `tensorrt` not installed

- `pycuda` not installed

- TensorRT version incompatible with system

**Scenario 3: Model-specific dependencies missing**

- SAM/SAM2 models require additional packages

- MediaPipe models need `mediapipe` package

- Vision-language models need transformers/tokenizers

**Scenario 4: CUDA dependencies missing**

- `pycuda` not installed for GPU operations

- CUDA toolkit not installed on system

- Incompatible CUDA version

### What To Check

1. **Check which backend you're using:**
   ```python
   from inference_models import AutoModel

   # See what backends are available
   AutoModel.describe_compute_environment()
   ```

2. **Review the error message:**

   - Error message specifies which dependency is missing

   - Lists required package extras to install

   - Indicates which backend needs the dependency

3. **Check your installation:**
   ```bash
   # Check installed packages
   pip list | grep onnxruntime
   pip list | grep tensorrt
   pip list | grep torch
   ```

### How To Fix

Visit our installation guide: [Installation Guide](../getting-started/installation.md) to find out how to
install dependencies for your specific use case.

---

## EnvironmentConfigurationError

**Invalid environment configuration detected.**

### Overview

This error occurs when the environment is configured incorrectly, typically when required environment variables are missing or when execution providers are not properly set up.

### When It Occurs

**Scenario 1: ONNX execution providers not configured**

- `ONNXRUNTIME_EXECUTION_PROVIDERS` environment variable not set and no `onnx_execution_providers` passed to `AutoModel.from_pretrained` method

- No execution providers specified in code

- Empty execution providers list

### What To Check

1. **Check ONNX execution providers:**
   ```python
   import os

   # Check if environment variable is set
   print(os.environ.get("ONNXRUNTIME_EXECUTION_PROVIDERS"))

   # Should be something like: "CUDAExecutionProvider,CPUExecutionProvider"
   ```

### How To Fix

**Set ONNX execution providers:**

```bash
# For CPU only
export ONNXRUNTIME_EXECUTION_PROVIDERS="CPUExecutionProvider"

# For GPU (CUDA)
export ONNXRUNTIME_EXECUTION_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"

# For TensorRT
export ONNXRUNTIME_EXECUTION_PROVIDERS="TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider"
```

**Or specify in code:**

```python
from inference_models import AutoModel

# Specify execution providers explicitly
model = AutoModel.from_pretrained(
    "yolov8n-640",
    backend="onnx",
    onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

---

## InvalidEnvVariable

**Environment variable has an invalid value.**

### Overview

This error occurs when an environment variable is set to an invalid value, typically when a boolean variable receives a non-boolean value.

### When It Occurs

**Scenario 1: Invalid boolean value**

- Environment variable expects "true" or "false"

- Received numeric value (0, 1)

- Received other string value

**Scenario 2: Wrong data type**

- Variable expects string but receives other type

- Type conversion fails

### What To Check

1. **Review error message:**

     * Shows which variable has invalid value

     * Shows what value was provided

     * Indicates expected format

### How To Fix

**Fix boolean environment variables:**

```bash
# ❌ Wrong - numeric values
export RUNNING_ON_JETSON=0

# ✅ Correct - use "true" or "false"
export RUNNING_ON_JETSON="false"
```

**Running on Roboflow platform?**

This is most likely a bug you need to [report](https://github.com/roboflow/inference/issues). 

