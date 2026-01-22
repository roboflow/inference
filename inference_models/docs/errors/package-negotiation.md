# Model Package Negotiation Errors

**Base Class:** `ModelPackageNegotiationError`

Model package negotiation errors occur when the system cannot select an appropriate model package for the current environment. These errors happen during the package selection phase, before actual model loading, and are typically caused by incompatible hardware, unsupported backends, invalid configurations, or environment introspection failures.

---

## ModelPackageNegotiationError

**Base class for all model package negotiation errors.**

This is the parent error class. Specific errors below provide detailed information about what went wrong during package negotiation.

---

## UnknownBackendTypeError

**Requested backend type is not recognized.**

### Overview

This error occurs when you specify a `backend` parameter value that is not supported by `inference-models`.

### When It Occurs

**Scenario 1: Typo in backend name**

- Misspelled backend name (e.g., `"pytorch"` instead of `"torch"`)

- Incorrect capitalization (backend names are case-insensitive but must match valid values)

**Scenario 2: Unsupported backend**

- Requesting a backend that doesn't exist

- Using a backend name from a different framework

### What To Check

1. **Verify supported backends:**
   ```python
   from inference_models import BackendType

   # List all supported backends
   print([b.value for b in BackendType])
   # Output: ['torch', 'torch-script', 'onnx', 'trt', 'hugging-face', 'ultralytics', 'mediapipe', 'custom']
   ```

2. **Check your backend parameter:**
   ```python
   # ❌ Wrong - typo
   model = AutoModel.from_pretrained("yolov8n-640", backend="pytorch")

   # ✅ Correct
   model = AutoModel.from_pretrained("yolov8n-640", backend="torch")
   ```

3. **Common mistakes:**

   - `"pytorch"` → Use `"torch"`

   - `"tensorrt"` → Use `"trt"`

   - `"hf"` → Use `"hugging-face"`

### How To Fix

**Fix typos:**
```python
from inference_models import AutoModel

# Use correct backend name
model = AutoModel.from_pretrained(
    "yolov8n-640",
    backend="torch"  # or "onnx", "trt", etc.
)
```

**Let auto-negotiation choose:**
```python
# Don't specify backend - let the system choose the best one
model = AutoModel.from_pretrained("yolov8n-640")
```

---

## UnknownQuantizationError

**Requested quantization type is not recognized.**

### Overview

This error occurs when you specify a `quantization` parameter value that is not supported by `inference-models`.

### When It Occurs

**Scenario 1: Typo in quantization name**

- Misspelled quantization type (e.g., `"float16"` instead of `"fp16"`)

- Incorrect format

**Scenario 2: Unsupported quantization**

- Requesting a quantization type that doesn't exist

- Using quantization names from other frameworks

### What To Check

1. **Verify supported quantization types:**
   ```python
   from inference_models import Quantization

   # List all supported quantization types
   print(list(Quantization.__members__.keys()))
   # Output: ['FP32', 'FP16', 'INT8']
   ```

2. **Check your quantization parameter:**
   ```python
   # ❌ Wrong - incorrect format
   model = AutoModel.from_pretrained("yolov8n-640", quantization="float16")

   # ✅ Correct
   model = AutoModel.from_pretrained("yolov8n-640", quantization="fp16")
   ```

3. **Common mistakes:**

   - `"float32"` → Use `"fp32"` or `Quantization.FP32`

   - `"float16"` → Use `"fp16"` or `Quantization.FP16`

   - `"int8"` → Use `"int8"` or `Quantization.INT8` (lowercase works)


### How To Fix

**Fix typos:**
```python
from inference_models import AutoModel, Quantization

# Use correct quantization name
model = AutoModel.from_pretrained(
    "yolov8n-640",
    quantization="fp16"  # or Quantization.FP16
)
```

**Use enum for type safety:**
```python
from inference_models import AutoModel, Quantization

# Recommended: use enum to avoid typos
model = AutoModel.from_pretrained(
    "yolov8n-640",
    quantization=Quantization.FP16
)
```

**Request multiple quantizations:**
```python
# System will try to find a package matching any of these
model = AutoModel.from_pretrained(
    "yolov8n-640",
    quantization=["fp16", "fp32"]
)
```

---

## InvalidRequestedBatchSizeError

**Requested batch size has an invalid value.**

### Overview

This error occurs when the `batch_size` parameter has an invalid format or value.

### When It Occurs

**Scenario 1: Tuple with wrong number of elements**

- Batch size tuple must have exactly 2 elements: `(min, max)`

- Provided tuple has more or fewer elements

**Scenario 2: Non-integer values**

- Batch size must be integer or tuple of integers

- Provided float, string, or other type

**Scenario 3: Invalid range**

- `max_batch_size` is less than `min_batch_size`

- Either value is <= 0

**Scenario 4: Wrong type**

- Batch size is not an integer or tuple

### What To Check

1. **Verify batch size format:**
   ```python
   # ✅ Valid formats
   batch_size = 1                    # Single value
   batch_size = 8                    # Single value
   batch_size = (1, 8)               # Range (min, max)

   # ❌ Invalid formats
   batch_size = (1, 4, 8)            # Too many elements
   batch_size = (1,)                 # Too few elements
   batch_size = 1.5                  # Not an integer
   batch_size = "8"                  # String instead of int
   batch_size = (8, 1)               # max < min
   batch_size = (0, 8)               # min <= 0
   ```

2. **Check error message for specific issue:**

   - "tuple of invalid size" → Wrong number of elements

   - "not integer values" → Non-integer in tuple

   - "max_batch_size is lower than min_batch_size" → Invalid range

   - "not integer but has type" → Wrong type for single value

### How To Fix

**Use single integer:**
```python
from inference_models import AutoModel

# Fixed batch size
model = AutoModel.from_pretrained(
    "yolov8n-640",
    batch_size=8
)
```

**Use valid range:**
```python
# Dynamic batch size range
model = AutoModel.from_pretrained(
    "yolov8n-640",
    batch_size=(1, 16)  # min=1, max=16
)
```

**Fix common mistakes:**
```python
# ❌ Wrong
batch_size = (8, 1)      # max < min
batch_size = (1, 4, 8)   # too many values
batch_size = "8"         # string

# ✅ Correct
batch_size = (1, 8)      # proper range
batch_size = (1, 8)      # exactly 2 values
batch_size = 8           # integer
```

---

## RuntimeIntrospectionError

**Failed to introspect the runtime environment.**

### Overview

This error occurs when the system cannot properly detect or validate the runtime environment, particularly on Jetson devices.

### When It Occurs

**Scenario 1: Jetson environment inconsistency**

- GPU device name indicates Jetson hardware (e.g., "orin")

- But L4T version cannot be determined

- Required environment variables not set

**Scenario 2: Container environment on Jetson**

- Running in Docker container on Jetson

- Container cannot access hardware information

- Missing environment variable declarations

### What To Check

1. **Check if running on Jetson:**
   ```python
   from inference_models.developer_tools import x_ray_runtime_environment

   env = x_ray_runtime_environment()
   print(f"Jetson type: {env.jetson_type}")
   print(f"L4T version: {env.l4t_version}")
   print(f"GPU devices: {env.gpu_devices}")
   ```

2. **Verify environment variables:**
   ```bash
   echo $RUNNING_ON_JETSON
   echo $L4T_VERSION
   echo $JETSON_MODULE
   ```

### How To Fix

**Set required environment variables (Jetson in container):**
```bash
# Required
export RUNNING_ON_JETSON=True
export L4T_VERSION=35.3.1  # Your L4T version

# Optional but recommended
export JETSON_MODULE="NVIDIA Jetson Orin Nano"
```

**In Docker:**
```bash
docker run \
  -e RUNNING_ON_JETSON=True \
  -e L4T_VERSION=35.3.1 \
  -e JETSON_MODULE="NVIDIA Jetson Orin Nano" \
  your-image
```

**In Python before loading model:**
```python
import os

# Set environment before importing inference_models
os.environ["RUNNING_ON_JETSON"] = "True"
os.environ["L4T_VERSION"] = "35.3.1"
os.environ["JETSON_MODULE"] = "NVIDIA Jetson Orin Nano"

from inference_models import AutoModel
model = AutoModel.from_pretrained("yolov8n-640")
```

**Find your L4T version:**
```bash
# On Jetson device (not in container)
cat /etc/nv_tegra_release

# Example output:
# R35 (release), REVISION: 3.1
# This means L4T version is 35.3.1
```

**Supported Jetson modules:**

- NVIDIA Jetson Orin Nano

- NVIDIA Jetson Orin NX

- NVIDIA Jetson AGX Orin

- NVIDIA Jetson IGX Orin

- NVIDIA Jetson Xavier NX

- NVIDIA Jetson AGX Xavier Industrial

- NVIDIA Jetson AGX Xavier

- NVIDIA Jetson Nano

- NVIDIA Jetson TX2

---

## JetsonTypeResolutionError

**Failed to determine the Jetson device type.**

**Inherits from:** `RuntimeIntrospectionError`

### Overview

This error occurs when the system detects a Jetson device but cannot determine which specific Jetson model it is.

### When It Occurs

**Scenario 1: Unknown Jetson module name**

- Device tree or environment variable contains unrecognized Jetson module name

- New Jetson model not yet supported

- Custom Jetson board with non-standard naming

**Scenario 2: Corrupted device information**

- `/proc/device-tree/model` file has unexpected format

- Environment variable `JETSON_MODULE` has invalid value

### What To Check

1. **Check device tree model:**
   ```bash
   cat /proc/device-tree/model
   ```

2. **Check environment variable:**
   ```bash
   echo $JETSON_MODULE
   ```

3. **Verify the value starts with a supported Jetson name** (see list above)

### How To Fix

**Set JETSON_MODULE explicitly:**
```bash
# Set to your specific Jetson model
export JETSON_MODULE="NVIDIA Jetson Orin Nano"
```

**In Docker:**
```bash
docker run \
  -e JETSON_MODULE="NVIDIA Jetson Orin Nano" \
  -e L4T_VERSION=35.3.1 \
  -e RUNNING_ON_JETSON=True \
  your-image
```

**If you have a new/unsupported Jetson model:**

- Report the issue at https://github.com/roboflow/inference/issues

- Include the output of `cat /proc/device-tree/model`

- Temporarily set `JETSON_MODULE` to the closest supported model

---

## NoModelPackagesAvailableError

**No compatible model packages are available for the current environment.**

### Overview

This error occurs when the package negotiation system cannot find any model package that matches your requirements and environment.

### When It Occurs

**Scenario 1: No packages announced by provider**

- Model metadata doesn't include any packages

- Model not ready on Roboflow platform

- Custom weights provider not configured correctly

**Scenario 2: Auto-negotiation rejected all packages**

- All available packages failed compatibility checks

- Missing required dependencies

- Too strict requirements (backend, quantization, batch size)

- Hardware incompatibility

**Scenario 3: Requested package ID not found**

- Specified `model_package_id` doesn't exist

- Typo in package ID

- Package removed or renamed by provider

### What To Check

1. **Check available packages:**
   ```python
   from inference_models import AutoModel

   # See what packages are available
   AutoModel.describe_model("yolov8n-640")
   ```

2. **Check your environment:**
   ```python
   from inference_models import AutoModel

   # See what backends/devices are available
   AutoModel.describe_compute_environment()
   ```

3. **Review error message for rejection reasons:**

      - Common reasons: missing dependencies, incompatible device, unsupported quantization

### How To Fix

**Scenario 1: No packages from provider**

   * If using Roboflow: wait for model to finish training/processing or check model status in Roboflow dashboard

   * If using custom provider: verify provider is configured correctly or check provider returns model metadata with packages


**Scenario 2: All packages rejected**

   * Install missing dependencies - check [Installation Guide](../getting-started/installation.md)

   * Relax requirements (if applicable for your use-case):
   ```python
   from inference_models import AutoModel
   
   # ❌ Too strict - might reject all packages
   model = AutoModel.from_pretrained(
       "yolov8n-640",
       backend="trt",           # Only TensorRT
       quantization="int8",     # Only INT8
       device="cuda:0"
   )
   
   # ✅ More flexible - let auto-negotiation choose
   model = AutoModel.from_pretrained(
       "yolov8n-640",
       device="cuda:0"  # Only specify what's essential
   )
   ```

   * Try different backend:
   ```python
   # If ONNX packages are rejected, try PyTorch
   model = AutoModel.from_pretrained(
       "yolov8n-640",
       backend="torch"
   )
   ```

**Scenario 3: Model Package ID not found**

* Check available package IDs:
   ```python
   from inference_models import AutoModel
   
   # List all packages and their IDs
   AutoModel.describe_model("yolov8n-640")
   ```

* Fix typo or use correct ID:
   ```python
   # Use the exact package ID from describe_model()
   model = AutoModel.from_pretrained(
       "yolov8n-640",
       model_package_id="correct-package-id"
   )
   ```

---

## AmbiguousModelPackageResolutionError

**Multiple model packages match the selection criteria.**

### Overview

This error occurs when you specify a `model_package_id` that matches multiple packages, which should never happen with a properly configured weights provider.

### When It Occurs

**Scenario 1: Duplicate package IDs from provider**

- Weights provider returns multiple packages with same ID

- This is a provider bug - package IDs must be unique

### What To Check

1. **Verify this is a provider issue:**
   ```python
   from inference_models import AutoModel

   # Check if packages have duplicate IDs
   AutoModel.describe_model("your-model-id")
   ```

2. **Check error message:**

   - Shows how many packages matched

   - Indicates this is likely a provider error

### How To Fix

**If using Roboflow:**

- This is a bug - please [report it](https://github.com/roboflow/inference/issues)

- Include model ID and error message

- Alternatively, temporarily don't specify `model_package_id` - let auto-negotiation choose

**If using custom provider:**

- Fix your provider to return unique package IDs

- Each package must have a distinct identifier

**Workaround:**
```python
from inference_models import AutoModel

# Don't specify model_package_id
# Let auto-negotiation select the best package
model = AutoModel.from_pretrained("your-model-id")
```
