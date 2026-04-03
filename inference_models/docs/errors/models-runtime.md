# Model Runtime Errors

**Base Class:** `ModelRuntimeError`

Model runtime errors occur during model execution when the model encounters issues processing input data or when runtime constraints are violated. These errors happen during the inference phase and are typically caused by incompatible input formats, device mismatches, or model execution failures.

---

## ModelRuntimeError

**Error during model execution or inference.**

### Overview

This error occurs when a model fails during runtime execution. This can happen due to invalid input formats, device incompatibilities, batch size mismatches, or internal model execution failures.

### When It Occurs

**Scenario 1: Invalid input type or format**

- Unsupported input type (not list, np.ndarray, or torch.Tensor)

- Empty input list provided

- Wrong tensor dimensions (expected 3 or 4, got different)

- Wrong number of color channels (expected 3)

- Unknown batch element type

**Scenario 2: Device mismatch**

- TensorRT model loaded on CPU device (requires CUDA)

- Model and input on different devices

- CUDA not available when required

**Scenario 3: Batch size inconsistency**

- Different batch sizes across input tensors

- Incompatible batch dimensions

- Batch size exceeds model limits

**Scenario 4: Input shape mismatch**

- Numpy array with wrong number of dimensions

- Torch tensor with incorrect shape

- Batched tensors not allowed but provided

- Missing required dimensions

### What To Check

1. **Check input type and format:**
   ```python
   import numpy as np
   import torch

   # Check input type
   print(f"Input type: {type(images)}")

   # For numpy arrays
   if isinstance(images, np.ndarray):
       print(f"Shape: {images.shape}")
       print(f"Dimensions: {len(images.shape)}")
       print(f"Channels: {images.shape[-1] if len(images.shape) == 3 else 'N/A'}")

   # For torch tensors
   if isinstance(images, torch.Tensor):
       print(f"Shape: {images.shape}")
       print(f"Device: {images.device}")

   # For lists
   if isinstance(images, list):
       print(f"List length: {len(images)}")
       if images:
           print(f"First element type: {type(images[0])}")
   ```

2. **Review error message:**

     * "Unsupported input type" → Wrong data type

     * "TRT engine only runs on CUDA" → Device mismatch

     * "different batch sizes" → Batch inconsistency
 
     * "incorrect number of dimensions" → Shape mismatch

### How To Fix

**Fix input type issues:**

```python
from inference_models import AutoModel
import numpy as np
import torch

model = AutoModel.from_pretrained("yolov8n-640")

# ❌ Wrong - unsupported type
images = "path/to/image.jpg"  # String not supported
result = model.predict(images)  # ModelRuntimeError!

# ❌ Wrong - unsupported type
images = ["path/to/image.jpg"]
result = model.predict(images)

# ✅ Or numpy array (HWC format, 3 channels)
image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
result = model.predict([image])

# ✅ Or torch tensor (CHW format for single image)
image = torch.rand(3, 640, 640)
result = model.predict(image)

# ✅ Or torch tensor (BCHW format for batch)
images = torch.rand(2, 3, 640, 640)
result = model.predict(images)
```

---

## Common Input Format Requirements

### Numpy Arrays

**Expected format:**
- **Shape:** `(height, width, channels)` - HWC format
- **Channels:** 3 (RGB or BGR, default BRG)
- **Dtype:** `uint8`
- **Value range:** 0-255 for uint8

```python
import numpy as np

# ✅ Correct numpy array format
image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Or from file
from PIL import Image
image = np.array(Image.open("image.jpg"))  # Automatically HWC format
```

### Torch Tensors

**Expected format:**
- **Shape (single image):** `(channels, height, width)` - CHW format
- **Shape (batch):** `(batch, channels, height, width)` - BCHW format
- **Channels:** 3 (RGB or BGR, default RGB)
- **Dtype:** `float32`
- **Value range:** 0.0-255.0

```python
import torch

# ✅ Correct torch tensor format (single image)
image = torch.rand(3, 640, 640)

# ✅ Correct torch tensor format (batch)
images = torch.rand(4, 3, 640, 640)

# Convert numpy (HWC) to torch (CHW)
import numpy as np
np_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
torch_image = torch.from_numpy(np_image).permute(2, 0, 1).float() / 255.0
```


### Lists

**Expected format:**
- **List of numpy arrays:** Each array in HWC format
- **List of torch tensors:** Each tensor in CHW format

```python
# ✅ List of numpy arrays
images = [
    np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
    np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8),
]

# ✅ List of torch tensors
images = [
    torch.rand(3, 640, 640),
    torch.rand(3, 480, 480),
]
```

---

## Backend-Specific Requirements

### TensorRT Backend

**Requirements:**
- **Device:** Must be CUDA (GPU)

```python
import torch
from inference_models import AutoModel

# ✅ Correct TensorRT usage
model = AutoModel.from_pretrained(
    "yolov8n-640",
    backend="trt",
    device="cuda"
)
```

### ONNX Backend

**Requirements:**
- **Device:** CPU or CUDA
- **Execution providers:** Must be configured
- **Flexibility:** Works on most platforms

```python
from inference_models import AutoModel

# ✅ ONNX on CPU
model = AutoModel.from_pretrained(
    "yolov8n-640",
    backend="onnx",
    device="cpu",
    onnx_execution_providers=["CPUExecutionProvider"]
)

# ✅ ONNX on GPU
model = AutoModel.from_pretrained(
    "yolov8n-640",
    backend="onnx",
    device="cuda",
    onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

### PyTorch Backend

**Requirements:**
- **Device:** CPU or CUDA
- **PyTorch installation:** Required
- **Model support:** Not all models available

```python
import torch
from inference_models import AutoModel

# ✅ PyTorch on CPU
model = AutoModel.from_pretrained(
    "sam-vit-b",
    backend="torch",
    device="cpu"
)

# ✅ PyTorch on GPU
if torch.cuda.is_available():
    model = AutoModel.from_pretrained(
        "sam-vit-b",
        backend="torch",
        device="cuda"
    )
```
