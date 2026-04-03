# Load Models Locally

This guide explains the three ways to load models from local storage in `inference-models`.

## Overview

The `inference-models` library supports three distinct approaches for loading models locally:

1. **[Custom Model Packages](#1-custom-model-packages)** - Run models with custom architectures not in the main package (both code and weights from local directory)
2. **[Locally Cached Packages](#2-locally-cached-packages)** - Load models from cached packages distributed via weights providers (useful for development)
3. **[Direct Checkpoint Loading](#3-direct-checkpoint-loading)** - Load training checkpoints directly (currently RF-DETR only)

Each approach serves different use cases - choose based on your needs.

## 1. Custom Model Packages

Load models with **custom architectures not in the main `inference-models` package**. This approach is especially valuable for production deployment of proprietary or experimental models.

### When to Use

- **Custom architectures** - Run models with architectures not submitted to the main package
- **Proprietary models** - Keep your model code and architecture private
- **Using `inference-models` as a deployment tool** - Leverage production-ready tooling (multi-backend support, model loading, preprocessing) and integration with the Roboflow ecosystem ([Workflows](https://inference.roboflow.com/workflows/about/), [InferencePipeline](https://inference.roboflow.com/quickstart/inference_pipeline/))

### Package Structure

A custom model package contains both the model implementation code and weights:

```
my_custom_model/
├── model_config.json    # Points to your model class
├── model.py            # Your model implementation
└── weights.pt          # Model weights (optional)
```

### Loading Custom Models

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "/path/to/my_custom_model",
    allow_local_code_packages=True
)
```

!!! warning "Security Warning"
    Only enable `allow_local_code_packages` for trusted sources. This allows execution of arbitrary Python code from the model package.

### Creating Custom Model Packages

#### Step 1: Create `model_config.json`

The config file specifies which Python module and class to load:

```json
{
  "model_module": "model.py",
  "model_class": "MyCustomDetector"
}
```

**Required fields:**

- **`model_module`** - Name of the Python file containing your model class (e.g., `"model.py"`)
- **`model_class`** - Name of the class in that module (e.g., `"MyCustomDetector"`)

#### Step 2: Implement Your Model Class

Your model class must comply with the standard `.from_pretrained(...)` classmethod schema that all models use:

**Example: Object Detection Model**

```python
from typing import List, Optional, Union
import numpy as np
import torch
from inference_models import ObjectDetectionModel, Detections
from inference_models.developer_tools import get_model_package_contents

class MyCustomDetector(ObjectDetectionModel[torch.Tensor, torch.Tensor]):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> "MyCustomDetector":
        """Load model from package directory.

        Args:
            model_name_or_path: Path to model package directory
            device: Device to load model on
            **kwargs: Additional arguments

        Returns:
            Initialized model instance
        """
        # Get model package contents
        package_contents = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["weights.pt", "config.json"]  # Files you need
        )

        # Load your model
        model = torch.load(package_contents["weights.pt"], map_location=device)

        return cls(model=model, device=device)

    def pre_process(self, images, **kwargs):
        # Your preprocessing logic
        pass

    def forward(self, pre_processed_images, **kwargs):
        # Your inference logic
        pass

    def post_process(self, model_results, **kwargs) -> Detections:
        # Your postprocessing logic
        pass
```

**Example: Classification Model**

```python
from typing import List, Optional, Union
import numpy as np
import torch
from inference_models import ClassificationModel, ClassificationPrediction, ColorFormat
from inference_models.developer_tools import get_model_package_contents

class MyClassificationModel(ClassificationModel[torch.Tensor, torch.Tensor]):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> "MyClassificationModel":
        # Load model package contents
        package_contents = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["weights.pt"]
        )

        # Initialize your model
        model = torch.load(package_contents["weights.pt"], map_location=device)

        return cls(model=model, device=device)

    @property
    def class_names(self) -> List[str]:
        return ["class1", "class2", "class3"]

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Your preprocessing logic
        pass

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        # Your inference logic
        pass

    def post_process(
        self,
        model_results: torch.Tensor,
        **kwargs,
    ) -> ClassificationPrediction:
        # Your postprocessing logic
        pass
```

!!! tip "Imports from `inference-models`"
    When creating custom models, use the public `inference-models` interface:

    **Base model classes:**

    - `from inference_models import ObjectDetectionModel, Detections`
    - `from inference_models import ClassificationModel, ClassificationPrediction`
    - `from inference_models import InstanceSegmentationModel`
    - `from inference_models import KeypointsDetectionModel`

    **Developer tools:**

    - `from inference_models.developer_tools import get_model_package_contents` - Load files from model packages
    - `from inference_models.developer_tools import x_ray_runtime_environment` - Runtime introspection

    **Utilities:**

    - `from inference_models import ColorFormat` - Image color format handling

    See [Core Concepts - Clear Public Interface](understand-core-concepts.md#1-clear-public-interface) for the complete public API.

## 2. Locally Cached Packages

Load models from **locally cached packages distributed via weights providers**. This approach is useful when developing changes to model code or before upstreaming implementation to the main repository.

### When to Use

- **Development and debugging** - Test changes to model implementations before contributing
- **Determining package contents** - Verify what files should be distributed by weights providers
- **Offline testing** - Work with cached models without network access

### How It Works

When you load a model from a weights provider (like Roboflow), `inference-models` downloads and caches the model package locally. Both `AutoModel` and specific model implementation classes can load from this cache.

### Loading from Cache

```python
from inference_models import AutoModel

# First load downloads and caches the model
model = AutoModel.from_pretrained("rfdetr-base")

# Subsequent loads use the cached version
# Default cache location: /tmp/cache/models-cache/
# Override with: export INFERENCE_HOME=/path/to/cache
```

### Direct Cache Access

You can also load directly from a cached package directory:

```python
from inference_models import AutoModel

# Load from specific cache directory
model = AutoModel.from_pretrained(
    "/tmp/cache/models-cache/rfdetr-base-6a8b9c2d/torch-fp32-batch1"
)
```

### Package Structure

This is an example structure - each model follows its own rules and there are no arbitrary files:

```
model_package/
├── model_config.json       # Auto-generated metadata
├── weights.pt              # Model weights
└── class_names.txt         # Class labels (if applicable)
```

!!! info "Roboflow Platform Models"
    Models trained on Roboflow Platform comply to a custom RF standard where runtime behavior is determined by `inference_config.json`. See [Understand Roboflow Model Packages](roboflow-model-packages.md) for details.

### Development Workflow

1. **Download model package** - Load model to populate cache
2. **Modify model code** - Edit implementation in your local repository
3. **Test with cached weights** - Load from cache directory to test changes
4. **Verify package contents** - Ensure all required files are present
5. **Upstream changes** - Submit PR with updated implementation

## 3. Direct Checkpoint Loading

Load **training checkpoints directly** without conversion or export. Currently supported for **RF-DETR models only**.

### When to Use

- **Seamless training-to-deployment** - Go from training to production instantly
- **Models trained outside Roboflow** - Use models trained with the [rf-detr repository](https://github.com/roboflow/rf-detr)
- **Rapid iteration** - Test freshly trained models without export steps

### Loading RF-DETR Checkpoints

```python
from inference_models import AutoModel

# Load RF-DETR checkpoint directly
model = AutoModel.from_pretrained(
    "/path/to/checkpoint_best.pth",
    model_type="rfdetr-base",  # Required: specify architecture
    labels=["class1", "class2", "class3"]  # Optional: your class names
)
```

**Required parameters:**

- **`model_type`** - RF-DETR architecture variant: `rfdetr-nano`, `rfdetr-small`, `rfdetr-base`, `rfdetr-medium`, `rfdetr-large`, `rfdetr-seg-preview`

**Optional parameters:**

- **`labels`** - Class names as a list or registered label set name (e.g., `"coco"`)

### Why This Matters

**Frictionless training-to-production workflow:**

- ✅ **No model conversion** - Use training checkpoints directly
- ✅ **No export step** - Skip ONNX/TensorRT export complexity
- ✅ **Instant deployment** - From training to production in seconds
- ✅ **Same API** - Identical interface for pre-trained and custom models

### Learn More

See the RF-DETR model documentation for complete training and deployment workflows:

- [RF-DETR Object Detection](../models/rfdetr-object-detection.md#trained-rf-detr-outside-roboflow-use-with-inference-models)
- [RF-DETR Instance Segmentation](../models/rfdetr-instance-segmentation.md#trained-rf-detr-segmentation-outside-roboflow-use-with-inference-models)

## Comparison Table

| Approach | Use Case | Code Required | Weights Location | When to Use |
|----------|----------|---------------|------------------|-------------|
| **Custom Model Packages** | Custom architectures | ✅ Yes (model.py) | Local directory | Production deployment of proprietary models |
| **Locally Cached Packages** | Standard architectures | ❌ No (uses library code) | Cache directory | Development, testing, offline work |
| **Direct Checkpoint Loading** | RF-DETR only | ❌ No (uses library code) | Checkpoint file | Training-to-deployment workflow |

## Next Steps

- [Understand Core Concepts](understand-core-concepts.md) - Understand the public interface and developer tools
- [RF-DETR Object Detection](../models/rfdetr-object-detection.md) - Learn about checkpoint loading
- [Supported Models](../models/index.md) - Browse available models

