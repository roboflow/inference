# Adding a Model

This guide walks you through the process of adding a new model to `inference-models`.

## Overview

Adding a model involves:

1. Understanding model implementation philosophy and base classes
2. Creating model implementation with `from_pretrained` contract
3. Registering model in the registry
4. Preparing model packages for registration in Roboflow weights provider
5. Adding tests
6. Writing documentation

## Step 1: Understanding Model Implementation Philosophy

Before writing code, you need to understand how models are organized in `inference-models`. The library provides base classes for common model categories, but you're not required to use them - the only hard requirement is implementing the `from_pretrained` contract.

### Model Categories and Base Classes

The library provides base classes for common computer vision tasks:

- **`ObjectDetectionModel`** - For object detection models (returns `Detections`)
- **`InstanceSegmentationModel`** - For instance segmentation (returns `InstanceDetections`)
- **`ClassificationModel`** - For image classification (returns `ClassificationPrediction`)
- **`KeyPointsDetectionModel`** - For keypoint detection (returns `KeyPoints`)
- **`SemanticSegmentationModel`** - For semantic segmentation
- **`DepthEstimationModel`** - For depth estimation
- **`TextImageEmbeddingModel`** - For embedding models (CLIP-like)
- **`StructuredOCRModel`** / **`TextOnlyOCRModel`** - For OCR tasks
- **`OpenVocabularyObjectDetectionModel`** - For open-vocabulary detection

These base classes provide a consistent interface with methods like `pre_process()`, `forward()`, and `post_process()`. If your model fits one of these categories, **consider extending the appropriate base class** - it will give you a consistent interface and make your model easier to use.

However, **you're not required to extend these base classes**. Some models have unique behaviors that don't fit neatly into these categories (like SAM for interactive segmentation, or vision-language models with multiple task modes). In such cases, it's perfectly acceptable to create a standalone model class.

### The `from_pretrained` Contract

The only hard requirement is that your model class must implement a `from_pretrained` class method:

```python
@classmethod
def from_pretrained(
    cls,
    model_name_or_path: str,
    **kwargs
) -> "YourModel":
    """Load model from a directory containing model files.

    Args:
        model_name_or_path: Path to directory with model files
        **kwargs: Additional parameters passed from AutoModel

    Returns:
        Initialized model instance
    """
    # Load model files from model_name_or_path
    # Initialize and return model instance
    pass
```

This method is called by `AutoModel` after downloading and caching model files. The `model_name_or_path` parameter points to a directory containing all the files specified in your model package metadata.

### When to Create a New Base Class

If you notice a growing number of models from the same category that don't fit existing base classes, consider creating a new base class. For example, if you're adding multiple gaze detection models, it might make sense to create a `GazeDetectionModel` base class that standardizes the interface.

The decision should be based on:

- **Common interface** - Do these models share similar inputs/outputs?

- **Reusable logic** - Is there preprocessing/postprocessing code that could be shared?

- **Growing category** - Are more models of this type likely to be added?

## Step 2: Create Model Implementation

### Directory Structure

Create a new directory for your model family:

```
inference_models/models/your_model/
├── __init__.py
├── model.py           # Main model class
├── preprocessing.py   # Input preprocessing (if needed)
├── postprocessing.py  # Output postprocessing (if needed)
└── README.md          # Model-specific notes
```

### Example: Extending a Base Class

If your model fits an existing category, extend the appropriate base class:

```python
from typing import List, Tuple, Union
import torch
import numpy as np
from inference_models import ObjectDetectionModel, Detections

class YourObjectDetectionModel(ObjectDetectionModel[torch.Tensor, dict, torch.Tensor]):
    """Your object detection model implementation."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs
    ) -> "YourObjectDetectionModel":
        # Load model files from model_name_or_path
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        return cls(model_path=model_name_or_path, device=device)

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self._class_names = []  # Load from metadata
        # Load model weights

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        # Preprocess images and return preprocessing metadata
        pass

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        # Run model inference
        pass

    def post_process(
        self,
        model_results: torch.Tensor,
        pre_processing_meta: dict,
        **kwargs
    ) -> List[Detections]:
        # Convert raw outputs to Detections
        pass
```

### Example: Standalone Model

For models with unique behaviors:

```python
from typing import List, Union
import torch
import numpy as np

class YourUniqueModel:
    """Model with unique interface not fitting standard categories."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs
    ) -> "YourUniqueModel":
        return cls(model_path=model_name_or_path, **kwargs)

    def __init__(self, model_path: str, **kwargs):
        # Initialize model
        pass

    def __call__(self, images, **kwargs):
        # Your custom inference logic
        pass

    # Add any task-specific methods your model needs
    def task_a(self, images, **kwargs):
        pass

    def task_b(self, images, **kwargs):
        pass
```

## Step 3: Register Model in Registry

After implementing your model, you need to register it so `AutoModel` can find and instantiate it.

### Understanding the Registry

The model registry (`inference_models/models/auto_loaders/models_registry.py`) maps a combination of:
 
- **Model architecture** (e.g., `"yolov8"`, `"sam"`, `"moondream2"`)

- **Task type** (e.g., `"object-detection"`, `"classification"`)

- **Backend type** (e.g., `BackendType.ONNX`, `BackendType.TORCH`)

to your model implementation class.

When `AutoModel.from_pretrained()` is called:

1. It retrieves model metadata from the weights provider

2. It looks up the implementation in the registry using `(model_architecture, task_type, backend)`

3. It calls `model_class.from_pretrained(model_package_cache_dir, **kwargs)`

### Choosing Model Architecture and Task Type

**Model Architecture** should identify the model family (e.g., `"yolov8"`, `"resnet"`, `"sam"`). Use the same architecture name across different backends and task types for the same model family.

**Task Type** should describe what the model does. Use existing task types when possible:
 
- `"object-detection"` - Detect and localize objects

- `"instance-segmentation"` - Segment object instances

- `"classification"` - Classify images

- `"keypoint-detection"` - Detect keypoints

- `"embedding"` - Generate embeddings

- `"vlm"` - Vision-language models

- `"depth-estimation"` - Estimate depth

- `"gaze-detection"` - Detect gaze direction

- `"open-vocabulary-object-detection"` - Open-vocabulary detection

- `"interactive-instance-segmentation"` - Interactive segmentation (SAM-like)

You can create new task types if your model doesn't fit existing categories, but **align with the metadata from your weights provider** - the task type in the registry must match the task type in model metadata.

### Adding Registry Entry

Edit `inference_models/models/auto_loaders/models_registry.py`:

```python
from inference_models.utils.imports import LazyClass

REGISTERED_MODELS: Dict[
    Tuple[ModelArchitecture, TaskType, BackendType], Union[LazyClass, RegistryEntry]
] = {
    # ... existing entries ...

    # Add your model - simple entry
    ("your-model", "object-detection", BackendType.ONNX): LazyClass(
        module_name="inference_models.models.your_model.model",
        class_name="YourModel",
    ),

    # If you have multiple backends, add entries for each
    ("your-model", "object-detection", BackendType.TORCH): LazyClass(
        module_name="inference_models.models.your_model.model_torch",
        class_name="YourModelTorch",
    ),
}
```

**Using `LazyClass`**: This defers importing your model until it's actually needed, keeping startup time fast.

### Advanced: Registry Entry with Features

If your model supports optional features (like fused NMS), use `RegistryEntry`:

```python
("your-model", "object-detection", BackendType.ONNX): RegistryEntry(
    model_class=LazyClass(
        module_name="inference_models.models.your_model.model",
        class_name="YourModel",
    ),
    supported_model_features={"nms_fused", "dynamic_batching"},
),
```

This allows the auto-loader to select the right implementation based on model package features.

### Exposing Your Model in `__init__.py`

If you created a new base class, export it :

```python
from inference_models.models.base.your_category import YourCategoryModel

__all__ = [
    # ... existing exports ...
    "YourCategoryModel",
]
```

### Using Developer Tools

When implementing `from_pretrained`, you can use developer tools to access model package contents:

```python
from inference_models.developer_tools import get_model_package_contents

@classmethod
def from_pretrained(cls, model_name_or_path: str, **kwargs):
    # Get all files from the model package
    package_contents = get_model_package_contents(model_name_or_path)

    # Access specific files
    weights_path = package_contents["model.onnx"]
    config_path = package_contents["config.json"]

    return cls(weights_path=weights_path, config_path=config_path, **kwargs)
```

## Step 4: Prepare Model Packages for Registration

After implementing your model, you need to prepare model package artifacts for registration in the Roboflow weights provider.

### Multiple Backend Implementations

When adding one model implementation, it's often a low-hanging fruit to add implementations and model packages for other backends. For example:

- If you implement a **PyTorch** model, you can often convert it to **ONNX** for broader compatibility
- ONNX models can sometimes be further optimized to **TensorRT** for GPU inference
- Consider which backends make sense for your model's use case

Adding multiple backend implementations increases the model's accessibility and allows users to choose the best backend for their deployment environment.

### For Internal Roboflow Contributors

Follow the internal procedure for model registration in the Roboflow weights provider. This includes:

1. Preparing model artifacts (weights, metadata, configuration files)
2. Uploading artifacts through internal API
3. Testing the model through the AutoModel interface

### For External Contributors

If you're an external contributor:

1. **Create an issue** in the repository describing your model contribution
2. **Deliver the artifacts** including:

     - Model weights files (e.g., `.pt`, `.onnx`, `.trt`)
   
     - Model metadata (architecture, task type, input dimensions, class names)
   
     - Any additional configuration files needed
   
     - Example usage and expected outputs
   
3. The Roboflow team will review and handle the registration process

## Step 5: Add Tests

Tests are essential for ensuring your model implementation works correctly. For detailed guidance on writing tests, see the [Writing Tests](writing-tests.md) guide.

## Step 6: Write Documentation

Model documentation should be added to the `inference_models/docs/models/` directory. The documentation structure follows the existing pattern:

- **Individual model pages** go in `inference_models/docs/models/your-model.md`
- **Model category index pages** are in `inference_models/docs/models/index.md` and category-specific pages
- Follow the existing documentation style and structure for consistency

Your documentation should cover:

- Model description and use cases

- Installation instructions

- Basic usage examples

- Model variants (if applicable)

- Performance characteristics

- Citation information (if applicable)

For more information:
- [Writing Tests](writing-tests.md) - Best practices for testing
- [Dependencies and Backends](dependencies-and-backends.md) - Managing dependencies

