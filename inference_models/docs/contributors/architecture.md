# Architecture Overview

This guide provides a deep dive into the `inference-models` architecture for contributors and advanced users.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Application                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Public API Layer                          │
│  - AutoModel.from_pretrained()                              │
│  - AutoModelPipeline.from_pretrained()                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Auto-Loading & Resolution Layer                │
│  - Model metadata retrieval                                 │
│  - Backend selection & negotiation                          │
│  - Package filtering & ranking                              │
│  - Dependency resolution                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Weights Provider Layer                     │
│  - Roboflow API client                                      │
│  - Local filesystem provider                                │
│  - Custom providers (extensible)                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Download & Cache Layer                    │
│  - File download with retry                                 │
│  - Hash verification                                        │
│  - Local caching                                            │
│  - File locking for concurrent access                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Model Registry                           │
│  Maps (architecture, task, backend) → Model Class          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Model Implementation Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ PyTorch      │  │ ONNX         │  │ TensorRT     │      │
│  │ Models       │  │ Models       │  │ Models       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Hugging Face │  │ MediaPipe    │                        │
│  │ Models       │  │ Models       │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. AutoModel Class

**Location**: `inference_models/models/auto_loaders/core.py`

The main entry point for model loading. Key responsibilities:

- Validate user inputs
- Coordinate the loading process
- Manage caching
- Handle errors gracefully

**Key Methods**:

```python
class AutoModel:
    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs) -> AnyModel:
        """Load a model with automatic backend selection"""
        
    @classmethod
    def describe_model(cls, model_id, **kwargs) -> None:
        """Show model information without loading"""
        
    @classmethod
    def describe_runtime(cls) -> None:
        """Show runtime environment information"""
```

### 2. Model Registry

**Location**: `inference_models/models/auto_loaders/models_registry.py`

Maps model specifications to implementation classes:

```python
REGISTERED_MODELS: Dict[
    Tuple[ModelArchitecture, TaskType, BackendType],
    Union[LazyClass, RegistryEntry]
] = {
    ("yolov8", "object-detection", BackendType.ONNX): LazyClass(
        module_name="inference_models.models.yolov8.yolov8_object_detection_onnx",
        class_name="YOLOv8ForObjectDetectionOnnx",
    ),
    ("yolov8", "object-detection", BackendType.TRT): LazyClass(
        module_name="inference_models.models.yolov8.yolov8_object_detection_trt",
        class_name="YOLOv8ForObjectDetectionTRT",
    ),
    # ... hundreds more entries
}
```

**LazyClass**: Defers imports until needed, reducing startup time.

### 3. Weights Providers

**Location**: `inference_models/weights_providers/`

Retrieve model metadata and download URLs.

#### Roboflow Provider

```python
def get_roboflow_model(model_id: str, api_key: Optional[str]) -> ModelMetadata:
    """Fetch model metadata from Roboflow API"""
    # 1. Call API endpoint
    # 2. Parse response
    # 3. Validate model packages
    # 4. Return structured metadata
```

**API Endpoint**: `https://api.roboflow.com/models/v1/external/weights`

**Response Structure**:
```json
{
  "modelMetadata": {
    "modelId": "yolov8n-640",
    "modelArchitecture": "yolov8",
    "taskType": "object-detection",
    "modelPackages": [
      {
        "type": "external-model-package-v1",
        "packageId": "yolov8n-640-onnx-fp32",
        "packageManifest": {...},
        "packageFiles": [...]
      }
    ]
  }
}
```

### 4. Backend Selection

**Location**: `inference_models/models/auto_loaders/core.py`

The backend selection algorithm:

```python
def select_best_backend(
    available_packages: List[ModelPackageMetadata],
    installed_backends: Set[BackendType],
    hardware_info: HardwareInfo,
    user_preference: Optional[BackendType]
) -> ModelPackageMetadata:
    """
    1. Filter packages by installed backends
    2. Filter by hardware compatibility
    3. Apply user preference if specified
    4. Rank by default preference order
    5. Return best match or raise error
    """
```

**Default Preference Order**:
1. TensorRT (if GPU available)
2. PyTorch
3. ONNX
4. Hugging Face
5. MediaPipe

### 5. Model Base Classes

**Location**: `inference_models/models/base/`

Abstract base classes defining model interfaces:

```python
class ObjectDetectionModel(ABC):
    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load model from path"""
        
    @abstractmethod
    def pre_process(self, images, **kwargs):
        """Preprocess inputs"""
        
    @abstractmethod
    def forward(self, pre_processed_images, **kwargs):
        """Run model inference"""
        
    @abstractmethod
    def post_process(self, model_results, **kwargs):
        """Post-process outputs"""
        
    def __call__(self, images, **kwargs):
        """Convenience method for inference"""
        return self.infer(images, **kwargs)
```

All models follow this pattern:
1. `pre_process`: Resize, normalize, convert to tensor
2. `forward`: Run backend-specific inference
3. `post_process`: Parse outputs, apply NMS, convert to standard format

### 6. Caching System

#### Auto-Resolution Cache

Caches backend selection decisions:

```python
class AutoResolutionCache:
    def __init__(self):
        self._cache: Dict[str, Tuple[ModelClass, str]] = {}
        
    def get(self, cache_key: str):
        """Retrieve cached resolution"""
        
    def set(self, cache_key: str, model_class, package_dir):
        """Store resolution result"""
```

**Cache Key**: Hash of (model_id, backend_preferences, environment)

#### Model Package Cache

Downloaded files are stored at:
```
~/.cache/inference-models/
├── {model_id}/
│   ├── {package_id}/
│   │   ├── model.{pt,onnx,engine}
│   │   ├── class_names.txt
│   │   └── model_config.json
```

### 7. Download System

**Location**: `inference_models/utils/download.py`

Features:
- Parallel downloads
- Retry with exponential backoff
- MD5 hash verification
- Atomic file operations (download to temp, then rename)
- File locking for concurrent access

```python
def download_files_to_directory(
    files: List[FileDownloadSpecs],
    target_directory: str,
    verify_hash: bool = True,
    on_file_created: Optional[Callable] = None
):
    """Download files with verification and callbacks"""
```

## Data Flow

### Model Loading Flow

```
User calls AutoModel.from_pretrained("yolov8n-640")
    ↓
Check auto-resolution cache
    ↓ (cache miss)
Retrieve model metadata from Roboflow API
    ↓
Parse metadata → ModelMetadata object
    ↓
Filter packages by installed backends
    ↓
Rank packages by preference
    ↓
Select best package → ModelPackageMetadata
    ↓
Check local cache for package files
    ↓ (cache miss)
Download package files to cache
    ↓
Verify file hashes
    ↓
Resolve model class from registry
    ↓
Instantiate model class
    ↓
Load weights into model
    ↓
Return model instance
```

### Inference Flow

```
User calls model(images)
    ↓
model.infer(images)
    ↓
pre_processed, metadata = model.pre_process(images)
    ↓
raw_predictions = model.forward(pre_processed)
    ↓
predictions = model.post_process(raw_predictions, metadata)
    ↓
Return predictions
```

## Extension Points

### Adding a New Model

1. **Implement model class** inheriting from appropriate base
2. **Register in model registry**
3. **Add to weights provider** (if pre-trained)
4. **Write tests**
5. **Document**

See [Adding Models](adding-models.md) for details.

### Adding a New Backend

1. **Define backend type** in `BackendType` enum
2. **Implement model classes** for the backend
3. **Add backend detection** logic
4. **Update package parsing** in weights provider
5. **Add to extras** in `pyproject.toml`

### Adding a New Weights Provider

```python
from inference_models.weights_providers.core import WEIGHTS_PROVIDERS

def my_provider(model_id: str, api_key: Optional[str]) -> ModelMetadata:
    # Implement retrieval logic
    return ModelMetadata(...)

WEIGHTS_PROVIDERS["my-provider"] = my_provider
```

## Next Steps

- [Multi-Backend System](backends.md) - Backend implementation details
- [Dependency Management](dependencies.md) - Managing extras and conflicts
- [Adding Models](adding-models.md) - Step-by-step guide
- [Testing](testing.md) - Testing strategies

