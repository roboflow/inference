# Writing Tests

This guide covers best practices for writing tests in `inference-models`.

## Test Types

The `inference_models` library uses three types of tests, each serving a different purpose:

### 1. Integration Tests

**Location:** `inference_models/tests/integration_tests/models/`

Integration tests verify that your model produces correct predictions with actual model packages. They:

- Load models using `from_pretrained()` with real model packages
- Run inference on test images
- Assert that predictions match expected outputs (bounding boxes, confidence scores, class IDs, etc.)
- Test different input formats (numpy arrays, torch tensors, single images, batches)
- Test model-specific features (custom image sizes, preprocessing options, etc.)

**Example:** `test_yolov8_object_detection_predictions_onnx.py` tests YOLOv8 ONNX models with various configurations.

**When to add:** Always add integration tests when implementing a new model. These are the most important tests.

### 2. E2E Platform Tests

**Location:** `inference_models/tests/e2e_platform_tests/`

E2E (end-to-end) platform tests verify the full auto-loading pipeline using `AutoModel.from_pretrained()`:

- Test the complete workflow: weights provider → package download → model instantiation
- Verify that models work correctly when loaded through the platform
- Typically simpler than integration tests, focusing on basic functionality

**Example:** `test_moondream2_e2e.py` tests loading Moondream2 via AutoModel and running basic inference.

**When to add:** Add an E2E test to verify your model integrates correctly with the AutoModel system.

### 3. Unit Tests

**Location:** `inference_models/tests/unit_tests/`

Unit tests verify individual components in isolation:

- Auto-loader logic (`models/auto_loaders/`)
- Model registry functionality
- Utilities and helper functions
- Weights provider logic

**When to add:** Add unit tests if you created new utilities, helper functions, or custom auto-loader logic.

## Test Organization

### Directory Structure

```
inference_models/tests/
├── integration_tests/
│   └── models/
│       ├── test_yolov8_object_detection_predictions_onnx.py
│       ├── test_yolov8_object_detection_predictions_torch.py
│       └── test_<your_model>_predictions_<backend>.py
├── e2e_platform_tests/
│   ├── test_moondream2_e2e.py
│   └── test_<your_model>_e2e.py
└── unit_tests/
    ├── models/
    │   └── auto_loaders/
    └── weights_providers/
```

### Test File Naming

- Integration tests: `test_<model_name>_predictions_<backend>.py`
- E2E tests: `test_<model_name>_e2e.py`
- Unit tests: `test_<component>.py`
- Test functions: `test_*`
- Test classes: `Test*`

## Writing Integration Tests

Integration tests are the most important tests for model implementations. Here's how to write them:

### Basic Integration Test Structure

```python
import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_letterbox_numpy(
    coin_counting_yolov8n_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_onnx import (
        YOLOv8ForObjectDetectionOnnx,
    )

    model = YOLOv8ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9608, 0.9449, 0.9339]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[1296, 528, 3024, 1979], [1172, 2632, 1376, 2847]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
```

### Key Elements of Integration Tests

1. **Use fixtures for test data** - Model packages and test images should be provided via pytest fixtures
2. **Test multiple input formats** - Test with numpy arrays, torch tensors, single images, and batches
3. **Assert on actual outputs** - Compare predictions against known-good outputs
4. **Use appropriate tolerances** - Use `atol` for floating-point comparisons
5. **Mark tests appropriately** - Use `@pytest.mark.slow`, `@pytest.mark.onnx_extras`, etc.

### Testing Different Input Formats

```python
# Test single numpy array
def test_single_numpy(model_package, test_image_numpy):
    model = YourModel.from_pretrained(model_package)
    predictions = model(test_image_numpy)
    assert len(predictions) == 1

# Test batch of numpy arrays
def test_batch_numpy(model_package, test_image_numpy):
    model = YourModel.from_pretrained(model_package)
    predictions = model([test_image_numpy, test_image_numpy])
    assert len(predictions) == 2

# Test single torch tensor
def test_single_torch(model_package, test_image_torch):
    model = YourModel.from_pretrained(model_package)
    predictions = model(test_image_torch)
    assert len(predictions) == 1

# Test stacked torch tensors
def test_batch_torch_stacked(model_package, test_image_torch):
    model = YourModel.from_pretrained(model_package)
    batch = torch.stack([test_image_torch, test_image_torch], dim=0)
    predictions = model(batch)
    assert len(predictions) == 2

# Test list of torch tensors
def test_batch_torch_list(model_package, test_image_torch):
    model = YourModel.from_pretrained(model_package)
    predictions = model([test_image_torch, test_image_torch])
    assert len(predictions) == 2
```

## Writing E2E Platform Tests

E2E tests verify the AutoModel integration:

### Basic E2E Test Structure

```python
import pytest
from inference_models import AutoModel


@pytest.mark.e2e_model_inference
def test_moondream2_e2e():
    """Test Moondream2 loads and runs via AutoModel."""
    # Load model through AutoModel
    model = AutoModel.from_pretrained("vikhyatk/moondream2")

    # Run basic inference
    image = "path/to/test/image.jpg"
    prompt = "Describe this image"

    result = model(image, prompt)

    # Basic assertions
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
```

## Writing Unit Tests

Unit tests verify individual components:

### Basic Unit Test Structure

```python
import pytest
import numpy as np
from inference_models.models.your_model import YourModel

def test_model_initialization():
    """Test that model can be initialized."""
    model = YourModel(
        weights_path="path/to/model.pt",
        device="cpu",
    )
    assert model is not None
    assert model.device == "cpu"

def test_single_image_inference():
    """Test inference on a single image."""
    model = YourModel(weights_path="path/to/model.pt", device="cpu")
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    predictions = model(image)

    assert len(predictions) == 1
    assert hasattr(predictions[0], 'xyxy')
    assert hasattr(predictions[0], 'confidence')
    assert hasattr(predictions[0], 'class_id')
```

### Testing Error Handling

```python
def test_missing_dependency_error():
    """Test that missing dependencies raise appropriate errors."""
    from inference_models.exceptions import MissingDependencyError
    
    with pytest.raises(MissingDependencyError):
        # Code that requires missing dependency
        pass

def test_invalid_input():
    """Test that invalid inputs raise ValueError."""
    model = YourModel(weights_path="path/to/model.pt", device="cpu")
    
    with pytest.raises(ValueError):
        model(None)  # Invalid input
```

## Fixtures

### Using Fixtures

```python
import pytest

@pytest.fixture
def sample_image():
    """Provide a sample image for testing."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

@pytest.fixture
def model():
    """Provide a model instance for testing."""
    return YourModel(weights_path="path/to/model.pt", device="cpu")

def test_with_fixtures(model, sample_image):
    """Test using fixtures."""
    predictions = model(sample_image)
    assert len(predictions) == 1
```

### Fixture Scope

```python
@pytest.fixture(scope="module")
def expensive_model():
    """Load model once per test module."""
    model = YourModel(weights_path="path/to/model.pt", device="cpu")
    yield model
    # Cleanup code here if needed
```

## Parametrized Tests

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_different_devices(device):
    """Test model on different devices."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model = YourModel(weights_path="path/to/model.pt", device=device)
    assert model.device == device

@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_different_batch_sizes(batch_size):
    """Test model with different batch sizes."""
    model = YourModel(weights_path="path/to/model.pt", device="cpu")
    images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(batch_size)
    ]
    
    predictions = model(images)
    assert len(predictions) == batch_size
```

## Pytest Markers

Use markers to categorize and control test execution:

### Available Markers

- `@pytest.mark.slow` - Tests that take significant time (e.g., loading large models, processing many images)
- `@pytest.mark.gpu_only` - Tests that require GPU hardware
- `@pytest.mark.cpu_only` - Tests that should only run on CPU
- `@pytest.mark.onnx_extras` - Tests for ONNX-specific functionality
- `@pytest.mark.torch_models` - Tests for PyTorch-specific functionality
- `@pytest.mark.e2e_model_inference` - End-to-end platform tests

### Using Markers

```python
@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_large_batch():
    """Test ONNX model with large batch (slow)."""
    # Test code here
    pass

@pytest.mark.gpu_only
def test_gpu_inference():
    """Test GPU inference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # Test code here
    pass

@pytest.mark.e2e_model_inference
def test_automodel_loading():
    """Test loading via AutoModel."""
    # Test code here
    pass
```

### Running Tests by Marker

```bash
# Skip slow tests
uv run pytest -m "not slow" tests/

# Run only GPU tests
uv run pytest -m "gpu_only" tests/

# Run only ONNX tests
uv run pytest -m "onnx_extras" tests/

# Run E2E tests
uv run pytest -m "e2e_model_inference" tests/
```

## Running Tests

### Run All Tests

```bash
uv run pytest tests/
```

### Run Specific Test File

```bash
uv run pytest inference_models/tests/integration_tests/models/test_your_model.py
```

### Run Specific Test Function

```bash
uv run pytest inference_models/tests/integration_tests/models/test_your_model.py::test_model_initialization
```

### Run with Coverage

```bash
uv run pytest --cov=inference_models --cov-report=html tests/
```

### Run in Parallel

```bash
uv run pytest -n auto tests/
```

## Best Practices

1. ✅ **Test one thing per test** - Each test should verify a single behavior
2. ✅ **Use descriptive names** - Test names should explain what they test
3. ✅ **Keep tests independent** - Tests should not depend on each other
4. ✅ **Use fixtures for setup** - Avoid duplicating setup code
5. ✅ **Test edge cases** - Empty inputs, large batches, invalid data
6. ✅ **Mark slow tests** - Allow fast test runs during development
7. ✅ **Clean up resources** - Use fixtures with cleanup or context managers
8. ❌ **Don't test implementation details** - Test behavior, not internals
9. ❌ **Don't skip tests without reason** - Fix or remove broken tests
10. ❌ **Don't use real API calls** - Mock external dependencies

## Next Steps

- [Development Environment](dev-environment.md) - Set up your test environment
- [Adding a Model](adding-model.md) - Add tests for new models
- [Core Architecture](core-architecture.md) - Understand what to test

