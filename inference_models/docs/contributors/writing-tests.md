# Writing Tests

This guide covers best practices for writing tests in `inference-models`.

## Test Organization

### Directory Structure

```
tests/
├── inference/
│   └── unit_tests/
│       └── models/
│           └── test_your_model.py
├── inference_cli/
│   └── unit_tests/
├── inference_sdk/
│   └── unit_tests/
└── workflows/
    └── unit_tests/
```

### Test File Naming

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

## Writing Unit Tests

### Basic Test Structure

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

def test_batch_inference():
    """Test inference on batch of images."""
    model = YourModel(weights_path="path/to/model.pt", device="cpu")
    images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    
    predictions = model(images)
    
    assert len(predictions) == 3
    for pred in predictions:
        assert hasattr(pred, 'xyxy')
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

## Marking Tests

### Slow Tests

Mark tests that take a long time:

```python
@pytest.mark.slow
def test_large_batch_inference():
    """Test inference on large batch (slow)."""
    model = YourModel(weights_path="path/to/model.pt", device="cpu")
    images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(100)
    ]
    predictions = model(images)
    assert len(predictions) == 100
```

Skip slow tests:

```bash
pytest -m "not slow" tests/
```

### GPU Tests

Mark tests that require GPU:

```python
@pytest.mark.gpu
def test_gpu_inference():
    """Test GPU inference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    model = YourModel(weights_path="path/to/model.pt", device="cuda")
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    predictions = model(image)
    assert len(predictions) == 1
```

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/inference/unit_tests/models/test_your_model.py
```

### Run Specific Test Function

```bash
pytest tests/inference/unit_tests/models/test_your_model.py::test_model_initialization
```

### Run with Coverage

```bash
pytest --cov=inference_models --cov-report=html tests/
```

### Run in Parallel

```bash
pytest -n auto tests/
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

