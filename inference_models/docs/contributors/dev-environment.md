# Development Environment

This guide walks you through setting up your development environment for contributing to `inference-models`.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver
- Git
- (Optional) CUDA-capable GPU for GPU backend development

!!! note "Python Version"
    `uv` will automatically install Python 3.12 when creating the virtual environment. No need to install Python separately.

## Setting Up the Environment

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the Repository

```bash
git clone https://github.com/roboflow/inference.git
cd inference/inference_models
```

### 3. Sync Dependencies

`uv` will automatically create a virtual environment and install dependencies:

```bash
# Install base package with test dependencies
uv sync --extra test

# Install with specific backends for CPU development
uv sync --extra test --extra torch-cpu --extra onnx-cpu

# Install with GPU backends (CUDA 12.8)
uv sync --extra test --extra torch-cu128 --extra onnx-cu12 --extra trt10

# Install with model-specific dependencies (e.g., MediaPipe)
uv sync --extra test --extra mediapipe
```

!!! tip "uv sync vs uv pip install"
    `uv sync` is the recommended way to install dependencies as it:

    - Creates a virtual environment automatically
    - Installs the package in editable mode
    - Locks dependencies for reproducibility
    - Is significantly faster than pip

## Running Tests

!!! note "Working Directory"
    All commands below assume you're in the `inference/inference_models` directory.

### Run All Tests

```bash
# From inference/inference_models directory
uv run pytest tests/
```

### Run Specific Test Suites

```bash
# Unit tests
uv run pytest tests/unit_tests/

# Integration tests
uv run pytest tests/integration_tests/
```

### Skip Slow Tests

```bash
uv run pytest -m "not slow" tests/
```

## Code Quality

!!! note "Working Directory"
    All commands below assume you're in the `inference/inference_models` directory.

### Format Code

```bash
# From inference/inference_models directory
black inference_models tests
isort --profile black inference_models tests
```

### Check Code Quality

```bash
# From inference/inference_models directory
black --check inference_models tests
isort --profile black --check inference_models tests
```

!!! tip "Contribution Idea"
    We'd love to have a unified code quality tool or script! If you're interested in improving the developer experience, consider creating a simple script or Makefile to run these commands together.

## Verifying Your Setup

!!! note "Working Directory"
    All commands below assume you're in the `inference/inference_models` directory.

Test that everything works:

```bash
# From inference/inference_models directory
uv run python -c "from inference_models import AutoModel; print('✅ Import successful')"
```

Run a simple inference test:

```bash
# From inference/inference_models directory
uv run python -c "
from inference_models import AutoModel
model = AutoModel.from_pretrained('yolov8n-640')
print('✅ Model loaded successfully')
"
```

## Next Steps

- [Core Architecture](core-architecture.md) - Understand the codebase structure
- [Adding a Model](adding-model.md) - Learn how to add a new model
- [Writing Tests](writing-tests.md) - Best practices for testing

