# Experimental version of `inference`

## Development Setup

This package uses `uv` for package management and build tooling. It is recommended to create a separate virtual environment to avoid dependency conflicts.

### 1. Create and Activate a Virtual Environment

Navigate to the `inference_experimental` directory and create a virtual environment using `uv`.

```bash
cd inference_experimental
uv venv
source .venv/bin/activate
```

### 2. Install Dependencies

Install the package in editable mode. This allows you to make changes to the code and have them reflected immediately without reinstalling.

```bash
uv pip install -e .
```

The project also includes optional dependencies for different hardware and backends (e.g., `onnx-cpu`, `torch-cu118`). You can install them as needed. For example, to install with `onnx-cpu` support:

```bash
uv pip install -e '.[onnx-cpu]'
```

### 3. Running Tests

You can run the test suite using `pytest`.

```bash
pytest
```

**Note for non-NVIDIA/TensorRT Systems (like macOS):**

Some tests are specific to environments with NVIDIA GPUs and TensorRT installed. These tests are marked with `trt_extras`. If you are on a system without these dependencies, you will need to exclude these tests when running `pytest`.

```bash
pytest -m "not trt_extras"
```

