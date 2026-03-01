# AGENTS.md

This guide governs the entire repository. If a subfolder provides its own
`AGENTS.md`, instructions there override this file for that subtree.

## Overview
Roboflow Inference is a set of Python packages that run computer vision models
locally and expose them via an HTTP API and command line interface. The repo
contains the core library, CLI, SDK, and Dockerfiles for building CPU or GPU
images. Target Python version is 3.10 (minimum 3.8).

## Project Structure
- `inference/` – core library with model loading and streaming utilities.
- `inference_cli/` – command line tools and server entry points.
- `inference_sdk/` – Python SDK for interacting with a running inference server.
- `docker/` – Dockerfiles used to build CPU and GPU images.
- `tests/` – unit and integration tests for all packages.
- `docs/` – mkdocs documentation source.

## Setup / Environment
Create a Python environment and install the repo in editable mode:

```bash
conda create -n inference-development python=3.10
conda activate inference-development
pip install -e .
# optional models
pip install -e ".[sam]"
```

Important environment variables (see `inference/core/env.py` for all):
| Variable           | Default            | Purpose                           |
|--------------------|--------------------|-----------------------------------|
| `PROJECT`          | `roboflow-platform`| Selects prod or staging behavior  |
| `ROBOFLOW_API_KEY` | `""`               | Enables authenticated requests    |
| `MODEL_CACHE_DIR`  | `/tmp/cache`       | Stores downloaded models          |
| `PORT`             | `9001`             | API port when running locally     |
| `NUM_WORKERS`      | `1`                | Number of server worker threads   |

Defaults above mirror the Dockerfiles in `docker/dockerfiles/`.

## Build & Running
Build a development image and start the server from the repository root:

```bash
docker build -t roboflow/roboflow-inference-server-cpu:dev \
    -f docker/dockerfiles/Dockerfile.onnx.cpu.dev .
docker run -p 9001:9001 \
    -v ./inference:/app/inference \
    roboflow/roboflow-inference-server-cpu:dev
```

## Testing
Unit tests live in package specific folders. Run them individually with:

```bash
pytest tests/inference/unit_tests/
pytest tests/inference_cli/unit_tests/
pytest tests/inference_sdk/unit_tests/
pytest tests/workflows/unit_tests/
```

To run the entire suite while skipping slow tests:

```bash
pytest -m "not slow" tests/
```

## Code Style
Format code with:

```bash
make style
```

Check linting and formatting with:

```bash
make check_code_quality
```

The repository follows PEP 8 and uses Black (88 characters), isort and flake8.

## Contribution / PR Guidelines
- Ensure all relevant tests pass before opening a pull request.
- Keep commit messages concise and in the present tense, e.g. "Add model loader".
- PR descriptions should explain what changed and why, list test commands run,
  and follow the templates in `.github`.
- Update documentation when applicable.

## Cursor Cloud specific instructions

### Running the server natively (without Docker)

The `docker/` directory name conflicts with the installed `docker` Python package,
so `uvicorn docker.config.cpu_http:app` will fail with a `ModuleNotFoundError`.
Work around this by copying the config file to a temp location and using `--app-dir`:

```bash
cp docker/config/cpu_http.py /tmp/cpu_http.py
PYTHONPATH=/workspace uvicorn --app-dir /tmp cpu_http:app --host 0.0.0.0 --port 9001
```

The server starts on port 9001 by default. CLIP and other core models load on first
request without needing `ROBOFLOW_API_KEY`. Models requiring Roboflow-hosted weights
(e.g. custom YOLOv8 models) need the `ROBOFLOW_API_KEY` environment variable.

### Linting, testing, building

See the **Testing**, **Code Style**, and **Build & Running** sections above.
All commands (`make check_code_quality`, `make style`, `pytest`) work directly
with `PATH=$HOME/.local/bin:$PATH` after `pip install -e .`.

### Gotchas

- System packages `libopencv-dev`, `libgdal-dev`, `libvips-dev`, and `cmake` must
  be installed for native extension compilation (opencv, pyvips, etc.).
- `uvicorn` is pinned to `<=0.22.0` in test requirements but `<=0.34.0` in http
  requirements; the installed version may vary. Both work for local dev.
- The second (exit-zero) `flake8` pass in `make check_code_quality` prints warnings
  but does not fail the build; only the first pass (syntax errors) is blocking.
