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

### Running the inference server locally (without Docker)

Use the `debugrun.py` script at the repo root. It creates the model registry, manager, and HTTP interface directly:

```bash
cd /agent/repos/inference
PORT=9001 CORE_MODEL_SAM_ENABLED=False CORE_MODEL_SAM3_ENABLED=False python3 debugrun.py
```

The server will be available at `http://localhost:9001`. Verify with `curl http://localhost:9001/info`.

Disable SAM/SAM3 models via env vars to avoid missing-dependency warnings unless you specifically need them.

### Testing

- Unit tests: `pytest tests/inference/unit_tests/` (1400+ tests, ~2 min)
- SDK tests: `pytest tests/inference_sdk/unit_tests/`
- CLI tests: `pytest tests/inference_cli/unit_tests/`
- Workflow tests: `pytest tests/workflows/unit_tests/`
- Lint: `make check_code_quality` (Black + isort + flake8). Note: there are pre-existing flake8 F824 warnings in the repo.

### Gotchas

- The `inference_cli server start` command is designed to manage Docker containers, not for local dev. Use `debugrun.py` instead.
- The `app` attribute does not exist as a module-level variable in `http_api.py` (it is created inside the `HttpInterface` class). Do not try to start with `uvicorn inference.core.interfaces.http.http_api:app`.
