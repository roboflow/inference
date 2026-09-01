# AGENTS.md

This guide governs the entire repository. If a subfolder provides its own
`AGENTS.md`, its instructions supplement this file for that subtree.

## Overview

Roboflow Inference is a set of Python packages that run computer vision models
locally and expose them through an HTTP API and command-line interface. The
repository contains the core library, CLI, SDK, and Dockerfiles for CPU and GPU
images. The target Python version is 3.10, with 3.8 as the minimum.

## Project structure

- `inference/` – core library with model loading and streaming utilities.
- `inference_cli/` – command-line tools and server entry points.
- `inference_sdk/` – Python SDK for interacting with a running server.
- `docker/` – Dockerfiles for CPU and GPU images.
- `tests/` – unit and integration tests for all packages.
- `docs/` – MkDocs documentation source.

## Setup and environment

Create a Python environment and install the repository in editable mode:

```bash
conda create -n inference-development python=3.10
conda activate inference-development
pip install -e .
# optional models
pip install -e ".[sam]"
```

Important environment variables are defined in `inference/core/env.py`.

| Variable | Default | Purpose |
|---|---|---|
| `PROJECT` | `roboflow-platform` | Selects production or staging behavior. |
| `ROBOFLOW_API_KEY` | `""` | Enables authenticated requests. |
| `MODEL_CACHE_DIR` | `/tmp/cache` | Stores downloaded models. |
| `PORT` | `9001` | API port when running locally. |
| `NUM_WORKERS` | `1` | Number of server worker threads. |

Defaults above mirror the Dockerfiles in `docker/dockerfiles/`.

## Build and run

Build a development image and start the server from the repository root:

```bash
docker build -t roboflow/roboflow-inference-server-cpu:dev \
    -f docker/dockerfiles/Dockerfile.onnx.cpu.dev .
docker run -p 9001:9001 \
    -v ./inference:/app/inference \
    roboflow/roboflow-inference-server-cpu:dev
```

## Testing

Run package-specific unit tests with:

```bash
pytest tests/inference/unit_tests/
pytest tests/inference_cli/unit_tests/
pytest tests/inference_sdk/unit_tests/
pytest tests/workflows/unit_tests/
```

Run the complete suite while skipping slow tests with:

```bash
pytest -m "not slow" tests/
```

## Code style

Format code with:

```bash
make style
```

Check linting and formatting with:

```bash
make check_code_quality
```

The repository follows PEP 8 and uses Black (88 characters), isort, and flake8.

## Contribution guidelines

- Ensure all relevant tests pass before opening a pull request.
- Keep commit messages concise and in the present tense, such as `Add model loader`.
- Explain what changed and why in PR descriptions, and list the tests run.
- Update documentation when applicable.

## Canonical repository rules

The files under `.cursor/rules/` are the canonical detailed instructions for
this repository. Do not copy or restate their contents in `AGENTS.md` files.
Before acting on a task, read every applicable rule file completely and follow
its instructions. If a task spans multiple categories, read all matching files.

Always read:

- `.cursor/rules/uv-package-management.mdc`

Before editing or reviewing Python code, read:

- `.cursor/rules/empty-lines.mdc`
- `.cursor/rules/function-call.mdc`
- `.cursor/rules/google-docstrings.mdc`
- `.cursor/rules/pathlib.mdc`
- `.cursor/rules/return-values.mdc`

Also read the following rule when its condition applies:

- `.cursor/rules/cli-options.mdc` for Python command-line interfaces.
- `.cursor/rules/pydantic-field-descriptions.mdc` for Pydantic models.
- `.cursor/rules/pr-description.mdc` when the user requests a Roboflow-format PR
  body or explicitly names that rule.
- `.cursor/rules/execution-engine-version-changelog.mdc` for behavior changes
  under `inference/core/workflows/execution_engine/`; the subtree
  `AGENTS.md` repeats this routing requirement at the point of use.
