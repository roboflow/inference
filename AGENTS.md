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
pip install -e ./inference_models -e ./workflows -e .
# optional models
pip install -e ./inference_models -e ./workflows -e ".[sam]"
```

Run development commands from the repository root so the checkout's SDK source
takes precedence over the installed SDK dependency. The unpublished development
umbrella includes SDK source; release wheels leave SDK ownership to `inference-sdk`.

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
(cd workflows && pytest tests/unit_tests/ tests/isolation/)
```

To run the entire suite while skipping slow tests:

```bash
pytest -m "not slow" tests/
(cd workflows && pytest -m "not slow" tests/)
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

## Workflows Package (roboflow-workflows)

The Workflows block library lives in `workflows/` as a standalone Python project.

### Project layout
```
workflows/
  roboflow_workflows/    # importable package (distribution: roboflow-workflows)
  tests/
    unit_tests/          # package unit tests
    isolation/           # isolation probe tests (WORKFLOWS_ISOLATION_WHEEL must be set)
  scripts/
    workflows_isolation_probe.py  # CLI tool, see below
  build_scripts/
    download_fonts.py    # canonical font downloader (shim at root build_scripts/)
  pyproject.toml
  uv.lock
  pytest.ini
  CHANGELOG.md
```

### Versioning

`roboflow-workflows` is published to PyPI separately from `inference` and
pinned by `requirements/requirements.workflows.txt`. Contributors: add an entry
under `## Unreleased` in `workflows/CHANGELOG.md` for any change in
`roboflow_workflows/`. Maintainers: at release, bump `version` in
`workflows/pyproject.toml`, the pin in `requirements/requirements.workflows.txt`,
the hardcoded `roboflow_workflows-<version>-py3-none-any.whl` in
`.github/workflows/*.yml`, and run `cd workflows && uv lock`. Publishing uses
`skip-existing`, so an unbumped version is silently not re-published.

### Font provisioning

Fonts are package-data in the `roboflow_workflows` wheel. Before building the
wheel, download them:

```bash
make download_fonts        # downloads to workflows/roboflow_workflows/.../fonts/assets/
```

Or directly:

```bash
python workflows/build_scripts/download_fonts.py
```

The root `build_scripts/download_fonts.py` is a thin shim that delegates to the
canonical script above.

### Building the wheel

```bash
make create_workflows_wheel    # downloads fonts + uv build
# wheel lands in dist/roboflow_workflows-*.whl
```

### Running package tests

From the repo root (after building the wheel):

```bash
cd workflows && pip install --find-links ../dist "$(ls ../dist/roboflow_workflows-*.whl)[test]"
cd workflows && python -m pytest tests/unit_tests tests/isolation
```

Set `ENABLE_TENSOR_DATA_REPRESENTATION=True` for tensor-native mode (matches CI knob).

### Running the isolation probe

```bash
python workflows/scripts/workflows_isolation_probe.py \
    --wheel dist/roboflow_workflows-*.whl \
    --find-links dist/
```

The probe creates a throwaway venv outside the checkout, installs the wheel,
blocks all `inference.*` imports, and verifies the package is standalone.

### Server tests that exercise workflows (retained)

```bash
python -m pytest tests/workflows/unit_tests
python -m pytest tests/workflows/integration_tests
```

### Generating the uv.lock (run remotely)

```bash
cd workflows && uv lock
```
