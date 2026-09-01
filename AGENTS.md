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

## Python implementation conventions

These conventions apply when editing Python throughout the repository.

### CLI options

- Use Click for new Python CLIs unless the existing script intentionally uses a
  different parser.
- Format `@click.option(...)` with every option argument and keyword on its own
  line, including nested constructors such as `click.IntRange`.

### Layout and calls

- Add an empty line after guard clauses and between validation, setup, computation,
  and result-construction groups. End every file with a final newline.
- When a function has one obvious primary argument, make secondary configuration
  arguments keyword-only. At call sites, pass that primary argument positionally and
  pass configuration with explicit names.
- When no primary positional argument is obvious, prefer explicit keyword calls for
  variables whose names match parameters. Preserve clear established third-party APIs.
- Prefer returning a meaningful named variable instead of directly returning a
  function call. Simple literals and already-bound values are fine.

### Public API docstrings

- Follow Google-style docstrings for public modules, classes, functions, and methods.
  Public means a name that does not start with `_` or a symbol intended for import by
  other packages/workflow blocks.
- Use a one-line summary, optional explanatory paragraph, then a blank line before
  `Args:`, `Returns:`, and caller-actionable `Raises:` sections.
- Document every parameter. Add `Returns:` whenever the function returns a value.
  Add `Raises:` only for exceptions callers should handle.
- Private helpers may use concise narrative docstrings or targeted comments.
- Do not use NumPy-style or Sphinx/reStructuredText parameter sections.

### Paths and Pydantic

- Prefer `pathlib.Path` and its methods to new `os.path` usage unless a string-only or
  compatibility boundary requires otherwise.
- Define Pydantic `BaseModel` fields with `Field`, including a useful `description` for
  every field and `examples` when a concrete value helps schema consumers. Match the
  detail level of the existing metadata models.

### Python environments and dependencies

- For new Python projects, use uv for environments, dependencies, locking, building,
  and publishing. In this existing repository, follow its established tooling unless
  the task explicitly migrates it.
- Prefer `uv sync`, `uv add`, `uv remove`, `uv lock`, and `uv run`; do not introduce a
  competing Poetry/Pipenv/bare-requirements workflow. Commit `uv.lock` when dependency
  changes update it.

## Roboflow PR description format

Apply this section only when the user asks for a PR body in Roboflow format.

Before drafting, inspect the branch log, diff, and status. Do not invent behavior or
test results. Ask for a missing Linear issue ID/title and the primary change type when
they cannot be established.

Use these exact sections:

```markdown
## What does this PR do?

Linked issue: [<issue-id>](https://linear.app/roboflow/issue/<issue-id>/<issue-slug>)

<Context and motivation, followed by a high-level explanation.>

**Main elements:**
- <component or area>

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Other:

## Testing

### Unit tests

- <coverage>

### Integration tests

- <scenario or None>

### Other

- <commands, benchmarks, or manual checks actually run>

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code where necessary, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings or errors
- [ ] I have updated the documentation accordingly (if applicable)
```

Check exactly one primary change type. Leave checklist boxes unchecked unless the user
confirmed them. Add `Additional Context` only for useful rollout notes, screenshots,
or migration details. When opening a PR, push only when requested and pass the body to
`gh pr create` without inventing a Linear link.
