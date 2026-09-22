# AGENTS.md

This guide governs the entire repository. If a subfolder provides its own
`AGENTS.md`, instructions there override this file for that subtree.

## Overview
Roboflow Inference is a set of Python packages that run computer vision models
locally and expose them via an HTTP API and command line interface. The repo
contains the core library, CLI, SDK, and Dockerfiles for building CPU or GPU
images. Supported Python versions are 3.10–3.12.

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
- Ensure all relevant tests pass before marking a pull request ready for review.
- Keep commit messages concise and in the present tense, e.g. "Add model loader".
- PR descriptions should explain what changed and why, list test commands run,
  and follow the templates in `.github`.
- Update documentation when applicable.

### Internal contributions (Roboflow team only)

These process requirements apply to internal contributors. External contributors
do not need access to Roboflow's Slack or Slab; follow the general guidelines above.

- Before substantial implementation of a major feature or structural change,
  prepare an implementation plan and share it in `#discuss-inference-release`
  for maintainer agreement on the approach. The
  [implementation plan guide](https://roboflow.slab.com/posts/inference-contributions-implementation-plan-1sb5nyqz)
  is the source of truth for the requirements and template. Agents can help
  investigate and draft it, but the contributor must understand and own the
  recommendation and unresolved questions. If the guide is inaccessible, ask
  the contributor for its contents rather than inventing requirements.
- A plan is generally unnecessary for documentation corrections or examples of
  existing functionality, added tests or regression coverage, contained fixes
  restoring established behavior, and local refactoring that preserves behavior
  and interfaces. New workflow blocks also qualify when they follow existing
  patterns and introduce no new execution behavior or execution-engine changes.
  Discuss changes to shared infrastructure, compatibility, security, or package
  dependencies with maintainers first, even for a small diff. If unsure, share a
  short description in the channel to establish whether a plan is needed.
- Keep unfinished work in a draft PR. Before marking it ready, inspect the diff,
  run relevant checks, record the commands and results (including limitations),
  and be available to address feedback. Plan exemptions do not waive testing or
  review. Contributors remain responsible for follow-up issues after merge.
- Use the optional [local pre-review skill](.claude/skills/review-local/SKILL.md)
  to check work before requesting CI review. In Claude Code, invoke
  `/review-local`; other agents can read and follow that file directly. Local
  findings are advisory and do not replace CI review or maintainer approval.
- Claude provides the first CI review. Address its findings and add the
  `claude-review` label to request another pass; the label is consumed when review
  starts, and new commits alone do not trigger another review. An eligible agent
  pass requests maintainer review through the Slack handoff bot. If disagreeing
  with a finding, explain why in a PR comment beginning
  `/maintainer-review <reason>` to escalate. The handoff requires an open,
  non-draft, same-repository PR; escalation also requires repository write access
  or higher. Coordinate in the linked Slack thread and record final approval in
  GitHub. See [handoff behavior](.github/maintainer-review-slack.md) for details.

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
