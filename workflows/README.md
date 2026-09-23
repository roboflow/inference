# roboflow-workflows

Roboflow Workflows execution engine and block library.

`roboflow-workflows` is the Python distribution for the Workflows engine that
powers [Roboflow Inference](https://github.com/roboflow/inference).  It can be
installed independently for use in standalone pipelines or embedded in the
full Inference server.

## Installation

```bash
pip install roboflow-workflows
```

Optional extras:

```bash
pip install "roboflow-workflows[enterprise]"  # enterprise sink blocks
pip install "roboflow-workflows[test]"        # test-only dependencies
```

## Local development (from this repository)

```bash
# Provision font assets (required by visualization blocks).
python workflows/build_scripts/download_fonts.py

# Install the package in editable mode with test + enterprise deps.
pip install -e './workflows[test,enterprise]'

# Run package unit and isolation tests.
cd workflows
python -m pytest tests/unit_tests tests/isolation

# Build a wheel from the same project directory (fonts already provisioned).
uv lock --check
uv build --out-dir ../dist
```

For editing `roboflow_workflows` source while running the full Inference
server, install both projects together from the repository root:

```bash
pip install -e ./inference_models -e './workflows[test,enterprise]' -e .
```

The dev images also support mounting `workflows/roboflow_workflows` at
`/app/roboflow_workflows`, alongside the existing `/app/inference` mount.
Provision local fonts first, since a bind mount replaces bundled image files.

## Font assets

Visualization blocks require pre-provisioned font files.  Font binaries are
downloaded by `workflows/build_scripts/download_fonts.py` (or
`make download_fonts` at the repository root) and are gitignored.  They are
embedded as package-data in the published wheel so that offline environments
work without a checkout.
