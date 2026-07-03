# Development Profiling

Developer-only Nsight Systems profiling helpers live here. The committed tooling
drives focused, isolated profiling targets; generated targets and traces live
under ignored `inference_profiling/`.

## Setup

Use the standard local inference development install from the repository root:

```bash
uv venv --python 3.10
uv pip install -e .
```

After that, run profiling commands through the `uv` environment.

## Smoke Run

Run the built-in deterministic target without Nsight first:

```bash
uv run python development/profiling/main.py \
  --config development/profiling/smoke_config.yaml \
  --run-id smoke-local
```

Print the matching Nsight command:

```bash
uv run python development/profiling/main.py \
  --config development/profiling/smoke_config.yaml \
  --run-id smoke-local \
  --print-nsys-command
```

The printed command is intended for copy/paste. The Python entrypoint does not
execute `nsys` itself.

## Docker

From a local GPU-capable Docker environment, mount the repository and run the
same Python or printed `nsys profile` command from the repository root. The
container must include Nsight Systems, PyTorch, and GPU access configured by the
developer.

## Generated Targets

Generated snippets should expose a `target` object or zero-argument factory with
the `ProfileTarget` interface from `development.profiling.registry`. Configure
them with file-path import syntax:

```yaml
target:
  name: my-profile
  import_path: inference_profiling/snippets/my_profile/target.py:target
```

Generated snippets may import `development.profiling.*` helpers. Production code
must not import these development-only modules.
