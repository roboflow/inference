# Profiling Snippet Extraction

Use this skill when the user asks to create an isolated profiling target from a
function, snippet, module, or line range.

## Repository Context

This skill targets the Roboflow Inference repository. The committed profiling
tooling lives under `development/profiling/` and is developer-only; it is excluded
from package distribution. Generated profiling artifacts live under ignored local
paths:

```text
inference_profiling/snippets/<profile-name>/
  target.py
  config.yaml
  README.md
  runs/<run-id>/manifest.yaml
  runs/<run-id>/trace.nsys-rep
```

The reusable profiling modules expected by generated snippets are:

- `development.profiling.main`: CLI and `run_profile(...)`
- `development.profiling.config`: Pydantic config models and YAML parsing
- `development.profiling.registry`: `ProfileTarget` protocol and target import
  resolution
- `development.profiling.data.base`: `DataRecord`
- `development.profiling.nvtx`: `profiling_range(...)` and
  `profiling_range_if_cuda(...)`

Run generated profiles from the repository root after standard local setup:

```bash
uv venv --python 3.10
uv pip install -e .
uv run python development/profiling/main.py --config <generated-config.yaml>
```

## Workflow

1. Read the referenced production code and nearby imports, types, helpers, and
   call sites needed to understand inputs, outputs, side effects, and device
   behavior.
2. Create `inference_profiling/snippets/<profile-name>/target.py`.
3. Copy or adapt only the code needed for the experiment into `target.py`.
   Generated snippets are local experiments; do not edit production code unless
   the user separately asks for a production change.
   When copying a function body, preserve its direct control flow and delegate
   helper calls back to production modules unless the helper itself is the
   profiling subject. Import copied-body dependencies from their actual source
   modules, not from nearby or similarly named model metadata modules.
4. Add NVTX ranges with `development.profiling.nvtx.profiling_range_if_cuda` or
   `profiling_range` around meaningful operations such as preprocessing, model
   execution, decoding, filtering, NMS, or postprocessing. When the user
   references a function name or marks the lines of a function definition, copy
   or adapt that function's direct body into the generated target and instrument
   meaningful operations inside that body. Do not only wrap a call to the
   selected function unless the user explicitly asks for black-box timing.
5. Implement a `ProfileTarget`-compatible class with:
   - `name`
   - `prepare(record, *, device)`
   - `run(prepared)`
   - `validate(output)`
   - `summarize(output)`
6. Expose a `target` object or zero-argument factory from `target.py`.
7. Create `inference_profiling/snippets/<profile-name>/config.yaml` with:
   - profile name
   - target name
   - `target.import_path`
   - data source settings
   - device, warmup, iterations, record loading mode, and seed
   - capture range and CUDA synchronization settings when needed
8. Create a short README in the generated snippet directory that documents:
   - provenance of the copied code
   - assumptions and intentional differences from production
   - required local files, models, or environment variables
   - smoke command without Nsight
   - full `nsys profile` command

## Target Contract

Generated `target.py` files must expose a `target` object or zero-argument
factory implementing this shape:

```python
from typing import Any

import torch

from development.profiling.data.base import DataRecord


class MyProfileTarget:
    name = "my-profile"

    def prepare(self, record: DataRecord, *, device: torch.device) -> Any:
        ...

    def run(self, prepared: Any) -> Any:
        ...

    def validate(self, output: Any) -> None:
        ...

    def summarize(self, output: Any) -> dict[str, Any]:
        ...


target = MyProfileTarget()
```

The runner validates outputs during warmup only. Measured iterations should not
include validation work in the captured NVTX range.

## Config Shape

Generated configs should use explicit file-path target imports so generated
snippet directories do not need to be Python packages:

```yaml
profile_name: my-profile
target:
  name: my-profile
  import_path: inference_profiling/snippets/my_profile/target.py:target
  profile_prepare: false
data_source:
  name: dummy
device: cuda
warmup: 10
iterations: 100
record_loading: eager
seed: 0
capture_range: profile-target
cuda:
  synchronize_before_warmup: true
  synchronize_after_warmup: true
  synchronize_before_capture: true
  synchronize_after_capture: true
  synchronize_each_iteration: false
```

For local image directories, use the `images` data source:

```yaml
data_source:
  name: images
  directory: /path/to/images
  decode: true
  limit: 4
  repeat: 100
```

The image source decodes to RGB NumPy arrays when `decode: true`.
Use an absolute path in generated YAML when the user supplies a shell shortcut
such as `~/images`, because config parsing passes paths directly to `Path(...)`
and does not perform shell expansion.

## Command Template

The smoke command should run without Nsight first:

```bash
uv run python development/profiling/main.py \
  --config inference_profiling/snippets/my_profile/config.yaml \
  --run-id smoke
```

Then print or document the Nsight command:

```bash
uv run python development/profiling/main.py \
  --config inference_profiling/snippets/my_profile/config.yaml \
  --run-id smoke \
  --print-nsys-command
```

## Sanity Checks

When practical, compare the generated snippet output against the production
function on a tiny fixture. Preserve dtype and device behavior unless the
experiment is intentionally profiling an alternative implementation. Document any
known differences in the generated README.

Use `record_loading: eager` when selected records can fit in memory. Use
`record_loading: lazy` only with `target.profile_prepare: true`, because lazy mode
does not retain records for precomputed preparation. Add a `seed` when the target
or data source uses randomness.

Run a smoke profile after generation when practical. It catches config and import
mistakes that lints may miss, and confirms validation against real local data.
If converting decoded NumPy images to torch tensors in `prepare()`, make the array
writable and contiguous before `torch.from_numpy(...)` to avoid read-only array
warnings.

Some profiling snippets importing `torchvision.transforms` through production
preprocessing may fail in environments where fake `torchvision::nms` registration
runs before the operator schema exists. For local snippets, it is acceptable to
define a minimal `torchvision::nms` schema before importing the production module
when the profile does not call NMS; document this as an environment workaround in
the README.

## Boundaries

Generated snippets may import `development.profiling.*` helpers because they are
local development artifacts. Production code must not import these helpers.
