# Profiling Development Tooling Implementation Plan

## Goal

Create developer-only profiling tooling at `development/profiling` that makes
Nsight Systems profiling repeatable without copying large blocks of production
code by hand. This tooling should stay out of the package distribution.

The tooling should eventually support two related workflows, but this document
only designs the first:

1. Extract a focused function or code snippet into an isolated profiling module,
   add NVTX ranges around important operations, and run it against reusable data
   sources. This is the MVP workflow.
2. Production/full-path profiling. This needs a separate future design document
   after the snippet workflow is proven.

This document is a design proposal only. It intentionally avoids implementing the
tooling, data readers, scripts, or skills.

## Current State

Ad hoc profiling is usually assembled as a one-off directory containing:

- a small NVTX helper
- candidate functions copied or adapted from production code
- a script that builds inputs and invokes candidate functions inside NVTX capture
  ranges

The new tooling should turn that pattern into a reusable scaffold. A generated
README should carry the self-contained examples, commands, and assumptions for a
profile target so the profile can be moved between branches without depending on
the original scratch benchmark directory.

## Proposed Tooling Layout

The MVP should separate committed reusable tooling from ignored/generated local
artifacts:

```text
development/profiling/          # committed reusable tooling
  main.py
  config.py
  registry.py
  nvtx.py
  data/
    base.py
    dummy.py
    images.py

inference_profiling/            # ignored local/generated artifacts
  snippets/
    <generated_profile_name>/
      target.py
      config.yaml
      README.md
      runs/
        <run-id>/
          manifest.yaml
          trace.nsys-rep

tests/development/profiling/
  test_nvtx.py
  test_config.py
  test_registry.py
  test_smoke_execution.py
```

Later phases may add:

```text
development/profiling/
  skills/
    trace_analysis/
      SKILL.md
  runners.py
  data/
    coco.py
    roboflow.py
    video.py
    transforms.py
```

Packaging note: top-level `development/` is currently excluded by `setup.py`.
That is the desired location because these tools are for local development and
should not ship in the `inference-development` package distribution. Commands
should run from the repository root, for example
`uv run python development/profiling/main.py ...`.

## Implementation Phases

The implementation should be staged so the basic profiling workflow is proven
before this becomes a broad profiling framework.

### Phase 1: Minimal MVP

Phase 1 should implement only the isolated snippet profiling path. It should be
usable on a local workstation and from a local Docker container, but it should not
try to solve remote execution, CI automation, or broad production-path profiling.

MVP scope:

- generated/local-only snippet targets
- `nvtx.py` with lazy NVTX imports and no-op CPU behavior
- `config.py` with YAML/PyYAML config parsing
- `registry.py` for built-in target lookup and selected-target import resolution
- `data/base.py` with the common `DataRecord` and data source interface
- dummy tensor/image data source in `data/dummy.py` for smoke tests and fully
  deterministic traces
- local image directory data source in `data/images.py`
- `skills/snippet_extraction/SKILL.md` for snippet extraction only
- one committed built-in smoke target for tests
- one local generated example snippet under
  `inference_profiling/snippets/example_candidate/`
- `main.py` to run warmup and measured iterations
- command printer that emits the `nsys profile` command without executing it
- run manifest writer that writes into
  `inference_profiling/.../runs/<run-id>/`
- README instructions for local workstation execution and local Docker execution
- basic tests for no-op NVTX behavior, config parsing, registry lookup, and a
  smoke execution without `nsys`

Out of MVP scope:

- COCO data loading
- video URL/path data sources
- production/full-path instrumentation
- broad inference-pipeline profiling
- remote runner integration
- CI integration
- trace-analysis skill and Nsight aggregation helpers
- multi-target benchmark orchestration
- rich statistical reporting beyond manifest metadata

### Phase 2: Data And Analysis Expansion

Phase 2 can add richer data sources and the first trace-analysis docs:

- COCO-format image-only and image-plus-annotation loaders, if actually needed
- video pipeline mode and video frame mode
- `skills/trace_analysis/SKILL.md` for reading `nsys` outputs and aggregating NVTX
  ranges
- optional helper commands for exporting `nsys stats`
- aggregation intent and metric-selection config for trace analysis
- batching support through a `Batcher` object that consumes a `DataSource` and
  yields deterministic batches without requiring every data source or target to
  implement batching directly
- target-specific parameters through an explicit target factory/config contract,
  rather than silently parsing unused `target.parameters`
- richer workload controls such as seeded shuffle, explicit record ids, and
  fuller variance/trial reporting once real traces exist
- future validation workflows may compare target outputs against recorded outputs
  or a reference target. Phase 1 only supports lightweight `target.validate(output)`
  checks before capture.
- richer CUDA device inference for modules and nested structures, if real targets
  need it
- Roboflow exact-version dataset loader with cache support, if local and
  COCO-format data are not enough

### Future: Production And Full-Path Profiling

Production/full-path profiling should move into a separate future design
document. The snippet profiler should not carry this complexity.

That future document should address instrumentation control, threading, async,
multiprocessing, worker initialization, distributed coordination, and NVTX naming
per worker/process. It must also define an import boundary that does not require
production modules to import `development.profiling.*`.

## Nsight Command Contract

The tooling should document and generate commands equivalent to:

```bash
nsys profile \
  -o inference_profiling/snippets/<profile-name>/runs/<run-id>/trace \
  --trace=cuda,osrt,nvtx \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --capture-range=nvtx \
  --capture-range-end=stop \
  --nvtx-capture=profile-target@* \
  -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
  --trace-fork-before-exec=true \
  uv run python development/profiling/main.py ...
```

Phase 1 should use one top-level capture range per profiling script invocation and `--capture-range-end=stop`. 
Future workflows may use `--capture-range-end=repeat` if the runner intentionally emits repeated top-level 
capture ranges and the analysis workflow is designed around multiple captures.

The Python entrypoint should not shell out to `nsys` by default. It should be the
script that users pass to `uv run` inside an `nsys profile` command. A helper can
print the full `nsys` command for copy/paste, but it should not execute it in the
initial implementation.

Phase 1 should document two execution environments:

- local workstation execution from the repository root
- local Docker execution with the repository mounted and GPU access configured

Docker support means documenting and smoke-testing the local container workflow.
It does not include remote runners, CI jobs, or managed benchmark orchestration.

## Core Concepts

### Import Boundaries

Phase 1 generated snippets are local development artifacts run from the
repository root, so they may import committed profiling helpers directly:

```python
from development.profiling.data.base import DataRecord
from development.profiling.nvtx import profiling_range_if_cuda
```

That import boundary is acceptable only for generated/local snippet profiling.
Production/full-path profiling must not require production code to import from
`development.profiling.*`. The separate future production/full-path design should
choose a safe integration model for production code instead of reusing this MVP
snippet import pattern.

### Profile Target

A profile target should follow one concrete Python interface. This keeps
generated snippets, reference targets, tests, manifests, and future
production-path scripts compatible.

```python
from typing import Any, Protocol

import torch

from development.profiling.data.base import DataRecord


class ProfileTarget(Protocol):
    name: str

    def prepare(self, record: DataRecord, *, device: torch.device) -> Any:
        ...

    def run(self, prepared: Any) -> Any:
        ...

    def validate(self, output: Any) -> None:
        ...

    def summarize(self, output: Any) -> dict[str, Any]:
        ...
```

The main script owns the profiling loop:

1. call `target.prepare(record, device=device)` outside the measured NVTX range
   unless the target config explicitly requests profiling preparation
2. call `target.run(prepared)` inside the top-level capture range
3. call `target.validate(output)` when validation is enabled
4. call `target.summarize(output)` to add lightweight output metadata to the run
   manifest

By default, preparation happens once per selected record, outside the repeated
warmup and measured loops. This keeps setup, image decoding, host-side conversion,
and other fixture-building work out of the target range. If preparation is part
of the behavior being investigated, such as CPU preprocessing or expensive input
conversion, the config should explicitly opt into profiling preparation:

```yaml
target:
  profile_prepare: true
```

When `profile_prepare` is true, the runner should call `prepare` inside the
warmup and measured loops so the trace reflects the full prepare-plus-run path.

Targets should avoid CUDA synchronization unless it is semantically required.
The runner owns the default synchronization policy described below.

Built-in/reference targets should be statically registered. Generated snippet
targets should not be auto-discovered or imported at startup. They may depend on
optional packages, local files, model weights, GPU-only modules, or experimental
imports, so importing every generated target could make even `--help` fail.

Generated target configs should include an explicit import path. The main script
should parse config and CLI options first, then import only the selected target:

```yaml
target:
  name: my-profile
  import_path: inference_profiling/snippets/my_profile/target.py:target
```

Generated target imports should use file-path import syntax in Phase 1.
This avoids requiring `inference_profiling/` or its children to be Python packages
and keeps ignored scratch artifacts out of normal module discovery. 
Module-path imports can be added later if a promoted target needs them.

The registry should resolve targets in two ways:

- statically registered built-in/reference targets by name
- selected generated targets by explicit `import_path`

Both paths should return a `ProfileTarget` instance or a zero-argument factory
that builds a `ProfileTarget` instance. The registry should not accept bare
functions, so the runner can always rely on `prepare`, `run`, `validate`, and
`summarize`.

### Data Source

A data source yields `DataRecord` objects that can be adapted into target inputs.
Phase 1 should pin down a minimal record shape in `data/base.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DataRecord:
    id: str
    image: Any | None = None
    path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)
```

The fields mean:

- `id`: stable record id used in manifests and trace summaries
- `image`: optional decoded image/frame payload
- `path`: optional local path to the image or source asset
- `metadata`: optional task metadata, such as annotations, prompts, detections,
  keypoints, class names, dimensions, or source video frame number
- `source`: original source information for trace/debug reproducibility

The first implementation should favor image arrays and torch tensors. It should
not require every target to consume the same format; the target adapter is
responsible for converting records into the prepared object consumed by
`ProfileTarget.run(...)`.

The Phase 1 simple image source should emit records with `id`, `path`, decoded
`image` when requested by config, and `source` metadata describing the image
directory or file list used for the run.

Phase 1 data sources should not download data. Dummy data must be generated locally,
and the local image source should only read files already present on disk.
Future dataset loaders may use caches and may download missing datasets only when
config explicitly allows it.

Phase 1 data-source priority should be:

1. dummy tensor/image source for fast smoke tests and deterministic traces
2. local image directory source for real local assets without network access
3. COCO-format source later, only if dummy and local image sources are not enough

Most profiling paths should use image-only records. Image-plus-annotation records
should be available for paths that naturally need prompts or labels, such as SAM
flows that consume an image plus boxes, detections, masks, or keypoints. The data
interface should make those elements easy to map into script inputs without
forcing every loader to expose annotations.

### Capture Region

The top-level profiling script should wrap the meaningful repeated invocation in
a consistent NVTX range name. The default should be `profile-target`, but users
must be able to override it in YAML config and CLI options. The printed Nsight
command must use the selected capture range name in `--nvtx-capture=<name>@*`.

Nested NVTX ranges should be named for the operation being measured, for example
`preprocess inputs`, `run model`, `decode predictions`, or `postprocess outputs`.

### CUDA Synchronization Policy

CUDA synchronization should be a first-class profiling decision because it can
make or break trace interpretation. The design should keep these concepts
separate:

- NVTX range boundaries: labels used by Nsight to identify regions of interest.
- CUDA stream synchronization: host-side blocking that waits for queued device
  work to finish.
- Host-side wall-clock timing: optional Python timing around work submitted from
  the host.

The runner should expose explicit synchronization config:

```yaml
cuda:
  synchronize_before_warmup: true
  synchronize_after_warmup: true
  synchronize_before_capture: true
  synchronize_after_capture: true
  synchronize_each_iteration: false
```

Default policy:

- synchronize before and after warmup to avoid warmup work leaking into capture
- synchronize before capture so the captured range starts from a clean device
  queue
- synchronize after capture so host-side summaries and manifests do not race with
  outstanding device work
- do not synchronize each iteration by default

Synchronizing inside every nested NVTX range changes the workload by serializing
otherwise asynchronous GPU work. It should be avoided unless the user is
intentionally profiling isolated operations and accepts that the trace no longer
represents the natural end-to-end execution pattern.

## NVTX Utilities

The shared utility module should provide:

- a `profiling_range_if_cuda(name, device=None, enabled=True)` context manager
- a `profiling_range(name, enabled=True)` context manager for
  non-device-specific ranges
- optional decorators for wrapping functions
- explicit device handling, plus at most a tiny helper for direct torch tensors
- a no-op fallback when CUDA, torch CUDA support, or NVTX is unavailable

The helper should preserve the existing behavior of only using
`torch.cuda.nvtx.range` when CUDA profiling is meaningful. Phase 1 should not
attempt broad device inference through modules or nested structures because it can
add complexity and overhead. Targets should pass `device` explicitly or pass a
direct tensor to a small helper when that is enough.

The implementation should import `torch.cuda.nvtx` lazily. Lazy import keeps
CPU-only developer environments from paying import or availability costs unless
profiling actually enters a CUDA NVTX range.

Example README snippet:

```python
from contextlib import nullcontext
from typing import Optional

import torch


def profiling_range_if_cuda(
    name: str,
    *,
    device: Optional[torch.device] = None,
    enabled: bool = True,
):
    if not enabled:
        return nullcontext()

    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return nullcontext()

    import torch.cuda.nvtx as nvtx

    return nvtx.range(name)
```

## Data Package

### Shared Interface

Each data source should implement a common loader interface:

- `from_config(config)` or equivalent constructor
- `iter_records()`
- optional `collate(records)` for batch-oriented targets
- optional `describe()` for trace metadata and run summaries

The main script should support a narrow Phase 1 workload plan rather than a broad
benchmark framework. The MVP workload plan should describe:

- deterministic record selection
- `limit`
- `warmup`
- `iterations`
- `repetitions`
- `device`
- capture range name
- CUDA synchronization policy: the explicit `cuda.*` settings described in the
  core concept above

Phase 1 should define loop semantics explicitly:

1. Select records once according to deterministic config.
2. If `target.profile_prepare` is false, prepare selected records once before warmup.
3. Run `warmup` passes over the prepared or prepare-plus-run workload outside the top-level capture range.
4. Open one top-level capture range.
5. For each repetition, run `iterations` measured passes over the selected records.
6. Close the top-level capture range.

In pseudocode, the measured region should be:

```python
with profiling_range(capture_range_name):
    for repetition_index in range(repetitions):
        with profiling_range(f"repetition {repetition_index}"):
            for iteration_index in range(iterations):
                with profiling_range(f"iteration {iteration_index}"):
                    for record_or_prepared in selected_records:
                        output = run_one(record_or_prepared)
```

The main script should still be a profiling driver, not a full benchmarking
framework. Its job is to produce deterministic workloads and traces that are easy
to analyze. Aggregation intent, metric selection, seeded shuffle, complex input
conversion, and richer trial/variance semantics belong in the trace-analysis
skill or a later config schema.

Batching should remain out of Phase 1. The initial runner should process one
`DataRecord` at a time so target behavior and trace boundaries are easy to
understand. A Phase 2 batching design should introduce a `Batcher` abstraction
instead of placing batching directly on every data source. The `Batcher` should
accept a `DataSource`, apply `batch_size`, `drop_last`, and batching-specific
determinism controls, then yield batch records or batch containers that targets
can adapt in `prepare(...)`. That design should define how manifests report
batch membership, how lazy data sources interact with partial final batches, and
whether target summaries are per-record, per-batch, or both.

### Future COCO-Format Source

The COCO-format loader is not part of Phase 1. If dummy and local image directory
sources are not enough, a later COCO-format loader should support:

- image-only iteration over COCO validation images
- image-plus-object-detection annotations when the target needs boxes/classes
- image-plus-instance-segmentation annotations when the target needs masks
- image-plus-keypoint annotations when the target needs keypoint prompts
- classification-style image iteration when labels are not required by the target

The loader should use a cache directory. If the requested COCO assets are missing
and download is enabled, it may download them into the cache. If download is
disabled, it should fail with a clear message that includes the expected local
paths.

Expected config fields:

- `images_dir`
- `annotations_path`
- `task`
- `image_ids` or `limit`
- `class_filter`
- `include_annotations`
- `cache_dir`
- `download_if_missing`

### Future Roboflow Dataset Source

Roboflow is not part of Phase 1. Keep it out of v1 to avoid complexity around API
keys, dataset exact versions, network access, cache invalidation, auth errors,
and format handling.

A future Roboflow loader should fetch the exact dataset version requested by the
user. The agent must not guess a dataset or silently resolve to `latest`.
Multiple identifier forms are acceptable as long as they resolve to a concrete
version, for example:

- workspace/project/version identifiers
- concrete dataset version ids
- exported dataset format when needed by the profiling target
- split selection, such as `valid`, `test`, or `train`
- local cache directory
- forced refresh

Authentication should use existing Roboflow environment conventions where
possible, especially `ROBOFLOW_API_KEY`.

The loader may use the inference SDK or any other existing repo helper that
simplifies setup, provided the selected helper fetches the exact requested
version and does not add profiling-only behavior to production SDK surfaces.

### Future Video Sources

Video support is not part of Phase 1. A later video source should cover two
profiling modes:

1. Pipeline mode, where a video URL or path is passed through to an
   inference-pipeline-like execution path.
2. Frame mode, where a video URL or path is decoded into image records for
   targets that consume images rather than video streams.

Expected config fields:

- `video_path` or `video_url`
- `start_frame` or `start_seconds`
- `end_frame` or `end_seconds`
- `stride`
- `max_frames`
- `resize`
- `color_format`
- `mode`, either `pipeline` or `frames`
- `cache_dir`

The initial frame-mode implementation can use OpenCV because it is already used
elsewhere in development scripts. The record should preserve frame number and
timestamp metadata so traces can be mapped back to source frames. URL inputs
should be cached when downloaded, so repeated profiling runs do not depend on
network stability.

## Main Profiling Script

The main script should be runnable with:

```bash
uv run python development/profiling/main.py \
  --target <target-name> \
  --data-source <data-source-name> \
  --config <path-to-profile-config.yaml> \
  --device cuda \
  --warmup 10 \
  --iterations 100
```

The script should:

1. Load profile configuration.
2. Build the requested data source.
3. Resolve the requested profile target.
4. If `target.profile_prepare` is false, prepare each selected record once by
   calling `target.prepare(record, device=device)` before warmup and capture.
5. Run warmup iterations outside the top-level capture range.
6. Run profiled iterations inside one top-level NVTX capture range. For each
   repetition, run the configured number of measured iterations over the selected
   records. Each iteration calls `target.run(prepared)` when preparation is
   precomputed, or `target.prepare(...)` followed by `target.run(...)` when
   `profile_prepare` is true.
7. Apply the configured CUDA synchronization policy before/after warmup,
   before/after capture, and optionally per iteration.
8. Optionally call `target.validate(output)` after profiled iterations.
9. Call `target.summarize(output)` for lightweight manifest metadata.
10. Write `manifest.yaml` under
   `inference_profiling/snippets/<profile-name>/runs/<run-id>/` when requested.

The script should be intentionally boring: deterministic inputs, minimal logging,
and no hidden network access. Future data sources that support downloads must
require explicit config such as `download_if_missing: true`.

The run manifest should describe what was executed, not try to replace Nsight
analysis. It should include the profile name, target name, data source, resolved
data cache paths, selected record ids, warmup/iteration/repetition settings,
capture range name, CUDA synchronization policy, device, git commit if available,
command line, expected trace output path, and summaries returned by
`target.summarize(output)`.

## Future Trace Analysis Skill

`skills/trace_analysis/SKILL.md` should guide an agent through reading and
interpreting Nsight outputs produced by this tooling.

The skill should support:

- locating the `.nsys-rep` or exported SQLite/stat files for a run
- running or recommending `nsys stats` exports when needed
- aggregating durations by NVTX range name
- comparing warmup, trial, and repeated-run behavior
- reporting variance and obvious non-determinism
- identifying CPU gaps, CUDA synchronization points, and range nesting issues
- mapping trace results back to the run manifest and generated README

This is not part of the MVP. The skill should be separate from the main profiling
script because trace analysis can require extra tools, exports, and judgement
that should not run inside the profiled process.

## Snippet Extraction Skill

`skills/snippet_extraction/SKILL.md` should guide an agent through creating an
isolated profiling target from a user-provided code snippet, function name, or
module and line range.

The skill should require the agent to:

1. Read the referenced production code and its nearby imports/types.
2. Identify all required inputs, outputs, side effects, and dependencies.
3. Create `inference_profiling/snippets/<profile-name>/target.py`.
4. Duplicate the selected code into `target.py` and add NVTX ranges around
   meaningful operations, preserving semantics.
5. In the same `target.py`, create a `ProfileTarget` implementation whose
   `prepare` method maps records from `data/` into the object consumed by `run`.
6. Expose a `target` object or zero-argument target factory from `target.py`.
7. Create `inference_profiling/snippets/<profile-name>/config.yaml` with data
   source, device, warmup, iteration, target
   settings, and `target.import_path`.
8. Create a short README explaining provenance, assumptions, and the exact
   `nsys profile` command to run.

The skill should explicitly avoid editing production code. It is for isolated
experiments where copied code is acceptable and expected.

The skill should also require a sanity check against the production function
where practical:

- compare outputs for a small fixture
- document known intentional differences
- preserve dtype/device behavior unless the goal is to profile an alternative
  implementation

Example README snippet for `target.py`:

```python
from typing import Any

import torch

from development.profiling.data.base import DataRecord
from development.profiling.nvtx import profiling_range_if_cuda


def candidate_postprocess(scores: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    device = scores.device

    with profiling_range_if_cuda("filter scores", device=device):
        keep = scores > 0.5

    with profiling_range_if_cuda("select boxes", device=device):
        selected_boxes = boxes[keep]

    with profiling_range_if_cuda("sort by confidence", device=device):
        order = torch.argsort(scores[keep], descending=True)
        result = selected_boxes[order]

    return result


class ExampleCandidateTarget:
    name = "example-candidate-postprocess"

    def prepare(
        self,
        record: DataRecord,
        *,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        prepared = {
            "scores": torch.as_tensor(record.metadata["scores"], device=device),
            "boxes": torch.as_tensor(record.metadata["boxes"], device=device),
        }

        return prepared

    def run(self, prepared: dict[str, torch.Tensor]) -> torch.Tensor:
        result = candidate_postprocess(
            scores=prepared["scores"],
            boxes=prepared["boxes"],
        )

        return result

    def validate(self, output: torch.Tensor) -> None:
        if output.ndim != 2:
            raise ValueError("Expected candidate output to be a 2D tensor.")

    def summarize(self, output: torch.Tensor) -> dict[str, Any]:
        summary = {
            "num_boxes": int(output.shape[0]),
            "shape": list(output.shape),
        }

        return summary


target = ExampleCandidateTarget()
```

## Configuration

Use YAML profile config files for reproducibility and ergonomics. PyYAML should
be used for config parsing.

Config should include:

- profile name
- target name
- target import path for generated targets
- whether to include `target.prepare(...)` in warmup and measured loops
- data source type and data-source-specific parameters
- device
- deterministic record selection and optional limit
- warmup count
- iteration count
- repetitions
- output validation behavior
- capture range name, defaulting to `profile-target`
- CUDA synchronization policy
- optional Nsight output directory

## Generated Artifacts

Each generated profiling target should include:

- `target.py` with copied or path-based target code and its `ProfileTarget`
  adapter
- `config.yaml`
- provenance notes
- command notes
- expected environment variables
- any known limitations
- `runs/<run-id>/manifest.yaml` for each run
- `runs/<run-id>/trace.nsys-rep` when Nsight produces a trace

Generated profiling targets live under `inference_profiling/` by default. This
directory is already ignored by git in this repo, so generated snippets,
manifests, traces, downloaded datasets, and temporary exports stay local.

## Testing Strategy

Phase 1 should test only behavior required for the MVP workflow:

- NVTX helpers return no-op contexts on CPU-only environments.
- CUDA-gated helpers call NVTX only when CUDA is available.
- YAML configs parse and validate.
- dummy and local image data sources yield deterministic records for tiny
  fixtures.
- built-in/reference target lookup resolves without importing generated snippets.
- selected generated target import resolves only the configured `import_path` and
  returns an object that implements `ProfileTarget`.
- the committed built-in smoke target can run through `main.py` without `nsys`.
- the command printer emits the expected `nsys profile` command.
- the run manifest writer records the target, data source, workload settings,
  capture range, CUDA synchronization policy, and expected trace path.

Later phases can add tests for COCO-format, Roboflow, video loaders, and
trace-analysis helpers.

Nsight itself should remain a manual integration step. The docs should include a
smoke command that runs the profiling script without `nsys` to verify the Python
path first.

## Generated Example Profile

The first generated example profile should be a small, self-contained local
target under `inference_profiling/snippets/example_candidate/`. It should not
depend on an existing benchmark directory and should not be used as the committed
test fixture unless tests generate it at runtime.

Example generated profile:

- use the dummy or local image data source
- adapt each record into tensors with `scores` and `boxes`
- run a simple copied post-processing candidate
- wrap each meaningful operation in NVTX ranges
- include a README with the exact `uv run` smoke command and full `nsys profile`
  command

Example README command:

```bash
uv run python development/profiling/main.py \
  --target example-candidate-postprocess \
  --data-source images \
  --config inference_profiling/snippets/example_candidate/config.yaml \
  --device cuda \
  --warmup 10 \
  --iterations 100
```

Example README Nsight command:

```bash
nsys profile \
  -o inference_profiling/snippets/example_candidate/runs/<run-id>/trace \
  --trace=cuda,osrt,nvtx \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --capture-range=nvtx \
  --capture-range-end=stop \
  --nvtx-capture=profile-target@* \
  -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
  --trace-fork-before-exec=true \
  uv run python development/profiling/main.py \
    --target example-candidate-postprocess \
    --data-source images \
    --config inference_profiling/snippets/example_candidate/config.yaml \
    --device cuda \
    --warmup 10 \
    --iterations 100
```