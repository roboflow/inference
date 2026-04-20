# SAM2/SAM3 Video Tracking — handoff notes

> Temporary doc for local testing + follow-up.  **Delete before merging.**
> If you're Claude Code picking up this branch locally, read this file
> first — it captures the state of the work and the concrete steps to
> take it over the finish line.

## Branches

Two branches exist; you almost certainly want the second one.

| Branch | What's on it | Status |
| --- | --- | --- |
| `claude/sam-video-tracking-uXeSH` | First attempt — put everything in the legacy `inference/` package (new `SegmentAnything2Video` / `SegmentAnything3Video` under `inference/models/sam*/`, workflow blocks under `inference/core/workflows/.../foundation/`).  Kept as a reference; **do not ship this one**. | Stale |
| `claude/sam-video-tracking-inference-models` | Re-implementation in the newer `inference_models/` subpackage, with `sam2video` + `sam3video` model ids.  **This is the branch to ship.** | Active |

## What the active branch does

Everything below refers to `claude/sam-video-tracking-inference-models`.

### New model classes (shared HF base)

```
inference_models/inference_models/models/
├── common/
│   └── hf_streaming_video.py          # HFStreamingVideoBase — all the
│                                      # streaming plumbing (session init,
│                                      # prompt / track, state_dict contract,
│                                      # output unpacking).  SAM2Video and
│                                      # SAM3Video each ~25 lines on top.
├── sam2_video/
│   ├── __init__.py
│   └── sam2_video_hf.py               # class SAM2Video — wraps
│                                      # transformers.Sam2VideoModel /
│                                      # Sam2VideoProcessor.  No text prompts.
└── sam3_video/
    ├── __init__.py
    └── sam3_video_hf.py               # class SAM3Video — wraps
                                       # transformers.Sam3VideoModel /
                                       # Sam3VideoProcessor.  Accepts
                                       # both bbox and text prompts.
```

The existing `inference_models/models/sam2_rt/` (`SAM2ForStream`) is
**untouched**.  It was added before this branch and hasn't been used
much in practice; leave it alone.

### Registry entries

```python
# inference_models/inference_models/models/auto_loaders/models_registry.py
("segment-anything-2-rt", INSTANCE_SEGMENTATION_TASK, BackendType.TORCH)
    -> SAM2ForStream               # untouched, legacy
("sam2video",             INSTANCE_SEGMENTATION_TASK, BackendType.HF)
    -> SAM2Video                   # new
("sam3video",             INSTANCE_SEGMENTATION_TASK, BackendType.HF)
    -> SAM3Video                   # new
```

### Workflow blocks

Both are `LOCAL`-only (raise `NotImplementedError` at block `__init__`
if `WORKFLOWS_STEP_EXECUTION_MODE=remote`), load the model via
`inference_models.AutoModel.from_pretrained(model_id, api_key=...)`,
and multiplex sessions across videos by keying state_dicts on
`video_metadata.video_identifier`.

```
inference/core/workflows/core_steps/models/foundation/
├── _streaming_video_common.py              # shared helpers
├── segment_anything2_video/v1.py
│   SegmentAnything2VideoBlockV1
│   type: roboflow_core/segment_anything_2_video@v1
│   default model_id: "sam2video"
└── segment_anything3_video/v1.py
    SegmentAnything3VideoBlockV1
    type: roboflow_core/sam3_video@v1
    default model_id: "sam3video"
```

Both support three prompt modes:
- `first_frame` — prompt once per session, track every subsequent frame
- `every_n_frames` — re-prompt every N frames (`prompt_interval`)
- `every_frame` — re-prompt every frame

### Public API shape (used by the workflow blocks)

```python
masks, obj_ids, state_dict = model.prompt(
    image=np_frame,
    bboxes=[(x1, y1, x2, y2), ...],    # optional
    text="person",                      # optional, SAM3 only
    state_dict=None,                    # or prior state to reuse
    clear_old_prompts=True,             # False to add to existing session
    frame_idx=frame_number,
)

masks, obj_ids, state_dict = model.track(
    image=np_frame,
    state_dict=state_dict,              # REQUIRED — no None
)
```

- `masks`: `np.ndarray` bool `(N, H, W)`
- `obj_ids`: `np.ndarray` int64 `(N,)`
- `state_dict`: opaque dict holding an `inference_session` handle.
  **Not serializable across processes** — keep it in memory.

## Testing

### Unit tests (no weights, already passing)

```bash
# inference_models side (28 tests)
cd inference_models
python -m pytest tests/unit_tests/models/test_sam2_video.py \
                 tests/unit_tests/models/test_sam3_video.py -W ignore

# workflow blocks side (30 tests) — run from repo root
cd ..
python -m pytest \
  tests/workflows/unit_tests/core_steps/models/foundation/test_segment_anything2_video.py \
  tests/workflows/unit_tests/core_steps/models/foundation/test_segment_anything3_video.py \
  tests/workflows/unit_tests/core_steps/models/foundation/test_streaming_video_common.py \
  -W ignore
```

Both suites pass as of commit `8eb9af4`.  Note: inference_models tests
**must be run from the `inference_models/` directory** — they use a
separate `pytest.ini` and a path-sensitive `conftest.py`.

### Integration tests (need weights)

```bash
# inference_models integration tests (GPU recommended)
cd inference_models
python -m pytest \
  tests/integration_tests/models/test_sam2_video_predictions.py \
  tests/integration_tests/models/test_sam3_video_predictions.py \
  -m slow -W ignore
```

These call `AutoModel.from_pretrained` with `sam2video` / `sam3video`
ids, which will hit the Roboflow weights provider.  You need the
packages uploaded first — see **Weight packages** below.

### E2E-ish: driving the actual workflow block

Once the weights are up, the fastest sanity check is a single-frame
call to the block:

```python
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3_video.v1 import (
    SegmentAnything3VideoBlockV1,
)
# plus the VideoMetadata + WorkflowImageData dance from the unit tests.

block = SegmentAnything3VideoBlockV1(
    model_manager=None, api_key=<your_key>,
    step_execution_mode=StepExecutionMode.LOCAL,
)
result = block.run(
    images=[frame0], boxes=None, model_id="sam3video",
    class_names=["person"], prompt_mode="first_frame",
    prompt_interval=30, threshold=0.0,
)
```

For a real pipeline test, point `InferencePipeline` at a short MP4
and drive it through a workflow that has the block wired.

## Weight packages — need upload

Two zips need to live at `https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/`:

| Filename | Model | Contents |
| --- | --- | --- |
| `sam2video.zip` | `SAM2Video` | Standard HF transformers export of a SAM2 video checkpoint |
| `sam3video.zip` | `SAM3Video` | Standard HF transformers export of SAM3 video |

**Expected flat layout** (both zips — each file should be at the root
of the zip, no top-level directory):

```
config.json
preprocessor_config.json
model.safetensors                         # or sharded:
    model-00001-of-NNNNN.safetensors
    model.safetensors.index.json
# SAM3 additionally (if the published checkpoint ships them):
tokenizer.json
tokenizer_config.json
special_tokens_map.json
video_processor_config.json
```

### How to produce these zips from the HF Hub

The HF repos (at time of writing):
- SAM2 video: `facebook/sam2.1-hiera-tiny` (or `large`/`base+`/`small`
  — pick whichever you want to ship as the default)
- SAM3 video: `facebook/sam3`

Fetch with either:

```bash
# Option 1: huggingface-cli
huggingface-cli download facebook/sam3 --local-dir ./sam3-hf
cd sam3-hf && zip -r ../sam3video.zip .

# Option 2: Python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="facebook/sam3", local_dir="./sam3-hf")
# then zip -r sam3video.zip ./sam3-hf/*
```

**Important**: the files must be at the root of the zip, not inside a
subdirectory.  `download_model_package` unzips into a directory and
calls `Sam3VideoModel.from_pretrained(that_directory)` — if the files
are nested, `from_pretrained` won't find them.

Quick way to verify the zip is laid out right:

```bash
unzip -l sam3video.zip | head -20
# First-column lines should be bare filenames like
#   config.json,  preprocessor_config.json,  model.safetensors
# not  sam3-hf/config.json  etc.
```

### Registering weights in the Roboflow model registry

You mentioned you have tooling for this in another repo.  The model
ids we need registered are:
- `sam2video` → task `instance-segmentation`, backend `hugging-face`
- `sam3video` → task `instance-segmentation`, backend `hugging-face`

The bucket URL pattern the test fixtures use is
`rf-platform-models/<model_id>.zip`; production weight resolution
probably follows a similar convention — double-check with the weights
provider tooling.

## Known gotchas

These caught me during implementation; worth knowing:

- **`clear_old_prompts` vs `clear_old_points`.**  The legacy
  `SAM2ForStream.prompt()` uses `clear_old_points` (SAM2's
  point-prompt vocabulary).  The new `HFStreamingVideoBase.prompt()`
  uses `clear_old_prompts` (which is more accurate now that text +
  boxes are also supported).  If you're bridging old and new code,
  watch for this.

- **`track(state_dict=None)` must raise.**  An earlier version of the
  helper silently created a fresh empty session for tracking —
  producing empty masks forever.  Fixed to raise `ModelRuntimeError`
  explicitly.  Don't regress this.

- **`dict.get(a) or dict.get(b)` on numpy arrays is a trap.**
  Truthiness of numpy arrays is ambiguous.  The
  `_unpack_processed_outputs` helper uses an explicit `_first_present`
  function for this reason — if you extend it, keep the pattern.

- **`inference_models` tests need the right cwd.**  `pytest.ini` sits
  in `inference_models/` and a relative path conftest expects to be
  invoked from there.  Running `python -m pytest inference_models/...`
  from repo root will fail collection for the unit tests.

- **Transformers SAM3 loading needs torchvision at import time.**  The
  transformers integration for SAM3 video does `import torchvision` at
  module load; if torchvision isn't installed, `from transformers
  import Sam3VideoModel` fails with a cryptic
  `ModuleNotFoundError: Could not import module 'Sam3VideoModel'`.
  Not a real issue in prod (it's a baseline dep of `inference_models`)
  but will confuse local env bootstrap.

- **Remote step execution intentionally unsupported.**  Both blocks
  raise `NotImplementedError` in `__init__` if
  `WORKFLOWS_STEP_EXECUTION_MODE=remote` — per-video session state
  cannot survive a remote boundary.  This is intentional; the check
  exists to fail at workflow-compile time rather than on first frame.

## Open questions / TODOs for this PR

- [ ] Upload `sam2video.zip` + `sam3video.zip` to
      `roboflow-tests-assets/rf-platform-models/`.
- [ ] Register `sam2video` / `sam3video` model ids in the Roboflow
      weights provider (via the tooling in the other repo).
- [ ] Decide default SAM2 checkpoint size.  The HF family has
      `tiny` / `small` / `base+` / `large`.  Tiny is fastest and what
      the legacy block defaults to; pick whichever for `sam2video`'s
      single shipped variant.  If we want multiple variants, register
      multiple model ids (e.g. `sam2video-tiny`, `sam2video-large`).
- [ ] Run integration tests on a GPU runner; lock in the shape
      assertions if they're too loose (currently they only assert
      shape, not numerical content).
- [ ] Smoke test with a real `InferencePipeline` + short MP4 (can't
      do this from the session — needs the weights uploaded first).
- [ ] Decide whether to ship both workflow blocks at once or start
      with just one (SAM2's HF streaming mode may be less mature than
      SAM3's; integration test results will tell).

## What the follow-up skill should do

Separate planned work (different session): write a Claude skill at
`.claude/skills/add-inference-model/` that automates the pattern we
just worked through.  Rough shape:

1. Interactive questions:
   - Model family name (e.g. `sam3video`)
   - Task type (pick from `TaskType` enum)
   - Backend (TORCH / ONNX / HF / TRT)
   - Upstream source for weights (HF repo id? local dir? user uploads
     later?)
   - Should a workflow block be generated too? (yes / no)
2. Scaffolds:
   - `inference_models/models/<family>/<family>_<backend>.py` with a
     template model class matching the chosen backend
   - Unit test file with mocks for the chosen backend
   - Integration test + conftest fixture
   - Registry entry in `models_registry.py`
   - Optional: workflow block under
     `inference/core/workflows/.../foundation/<family>/v1.py` + test
     + loader registration
3. v1 punts on weight upload — asks the user for a path to a local
   HF-cache directory, zips it into the right layout, and prints the
   `gsutil cp` / registration command for the user to run.
4. v2 (later): actually fetch HF weights with `snapshot_download`, zip
   them, upload to the bucket, call the registry tooling.

Existing docs that cover the pattern:
- `inference_models/docs/contributors/adding-model.md`
- `inference_models/docs/contributors/architecture.md`
- `inference_models/docs/contributors/writing-tests.md`

Good reference PR the user pointed at: <https://github.com/roboflow/inference/pull/2227>
(adds a HF-backed model the manual way).
