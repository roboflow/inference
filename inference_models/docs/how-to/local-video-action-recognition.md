# Action recognition on a local video

`ActionRecognitionVideoPipeline` accepts a loaded action-recognition model and a
local file. It samples and prepares windows using the model's recorded contract,
runs inference, and returns a timeline in source-video coordinates.
Only `inference-models` and its model dependencies are required.

This API is included in the next release. To try it from this checkout:

```bash
uv pip install -e ./inference_models
```

Use a compatible Torch installation for your device; see the
[installation guide](../getting-started/installation.md).
Set `ROBOFLOW_API_KEY` when loading a trained Roboflow model.

## V-JEPA: trained action labels and confidence

Replace `my-project-abc123/2` with your trained V-JEPA model ID and `video.mp4`
with a local file:

```python
import os
from pathlib import Path

from inference_models import ActionRecognitionVideoPipeline, AutoModel

model = AutoModel.from_pretrained(
    "my-project-abc123/2",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)
pipeline = ActionRecognitionVideoPipeline(model)
result = pipeline.run(
    Path("video.mp4"),
    confidence="best",
    include_candidates=True,
    on_progress=lambda progress: print(
        f"{progress.windows_processed}/{progress.windows_total} windows"
    ),
)
for prediction in result.timeline:
    print(prediction.to_dict())
print(result.source_fps, result.frame_count, result.windows_classified)
print(result.confidence_threshold, result.per_class_confidence_thresholds)
print(result.span_semantics)
```

Use `class_names=["walk", "run"]` to select labels from the trained vocabulary.
Class IDs remain indices in the original vocabulary. Omitted confidence or
`"default"` selects the package threshold; `"best"` uses model evaluation
recommendations; a number from 0 to 1 selects a global threshold.
Filtering happens before same-class spans merge. Merged confidence is the maximum
score. V-JEPA reports `span_semantics="class_union"`.

`result.candidates` is `None` unless scored candidates are requested. Candidates
are unfiltered, unmerged source-coordinate spans, including those below the
selected threshold. The final timeline remains filtered.

## Cosmos: fine-tuned or open-vocabulary action recognition

```python
from inference_models import ActionRecognitionVideoPipeline, AutoModel

model = AutoModel.from_pretrained("nvidia/cosmos-3-edge")
result = ActionRecognitionVideoPipeline(model).run("video.mp4")
for prediction in result.timeline:
    print(prediction.to_dict())
```

The pipeline wraps a loaded Cosmos reasoner without loading its weights again.
For a trained action-recognition package, replace the model ID with your Roboflow
model ID and pass `api_key`. Fine-tuned packages use their recorded sampling and
vocabulary; the base reasoner uses its existing whole-clip, open-vocabulary
behavior. `class_names` selects labels for a fine-tuned vocabulary and follows
the model's existing behavior for open-vocabulary inference.

Cosmos spans are unscored: confidence options are ignored, `candidates` and
effective thresholds are `None`, and predictions omit confidence in `to_dict()`.
It reports `span_semantics="instances"`. See [Cosmos requirements](../models/cosmos3.md)
for GPU and Transformers requirements.

## Interpret the result

Each prediction has `class_name`, `class_id`, `start_frame_idx`,
`end_frame_idx`, and optional `confidence`. Frame indices are zero-based and
inclusive in the original video. Convert them to seconds with
`start_frame_idx / result.source_fps` and
`(end_frame_idx + 1) / result.source_fps`. Open-vocabulary classes have ID `-1`.

`windows_processed` includes skipped short windows; `windows_classified` counts
completed model calls. `frame_count` starts with container metadata and is
corrected if decoding observes an earlier endpoint. Sampling uses nominal FPS;
this API does not use variable-frame-rate timestamps. Short windows, overlap,
padding and frame preparation follow the loaded model's contract.

## Consume windows incrementally

For long files, process each window without collecting the full timeline or all
candidates. The context manager closes the decoder even if the loop breaks or
the output sink raises an error:

```python
import json
from dataclasses import asdict
from pathlib import Path

with Path("windows.jsonl").open("w") as sink:
    with pipeline.stream("video.mp4", include_candidates=True) as windows:
        for window in windows:
            sink.write(json.dumps({
                "predictions": [p.to_dict() for p in window.predictions],
                "candidates": (
                    None if window.candidates is None
                    else [p.to_dict() for p in window.candidates]
                ),
                "progress": asdict(window.progress),
            }) + "\n")
```

Events contain filtered **unmerged** source-coordinate predictions and optional
raw candidates, with no frame pixels or accumulated history. Overlapping windows
can report overlapping spans. To reproduce the collector, call
`window.merge_into(timeline)` on a caller-owned list, then sort it by
`(prediction.start_frame_idx, prediction.class_id)`. Bound that list yourself.

## Cancellation, deadlines and limits

```python
from threading import Event

from inference_models import (
    ActionRecognitionVideoPipeline,
    VideoProcessingControl,
    VideoProcessingLimits,
)

cancelled = Event()  # Another thread may call cancelled.set().
control = VideoProcessingControl(timeout_seconds=600, is_cancelled=cancelled.is_set)
pipeline = ActionRecognitionVideoPipeline(
    model,
    limits=VideoProcessingLimits(max_candidates=50_000, max_result_bytes=16 * 1024**2),
)
result = pipeline.run("video.mp4", control=control)
```

The deadline begins when the control is constructed. Local execution has no
deadline by default. Checks run during probing, planning, decoding and between
model calls. An active model/device operation is allowed to finish before cleanup.
The caller owns the loaded model; closing a run does not unload it.

| Limit | Default | Scope |
|---|---:|---|
| `max_windows` | 10,000 | Planned windows per video |
| `max_sampled_frames` | 1,000,000 | Planned sample indices, including overlap |
| `max_retained_frame_bytes` | 1 GiB | Prepared decoder frame buffer |
| `max_candidates` | 250,000 | Collected candidates or one incremental window; also caps model spans |
| `max_timeline_entries` | 100,000 | Collected timeline or one incremental window's predictions |
| `max_result_bytes` | 64 MiB | Encoded result data or one incremental window's data |

All limits must be positive integers. Planning and decoder bounds apply in both
modes. Incremental mode permits total output larger than collection limits while
bounding each event. The pixel bound excludes decoder internals, model weights,
activations and frames retained by the caller or model. Encoded result limits are
conservative and do not measure total Python heap usage.

Limits raise `VideoProcessingLimitError` instead of truncating output. A failed
`run()` returns no partial result; a stream consumer may already have received
earlier events. Cancellation raises `VideoProcessingCancelledError`; an expired
deadline raises `VideoProcessingTimeoutError`. Unusable local input raises
`VideoInputError`. See [video errors](../errors/video-processing.md).

The HTTP adapter delegates to this same pipeline after handling URL/base64 input.
It retains server duration, deadline and response policies and translates package
results into the existing HTTP schema (including its `class` alias). Workflows
retains its live-stream sampler and state, while sharing coordinate conversion.
