# SAM3 Video - Text-Prompted Concept Tracking

SAM3 Video is the streaming video head of Meta AI's Segment Anything Model 3. Unlike visually
prompted trackers (SAM2 video), it is prompted with **text**: name the concepts you care about
once, and every frame runs a fused detect-and-track step that segments all matching objects and
keeps their identities stable across frames.

## Overview

SAM3 Video provides open-vocabulary video object tracking:

- **Text/concept prompts** - Track every "person", "forklift", "shopping cart" without an
  upstream detector; each prompt is an independent concept
- **Fused detect-and-track** - Objects entering the scene mid-stream are picked up
  automatically; no re-prompting needed
- **Stable object ids** - Each tracked object keeps its id for the lifetime of the session
- **Per-object detection scores** - Real confidences, usable for thresholding
- **Per-concept labeling** - Each frame reports which objects belong to which prompt
- **Streaming-first** - Consumes frames one at a time from memory (webcam, RTSP, frame loops);
  no video file required upfront

!!! info "Looking for box-prompted video tracking?"
    To seed tracking from detector boxes instead of text, use the
    [SAM2 video tracker](sam2-interactive-segmentation.md) (`sam2video` model family).

## License

**SAM License** (Meta Platforms) - see the `LICENSE` file shipped with the model package.

## Pre-trained Model IDs

SAM3 Video requires a **Roboflow API key**.

| Model | Model ID |
|-------|----------|
| SAM3 Video (single checkpoint) | `sam3video` |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `hugging-face` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

!!! warning "GPU Recommended"
    The checkpoint is ~860M parameters. CPU inference works but is significantly slower;
    a CUDA GPU is recommended for real-time streams.

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ No custom training |
| **Upload Weights** | ❌ Not applicable |
| **Serverless API (v2)** | ❌ Not available (stateful sessions cannot cross request boundaries) |
| **Workflows** | ✅ `roboflow_core/sam3_video@v1` block (local execution with `InferencePipeline`) |
| **Edge Deployment (Jetson)** | ⚠️ Experimental |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Example

Call `prompt()` once with your concepts, then `track()` on every subsequent frame, threading
the returned `state_dict` through. Detection runs continuously - new matching objects appear
with fresh ids without re-prompting.

```python
import cv2
import supervision as sv
from inference_models import AutoModel

model = AutoModel.from_pretrained("sam3video", api_key="your_api_key")

mask_annotator = sv.MaskAnnotator(opacity=0.7, color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
video = cv2.VideoCapture("video.mp4")

state = None
while True:
    is_ok, frame = video.read()
    if not is_ok:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if state is None:
        # Register concept prompts and run the first streaming step
        result = model.prompt(image=rgb_frame, text=["person", "dog"])
    else:
        # Subsequent frames: fused detect-and-track against the same session
        result = model.track(image=rgb_frame, state_dict=state)
    state = result.state_dict

    # Label each object with the concept that claimed it
    object_id_to_label = {
        obj_id: prompt
        for prompt, obj_ids in result.prompt_to_object_ids.items()
        for obj_id in obj_ids
    }
    detections = sv.Detections(
        xyxy=result.boxes,
        mask=result.masks,
        confidence=result.scores,
        tracker_id=result.object_ids,
    )
    labels = [
        f"#{obj_id} {object_id_to_label.get(obj_id, '')} {score:.2f}"
        for obj_id, score in zip(result.object_ids, result.scores)
    ]

    annotated = mask_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)
    cv2.imshow("SAM3 Video", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
```

### Per-frame result

Every streaming step returns a `SAM3VideoFrameResult`:

| Field | Type | Description |
|-------|------|-------------|
| `masks` | `(N, H, W)` bool | Binary masks at the input frame's resolution |
| `object_ids` | `(N,)` int64 | Object ids, stable across the session's frames |
| `scores` | `(N,)` float32 | Per-object detection scores |
| `boxes` | `(N, 4)` float32 | Boxes in `xyxy`, derived from the masks |
| `prompt_to_object_ids` | `dict[str, list[int]]` | Maps each text prompt to the object ids it currently claims |
| `state_dict` | `dict` | Opaque session handle - pass into the next `track()` call |

### Session semantics

- **Text-only by design.** `prompt()` accepts text concepts exclusively - there is no box or
  point parameter. Box-prompted video tracking is a separate model family
  (`Sam3TrackerVideoModel`), not wrapped by `sam3video`.
- **One session per stream.** `prompt(..., clear_old_prompts=True)` (the default) starts a
  fresh session; pass `clear_old_prompts=False` with an existing `state_dict` to add concepts
  to an ongoing session.
- **`state_dict` is not serializable.** It holds a live inference session with device tensor
  references - keep it in process memory and never pickle it or send it across processes.
- **Long streams.** The session accumulates per-object state over time; for very long-running
  streams, periodically re-seeding with a fresh `prompt()` call bounds memory at the cost of
  resetting object ids.

## Workflows

The [`roboflow_core/sam3_video@v1`](https://inference.roboflow.com/workflows/blocks/sam_3_video/)
block wraps this model for video workflows: text prompts via `class_names`, one tracking
session per `video_identifier`, per-frame class labels from `prompt_to_object_ids`, and
detection scores exposed as `confidence`. It requires local step execution (drive it with
`InferencePipeline`) - see the [SAM3 docs](https://inference.roboflow.com/foundation/sam3/)
for a full workflow example.
