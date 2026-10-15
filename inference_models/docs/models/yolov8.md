# YOLOv8 TensorRT - Selectable Postprocessing

`YOLOv8ForObjectDetectionTRT` can decode raw model output with one of two
postprocess implementations. The default is unchanged.

| ID | Behavior |
|----|----------|
| `base` (default) | The original decoder: per-image `torchvision.ops.batched_nms`, rescale, round, `int32`. |
| `auto` | Currently identical to `base`. |
| `segmented` | Opt-in. The same per-image computation for a whole batch, with one Triton NMS kernel pass. |

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "yolov8n-640",
    backend="trt",
    postprocess_implementation="segmented",
)
```

`postprocess_implementation` is accepted by
`YOLOv8ForObjectDetectionTRT.from_pretrained` and its constructor. Other
backends do not define this option. An unknown ID raises `ModelRuntimeError`
before the engine loads.

## When `segmented` runs

Each call selects in three steps. Any failed step runs `base` instead and
records the reason.

| Step | Requirement |
|------|-------------|
| Static | CUDA device; `triton` installed; `torchvision` exactly `0.28.0` (a `+local` suffix is ignored); `torch.nonzero_static` present; the model is exactly `YOLOv8ForObjectDetectionTRT`, not a subclass such as YOLOv9 or YOLOv11. |
| Request | Non-fused `(B, 4 + C, N)` `float32` output on the context device, `B >= 2`, one metadata entry per image; torch default dtype `float32`; `max_detections` a non-negative integer; `iou_threshold` finite in `[0, 1]`; scalar or `(C,)` confidence threshold; metadata offsets, scales and sizes are host numbers. |
| Runtime | One-time kernel calibration on the device succeeded. |

The checks read shapes, dtypes, devices and host values only. They never read
GPU tensor values.

Triton is imported only by the runtime step. A missing Triton package falls back
in the static step. A broken Triton installation, a kernel compilation error or
a CUDA error raises; the call fails and nothing falls back.

## Inspecting the selection

```python
detections = model(images)

model.postprocess_selection
# {"requested_id": "segmented", "effective_id": "segmented",
#  "fallback_reason": None, "routes": {...}, "scratch_bytes": ...,
#  "scratch_limit_bytes": ..., "kernels": {...}, "calibration": {...}}

model.postprocess_ready_event  # completion event of the last call
```

Both properties are per thread and describe the last successful call. They are
`None` before the first call and after a failed call. A fallback is also logged
once per distinct reason.

`routes` counts images per NMS route in the call:

- `images_kernel`: the Triton kernel pass.
- `library_vanilla_branch_images`: images large enough for torchvision's own
  non-coordinate-trick branch; they run `torchvision.ops.batched_nms`.
- `library_scratch_limit_images`: images of a call whose IoU bitmask would exceed
  64 MiB; they run `torchvision.ops.batched_nms`.
- `images_empty`: images without candidates.

## Numerical behavior

`segmented` is designed to return the same values, dtypes, shapes and order as
`base`. It repeats the base thresholding, the per-image class-offset boxes of
torchvision 0.28.0, its stable descending sort and its float32 IoU test.

The CUDA build of torchvision does not fix whether the compiler fused IoU
multiply-adds. The first supported call on a device therefore compares six
arithmetic variants with the installed `torchvision.ops.nms`:

- thresholds 0.3, 0.45, 0.5 and 0.7;
- two box scales: sides 1-64 px near the origin, and sides 10-600 px at
  coordinates 1e4-1e5 (class-offset magnitudes);
- 32768 pairs per round with IoU a few float32 ulp from the threshold; the
  oracle checks up to 256 disagreeing and 32 agreeing pairs per round.

Exactly one variant must match every checked pair. Otherwise the device records
the calibration as unqualified and every call falls back to `base`. A match is
empirical evidence on these samples, not a proof of equality on every input.
The calibration takes seconds and runs once per device for each model instance
(or each separately constructed `SegmentedPostprocessor`). Instances do not share
calibration state.

## Streams and completion

Both implementations run on `torch.cuda.current_stream(device)` of the calling
thread. The model's `post_process` waits once, on one completion event, and
returns completed detections. There is no separate postprocess stream.

`segmented` output tensors are views of buffers allocated for that call. Later
calls never overwrite them, and the model output is not modified.

`segmented` still waits for the host inside a call: for candidate counts, for
the rescale constants upload and for each image on a library route.
