# V1 vs V2 engine comparison (`compare_ee.py`)

Runs one detector + boxes + labels workflow three ways on the same loaded
CPU ONNX `yolov8n-640` model:

| Mode          | Engine                     | Blocks                                              |
|---------------|----------------------------|-----------------------------------------------------|
| `v1`          | V1 `ExecutionEngine`       | stock tensor `object_detection@v1`, `bounding_box_visualization@v1`, `label_visualization@v1` |
| `v2_serial`   | V2 `session.run`           | `detection_blocks.py` via `workflows/live_detection.json` |
| `v2_pipeline` | V2 `session.pipeline`, x2  | same, compiled as phases                            |

This compares engine plus blocks, not schedulers alone.

## Run

From the repository root, with nothing else running:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:workflows:inference_models:stream_vision \
    python development/workflows-2.0/08-live-detection/compare_ee.py \
    --frames 100 --repeats 3 --output /tmp/compare_ee.json
```

Default input: `workflows/tests/assets/dogs.jpg` resized to 1920x1080.
Use `--image`, `--width`, `--height` for another still image.

## What it does

1. Loads the model once. V1 reaches it through `ForwardingModelsProvider`,
   which calls the real model (no model manager, no mock).
2. Checks parity outside timing: predictions and annotated pixels of both V2
   modes must equal V1 exactly, on the image and on a blank image. The source
   tensor must stay unchanged. Any mismatch stops the script.
3. Warms up every mode (5 runs x 2 frames = 10 frames), then measures
   `--repeats` rounds; the mode order rotates each round.
4. Per frame, the clock runs from creating the input wrapper to the result
   being ready. Pipeline mode keeps at most 2 frames in flight.

Output: per-run FPS, median and p95 latency, and a `setup` section with the
model backend, threads, block modules, timing boundary and known differences.
The console summary prints median FPS only; latency is in the per-run lines.

## Reading the numbers

- **FPS** = measured frames / wall time of the timed loop. Read it for
  throughput: how many frames per second the mode sustains.
- **Latency** = one frame's input wrapper creation to its result ready. Read
  it for per-frame delay.
- Serial modes: FPS ≈ 1000 / mean latency.
- Pipeline mode: 2 frames overlap, so FPS can be up to 2x 1000 / latency.
  Latency includes waiting behind the other in-flight frame. Higher FPS with
  equal or higher latency is expected, not an error.
- Each measured pipeline run opens and closes one pipeline. That startup and
  shutdown cost is inside the run's wall time, once per `--frames` frames.
