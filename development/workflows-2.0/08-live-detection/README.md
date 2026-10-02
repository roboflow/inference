# Live detection: camera → detector → boxes → labels → window

From the inference checkout, in the existing `roboflow-inference-new` environment:

```sh
conda activate roboflow-inference-new
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:workflows:inference_models:stream_vision \
  python development/workflows-2.0/08-live-detection/run_demo.py
```

This opens camera 0 and shows the annotated stream with speed statistics.
Press `q` or `Esc`, close the window, or press Ctrl-C to stop. On the first run
macOS asks for camera access for your terminal. If the camera does not open,
allow it in System Settings > Privacy & Security > Camera and run again.

The model is `yolov8n-640` from the Roboflow platform, loaded once with
`AutoModel.from_pretrained(..., backend="onnx", device="cpu",
onnx_execution_providers=["CPUExecutionProvider"])`. The first run downloads
it. Images stay CPU tensors (CHW, RGB, uint8). The platform's TorchScript
packages are rejected as untrusted, so there is no `--backend torch-script`.
MPS is not offered: no MPS path of this experiment has been run.

Use the ONNX Runtime range declared by `inference_models`' CPU ONNX extra:
`onnxruntime>=1.15.1,<1.23.0`. This demo uses **1.22.1**. If it is missing:

```sh
python -m pip install 'onnxruntime>=1.15.1,<1.23.0'
```

## Other ways to run it

Append to the command above:

| Flags | What happens |
| --- | --- |
| `--mode serial` | One `session.run` at a time instead of the pipeline |
| `--max-in-flight 3` | Pipeline with three frames in flight (default 2) |
| `--camera 1` | Another camera |
| `--confidence 0.6` | Detection threshold (default 0.4) |
| `--video clip.mp4` | A video file; every frame goes through the workflow (a window may skip showing a result, counted as `result` drop) |
| `--headless --duration 20` | No window; prints statistics after 20 s |
| `--frames 300` | Stop after 300 results |
| `--image photo.jpg --output-dir /tmp/live` | One image, no window; prints the detections, writes `annotated.png` and `stats.json` |

With `--output-dir`, every mode writes `stats.json`. Camera and video frames
are never written.

## What the overlay shows

```
pipeline x2 | YOLOv8ForObjectDetectionOnnx | CPUExecutionProvider | model cpu | images cpu
output 24.8 FPS | processed 24.9 FPS | camera 30.0 FPS
capture->present 71.3 ms (p95 88.0)
median ms: pre 6.1 | model 21.4 | post 1.2 | boxes 0.9 | labels 1.1
1920x1080 | detections 3 | dropped: camera 112 result 0
```

(Illustrative layout, not a measurement.)

| Value | Meaning |
| --- | --- |
| Header | Mode, and the backend the loaded model actually uses (model class, ONNX Runtime providers, device), read from the model |
| `output FPS` | Frames shown per second, over the last 2 s. Each result counts once |
| `processed FPS` | Workflow runs finished per second |
| `camera FPS` | Frames the source delivered per second |
| `capture->present` | From right after `read()` returned to after the `waitKey` that paints the frame. Sensor and driver latency are not included. Median and p95 of the last 120 frames |
| `pre`, `model`, `post` | The detector's `pre_process`, `forward` and `post_process`, wall time observed inside each call |
| `boxes`, `labels` | The two painters, wall time observed inside each call |
| `dropped: camera` | A newer camera frame replaced one the graph had not taken yet (busy) |
| `dropped: result` | A newer result replaced one the window never showed |

FPS is never computed from model time. Headless runs report `output` as
`null` because nothing is shown.

Reading the numbers:

- Stage times are observed wall times. In pipeline mode stages overlap, and
  ONNX Runtime's and torch's CPU thread pools can compete for the same cores.
  So stage times can include contention, and they do not add up to the latency.
- Pipeline mode is not assumed faster than serial. A larger `--max-in-flight`
  can add age (`capture->present`) without more FPS. Compare
  `--mode serial`, `--max-in-flight 1`, `2` and `3` on your machine.
- The pipeline takes a frame only while fewer than `--max-in-flight` runs are
  pending; waiting frames are replaced (`dropped: camera`). If the pipeline
  accepts no frame for 1 s, the run stops with a "Pipeline stalled" error
  instead of dropping it.
- Warmup is three runs on a blank 640x480 image before the camera opens; they
  are not counted. They find no objects, so the first real frames still pay
  cold costs (camera-size resize, first boxes and label sprites). Short runs
  and the single `--image` sample include them.
- `stats.json` records the configured ONNX Runtime `intra_op_num_threads` and
  `inter_op_num_threads` (0 means ONNX Runtime chooses, not zero threads) and
  `torch_threads`.

## How it runs

```
capture thread   read() → frames slot (1 frame; camera: newest wins, files: wait)
graph thread     BGR→RGB tensor (one copy) → session.run   (serial)
                                           → pipeline.submit (< max-in-flight pending)
                 → results slot (1 result, newest wins)
main thread      RGB→BGR (one copy) → overlay → imshow / waitKey
```

The main thread never runs the workflow, in either mode. On exit the window
closes first. A running inference then finishes (it cannot be interrupted),
and the capture thread releases the camera.

The workflow is [`workflows/live_detection.json`](workflows/live_detection.json).
Serial mode compiles each block's `run`; pipeline mode compiles the detector's
phases. Both use the same blocks from `detection_blocks.py`.

## Files

| File | Content |
| --- | --- |
| `run_demo.py` | Command line, model loading, compilation, warmup |
| `live.py` | Capture, graph and display threads; slots; shutdown |
| `stats.py` | Rolling, bounded statistics and the overlay text |
| `detection_blocks.py` | Detector, box and label blocks |
| `workflows/live_detection.json` | The workflow |
| `tests/` | Runner and block tests with fake models (no download, no camera) |

Tests:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:workflows:inference_models:stream_vision \
  python -m pytest development/workflows-2.0/08-live-detection/tests
```
