# V-JEPA 2.1 - Action Recognition

V-JEPA 2.1 identifies trained action labels and their frame ranges in a video.
An action span is a range of frames assigned an action label.
The model can report several actions at the same time.

Send a video to the Inference HTTP API to receive a timeline of action spans.
The server samples frames, processes overlapping windows, and merges predictions for the same action.
A window is a group of sampled frames processed together.
You do not need to split the video yourself.

## License

The adapted Meta architecture uses the [MIT License](https://github.com/roboflow/inference/blob/27ec46f2cb79601cc54f9fc87f527a1e2ae49cc3/inference_models/inference_models/models/vjepa2_1/LICENSE).
See [upstream attribution](https://github.com/roboflow/inference/blob/27ec46f2cb79601cc54f9fc87f527a1e2ae49cc3/inference_models/inference_models/models/vjepa2_1/UPSTREAM.md) for the source files and adaptation details.

## Model IDs

Use the ID of your trained Roboflow model, such as `my-project-abc123/2`.
Supply a Roboflow API key with access to that model.
The supported model type is `vjepa2-1-vitb-384`.
That type identifies the architecture, rather than a public pretrained model ID.
The loader requires an exported action-recognition package with trained weights and its class list.
It does not download an upstream checkpoint when weights are missing.

V-JEPA reports the classes recorded in your model package.
Use `class_filter` to report a subset of those classes.
The filter cannot introduce new action labels.

## Installation and supported integrations

Use an Inference server and `inference-models` version that include this V-JEPA integration.
For development, install both packages from the same checkout.
Do not assume that an older published package includes the model.

For a CUDA 12.8 environment, activate your Python environment.
From the repository root, install the model package:

```bash
uv pip install -e "./inference_models[torch-cu128]"
```

Choose the Torch extra that matches your environment.
Available extras include `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-cu130`, and `torch-jp6-cu126`.
These extras select dependencies. They do not establish performance or deployment readiness on each device.
GPU numerical agreement and performance are not yet validated for release.

| Integration | Supported interface |
|-------------|---------------------|
| Backend | Torch; no ONNX or TensorRT path |
| HTTP API | Video clips through `/infer/action_recognition` or the legacy model route |
| Workflows | Streaming frames through the Action Recognition Model block, `roboflow_core/roboflow_action_recognition_model@v1` |
| Direct model calls | Prepared RGB frames from one sampled window |

Hosted API availability depends on the deployed server version.

## Usage example: classify a video

This example assumes that an Inference server with V-JEPA support runs at `http://localhost:9001`.
Before starting that server, set `USE_INFERENCE_MODELS=True` and `ACTION_RECOGNITION_ENABLED=True` in its environment.
Install the example's HTTP client in your active Python environment:

```bash
uv pip install requests
```

Set `ROBOFLOW_API_KEY` in your environment.
Replace the model ID and video URL with your own values.
The video URL must be accessible from the server.

```python
import os

import requests

response = requests.post(
    "http://localhost:9001/infer/action_recognition",
    json={
        "api_key": os.environ["ROBOFLOW_API_KEY"],
        "model_id": "my-project-abc123/2",
        "video": {"type": "url", "value": "https://example.com/video.mp4"},
        "confidence": "default",
    },
    timeout=600,
)
response.raise_for_status()
result = response.json()

for span in result["timeline"]:
    print(
        span["class"],
        span["start_frame_idx"],
        span["end_frame_idx"],
        span["confidence"],
    )
```

Prefer a video URL for full clips.
The API also accepts `{"type": "base64", "value": "..."}` for short clips.
Base64 increases the payload size and holds the request in memory, so large clips can exceed gateway limits.

## Prediction format

The HTTP response describes the submitted video and its action timeline:

| Field | Meaning |
|-------|---------|
| `timeline` | Filtered, merged action spans |
| `source_fps` | The video's nominal frames per second (FPS) |
| `frame_count` | Number of source frames in the video |
| `windows_classified` | Number of windows processed by the model |
| `span_semantics` | `"class_union"` for V-JEPA: same-class spans combine rather than identify separate action instances |
| `confidence_threshold` | Effective uniform threshold, or `null` when thresholds differ by class |
| `per_class_confidence_thresholds` | Effective thresholds by class when model-evaluation recommendations apply |
| `candidates` | Unmerged scored predictions when `include_candidates` is `true` |

Each timeline entry contains `class`, `class_id`, `start_frame_idx`, `end_frame_idx`, and `confidence`.
The `class_id` is the label's position in the model's class list.
Frame indices start at zero and refer to the original video.
Both boundaries are inclusive: a span from frame 30 to frame 59 contains 30 frames.
At 30 FPS, that span covers the interval from 1.0 seconds to 2.0 seconds.
Convert its start with `start_frame_idx / source_fps` and its exclusive end with `(end_frame_idx + 1) / source_fps`.

Continuous model boundaries round outward to source frames.
The server merges touching or overlapping spans with the same class.
Spans with different classes can overlap, and separate spans do not count distinct action instances.

## Confidence and merging

A confidence threshold is the minimum score required to retain a prediction.
Set `confidence` in the HTTP request, the legacy model route, or the Action Recognition Model workflow block:

| Value | Threshold selection |
|-------|---------------------|
| Omitted or `"default"` | Use the threshold recorded in the model package |
| `"best"` | Use model-evaluation recommendations by class, then the global recommendation, then the package default |
| Number from 0 to 1 | Use that number for all classes, overriding recommendations |

Filtering happens before merging.
For example, two same-class predictions cover frames 30–59 at 0.7 and frames 45–74 at 0.9.
They merge into frames 30–74 at 0.9.
At a threshold of 0.8, only the second prediction remains, so the result covers frames 45–74.

When two candidate spans are merged, the merged span receives the higher of their confidence scores.
The same rule applies when more than two spans merge: the result keeps the highest contributing score.
Do not interpret that score as the probability that the action occurs throughout the merged span.

### Inspect predictions before filtering and merging

Set `include_candidates` to `true` in the HTTP request to receive scored, unmerged candidates.
These include predictions below the selected threshold.
Use these candidates to compare thresholds without another model call.
The returned `timeline` still uses the selected threshold.
Candidate output grows with video length and stays in memory for the request.

Cosmos uses the same action-recognition API, but returns no scored candidates.
It rejects numeric thresholds and `"best"`.
For Cosmos, omit confidence or use `"default"`.

## Video sampling and streaming

The model package records the sampling rate, window length, overlap, and default confidence threshold.
HTTP inference uses overlapping windows and aligns the final window with the video's end.
Workflows uses a rolling window of sampled frames and clips predictions to frames already observed.

Sampling uses the source's nominal FPS.
Variable-frame-rate videos can therefore sample at different times from timestamp-based training.
Use constant-frame-rate input when the timing must match training.

The workflow block can classify a partial window during startup.
There is no separate final model call when a stream ends.
The final window aligned to the video's end applies to HTTP clip inference, rather than a streaming flush.

## Input preparation and model limits

Send full frames. V-JEPA resizes each frame directly to the square resolution recorded in the package, without cropping.
This keeps the full field of view, but changes the aspect ratio of nonsquare frames.
Normalization also comes from the package.

The recorded network input must satisfy these limits:

| Parameter | Limit |
|-----------|-------|
| Square side | 64–1080 pixels, divisible by 16; default 384 |
| Frames per network window | 2–256, with an even frame count |
| Tokens per network window | At most 81,920: `(frames / 2) * (side / 16) ** 2` |

Short input windows repeat the final frame to fill the recorded network window.
Predictions remain clipped to the observed input duration.
Direct model calls require RGB frames with red, green, and blue channels.
Each channel uses `uint8` pixel values from 0 to 255.
The sampling FPS must match the value recorded in the package.
NumPy frames use height, width, and channel order. Torch frames use channel, height, and width order.
Direct outputs use sampled-window coordinates, rather than the HTTP response's source-frame coordinates.

The loader accepts the versioned `frame_anchored_multilabel_spans_v1` method exported by `roboflow-train`.
Packages must contain `model.safetensors`, `inference_config.json`, and `class_names.txt`.
The loader rejects incompatible methods and missing weights.
The model stores parameters as 32-bit floating-point values (FP32).
On CUDA, it uses the computation precision recorded in the package.
No temporal cache or causal-attention path is enabled.

## Memory and stream lifecycle

### Stored frames

HTTP decoding and Workflows resize frames before retaining them, including workflows with tensor inputs.
Stored frames are CPU RGB `uint8` arrays.
Normalization and GPU transfer happen during inference.
Direct model calls use the same resize, and already prepared arrays avoid another resize.

Each stored frame holds `3 * side * side` bytes of pixels.
For example, 160 frames at 512×512 hold 120 MiB of pixel data.
This is the window's pixel payload, rather than a limit on process or GPU memory.
Decoding temporarily holds a source frame, and inference also allocates normalized tensors and model activations.
Concurrent calls and source frames retained by your application add to peak memory.
The token limit does not guarantee a fixed memory peak.

The shared model interface uses `frame_storage_transform` to prepare stored frames.
V-JEPA supplies its square resize through this hook.
Cosmos and models without the hook keep the existing longest-side resize policy.

### Buffers and resets

The HTTP decoder shares prepared frames between overlapping windows and releases frames when no remaining window needs them.
Workflows discards sampled frames as they leave the rolling window.
It tracks at most 256 streams and retains at most 5,000 timeline entries per stream, plus a copied output snapshot.

Changing the model clears all stream bookkeeping.
Changing the class filter, window, stride, or confidence resets the affected stream.
Moving backward in frame numbers also resets that stream.
Stream eviction discards the stream's buffer and timeline.
There is no explicit end-of-stream flush or immediate cleanup signal.
