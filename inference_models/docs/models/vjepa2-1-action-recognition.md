# V-JEPA 2.1 - Action Recognition

V-JEPA returns trained action labels, frame ranges, and confidence scores.
Several actions can overlap in time.
Use your trained Roboflow model ID and an API key with access to it.
The supported model type is `vjepa2-1-vitb-384`, not a public pretrained model ID.

The adapted architecture uses the [MIT License](https://github.com/roboflow/inference/blob/27ec46f2cb79601cc54f9fc87f527a1e2ae49cc3/inference_models/inference_models/models/vjepa2_1/LICENSE).
See [upstream attribution](https://github.com/roboflow/inference/blob/27ec46f2cb79601cc54f9fc87f527a1e2ae49cc3/inference_models/inference_models/models/vjepa2_1/UPSTREAM.md).

## Usage

Use matching Inference and `inference-models` versions that include this integration.
Only the Torch backend is supported.
HTTP clips use `/infer/action_recognition`.
Streaming uses the Action Recognition Model workflow block.

For this example, run a compatible server at `http://localhost:9001` with `USE_INFERENCE_MODELS=True`, `ACTION_RECOGNITION_ENABLED=True`, and `VJEPA2_1_ENABLED=True`.
`VJEPA2_1_ENABLED` defaults to `True`. Set it to `False` and restart the server to block V-JEPA model loading through HTTP and Workflows, without disabling Cosmos.
Install `requests` in your Python environment.
Set `ROBOFLOW_API_KEY`.
Replace the model ID and video URL.
The server must be able to access the video URL.

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
print(response.json()["timeline"])
```

Each timeline entry contains `class`, `class_id`, `start_frame_idx`, `end_frame_idx`, and `confidence`.
Frame indices refer to the original video, start at zero, and include both boundaries.
Convert them to seconds with `start_frame_idx / source_fps` and `(end_frame_idx + 1) / source_fps`.
Prefer URLs over base64 for full clips to avoid large request payloads.

## Differences from Cosmos action recognition

| Behavior | V-JEPA | Cosmos |
|----------|--------|--------|
| Labels | Trained vocabulary only. `class_filter` selects a subset | Fine-tuned vocabulary or free-form zero-shot labels |
| Confidence | Scored spans and threshold controls | Unscored spans. Confidence overrides are ignored |
| Span meaning | Same-class spans combine as `class_union`, not separate action instances | `instances` |
| Frame preparation | Direct square resize without cropping | Aspect-ratio-preserving resize |
| HTTP windows | Recorded overlap and an end-aligned final window | Default nonoverlapping fine-tuned windows, or a whole clip in zero-shot mode |

For V-JEPA, filtering happens before touching or overlapping same-class spans merge.
When two candidate spans merge, the result receives the higher of their confidence scores.

## Confidence

A threshold is the minimum confidence score to retain.

| `confidence` | Selection |
|--------------|-----------|
| Omitted or `"default"` | Package threshold |
| `"best"` | Model-eval per-class thresholds, then the global recommendation, then the package threshold |
| Number from 0 to 1 | Explicit override for every class |

For requests to the legacy HTTP endpoint `/{dataset_id}/{version_id}`, confidence values of 1 or more are treated as percentages, so `1` means 1%.
The newer, recommended HTTP endpoint `/infer/action_recognition` uses fractions, so `1` means 100%.

Set `include_candidates: true` to return unmerged predictions, including those below the threshold.
The `timeline` remains filtered.
Candidate output grows with video length and stays in memory for the request.
HTTP defaults to at most 250,000 candidates and a 64 MiB encoded response.
The server administrator can change `MAX_ACTION_RECOGNITION_CANDIDATES` and `MAX_ACTION_RECOGNITION_RESPONSE_BYTES` to positive values.
Requests that exceed either limit fail with HTTP 413 and return no partial results.
The server rejects known oversized candidate requests before model calls and applies both limits as results accumulate.

## Inputs and limits

The package records normalization, sampling frames per second (FPS), window length, overlap, and the default threshold.
Send full frames. The server resizes them without cropping.

| Recorded input | Limit |
|----------------|-------|
| Square side | 64–1080 pixels, divisible by 16. Default 384 |
| Frames per window | Even count from 2 to 256 |
| Tokens per window | At most 81,920: `(frames / 2) * (side / 16) ** 2` |

Direct calls require RGB `uint8` frames: NumPy HWC or Torch CHW, at the recorded sampling FPS.
Their output coordinates refer to the sampled window, not the original video.
Short windows repeat the final frame, but predictions stay within observed duration.
Sampling uses nominal source FPS. Variable-rate videos can differ from timestamp-based training.
Workflows has no separate final model call when a stream ends.

## Memory and lifecycle

HTTP and Workflows store resized CPU `uint8` frames before normalization and GPU transfer.
A nominal 160-frame window at 512×512 holds 120 MiB of pixels.
The token limit does not bound total memory: decoding, weights, activations, and concurrent calls add further costs.

HTTP releases frames after later windows no longer need them.
Workflows discards old samples and caps retained state at 256 streams and 5,000 timeline entries per stream, plus an output snapshot.
Model changes clear all streams.
Filter, window, stride, confidence changes, frame rewinds, and eviction discard affected state.
There is no explicit stream-end cleanup.
