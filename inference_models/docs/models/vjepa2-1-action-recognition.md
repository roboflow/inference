# V-JEPA 2.1 action recognition

Load `vjepa2-1-vitb-384` packages from the platform with the Torch backend.
The package must contain `model.safetensors`, `inference_config.json`, and `class_names.txt`.
The loader rejects incompatible methods and missing weights instead of downloading an upstream checkpoint.

This release supports the versioned `frame_anchored_multilabel_spans_v1` method from roboflow-train.
Frames resize directly to the square resolution recorded in the package, without cropping.
The default is 384×384. The side must be 64 to 1080 pixels and divisible by 16.
Windows must contain 2 to 256 frames, with an even frame count.
Each window must fit the 81,920-token product limit: `(frames / 2) * (side / 16) ** 2`.
This limit does not guarantee a fixed GPU memory peak. Input size, precision, and concurrent calls also affect memory use.
Inference uses the normalization recorded in the package.
The recorded FPS, window length, overlap, and confidence threshold drive inference.
The shared action-recognition adapter and video decoder also serve Cosmos.
V-JEPA enables overlapping windows and an end-aligned final window through that shared path.

## Confidence and spans

Set `confidence` between 0 and 1, `"best"`, or `"default"` in `/infer/action_recognition`, the legacy model route, or the action-recognition workflow block.
`"best"` uses model-eval per-class thresholds, then the global recommendation, then the package default, like detection models.
Omit confidence or use `"default"` to use the threshold recorded in the model package.
An explicit numeric threshold overrides all recommendations.
Filtering happens before touching or overlapping same-class spans merge, and different classes can overlap.
Merged confidence is the maximum candidate confidence, not a calibrated probability for the merged span.

Set `include_candidates: true` to receive scored, unmerged candidates for threshold replay.
The returned `timeline` still uses the selected threshold, while `candidates` includes candidates below it.
Cosmos returns no scored candidates and rejects numeric thresholds and `"best"`. Omitted confidence and `"default"` leave its behavior unchanged.
The response identifies V-JEPA span semantics as `class_union`.

The public response uses inclusive source-frame indices, with continuous boundaries rounded outward to source frames.
Sampling uses the source's nominal FPS, like the shared Cosmos video path, so variable-rate inputs can differ from timestamp-based training.
Streaming keeps the existing cadence and short-window warmup, and clips spans to observed frames.
It does not perform a separate final call when a stream ends.

GPU numerical parity and performance require staging validation before release.
The model retains FP32 parameters and uses the package's recorded autocast precision on CUDA.
No temporal cache, causal attention, or TensorRT path is enabled.

## Retained frames

The shared action-recognition model interface exposes an optional `frame_storage_transform`.
V-JEPA supplies its recorded direct-square resize through this hook.
HTTP decoding and Workflows apply it before retaining frames, including tensor-input workflows.
The hook returns CPU RGB uint8 arrays. Normalization and GPU transfer occur only during inference.
Direct model calls use the same resize. Already prepared arrays pass through without another resize.
Cosmos and models without the hook retain the existing longest-side resize policy.

Each stored V-JEPA frame holds `3 * side * side` bytes of pixel data.
At 160 frames and 512×512, a full window holds 120 MiB of pixels instead of source-resolution pixels.
This describes the nominal window payload, not a total process or GPU memory guarantee.
Decoding temporarily holds a source frame, and inference creates normalized tensors and model activations.
Concurrent calls and caller-owned source frames also contribute to the peak.

The HTTP decoder shares prepared frames between overlapping windows and releases frames after no remaining window needs them.
Workflows discards sampled frames as they leave the rolling window.
It tracks at most 256 streams and retains at most 5,000 timeline entries per stream, plus a copied output snapshot.
Changing the model clears stream bookkeeping. Changing the filter, window, stride, or confidence, or moving backward in frame numbers, resets the affected stream.
Stream eviction discards that stream's buffer and timeline. There is no explicit end-of-stream flush or immediate cleanup signal.
Optional HTTP candidates remain request-local and grow with video length.
