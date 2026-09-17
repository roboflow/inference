# V-JEPA 2.1 action recognition

Load `vjepa2-1-vitb-384` packages from the platform with the Torch backend.
The package must contain `model.safetensors`, `inference_config.json`, and `class_names.txt`.
The loader rejects incompatible methods and missing weights instead of downloading an upstream checkpoint.

This release supports the versioned `frame_anchored_multilabel_spans_v1` method from roboflow-train.
Frames resize directly to 384×384 without cropping, using the normalization recorded in the package.
The recorded FPS, window length, overlap, and confidence threshold drive inference.
The shared action-recognition adapter and video decoder also serve Cosmos.
V-JEPA enables overlapping windows and an end-aligned final window through that shared path.

## Confidence and spans

Set `confidence` between 0 and 1 in `/infer/action_recognition`, the legacy model route, or the action-recognition workflow block.
Omit it to use the threshold recorded in the model package.
Filtering happens before touching or overlapping same-class spans merge, and different classes can overlap.
Merged confidence is the maximum candidate confidence, not a calibrated probability for the merged span.

Set `include_candidates: true` to receive scored, unmerged candidates for threshold replay.
The returned `timeline` still uses the selected threshold, while `candidates` includes candidates below it.
Cosmos returns no scored candidates and rejects an explicit confidence threshold.
The response identifies V-JEPA span semantics as `class_union`.

The public response uses inclusive source-frame indices, with continuous boundaries rounded outward to source frames.
Sampling uses the source's nominal FPS, like the shared Cosmos video path, so variable-rate inputs can differ from timestamp-based training.
Streaming keeps the existing cadence and short-window warmup, and clips spans to observed frames.
It does not perform a separate final call when a stream ends.

GPU numerical parity and performance require staging validation before release.
The model retains FP32 parameters and uses the package's recorded autocast precision on CUDA.
No temporal cache, causal attention, or TensorRT path is enabled.
