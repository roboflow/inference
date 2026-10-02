# Thor multi-stream detection benchmark

Runs N RTSP streams through Jetson NVDEC into one YOLOv8n TensorRT detector
(detection → boxes → labels) and writes throughput, fairness, frame age and
drop counts. Jetson Thor only: it needs CUDA and the native Jetson tensor
bridge (ABI 7). There is no CPU decoder fallback.

Run from the inference checkout with the sources on `PYTHONPATH`:

```sh
PYTHONPATH=".:workflows:inference_models:stream_vision${PYTHONPATH:+:$PYTHONPATH}" \
  python development/workflows-2.0/09-thor-multistream/run_benchmark.py \
    --mode v2_pipeline --sources 8 --max-in-flight 2 \
    --warmup-seconds 15 --duration-seconds 60 \
    --output-dir /tmp/thor/v2_pipeline-8
```

Keep the image's existing `PYTHONPATH`: the Jetson 1.7.2 image supplies its
custom OpenCV under `/opt/opencv/python`. This example explicitly requests
TensorRT with `allow_untrusted_packages=True`; it never falls back to ONNX.
Model warmup runs before sources open, followed by the live warmup window.

Source count and pipeline depth are independent. Every primary workflow call
uses model batch 1. `--sources 16 --max-in-flight 2` means sixteen camera
readers feeding one shared model, with at most two frames being processed.
Finished images stay in memory; display and video encoding are outside this
benchmark.

Modes, one per process:

| `--mode` | Engine and image representation |
| --- | --- |
| `v1_numpy` | V1 engine, stock blocks, CPU numpy images (the CUDA frame is copied to the host inside the timed call) |
| `v1_tensor` | V1 engine, stock blocks, CUDA tensor images |
| `v2_serial` | V2 engine, one run at a time |
| `v2_pipeline` | V2 engine, up to `--max-in-flight` runs in flight |

Check source capacity without a model:

```sh
PYTHONPATH=".:workflows:inference_models:stream_vision${PYTHONPATH:+:$PYTHONPATH}" \
  python development/workflows-2.0/09-thor-multistream/run_benchmark.py \
    --decoder-only --sources 16 --duration-seconds 30 --output-dir /tmp/thor/decode-16
```

`--help` lists all options and works without CUDA. Sources default to
`rtsp://192.168.10.55:8554/hd/cam_thor_{index}`; change with `--url-template`.

## Output

A progress line every 5 s, then a short summary. The exit code is 0 only for a
full window with no host-copy fallback and clean shutdown.

| File | Content |
| --- | --- |
| `summary.json` | Config, backend/model/decoder facts, first-frame tensor layout, per-source throughput and fairness, latency quantiles, drops per layer, native bridge counters, process CPU, stop report |
| `frames.csv` | One row per completed frame: phase, source, frame id, timestamps, latencies, detection count |

Timestamp meaning (full text in `summary.json` → `timestamp_semantics`):

| Name | Taken |
| --- | --- |
| `reader_wallclock_s` | Wall clock before the VideoSource grab. The bridge exposes no source PTS |
| `arrival_ns` | Runner reader received the frame from the VideoSource buffer |
| `admit_ns` | Frame handed to the backend |
| `ready_ns` | Backend result is complete, including its GPU work |

`age_ms` = ready − arrival: reader-receipt-to-ready time. It excludes time in
RTSP, decoding and the preceding native/VideoSource buffers. It is not
camera-to-display latency.

Each source keeps one latest frame. Drops are counted in three places: native
bridge slot, VideoSource buffer and runner slot. Frames lost before the
bridge (network, RTSP jitter buffer, decoder) are not visible here.

## Output parity and physical batching

Use the same `PYTHONPATH` as above. Capture sixteen held GPU-decoded frames,
compare all modes in separate processes, and stress overlapping V2 calls:

```sh
python development/workflows-2.0/09-thor-multistream/check_parity.py run \
  --output-dir /tmp/thor/parity --frames 16 --depth 4 --rounds 4
```

The report preserves every mismatch and saves comparison images. Tensor modes
require exact predictions and pixels. The existing plain and tensor box
renderers use slightly different border geometry; even the one-pixel colour
check can fail at overlapping borders. Inspect the saved difference rather
than assuming a nonzero exit means the detector or pipeline is wrong.

Reuse the fixture for a separate stock V1 tensor batch-8 diagnostic:

```sh
python development/workflows-2.0/09-thor-multistream/batch_diagnostic.py \
  --fixture /tmp/thor/parity/fixture.pt --output /tmp/thor/batch8.json \
  --batch-size 8 --model-only
```

This checks actual model forward batch sizes and compares batched outputs
with single-frame outputs. Its timing excludes live sources and admission;
compare its batch-8 result with its own batch-1 result.

For a live physical-batch control, use the same source and timing options
explicitly (the two runners have different defaults):

```sh
python development/workflows-2.0/09-thor-multistream/run_batched.py \
  --mode v1_tensor --sources 16 --batch-size 8 \
  --duration-seconds 30 --warmup-seconds 10 \
  --output-dir /tmp/thor/live-batch8
```

`v1_numpy` is also supported. At most eight frames are admitted; one worker
collects up to eight, with a 3 ms timeout for a partial batch, and runs one
stock V1 workflow call. Per-frame service includes collection wait and the
whole batch. `batches.csv` records batch service including image conversion
and GPU readiness. `--record-forward-batches` additionally records actual
model forward sizes; its small diagnostic wrapper runs during timing.
Check the detailed stop report and that native counters are present as well
as the exit code before accepting a measurement.

## Files

| File | Content |
| --- | --- |
| `run_benchmark.py` | Command line, admission loop, shutdown |
| `sources.py` | Forced NVDEC sources, latest-frame slots, round-robin admission |
| `metrics.py` | Timing records, `frames.csv`, summary figures |
| `backends.py`, `gpu_blocks.py` | V1/V2 detection backends and GPU-ready painter blocks |
| `check_parity.py` | Capture a fixture, compare outputs and stress concurrent submissions |
| `batch_diagnostic.py` | Held-frame stock V1 physical batching comparison |
| `run_batched.py` | Live stock V1 physical batching control, with batch-size and service records |
