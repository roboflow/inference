# Video cell benchmark harness

This directory turns the measurement tracks in
[MULTI_CELL_SCALING_RFC.md](../MULTI_CELL_SCALING_RFC.md) into repeatable inputs
and machine-readable results. The provisional workload classes are hypotheses,
not scheduler settings or certified capacity.

Do not run active load against production. Use local MediaMTX, staging, or a
dedicated performance cell after the environment owner approves the test load.

## Relay publisher/reader matrix

`run_relay_benchmark.py` replays an existing encoded video in real time with
`ffmpeg -re -stream_loop -1 -c copy`, publishes one path per source, and starts
the requested number of reader processes per path. This keeps encoder CPU out of
the measurement. Optional Prometheus endpoints are sampled into the JSON report;
labels are aggregated away.

Place fixtures under the ignored `fixtures/` directory and record their source,
codec, resolution, FPS, bitrate, and GOP alongside each result. The report also
records fixture SHA-256, host details, ffmpeg version, parameters, sanitized
process diagnostics, and aggregate metric samples.

For a local relay:

```bash
export BENCH_PUBLISH_URL_TEMPLATE='rtsp://127.0.0.1:8554/{stream}'
export BENCH_READ_URL_TEMPLATE='rtsp://127.0.0.1:8554/{stream}'
export BENCH_MEDIAMTX_METRICS_URL='http://127.0.0.1:9998/metrics'

python development/video_poc/benchmarks/run_relay_benchmark.py \
  --config development/video_poc/benchmarks/relay-matrix.example.json \
  --scenario smoke-1x1 \
  --dry-run \
  --output-dir development/video_poc/benchmarks/results
```

Remove `--dry-run` after inspecting the redacted commands. For a shared cell,
obtain authorized per-stream URL templates through the control plane or use a
purpose-built performance-cell credential. Keep credentials in environment
variables and never commit generated configuration or job files.

The first relay progression should be `1x1`, then source-count ramps with one
reader, then reader fan-out at a fixed source count. Stop at the first SLO breach
or infrastructure saturation rather than jumping directly to the largest case.

## Workflow packing corpus

[`workflows/manifest.json`](workflows/manifest.json) defines five initial
profiles:

| Profile | Hypothesis | What it isolates |
|---|---|---|
| `cpu-blur` | light | Decode + workflow + visualization without a model |
| `single-detection` | light | Small detector and bounding boxes; same-model sharing baseline |
| `detection-tracking` | medium | Small detector + stateful ByteTrack + trace rendering |
| `dual-detection` | heavy | Two model sessions and comparison visualization |
| `semantic-segmentation` | exclusive | Segmentation + mask rendering; conservative until measured |

Generate standalone job files for one worker with:

```bash
python development/video_poc/benchmarks/build_processor_jobs.py \
  --profile single-detection \
  --repeat 4 \
  --source-url-template 'rtsp://127.0.0.1:8554/shared-source' \
  --output-dir development/video_poc/benchmarks/generated-jobs
```

The command prints a processor invocation with repeated `--job-file` arguments
and `--max-jobs` set to the generated count. Publish the referenced sources
before starting the worker. A shared source isolates processor packing; unique
`{stream}` paths measure the normal many-source case.

Run homogeneous concurrency `1/2/4/6/8` first, then mixed profiles. Repeat runs
with watched output publishing disabled and enabled. Record processor `/metrics`,
DCGM, container CPU/memory/network, image/runtime versions, input resolution/FPS,
and whether model weights were cold or cached.

## Aggregate processor metrics

The processor endpoint exposes process-lifetime aggregates:

- active jobs, capacity, available slots, tier, and retiring state;
- jobs started and terminal outcomes by `stream|batch` mode;
- frames processed (delivered FPS is `rate(...frames_processed_total)`);
- job-start, time-to-first-result, and decode-to-result histograms;
- active annotated-output publishers by `whip|rtsp` transport.

Labels are deliberately restricted to finite enums. Workspace, source, job,
workflow, and model identity belong in the benchmark report or controlled logs,
not Prometheus labels.
