# Video cell benchmark harness

This directory turns the measurement tracks in
[MULTI_CELL_SCALING_RFC.md](../MULTI_CELL_SCALING_RFC.md) into repeatable inputs
and machine-readable results. The provisional workload classes are hypotheses,
not scheduler settings or certified capacity.

Record every direct staging mutation in
[`STAGING_EXPERIMENT_LEDGER.md`](STAGING_EXPERIMENT_LEDGER.md). Useful changes
must be committed and pushed with immutable image/config references; rejected
changes must be reverted and documented so staging does not become the only
copy of an experiment.

When only POC worker files change, build
[`processor/Dockerfile.overlay`](../processor/Dockerfile.overlay) on the exact
currently deployed processor tag or digest. Rebuilding from an
`inference-server:latest` base changes the runtime and invalidates a controlled
A/B comparison.

`processor/cloudbuild.overlay.yaml` builds GPU and CPU overlays together. All
four image substitutions and `_GIT_SHA` are required; use full immutable input
and output references. For example, the initial telemetry comparison uses the
currently deployed `video-processor:2e4a97ee5` and
`video-processor-cpu:2e4a97ee5` as its two bases.

Prefer the staging-only wrapper, which rejects mutable bases, non-staging
repositories, and shortened commit tags. It prints a dry plan by default; add
`--execute` only after authenticating `gcloud`. The result JSON contains the
resolved registry digests needed by the infra stack:

```bash
python development/video_poc/processor/build_staging_overlays.py \
  --gpu-base 'us-central1-docker.pkg.dev/roboflow-staging/video-proc/video-processor@sha256:REPLACE' \
  --cpu-base 'us-central1-docker.pkg.dev/roboflow-staging/video-proc/video-processor-cpu@sha256:REPLACE' \
  --git-sha REPLACE_WITH_FULL_PROCESSOR_COMMIT \
  --result-json development/video_poc/benchmarks/results/overlay-build.json
```

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
| `single-detection` | light | YOLOv8 Nano 640 and bounding boxes; same-model sharing baseline |
| `single-detection-cpu` | light | Exact same detector/spec on CPU for model-for-model economics |
| `detection-tracking` | medium | Small detector + stateful ByteTrack + trace rendering |
| `dual-detection` | heavy | Two model sessions and comparison visualization |
| `instance-segmentation` | exclusive | Instance segmentation + mask rendering; conservative until measured |

The staging canonical ID for YOLOv8 Nano 640 is
`microsoft-coco-obj-det/8`. Do not substitute the generic `yolov8n-640`
alias in staging: the current Inference alias table resolves it to the
production-era `coco/3` resource. Keep model IDs environment-specific and
validate the catalog mapping before carrying a matrix between staging and
production.

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

### Staging service API runner

`run_api_workflow_corpus.py` runs the same manifest through the workspace video
service API, without the UI. It only accepts staging API hosts, reads the key from
an environment variable, redacts unexpected job fields from reports, and defaults
to a no-network dry run. Generated reports are written under the ignored
`results/` directory.

Prefer an uploaded file source whose API status is `ready` for repeatable corpus
runs. A connector-local file is useful when measuring one shared relay stream,
but it depends on that exact file still existing on the connector host; a stale
source record will fail when the processor follows its download URL.

Use a dedicated staging service identity with `video-inference-job:read` and
`video-inference-job:create`. Never put its key on the command line:

```bash
export VIDEO_BENCHMARK_API_KEY='...'

python development/video_poc/benchmarks/run_api_workflow_corpus.py \
  --workspace my-staging-workspace \
  --list-sources
```

Inspect a deterministic four-job plan before starting it:

```bash
python development/video_poc/benchmarks/run_api_workflow_corpus.py \
  --workspace my-staging-workspace \
  --source-id SOURCE_ID \
  --profile single-detection \
  --repeat 4 \
  --run-id single-detection-c4-001
```

Run the plan for 60 measured seconds and cancel every job afterward:

```bash
python development/video_poc/benchmarks/run_api_workflow_corpus.py \
  --workspace my-staging-workspace \
  --source-id SOURCE_ID \
  --profile single-detection \
  --repeat 4 \
  --run-id single-detection-c4-001 \
  --duration-seconds 60 \
  --execute
```

Append `--publish-output` for the annotated-output variant. Repeat `--profile`
with different values to create a mixed workload. The runner starts the group in
parallel, waits until every job is running, records API job statistics, cancels
all non-terminal jobs on success, failure, or interruption, and returns non-zero
for startup timeouts, early terminal states, or incomplete cleanup. Repeated
copies receive distinct non-executable benchmark metadata because the service
correctly prevents the exact same workflow identity from running twice on one
source; their steps and model IDs remain identical for model-sharing tests.

For capacity and noisy-neighbor experiments, use explicit workload counts and
staged arrivals. This example establishes twelve light streams, records a
60-second baseline, then introduces one segmentation workflow on the same worker:

```bash
python development/video_poc/benchmarks/run_api_workflow_corpus.py \
  --workspace rf-inference-benchmark \
  --source-id SOURCE_ID \
  --workload single-detection=12 \
  --workload instance-segmentation=1@60 \
  --max-fps 15 \
  --require-single-processor \
  --duration-seconds 300 \
  --run-id light12-then-segmentation-001 \
  --execute
```

`--require-single-processor` makes the run fail if the jobs spread across pods;
without that assertion the result is not a single-worker packing measurement.
The service API must support the job's `maxFps` field before FPS sweeps are run.

`run_api_experiment_matrix.py` executes a resumable sequence of scenarios in
fresh child processes and writes an incrementally updated suite ledger. It
records each child before spawn and forwards SIGINT/SIGTERM so the child can
cancel captured jobs. `--resume` is allowed only with the exact matrix digest;
a completed child report is reconciled, while an ambiguous in-progress child is
refused until its exact-run cleanup tool has been used. Start by
copying `workflow-matrix.staging.example.json`, selecting a staging fixture
source, and dry-running the suite:

```bash
python development/video_poc/benchmarks/run_api_experiment_matrix.py \
  --matrix /path/to/workflow-matrix.staging.json \
  --suite-id gpu-current-worker-001
```

Add `--execute` only after checking the plans, staging worker capacity, and
available GPU headroom. The example matrix covers the 1/4/8/12/15/18/24 light
curve, delayed heavy arrivals, and output publishing. Suite and run reports stay
under the ignored `results/` directory and never contain the API-key value.

### Checked-in staging suites

The `matrices/` directory keeps the next measurement program reviewable and
repeatable. Every file is staging-only and contains API-key environment-variable
names rather than credentials:

The authoritative gate order and evidence required to advance between suites is
[`STAGING_CAMPAIGN.md`](STAGING_CAMPAIGN.md). It covers the controlled curves,
output/workflow mix, cross-workspace fairness, MMP/MPS, recovery, soaks, and
distributed relay/networking work as one campaign rather than independent
successful-looking runs.

| Matrix | Purpose |
|---|---|
| `gpu-ingest-gate.staging.example.json` | Two c1 repetitions at unbounded, 5, 10, and 15 FPS; run once per decoder/runtime variant before capacity tests |
| `gpu-controlled-fps.staging.example.json` | Two repetitions of the 1/4/8/12/18/24 GPU concurrency curve at 5, 10, and 15 FPS |
| `cpu-controlled-fps.staging.example.json` | Two repetitions of the 1/2/3/4 CPU concurrency curve at 5, 8, 10, and 15 FPS using the same YOLO model as GPU |
| `multi-workspace-fairness.staging.example.json` | Balanced tenants, delayed peer arrival, heavy-workflow arrival, reverse arrival order, and output-publisher noisy neighbors |
| `output-overhead.staging.example.json` | Matched output-off/output-on pairs for GPU detection, GPU tracking, and CPU detection |
| `long-soak.staging.example.json` | Independently selectable 15-minute, 1-hour, 4-hour, and 12-hour GPU and CPU stability gates |

Replace only the explicit `REPLACE_WITH_...` workspace/source values, then dry
run the exact scenario selection before using `--execute`. In particular, run
soak gates one at a time and advance only after the shorter gate passes:

```bash
python development/video_poc/benchmarks/run_api_experiment_matrix.py \
  --matrix development/video_poc/benchmarks/matrices/long-soak.staging.example.json \
  --scenario gpu-soak-15m \
  --suite-id gpu-soak-15m-001
```

The controlled-FPS and output suites use `VIDEO_BENCHMARK_API_KEY`. The
cross-workspace suite uses `VIDEO_BENCHMARK_API_KEY_A` and
`VIDEO_BENCHMARK_API_KEY_B`. Export values only in the runner environment; do
not substitute them into a matrix.

For cross-workspace noisy-neighbor and fairness runs, follow
[`MULTI_WORKSPACE_FAIRNESS.md`](MULTI_WORKSPACE_FAIRNESS.md). Those scenarios
keep credentials in separate environment variables, persist only safe labels,
and can assert either same-worker packing or deliberate distribution. For
unattended runs, interruption recovery, and exact-run cleanup, read
[`UNATTENDED_RECOVERY.md`](UNATTENDED_RECOVERY.md) before enabling `--execute`.
Both runners write compact atomic checkpoints after every successful start and
at bounded poll intervals; staging-only janitors act only on captured job IDs.

After a recovery baseline is clean, use the dry-run-first
[`fault_injection/`](fault_injection/) controller to exercise processor loss
during startup or steady state and relay loss during steady state. Rendering is
read-only. Its `--execute` mode deletes one exact staging pod after a UID
recheck, so every live run still requires explicit staging-cluster approval.
Pair processor fault runs with `--recovery-timeout-seconds`; ordinary capacity
runs leave it at zero so unexpected requeues remain hard failures.

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
