# Staging L40S runtime capacity campaign

This campaign finds the maximum number of live MediaMTX streams that one L40S
can process with the exact staging YOLOv8 Nano workflow. It compares the
original processor/runtime, Inference v1.4 tensor workflows with PyAV, and the
same v1.4 tensor workflows with NVDEC. MMP, process isolation, and autobatching
are out of scope. Production reads, writes, and applies are forbidden.

## Fixed workload

- cluster: `ck8s-stg`; namespace: `video-proc`; GPU pool only;
- source: workspace `thomas-workspace`, connector source
  `9g7UzPcDyVBFBJ0dLei6`, `traffic.mp4`, connector-local ID
  `file:traffic.mp4`;
- connector identity at preflight:
  `conn-Thomass-Mac-F2YCM646DL.local`, source status `connected`;
- matrix: [`matrices/l40s-runtime-capacity.staging.json`](matrices/l40s-runtime-capacity.staging.json);
- workflow profile: `single-detection`; exact model
  `microsoft-coco-obj-det/8`; bounding-box visualization remains in the exact
  workflow, but watched output publishing is disabled;
- source rate: native/unbounded. Do not pass `maxFps`; this avoids the separate
  uploaded-file cadence experiment and measures the active connector path;
- one worker, one L40S, `MAX_CONCURRENT_JOBS=24`, no recovery tolerance;
- each point has a 180-second measurement, 10-second analysis warmup, and two
  repetitions. Run `c1,c2,c4,c8,c12,c16,c24` one scenario at a time so the
  fail-fast rule can stop the curve.

Before every scenario, re-check that the source is still `connected`, exactly
one ready GPU worker exists, no prior `pool=working` pod remains, and the
Deployment image/runtime identity matches the selected variant. A scenario is
invalid if jobs spread across workers.

## Immutable variants

| Variant | Image and source | Runtime contract |
|---|---|---|
| A — original | `video-processor-telemetry@sha256:50d4c922f5cd760f43fd982e04819c9a9ad18a1e17a43f67268ff8f917c80e6a`; `c63f9720c25a27e7aa290cea601b09590a6de9f2` | original inference runtime and processor, low-latency PyAV, ndarray workflow path; no v1.4 tensor flags |
| B — v1.4 PyAV | `video-processor-nvdec@sha256:214196ff30e8ac912830617138d32789c08456349528e0dd44e42cba7e8ac326`; `6ca38194bdc3c312c0adf6a3a275b9014c79f4b6` | tensor workflow serializer/model path, CUDA image device, adaptive backpressure, PyAV ingest |
| C — v1.4 NVDEC | same exact image and source as B | same flags as B; only ingest mode changes to fail-loud `gstreamer_cuda` and the telemetry label names the NVDEC leg |

Artifact Registry returned both exact digests and SLSA build level 3 during
preflight. Variant C already passed the disposable L40S capability probe for
CUDA/GStreamer tensor construction. Its c1 report must additionally show
`hardwareDecodeVerified=true`, the GStreamer CUDA producer, tensor-bridge
counters, and zero image-output host materializations; otherwise C fails before
the capacity curve.

The server-side-dry-run-validated patches are in
[`runtime_variants/`](runtime_variants/). Apply only after any active staging
experiment is complete:

```bash
kubectl --context ck8s-stg -n video-proc get deployment video-processor-pool -o yaml \
  > development/video_poc/benchmarks/results/pre-campaign-deployment.yaml

kubectl --context ck8s-stg -n video-proc patch deployment video-processor-pool \
  --type=strategic \
  --patch-file development/video_poc/benchmarks/runtime_variants/l40s-capacity-a-legacy.yaml

kubectl --context ck8s-stg -n video-proc rollout status \
  deployment/video-processor-pool --timeout=10m
```

Use the corresponding B or C patch for later legs. After every rollout, verify
the pod `imageID`, runtime environment, one ready replica, and no working pods
before c1. Record the pre-mutation revision and rendered template SHA-256 in the
experiment ledger. At the current preflight snapshot the workload UID is
`f001fcc0-4cc7-44a5-9b5f-c6f1010a1e4d`, generation 13, revision 12, but these
must be refreshed because the FPS-limiter experiment was active.

Immediate rollback uses the exact revision captured immediately before the
leg. If no intervening staging change occurs, the pre-campaign rollback is:

```bash
kubectl --context ck8s-stg -n video-proc rollout undo \
  deployment/video-processor-pool --to-revision=12
kubectl --context ck8s-stg -n video-proc rollout status \
  deployment/video-processor-pool --timeout=10m
```

Never infer a rollback revision after mutation: capture it first. Reconcile the
final retained staging variant in infra Git; do not apply production.

## Execution and evidence

Keep the staging API key in the process environment only. Execute one selected
scenario at a time, using a variant-specific suite ID:

```bash
python development/video_poc/benchmarks/run_api_experiment_matrix.py \
  --matrix development/video_poc/benchmarks/matrices/l40s-runtime-capacity.staging.json \
  --scenario connector-native-c01 \
  --suite-id l40s-a-legacy-20260813 \
  --execute
```

After each repetition, collect the exact measurement-window resource evidence:

```bash
python development/video_poc/benchmarks/collect_staging_capacity_telemetry.py \
  development/video_poc/benchmarks/results/api-corpus-RUN_ID.json
```

The collector is read-only and refuses any context other than `ck8s-stg`. It
joins cAdvisor CPU/memory and DCGM GPU utilization, framebuffer use, decoder,
encoder, and memory-copy utilization to the exact processor pod recorded in the
report. It also captures MediaMTX CPU/memory, reader count, ingress/egress, and
RTSP packet-loss/error rates. The API report supplies per-stream FPS, fairness,
startup, placement, attempts, frame latency, captured/decoded/dropped/inferred
counters, tensor bridge counters, and output host materializations.

Analyze the reports for one variant together with a 10-second warmup:

```bash
python development/video_poc/benchmarks/analysis/report.py \
  --format json --warmup-seconds 10 \
  development/video_poc/benchmarks/results/api-corpus-l40s-a-legacy-*.json
```

Delete only exact cancelled-job pods if a worker fails to observe cancellation,
after matching its job annotation to the report. Wait for one healthy ready pod
before continuing.

## Low-latency, no-lag SLO and stop rule

Each variant's two c1 repetitions establish its delivered-FPS baseline. Both c1
runs must be operationally clean, have frame-latency p95 at or below 50 ms,
p99 at or below 150 ms, and first result within 30 seconds. A higher point
passes only when both repetitions satisfy all of these:

- every job remains on one processor with no retry, recovery, counter reset,
  prolonged zero-FPS interval, pipeline error, or restart;
- every stream delivers at least 90% of that variant's median c1 FPS, aggregate
  FPS reaches at least `0.90 * concurrency * median_c1_fps`, cohort FPS spread
  is at most 10%, and Jain fairness is at least 0.99;
- per-frame decode-to-result p95 is at most 50 ms and p99 at most 150 ms;
- output host materializations remain zero with output publishing disabled;
- processor CPU p95 is below 7.2 cores and memory max below 28.8 GiB;
- GPU-utilization, decoder-utilization, and memory-copy-utilization p95 remain
  below 90%, and framebuffer use stays below 41,461 MiB;
- MediaMTX CPU p95 is below 3.6 cores, memory max below 3.6 GiB, reader count
  covers the jobs, and RTSP packet-loss/error rates remain zero.

Captured/decoded/drop ratios are reported but are not an independent failure:
freshest-frame processing intentionally drops work to prevent lag. Delivered
FPS retention plus latency are the gate.

Stop a variant immediately if c1 fails. Otherwise continue the stepped curve
until two concurrency points fail; do not execute any higher point after the
second failure. The recommended maximum is the highest point for which both
repetitions pass, never the admission ceiling or a single lucky run.
