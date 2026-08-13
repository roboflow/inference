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
  independent repetitions. The matrix intentionally executes one repetition
  per suite ID. Repeat the complete preflight before the second suite instead
  of allowing one process to start it after a fixed cooldown. Run
  `c1,c2,c4,c8,c12,c16,c24` one scenario at a time so the fail-fast rule can
  stop the curve.

Before every scenario, re-check that the source is still `connected`, exactly
one ready GPU worker exists, no prior `pool=working` pod remains, and the
Deployment image/runtime identity matches the selected variant. A scenario is
invalid if jobs spread across workers.

## Resume checkpoint (2026-08-13)

Do not repeat completed A or B points. The authoritative threaded results are:

| Variant | Point | State | Evidence |
|---|---:|---|---|
| A — original PyAV | c1, c2 | pass x2 | strict maximum so far is c2 |
| A — original PyAV | c4, c8 | fail x2 | second failed point reached; A is complete |
| B — v1.4 tensor PyAV | c1, c2 | pass x2 | c1 median 59.3825 FPS; c2 retains approximately 59 FPS per stream |
| B — v1.4 tensor PyAV | c4 | fail x2 | 120.096/119.762 aggregate FPS, approximately 30 FPS per stream, p95 100 ms |
| B — v1.4 tensor PyAV | c8 | next | run twice; it is B's second fail-fast point |
| C — v1.4 tensor NVDEC | c1 | after B | re-run the decoder validity gate, then start its independent curve |
| D/E/F — one process per job | c1/c2 topology gate | after C | use only rebuilt immutable overlays recorded in `JOB_PROCESS_MATRIX.md` |

B c4 failed throughput and latency while CPU p95 was approximately 2.23 cores,
GPU utilization was 17-18%, framebuffer use was 2,024 MiB, decoder utilization
was 0%, copy utilization was 1%, and MediaMTX reported four readers with zero
packet loss/errors. It is not a whole-pod, GPU, or relay saturation result.

Exclude these non-measurements from conclusions:

- the original B c4 attempt made while the workspace route was overwritten;
- the automatic B c4 r2 attempt that split across two pool pods before
  measurement; and
- every B c8 preflight made while the source list was empty. Those attempts
  created no c8 jobs.

The current B worker image predates bounded threaded cancellation cleanup. A
cancelled run can leave `activeJobs=1`, `activeDomains=0`, and `pool=working`
even after all pipelines release. After every B/C repetition, wait for the
exact worker to retire. If it remains stuck, match its annotation to only
cancelled jobs, verify `activeDomains=0`, delete only that exact stale pod, wait
for deletion, then require one ready worker and zero working pods. Never begin
the second repetition while the previous worker exists.

## Dedicated control API deployment gate

The capacity runner must use the staging API Hosting route:
`https://api.roboflow.one`. Processor claim/status and connector,
relay, and fleet routes intentionally remain on `light-v2-device`; do not
change the worker's `RF_API_URL` or the connector's internal control-plane URL.

Resume only after the same exact Roboflow ref has deployed both artifacts:

1. `light-v2-video` via the **Deploy Function** workflow with
   `deploy_only=light-v2-video` and a staging-only target coordinated by the
   owner;
2. the `api` Firebase Hosting target via **Deploy API** with
   `deploy_target=hosting` and a staging-only target; and
3. both deployment runs are terminal-success, the US staging function is
   Ready, and Hosting serves it. A function-only deploy does not update the
   public route; a Hosting-only deploy before the function exists is broken.

The required rewrite order is:

```text
/*/video-sources/** -> light-v2-video
/*/video-jobs/**    -> light-v2-video
**                  -> light-v2-api
```

Before deployment, the observed and expected staging state is:

- `light-v2-video` does not exist;
- `api.roboflow.one`, `roboflow-api-staging.web.app`, and the
  `firebaseapp.com` alias return
  JSON 404 `Endpoint not found.` for both benchmark workspaces; and
- direct `light-v2-device` calls return the old misleading HTTP 200 empty list.

After deployment, verify all of these without printing or persisting keys:

- Hosting GET for `thomas-workspace` returns HTTP 200 and includes connected
  connector source `traffic.mp4`, ID `9g7UzPcDyVBFBJ0dLei6`, connector
  `conn-Thomass-Mac-F2YCM646DL.local`;
- Hosting GET for `rf-inference-benchmark` returns HTTP 200 and includes
  connected `vehicles.mp4`; keep it isolated and do not substitute it into the
  authoritative A-F traffic curve;
- a controlled short c1 API smoke can create, read, and cancel a job with the
  dedicated workspace credential and a unique idempotency key; exclude it from
  capacity results;
- runtime telemetry for the Thomas job reports H.264, 1280x720, 60/1 FPS,
  output host materializations zero, and the expected immutable worker; and
- exactly one ready GPU worker exists with no working pod before B c8 r1.

The isolated fixture is already deployment-ready: its connector Deployment is
1/1 on immutable image
`rf-video-connector@sha256:5a81d357bab2490db782a409203da70dd89d0df642ad142780d86b05b50911c4`,
has zero restarts, and discovers `file:vehicles.mp4`. Both it and the Thomas
connector continue to receive HTTP 200 healthchecks on the internal device
surface. Empty workspace lists are therefore a public-route constraint, not a
fixture-discovery failure.

The exact resume sequence is:

1. deploy and verify the dedicated function plus Hosting rewrite;
2. run the excluded short API/c1 control smoke;
3. run B c8 r1, reconcile the exact worker, repeat the full preflight, then run
   B c8 r2; c8 is B's second failed point if both repetitions fail, so do not
   run B c12+ in that case;
4. roll C using the existing immutable B/C image and change only the decoder
   configuration; run C c1 twice only after CUDA/NVDEC validity is proven, then
   continue its stepped curve and stop after c1 failure or two failed points;
5. build and smoke the new D/E/F overlays containing bounded cancellation
   cleanup, fill the immutable placeholders in
   `../experiments/process_isolation/JOB_PROCESS_MATRIX.md`, and obtain review;
6. run D, E, and F separately, each beginning with the process PID,
   cancellation, crash-containment, credential-redaction, and c1/c2 topology
   gates before any capacity point.

The first resumed B c8 repetition,
`l40s-b-v14-pyav-c08-r1-20260813-connector-native-c08-r1`, is preserved as an
operational failure rather than a capacity sample. All eight jobs initially ran
on one processor, but MediaMTX repeatedly reported slow TCP readers and dropped
large frame bursts; one PyAV pipeline then lost its RTSP session and the job
requeued during measurement. Before that failure, the 10-second-warmup window
delivered `70.537 FPS` aggregate (`8.675`-`8.942 FPS` per stream), with Jain
fairness `0.999891`, cohort spread `3.0%`, and frame-latency p95 buckets of
`250`-`500 ms`. Resource evidence shows processor CPU p95 `2.242` cores,
memory p95 `2.074 GB`, GPU p95 `12%`, framebuffer max `3524 MiB`, decoder
`0%`, memory-copy p95 `1%`, and MediaMTX CPU p95 `0.021` cores. This rules out
whole-pod CPU, memory, GPU, VRAM, decoder, relay CPU, and aggregate relay
bandwidth saturation; it is direct evidence of reader/pipeline backpressure in
the shared-process PyAV topology. Repeat c8 after exact-worker reconciliation
to establish whether the operational failure reproduces.

## Immutable variants

| Variant | Image and source | Runtime contract |
|---|---|---|
| A — original | `video-processor-telemetry@sha256:50d4c922f5cd760f43fd982e04819c9a9ad18a1e17a43f67268ff8f917c80e6a`; `c63f9720c25a27e7aa290cea601b09590a6de9f2` | original inference runtime and processor, low-latency PyAV, ndarray workflow path; no v1.4 tensor flags |
| B — v1.4 PyAV | `video-processor-nvdec@sha256:214196ff30e8ac912830617138d32789c08456349528e0dd44e42cba7e8ac326`; `6ca38194bdc3c312c0adf6a3a275b9014c79f4b6` | tensor workflow serializer/model path, CUDA image device, adaptive backpressure, PyAV ingest |
| C — v1.4 NVDEC | same exact image and source as B | same flags as B; only ingest mode changes to fail-loud `gstreamer_cuda` and the telemetry label names the NVDEC leg |

A cancellation-safe in-process v1.4 overlay is also available at
`video-processor-inprocess-v14@sha256:df170c8b6e569f9bffb23540fc3ace143c6e5b92b2ef1ee48a9719de24df8148`
(Cloud Build `9e6e4d38-8230-4eb9-9454-a99d4cc95a2a`, source
`8b6709e66c7092c0599bf7e24edd870fa1c18e4e`, credential-free import smoke
`f811a2aa-c9e7-45d8-b072-9eaec0cffd2d` passed). It carries bounded
cancellation cleanup and current `maxFps` worker wiring over the exact B/C base.
Do not substitute it mid-curve: B and C intentionally retain the same older
worker image so decoder mode is their only runtime difference. Use the overlay
for a separate cancellation/capped-input validation after the immutable B/C
comparison.

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
experiment ledger. The read-only 2026-08-13 resume snapshot is generation 15,
revision 14, with B image digest `214196ff...8ac326`, one ready L40S worker,
and no working pod. This is evidence, not a future rollback value; refresh it
immediately before every mutation.

Immediate rollback uses only the revision captured immediately before the leg:

```bash
CAMPAIGN_PRE_ROLLOUT_REVISION="$({
  kubectl --context ck8s-stg -n video-proc get deployment video-processor-pool \
    -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}'
})"
test -n "$CAMPAIGN_PRE_ROLLOUT_REVISION"

kubectl --context ck8s-stg -n video-proc rollout undo \
  deployment/video-processor-pool \
  --to-revision="$CAMPAIGN_PRE_ROLLOUT_REVISION"
kubectl --context ck8s-stg -n video-proc rollout status \
  deployment/video-processor-pool --timeout=10m
```

Never infer a rollback revision after mutation: capture it first. Reconcile the
final retained staging variant in infra Git; do not apply production.

## Execution and evidence

Keep the staging API key in the process environment only. Execute one selected
scenario at a time, using a variant- and repetition-specific suite ID. The
matrix contains one repetition deliberately; repeat the preflight and use a new
suite ID for r2:

```bash
python development/video_poc/benchmarks/run_api_experiment_matrix.py \
  --matrix development/video_poc/benchmarks/matrices/l40s-runtime-capacity.staging.json \
  --scenario connector-native-c08 \
  --suite-id l40s-b-v14-pyav-c08-r1-YYYYMMDD \
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
