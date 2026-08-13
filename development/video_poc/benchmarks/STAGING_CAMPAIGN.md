# Staging video benchmark campaign

This is the gated execution order for the video capacity program. It is scoped
to the Crusoe staging cell (`ck8s-stg`, namespace `video-proc`) and the isolated
MMP namespace (`video-proc-bench-mmp`). Production is never an implicit next
step.

Every live mutation gets a row in `STAGING_EXPERIMENT_LEDGER.md` before it is
applied. Every image is deployed by registry digest, every API suite uses an
immutable matrix digest and unique suite ID, and every gate finishes cleanup
before the next gate starts.

## Gate 0: prerequisites

All must be true before a controlled run:

- `gcloud projects describe roboflow-staging` succeeds non-interactively;
- `kubectl --context ck8s-stg get namespace video-proc` succeeds;
- Roboflow service API PR #14376 and its `maxFps` follow-up #14402 are deployed
  to staging, and a one-job report echoes the requested finite `maxFps`;
- GPU and CPU telemetry overlays are pushed and their registry digests recorded;
- infra PR #2454 is applied to staging or the equivalent live configuration is
  reconciled in the ledger;
- MediaMTX, GPU processor, CPU processor, kubelet/cAdvisor, and DCGM scrape
  targets are healthy for the intended Prometheus window;
- `VIDEO_BENCHMARK_API_KEY` is set only in the runner environment and
  `traffic.mp4` is still a `ready` uploaded fixture in the chosen staging
  workspace;
- no prior suite has an ambiguous running checkpoint; use the exact-run janitor
  before continuing.

The first smoke is one GPU and one CPU YOLOv8 Nano stream at 5 FPS for 60
seconds. Require one attempt, positive frame deltas, finite latency, matching
runtime/image identity, clean cancellation, and no processor/container restart.

Before any concurrency curve, run `gpu-ingest-gate.staging.example.json`
separately for the v1.4 PyAV and NVDEC/tensor worker variants. The buffering
configuration must be identical (`freshest`, `DROP_OLDEST`, decoding queue 1).
Require every controlled c1 run to deliver at least 90% of requested FPS. For
NVDEC additionally require `GstreamerCudaVideoFrameProducer`, verified CUDA
frames, advancing bridge CUDA-map counters, zero host-pixel maps/copies, and no
image-output host materialization while output is disabled. A failed c1 gate
blocks the larger packing matrix.

## Gate 1: controlled single-workload capacity

Run `gpu-controlled-fps.staging.example.json` and
`cpu-controlled-fps.staging.example.json` with output disabled. Use two
repetitions per point and always run lower concurrency before higher
concurrency. Stop increasing a curve after two consecutive points fail the same
SLO; do not consume the remaining high-concurrency points merely because the
admission ceiling permits them.

The strict provisional gate is:

- every job succeeds with one attempt and stays on one processor;
- delivered FPS per stream retains at least 90% of the c1 baseline;
- cohort FPS spread is at most 10%;
- sampled/histogram latency p95 is at most 50 ms;
- no frame-counter reset, recovery event, worker restart, CUDA error, OOM, or
  output stall;
- resource samples cover the measurement window.

Analyze each completed family with `analysis/report.py` and
`analysis/recommendations.py`. A point is a lower bound until a failing point
above it is observed. Keep a separately labeled relaxed 75 ms CPU result; never
silently mix it into the strict class.

## Gate 2: output and workflow mix

Run `output-overhead.staging.example.json` as matched off/on pairs. Compare
delivered FPS, latency, CPU, GPU, relay egress, publisher count, and failures.
Then run delayed arrivals from the mixed-workload corpus: detection plus
tracking, dual detection, and instance segmentation. Record both incumbent
degradation and arriving-job startup time.

Do not advance if output publication causes an unexplained frame-rate collapse,
relay reconnect loop, or unbounded memory growth.

## Gate 3: cross-workspace fairness

Use `multi-workspace-fairness.staging.example.json` with two distinct staging
workspace API keys and sources. First prove balanced same-model placement, then
delayed peer arrival, light/heavy noisy-neighbor cases, reverse arrival order,
and publisher noise. Reports use safe tenant labels; workspace IDs and keys are
not persisted.

This gate measures current shared-process behavior. It does not claim process
or CUDA-context isolation. Require per-tenant throughput/latency, Jain fairness,
startup, placement, attempts, runtime identity, and resource evidence.

## Gate 4: MMP isolation and MPS

Build inference PR #2788 with its 200 GiB Cloud Build config. Deploy the
digest-only renderer into `video-proc-bench-mmp`; never replace the normal video
worker pool with this image.

Run in this order on one exclusive L40S with a 4 GiB memory-backed `/dev/shm`:

1. capability probe with MPS disabled;
2. same-model shared backend, concurrency 1/2/4/8/12/16/24/32;
3. same weights with tenant-specific `instance` values and separate backends;
4. mixed detection/segmentation models;
5. repeat the exact warm/cold/cache sequence with MPS enabled;
6. cancel one client and terminate one model worker under load;
7. run the winning non-MPS and MPS configurations for at least 15 minutes.

The comparison must include throughput, p50/p95/p99 latency, per-tenant
fairness, errors, model loads, worker PIDs, slot pressure, batching, GPU/VRAM,
power, and cache state. MPS is a scheduling/throughput experiment, not a tenant
security boundary. L40S does not support MIG.

## Gate 5: processor and relay recovery

Use the dry-run-first fault controller only after its rendered plan shows the
exact `ck8s-stg` API server, run ID, job ordinal, namespace, and captured pod.
Run one-job startup loss, one-job steady-state loss, then relay replacement.
Processor evidence must join the exact old/new pod identities to a complete
runner report and verified post-replacement frame progress.

Capacity suites keep recovery tolerance at zero. Fault suites use the explicit
bounded recovery window and are excluded from capacity certification. Relay
replacement alone does not prove media continuity; pair it with frame-identity
evidence before making a continuity claim.

## Gate 6: soak progression

For the selected strict capacity point, run the 15-minute, 1-hour, 4-hour, and
12-hour scenarios from `long-soak.staging.example.json` in that order. A failed
or ambiguous shorter gate blocks every longer gate. Preserve time-series
resource evidence and check drift, latency tails, reconnects, restarts, model
reloads, memory growth, GPU errors, and frame-counter continuity.

## Gate 7: relay and multi-cell networking

Run the distributed relay harness only after the benchmark load-generator pool
has adequate CPU and a recorded instance/VPC bandwidth entitlement. Separate
publish-copy, read-copy, and read-decode curves. Record actual placement,
provider/cell, VM class, LB/CNI path, TTFF, stalls, reconnects, achieved bitrate,
CPU, and network counters.

Test colocated first, then cross-zone/datacenter/cloud and on-prem paths. These
results feed cell selection, source home-cell stickiness, dedicated workspace
cells, relay shard size, and ingress/egress admission. Do not infer a MediaMTX
limit from a node or load generator that saturated first.

## Final decision artifacts

The campaign is complete only when the ledger and committed summaries support:

- strict and relaxed CPU worker classes by workload/FPS;
- GPU streams per worker and the first failing boundary;
- output and heavy-workflow multipliers;
- cross-workspace fairness and isolation tradeoffs;
- MMP shared-vs-isolated backend and MPS recommendations;
- startup/steady recovery and continuity bounds;
- relay/node/cell capacity with the first saturated resource identified;
- cost inputs and a product allocation policy that does not confuse throughput
  with pricing;
- deployable staging configuration and an explicit, separately approved
  production proposal.
