# Staging CPU worker sizing and topology campaign

This campaign decides whether CPU jobs should use several right-sized pod
classes, how many live jobs each class can admit, and whether one OS process per
job improves isolation or capacity. It is staging-only. Files in this directory
do not authorize a build, API run, Deployment patch, pod deletion, or rollout.

## What is known, and what is only preliminary

A read-only inspection on 2026-08-13 found the live `ck8s-stg/video-proc`
`video-processor-pool-cpu` Deployment at UID
`cdf161b8-2b7d-43d5-ae09-c5c52ed14d21`, generation/revision `7/7`. It had one
ready `Burstable` pod with:

- exact image
  `video-processor-cpu-telemetry@sha256:106cf74b916f55d1cfc4322d9b361fb1fbed86889dd8d8eb435904c950801d6a`;
- reported source `c63f9720c25a27e7aa290cea601b09590a6de9f2` and variant `thread`;
- requests `4 CPU / 8 GiB`, limits `8 CPU / 16 GiB`, `MAX_CONCURRENT_JOBS=4`;
- `ONNX_INTRA_OP_THREADS=2`, `OMP_NUM_THREADS=2`;
- placement on an AMD `c1a.16x` default-pool node with 16 allocatable CPUs and
  about 64 GiB memory; and
- idle use of approximately `6m CPU / 719 MiB` at that instant.

The checked-in experiment ledger records ignored reports
`cpu-single-detection-c{1,2,3,4}-20260812a`, all using the same
`microsoft-coco-obj-det/8` YOLOv8 Nano 640 workflow, output off, one worker, and
90-second windows:

| Jobs | Aggregate FPS | Spread | Reported latency | Average CPU |
|---:|---:|---:|---:|---:|
| 1 | 9.417 | n/a | 39.12 ms | 1.287 cores |
| 2 | 18.570 | 2.9% | 43.38 ms | 2.360 cores |
| 3 | 27.403 | 3.5% | 55.72 ms | not retained |
| 4 | 30.842 | 37.8% | 190.58 ms | 4.953 cores; reported 8-core-limit contact |

That run suggests a strict-SLO knee between two and three jobs and saturation
at four. It is **not a current certification**: the raw reports and resource
files are absent, the recorded image is only mutable tag
`video-processor-cpu:2e4a97ee5`, and the old evidence has no immutable report
hash, histogram-source gate, CFS-throttling series, process IDs, counter-reset
gate, or current-image comparison. Its pod request/limit manifest and OS-process
topology were not retained either; the noted 8-core contact does not prove a
specific current size class. Use it only to order the new curve.

## Candidate classes

[`size_classes.json`](size_classes.json) defines three hypotheses:

| Class | Guaranteed CPU/RAM | Admission | ONNX threads/session | Initial hypothesis |
|---|---:|---:|---:|---|
| `cpu-2` | 2 / 4 GiB | 1 | 2 | one light job |
| `cpu-4` | 4 / 8 GiB | 2 | 2 | up to two light jobs or one medium job |
| `cpu-8` | 8 / 16 GiB | 4 | 2 | up to four light jobs, or a measured heavy/exclusive job |

Requests equal limits so a class comparison does not silently rely on burst
CPU. This is Guaranteed QoS, not dedicated cores: the node may still host other
pods. The collector records node CPU contention, but this campaign neither
controls nor claims an exclusive CPU Manager policy. Results characterize the
observed shared-node environment. The admission values are experimental
ceilings, not capacity claims.

Keeping two intra-op threads per model session makes total potential ONNX work
match the class at full light-job admission. Do not tune both pod size and
intra-op threads in one comparison. If thread-count tuning is needed, add a
separate fixed-size A/B after this curve.

## Runtime topology A/B

Run every qualifying class with both:

- `thread`: the current worker topology, multiple pipelines in one interpreter;
- `process`: one spawned OS process per job, with decode, workflow, model, and
  output publishing inside the child. Frames and tensors do not cross IPC.

The process leg requires a new immutable CPU overlay. The observed thread image
predates the reviewed process cleanup source and must never be relabeled as the
process leg. Build a thin overlay from the exact observed digest using a
reviewed descendant of
`008d5e64b27d19c7c5da6334ec9497ba756827ad`, record its Cloud Build ID and
resolved digest, and run the credential-free image smoke before rollout. A
suggested helper is dry-run by default and materializes a clean temporary
context solely from `git show REVISION:path`. It records a SHA256 for every
tracked processor input. `--execute` is a staging external write and must not
be used without separate authorization:

```bash
python development/video_poc/benchmarks/cpu_sizing/build_process_image.py \
  --revision REVIEWED_GIT_SHA \
  --output-dir development/video_poc/benchmarks/cpu_sizing/provenance/REVIEWED_GIT_SHA
```

After authorization, add `--execute`. The retained provenance validates the
Cloud Build per-file input hashes against the exact Git bytes, substitutions,
resolved uploaded source, resulting image digest, Artifact Registry digest,
and credential-free smoke build. Merely recording a build ID or digest is not
sufficient.

## Exact patches and preflight

[`render_variant.py`](render_variant.py) produces a minimal strategic-merge
patch. Thread patches default to the observed immutable digest. Process patches
fail closed unless `--process-provenance` supplies a committed record under
`cpu_sizing/provenance/` binding the new digest to the exact base, descendant
source revision, successful Cloud Build, and passed image smoke:

The checked-in `patches/cpu-{2,4,8}-thread.json` files are exact renderer
outputs for the observed baseline. Tests fail if they drift. Process patches
are intentionally rendered only after the new reviewed digest exists, so no
placeholder or thread-only image can be accidentally applied as process mode.

```bash
python development/video_poc/benchmarks/cpu_sizing/render_variant.py \
  --size cpu-4 --topology thread --output /tmp/cpu-4-thread.json

python development/video_poc/benchmarks/cpu_sizing/render_variant.py \
  --size cpu-4 --topology process \
  --process-provenance /path/to/retained-cpu-process-provenance.json \
  --output /tmp/cpu-4-process.json
```

Before every separately authorized patch:

1. stop all staging benchmark runners and confirm no run is between repetitions;
2. require context `ck8s-stg`, namespace `video-proc`, the exact Deployment UID,
   exactly one desired/updated/ready/available CPU pod, zero unavailable replicas,
   zero `pool=working` pods, and no
   active API job;
3. capture full Deployment YAML, revision, pod `imageID`, env, resources, node,
   and node allocation as the rollback anchor;
4. run `kubectl patch --type=strategic --dry-run=server -o yaml` and inspect that
   only image, the eight named experiment env vars, resources, and the rollout
   annotation change; and
5. for process mode, prove the image runtime reports the reviewed source SHA and
   passed the credential-free process-module/protocol image smoke. Also require
   `git merge-base --is-ancestor 008d5e64b27d19c7c5da6334ec9497ba756827ad REVIEWED_GIT_SHA`.
   A digest alone does not prove topology.

After rollout require the intended immutable `imageID`, no restarts, exact
resource requests/limits, `status.runtimeVariant`, and
`jobExecutionMode=thread|process` before submitting any job. Roll back to the
captured Deployment revision between topology legs; do not infer rollback from
a tag.

Run [`verify_rollout_patch.py`](verify_rollout_patch.py) before every authorized
patch. It binds the target to the audited Deployment UID, requires a fully ready
CPU pool and zero working pods, rejects placement or extended-resource drift,
asks the API server for a dry-run render, and rejects any change outside the
image, eight experiment env vars, resources, and rollout annotation. A passing
preflight is still not write authorization and cannot detect a job submitted
after the check; pause submitters for the whole transition.

```bash
python development/video_poc/benchmarks/cpu_sizing/verify_rollout_patch.py \
  /tmp/cpu-4-thread.json \
  --expected-current-image "$CAPTURED_ROLLBACK_IMAGE"
```

Retain its JSON output; it hashes the live Deployment, server render, and patch.

Before any process-mode capacity point, execute the c2 selective-cancellation
and child-crash procedure from
[`../../experiments/process_isolation/JOB_PROCESS_MATRIX.md`](../../experiments/process_isolation/JOB_PROCESS_MATRIX.md)
against the exact CPU digest and size. Retain the raw c2 topology report,
process snapshot, selective-cancellation observation, child-crash observation,
and final status snapshot together with the exact process patch and catalog.
[`validate_process_gate.py`](validate_process_gate.py) `generate` hashes each
input and derives the supervisor/child relationship, distinct per-job PIDs,
sibling frame advancement after cancellation and crash, sanitized crash
failure, exact runtime identity, and `activeJobs=0`. Its `validate` command
re-reads and re-hashes every raw artifact and recomputes the result; an
operator-authored list of passing booleans is not accepted. A GPU D/E/F gate or
an image import smoke does not substitute for this CPU runtime gate.

The raw schemas are intentionally small and sanitizable: every artifact has a
`runtime` block (`image`, `revision`, `variant`, `podUid`); lifecycle artifacts
have `targetJobId`, `siblingJobId`, `before.jobs`, and `after.jobs`, where each
job retains only `id`, `state`, and `frames`; the crash artifact also retains a
sanitized `failure.message`. Each lifecycle artifact also retains the injected
`action` bound to the target child PID plus exact pre/post process tables; the
crash action must show successful `SIGKILL` delivery, and `observedExit` must
bind the resulting child exit to that PID. Cleanup retains `activeJobs` and a final process
table proving the supervisor remains with no child PIDs. The process
snapshot retains only `supervisorProcessId` and `processes` entries with
`pid`, `ppid`, and sanitized `argv`. Example:

```bash
python development/video_poc/benchmarks/cpu_sizing/validate_process_gate.py generate \
  --topology-report evidence/topology.json \
  --process-snapshot evidence/processes.json \
  --cancellation-observation evidence/cancel.json \
  --crash-observation evidence/crash.json \
  --cleanup-status evidence/cleanup.json \
  --expected-patch evidence/cpu-4-process.json \
  --catalog evidence/size_classes.json \
  --output evidence/process-gate.json
python development/video_poc/benchmarks/cpu_sizing/validate_process_gate.py \
  validate evidence/process-gate.json
```

## Workload and curve order

[`cpu-capacity.staging.json`](cpu-capacity.staging.json) uses the verified
staging connector fixture and two repetitions per point. It adds CPU-tier forms
of the exact tracking, segmentation, and dual-model workflow specifications;
their CPU viability is intentionally unproven.

For each topology:

1. On `cpu-8`, run the c1 gates `cpu-blur-c01`, `cpu-fps05-c01`,
   `cpu-tracking-c01`, `cpu-segmentation-c01`, and `cpu-dual-model-c01`.
   Parse/start failure, model incompatibility, OOM, or target miss is a result;
   do not silently substitute a smaller workflow.
2. Run the 5-FPS curve first: `cpu-2` gets c1; `cpu-4` gets c1/c2; `cpu-8`
   gets c1/c2/c3/c4. Stop a class after two failed concurrency points.
3. Repeat at 8 FPS. Advance to 10 and 15 FPS only while c1 passes; higher target
   rates are different service tiers, not required for a lower-rate class.
4. Run unbounded c1 through each class for per-core scaling, then up to each
   class's admission ceiling. Unbounded runs diagnose saturation; they do not
   certify a target-FPS service tier.
5. Only after homogeneous points pass, run `cpu-mix-detect-track-c02`,
   `cpu-mix-detect-segment-c02`, and `cpu-mix-detect2-track-c03` on classes with
   sufficient admission. The medium/heavy workload starts after 60 seconds;
   require at least ten baseline samples for every incumbent and compare its
   delivered FPS before arrival with the measurement phase. Retention must be at
   least 90%; missing baseline coverage fails rather than passing vacuously.
   Run [`analyze_arrival.py`](analyze_arrival.py) on each raw report and retain
   its result next to the ordinary capacity analysis.

Dry-run exact selections before adding `--execute`, for example:

```bash
python development/video_poc/benchmarks/run_api_experiment_matrix.py \
  --matrix development/video_poc/benchmarks/cpu_sizing/cpu-capacity.staging.json \
  --scenario cpu-fps05-c01 --scenario cpu-fps05-c02 \
  --suite-id cpu-4-thread-fps05-001
```

## Validity and certification gates

Every report must retain the API report and the paired output from
[`collect_cpu_telemetry.py`](collect_cpu_telemetry.py). Certify a point only
when both repetitions prove:

- the actual connector stream is H.264 1280x720 at 60 FPS, source is connected,
  output publishing is off, measured window is at least 180 seconds, and the
  first 10 seconds are excluded;
- the exact image digest, source revision, runtime variant, size, topology,
  workflow specification, model IDs, and target FPS match;
- all jobs use one processor, no requeue/recovery/migration/counter reset occurs,
  and cleanup returns `activeJobs=0`;
- delivered per-job FPS is at least 90% of target, histogram-derived frame p95
  is at most 50 ms, cohort spread is at most 10%, and Jain fairness is at least
  0.98;
- thread mode reports one common worker process ID; process mode reports one
  distinct child process ID per job and cancellation/crash tests prove sibling
  survival before capacity claims are accepted;
- processor CPU p95 is at most 90% of its class, memory max is at most 80% of
  limit, CFS throttled-period ratio is at most 1%, restart delta is zero, and no
  OOM/eviction appears; and
- relay loss/error counters remain zero so CPU conclusions are not network
  conclusions.

The CPU collector extends lifecycle queries through collection time and records CPU
and memory requests/limits, QoS, node/image labels, CFS throttled-period ratio,
thread count, restart delta, OOM/eviction state, and the final active-job gauge.
Its `certificationEvidence.evidenceComplete` is fail-closed: any absent series,
wrong size/topology/PID identity, nonzero restart/OOM/eviction, non-Guaranteed
QoS, or nonzero final active jobs prevents certification. Prometheus currently
exports the required staging metrics, but every run must still prove coverage.
Run it after at least one post-cleanup scrape (normally 35 seconds); it extends
lifecycle queries to collection time while keeping CPU, memory, throttling, and
node-contention summaries bounded to the measurement window. An immediate run
that has not yet observed `activeJobs=0` correctly remains non-certifying.
The collector requires `--expected-patch`; it derives the expected immutable
image, source revision, topology, and size from that exact patch rather than
trusting the run's self-reported variant. OOM/eviction reason series are not
fabricated with `vector(0)`: their absence means no reason was emitted only
after exact-pod image, QoS, restart, active-job, and resource coverage passes.

If target FPS and latency pass but the 10% CPU headroom gate fails, call the
point a throughput observation, not safe capacity. Compare classes by aggregate
successful FPS per requested core, memory per active job, startup/TTFR,
tail latency, throttling, fairness, and failure containment—not aggregate FPS
alone.

The desired outcome may be one class, multiple classes, or GPU-only routing for
some workflows. Do not pre-assign segmentation or dual-model workflows to CPU
until their c1 gates pass and cost-per-successful-frame is compared with the
matching GPU result.
