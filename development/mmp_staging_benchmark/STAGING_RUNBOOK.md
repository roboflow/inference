# Staging-only MMP and raw-MPS runbook

This runbook is for the isolated `video-proc-bench-mmp` namespace on
`ck8s-stg`. Every `apply`, `exec` that starts/stops MPS, model-worker kill, and
delete requires separate staging-write authorization. Nothing here authorizes
production or changes the normal `video-proc` pool.

## Experiment boundaries

Do not merge these results:

| Phase | Process topology | Input path | MPS | CUDA IPC |
|---|---|---|---|---|
| MMP baseline | 4 HTTP workers -> one MMP -> one model subprocess per routing key | local JPEG -> HTTP -> CPU SHM -> worker decode/upload | off | no |
| MMP raw MPS | exact baseline topology and image | exact same JPEG path | raw, same pod | no |
| Video D/E/F | supervisor -> one full decoder/workflow/model/publisher process per job | live MediaMTX stream | off | not applicable |

The MMP artifact contains no InferencePipeline or video processor. It cannot
answer live-stream freshness, NVDEC, output, or per-job failure-containment
questions. Conversely, D/E/F has no shared MMP or cross-process batching.

## Known immutable inputs

- PR head: `5f02db12ebdda013ff92e6607f92f88c7f9582ec`
- image: `us-central1-docker.pkg.dev/roboflow-staging/video-proc/mmp-benchmark@sha256:6a6592f77e0eb1d3bfc8b82d7add6a7206e946a437a072f7f3a58cf693b1716d`
- Cloud Build: `f0bd00b6-f0d1-410a-8ba6-bf9f8e8ccfd0`
- target: one exclusively allocated L40S GPU in Crusoe staging
- SHM: `128 * (20 MiB + 64 B) + 64 B = 2,684,362,816 B`; the 1.25
  reserve gate is `3,355,453,520 B`, so the 4 GiB memory-backed `/dev/shm`
  passes. This tmpfs consumes pod memory and is covered by the 32 GiB limit.
- first decoder axis: `imagecodecs`; `nvjpeg` is a separate later axis.
- fixture: `tests/workflows/integration_tests/execution/assets/dogs.jpg`, SHA-256
  `83bfa4e706f274ce1da7309cec6374d542f9938b3538481035588681cdaff139`.
- initial MPS mode: no active-thread percentage cap. A later 50/25-percent
  isolated-process sensitivity is not part of the matched baseline.

The registry provenance binds the digest to the uploaded Cloud Build source
tar and revision build argument, but not to a Git source URI. Preserve the
in-image `python-packages.txt` hash and do not call it cryptographic Git-tree
equivalence.

## Safety and tenancy boundary

The renderer emits only a Namespace and Deployment—no Service or Ingress. API
keys stay in the local shell and reach the pod only as Bearer request headers
over an explicit loopback port-forward. Never create an API-key Secret for the
server container.

The HTTP layer validates the key and checks model access, but discards the
returned workspace before MMP scheduling. MMP routing is
`model_id[:instance]`; `instance` is caller supplied. The scheduler, SHM pool,
and model queue are not workspace-aware. Any valid key can currently view
global metrics/model identities and invoke global unload endpoints. Therefore:

- this pod is a trusted, operator-only experiment;
- tenant labels are pseudonymous report labels, not security boundaries;
- shared-backend runs prove cross-key batching behavior, not tenant isolation;
- isolated `instance` runs prove process separation by routing key, not an
  authenticated workspace boundary;
- raw MPS is a throughput/scheduling experiment, not memory isolation or
  fairness enforcement.

## Gate 0: local render only

These commands do not contact Kubernetes:

```bash
python development/mmp_staging_benchmark/validate_staging_plan.py \
  --matrix development/mmp_staging_benchmark/matrix.staging.json \
  --run-prefix mmp-capacity-001 \
  --output /tmp/mmp-capacity-plan.json

python development/mmp_staging_benchmark/render_staging_deployment.py \
  --image us-central1-docker.pkg.dev/roboflow-staging/video-proc/mmp-benchmark@sha256:6a6592f77e0eb1d3bfc8b82d7add6a7206e946a437a072f7f3a58cf693b1716d \
  --source-revision 5f02db12ebdda013ff92e6607f92f88c7f9582ec \
  --run-id mmp-capacity-001-no-mps \
  --output /tmp/mmp-no-mps.json
```

Verify locally that the render has one GPU, `Recreate`, `gpu_type=L40S`, 4 GiB
memory-backed `/dev/shm`, `NVIDIA_MPS=0`, no Secret refs, and no Service or
Ingress. After staging-write authorization, run server-side dry-run and inspect
the entire diff before apply:

```bash
kubectl --context ck8s-stg apply --server-side --dry-run=server \
  -f /tmp/mmp-no-mps.json -o yaml
```

## Gate 1: non-MPS capability and one-request smoke

After the separately authorized apply, save exact workload identity:

```bash
kubectl --context ck8s-stg -n video-proc-bench-mmp rollout status \
  deployment/mmp-benchmark-server --timeout=15m
kubectl --context ck8s-stg -n video-proc-bench-mmp get deployment,pod -o yaml \
  > development/mmp_staging_benchmark/results/mmp-capability-001-workload.yaml
```

Run the read-only capability probe in the ready pod and copy its report before
any load. Require: one L40S, expected digest/source label, CUDA available,
compute mode `Default`, both MPS binaries, MIG `N/A`, writable SHM, and
`satisfies_recommended=true`.

```bash
kubectl --context ck8s-stg -n video-proc-bench-mmp exec \
  deployment/mmp-benchmark-server -- \
  python /opt/mmp-benchmark/capability_probe.py \
    --require-gpu --require-mps --require-shm \
    --output /results/capability-no-mps.json
kubectl --context ck8s-stg -n video-proc-bench-mmp cp \
  deployment/mmp-benchmark-server:/results/capability-no-mps.json \
  development/mmp_staging_benchmark/results/capability-no-mps.json
```

The prior read-only staging probe observed driver `570.133.20`, CUDA `12.4`,
Torch `2.6.0+cu124`, both MPS binaries, MIG `N/A`, and Kubernetes sharing
strategy `none`. Treat that only as prerequisite evidence; repeat it on the
actual benchmark pod.

Capture the pod as JSON (not only YAML), record the exact `spec.nodeName` and
single GPU UUID from the capability report, then re-render every measured
Deployment with `--node-name SPEC_NODE_NAME`. The non-MPS/MPS comparison is
invalid if either the node or GPU UUID differs from this capability baseline.

```bash
python development/mmp_staging_benchmark/render_staging_deployment.py \
  --image IMAGE_AT_EXACT_DIGEST \
  --source-revision 5f02db12ebdda013ff92e6607f92f88c7f9582ec \
  --run-id mmp-capacity-001-no-mps \
  --node-name "${BASELINE_NODE}" \
  --output /tmp/mmp-no-mps-pinned.json
```

With separate authorization for the state-changing MPS smoke, run
`--start-stop-mps` only while the server deployment itself has
`NVIDIA_MPS=0`. Require start, server-list query, and stop success. Do not run a
second MPS daemon inside the MPS-enabled deployment.

Start an explicit local port-forward. No server port may be published:

```bash
kubectl --context ck8s-stg -n video-proc-bench-mmp port-forward \
  deployment/mmp-benchmark-server 18000:8000
```

Use two distinct staging-workspace keys only in the local environment. Run c1,
then c2, before any larger point. The report hashes the fixture; every later
point must match that SHA-256.

## Gate 2: matched non-MPS matrix

For each phase and allowed concurrency in `matrix.staging.json`:

1. capture the exact ready pod and node;
2. render one point with `render_run_spec.py`;
3. run it twice in ascending concurrency order;
4. save the server log for the exact measurement interval;
5. apply `analyze_report.py`;
6. stop after two consecutive points fail the same strict gate.

The strict capacity gate is 100% success, per-client p95 <= 50 ms, Jain >=
0.95 for equal-work same-model runs, zero pool-full rejects, one stable worker
PID per expected routing key, no CUDA/Xid error, and >= 90% metrics coverage.
Capacity is the highest point passing twice with the next point failing twice.
Mixed-model Jain is descriptive only because model costs differ.

Do not report capacity from an individual `analyze_report.py` result. Feed all
paired analyzer outputs for a phase to `analyze_curve.py`; it rejects missing,
split, duplicate-run, and right-censored boundaries.

Run order:

1. `mmp-shared-no-mps`: one routing key, one model subprocess, cross-key
   requests eligible for auto-batching;
2. `mmp-isolated-no-mps`: two routing keys and two model subprocesses loading
   the same weights;
3. `mmp-mixed-no-mps`: independent detection and segmentation subprocesses.

Preserve cold and warm runs separately. Restarting the disposable Deployment
clears model processes and its `emptyDir` cache; keeping it alive preserves
both. Never silently compare a cold point with a warm point.

## Gate 3: matched raw-MPS matrix

Copy all reports and logs, then render a new run ID with `--mps`. Server-side
dry-run and apply again require staging-write authorization. Require the same
image digest, node class, decoder, fixture digest, cache classification, slots,
batch window, durations, and concurrency points. Then repeat the three phases
in the same order.

Raw MPS uses the host-injected `nvidia-cuda-mps-control/server` binaries and
private pipe/log volumes. It does not use Kubernetes device-plugin MPS sharing;
the L40S node advertises that feature as unavailable. It also does not use CUDA
IPC—the current subprocess argument is reserved and unused.

## Gate 4: fairness and failure sensitivity

Only after both capacity curves:

- equal-work cross-key comparison at the selected knee;
- asymmetric 1:N noisy-neighbor load, reported per client without treating raw
  Jain as equal-work fairness;
- cancel one local client and verify the peer continues;
- terminate one model worker by its observed PID and require one bounded reload,
  stable peer progress, no server restart, and no leaked SHM slot;
- optional isolated-backend MPS active-thread 50/25-percent sensitivity,
  labelled separately from the uncapped MPS comparison;
- 15-minute winning non-MPS and MPS soaks before longer campaign soaks.

MPS has no tenant scheduler. Any fairness improvement is observational and
must not be described as enforcement.

The analyzer requires the interval server log, exact pod JSON, capability
report, baseline node, and baseline GPU UUID. It rejects container/model-worker
restarts, a runtime image digest mismatch, changed runtime flags, or mismatched
pod annotations:

```bash
kubectl --context ck8s-stg -n video-proc-bench-mmp get pod "${SERVER_POD}" \
  -o json > development/mmp_staging_benchmark/results/pod.json
python development/mmp_staging_benchmark/analyze_report.py REPORT.json \
  --phase mmp-shared-no-mps \
  --server-log SERVER.log \
  --pod-evidence development/mmp_staging_benchmark/results/pod.json \
  --capability-report development/mmp_staging_benchmark/results/capability-no-mps.json \
  --expected-node "${BASELINE_NODE}" \
  --expected-gpu-uuid "${BASELINE_GPU_UUID}"
```

## Cleanup and rollback

First stop the local runner and port-forward, copy `/results`, workload YAML,
logs, reports, and analyzer output, and verify no process still writes a report.
The exact rollback is deletion of the dedicated Deployment and namespace; it
does not touch `video-proc`. This deletion is a staging write and requires
authorization. Verify the namespace is gone and that the normal staging video
pool image, ready/working counts, and configuration never changed.
