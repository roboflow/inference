# Staging MMP capacity experiment

This directory packages the `feat/new-model-manager` inference server into a
staging-only GPU image and provides a capability probe, local-only concurrent
runner, immutable matrix/renderers, strict analyzer, and operator runbook:

- `capability_probe.py` verifies the L40S/CUDA/MPS runtime and `/dev/shm`
  geometry before a test.
- `run_concurrent_clients.py` drives same-model or mixed-model clients and
  records per-tenant throughput/latency/errors plus MMP batching, model-load,
  slot, GPU-utilization, and VRAM evidence.

The image is deliberately separate from the video POC image. It exercises the
new `inference_server` -> MMP -> subprocess model worker path directly and is
not a deployable production artifact.

## Build

Build from the repository root. The CUDA builder/runtime and `uv` images are
pinned to amd64 manifest digests. Set `SOURCE_REVISION` to the exact git SHA;
use that SHA in the image tag as well.

```bash
REVISION=$(git rev-parse HEAD)
IMAGE=us-central1-docker.pkg.dev/roboflow-staging/video-proc/mmp-benchmark:${REVISION}

docker buildx build --platform linux/amd64 \
  --build-arg SOURCE_REVISION="${REVISION}" \
  -f development/mmp_staging_benchmark/Dockerfile \
  -t "${IMAGE}" \
  --push .
```

Record the registry digest returned by `buildx`; deploy the resulting
`IMAGE@sha256:...`, not the mutable tag.

For Cloud Build, which avoids relying on a developer Docker Desktop disk, use
the checked-in 200 GiB build configuration. Both substitutions are required:

```bash
gcloud builds submit \
  --project roboflow-staging \
  --config development/mmp_staging_benchmark/cloudbuild.yaml \
  --substitutions \
_SOURCE_REVISION="${REVISION}",_IMAGE="${IMAGE}" \
  .
```

## Required pod shape

This image must run in a dedicated staging pod that exclusively owns one GPU.
It must not be placed beside the normal video processor in the same pod.

The MMP pool consumes approximately:

```text
INFERENCE_N_SLOTS * (INFERENCE_INPUT_MB MiB + 64-byte slot header)
```

The experiment defaults are `128 * 20 MiB`, or about 2.5 GiB. Mount a
memory-backed volume at `/dev/shm` with at least 4 GiB:

```yaml
volumeMounts:
  - name: dshm
    mountPath: /dev/shm
volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 4Gi
```

Docker's default 64 MiB `/dev/shm` is insufficient. The process may fail with
`ENOSPC` during startup or `SIGBUS` when a mapped page is touched.

Suggested environment for the first L40S smoke test:

```yaml
env:
  - {name: API_BASE_URL, value: https://api.roboflow.one}
  - {name: NUM_WORKERS, value: "4"}
  - {name: INFERENCE_N_SLOTS, value: "128"}
  - {name: INFERENCE_INPUT_MB, value: "20"}
  - {name: INFERENCE_BATCH_MAX_SIZE, value: "0"}
  - {name: INFERENCE_BATCH_MAX_WAIT_MS, value: "5"}
  - {name: NVIDIA_MPS, value: "0"}
```

Mount model cache storage at `/models/cache` if cold-load results should be
separated from repeated warm-cache capacity runs.

Render the dedicated Crusoe staging Deployment only after the pushed image has
an immutable registry digest. The renderer refuses tags, production
repositories, and malformed source revisions. It creates only the isolated
`video-proc-bench-mmp` namespace and one Recreate Deployment pinned to an L40S;
it does not modify the normal `video-proc` worker pool.

```bash
python development/mmp_staging_benchmark/render_staging_deployment.py \
  --image "${IMAGE}@sha256:REGISTRY_DIGEST" \
  --source-revision "${REVISION}" \
  --run-id mmp-capability-001-no-mps \
  --output /tmp/mmp-staging.json

kubectl --context ck8s-stg apply -f /tmp/mmp-staging.json
```

The Deployment intentionally has no Service or Ingress, and the renderer no
longer injects API keys into the server container. Keep workspace keys only in
the local runner environment, connect with `kubectl port-forward`, and copy
reports to a local ignored results directory. Re-render with `--mps` only after
the non-MPS capability run passes. Never expose this API as a multi-tenant
service: any valid key can currently see global MMP metrics/model identities and
invoke global model lifecycle endpoints.

## Capability and MPS smoke

Run the non-mutating probe first:

```bash
python /opt/mmp-benchmark/capability_probe.py \
  --require-gpu --require-shm --output /results/capability.json
```

The NVIDIA container runtime must inject both `nvidia-cuda-mps-control` and
`nvidia-cuda-mps-server`. The image sets
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`, but the binaries ultimately come
from the host driver/runtime integration, not the CUDA userspace image. If
they are absent, fix the dedicated staging runtime before setting
`NVIDIA_MPS=1`.

Only in a dedicated, single-GPU experiment pod, exercise daemon startup and
shutdown:

```bash
python /opt/mmp-benchmark/capability_probe.py \
  --require-gpu --require-mps --require-shm --start-stop-mps \
  --output /results/mps-capability.json
```

`NVIDIA_MPS=1` makes the existing `inference_server.server` launcher own the
MPS daemon for the lifetime of the server. MPS must be compared with the same
image, models, client matrix, batch settings, GPU, and cache state as the
non-MPS run.

## Benchmark clients

API keys are read from environment variables and are never serialized into
the report. A tenant label is only report metadata; using distinct keys from
distinct staging workspaces is what creates a real cross-workspace test.

```bash
export RF_BENCH_TENANT_A_KEY=...
export RF_BENCH_TENANT_B_KEY=...

kubectl --context ck8s-stg -n video-proc-bench-mmp port-forward \
  deployment/mmp-benchmark-server 18000:8000

python development/mmp_staging_benchmark/render_run_spec.py \
  --matrix development/mmp_staging_benchmark/matrix.staging.json \
  --total-concurrency 8 \
  --run-id mmp-shared-no-mps-c08-r1 \
  --phase mmp-shared-no-mps \
  --server-pod "${SERVER_POD}" \
  --server-node "${SERVER_NODE}" \
  --output /tmp/mmp-point.json

RF_BENCH_TENANT_A_KEY=... RF_BENCH_TENANT_B_KEY=... \
python development/mmp_staging_benchmark/run_concurrent_clients.py \
  --spec /tmp/mmp-point.json \
  --image tests/workflows/integration_tests/execution/assets/dogs.jpg \
  --output development/mmp_staging_benchmark/results/same-model.json \
  --fail-on-errors
```

The point renderer derives the harness revision itself and refuses a dirty
checkout; it cannot be labeled with a caller-supplied SHA.

For a mixed-model test, use `spec.mixed-model.example.json`. Every client has
its own tenant ID, API-key environment variable, model, instance, concurrency,
optional target FPS, and model parameters. `target_fps: 0` means closed-loop
maximum throughput.

The current MMP worker boundary is the routing key, not the authenticated
workspace, tenant label, or API key. The server validates the Bearer key and
checks per-request model access, but workspace identity is not carried into the
MMP scheduler. Two clients with the same `model_id` and empty `instance` intentionally
share one model subprocess and can be auto-batched together; this is what
`spec.same-model.example.json` measures. It is **not** cross-workspace process
isolation. Use `spec.same-model-isolated.example.json` to give each tenant a
distinct `instance`. The MMP then creates separate routing keys and separate
model subprocesses while loading the same weights. Comparing those two specs
with and without MPS measures the real isolation/VRAM/throughput tradeoff.

The runner accepts only numeric loopback IPs over plain HTTP, refuses redirects,
and rejects DNS names (including `localhost`), private addresses, Kubernetes
service DNS, `*.roboflow.one`, and arbitrary public hosts. The campaign uses
`127.0.0.1:18000` through the explicit port-forward above.

This transport is JPEG bytes in a CPU-owned shared-memory slot. The model
worker decodes those bytes and uploads model input to the GPU. CUDA IPC is not
implemented in this path: `use_cuda_ipc` in `SubprocessBackend` is currently a
reserved, unused argument. `INFERENCE_DECODER=imagecodecs` is the required
first comparison; a separately labelled `nvjpeg` axis may follow, but it must
not be mixed into the raw-MPS A/B.

The JSON report includes:

- per-client and aggregate request counts, delivered FPS, p50/p95/p99/max
  latency, status/error counts, and Jain fairness;
- first-success/cold-load timing by model;
- one-second client time buckets for degradation and recovery analysis;
- initial/final and periodic `/v2/server/metrics` snapshots;
- MMP slot pressure/rejects and per-model batching/worker-process fields;
- GPU utilization, memory, power, and per-model VRAM attribution exposed by
  MMP/NVML;
- the source revision and image reference supplied by the image build/runtime.

## Reproducible dry-run

The checked-in matrix pins the already-built exact image digest, workload axes,
and strict gates. Validate it and render the matched non-MPS/MPS Deployments
without touching a cluster:

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

After separate staging authorization, server-side dry-run the exact output
before applying it. The Deployment must remain Service/Ingress-free and own one
exclusive L40S. Capture the namespace and Deployment before mutation; deleting
the dedicated namespace is the complete rollback.

## First staging sequence

1. Start one dedicated L40S pod with `NVIDIA_MPS=0` and the 4 GiB `/dev/shm`.
2. Save the capability report, image digest, node, GPU UUID, driver, and cache
   state.
3. Keep the same image digest, node/GPU, decoder, JPEG digest, cache state,
   batch window, duration, and warmup. Run the shared-backend points from
   `matrix.staging.json`, twice each in ascending order.
4. Run same-weight isolated routing keys, then the mixed detection/segmentation
   points. `instance` is a benchmark routing convention—not an authenticated
   workspace or security boundary.
5. Repeat the exact shared/isolated/mixed sequence with `NVIDIA_MPS=1`.
   MPS adds same-pod CUDA kernel concurrency; it provides neither memory
   isolation nor tenant-aware fairness.
6. Re-run the winning configurations as a long soak and inject one client
   cancellation/model-worker failure before considering video integration.

Apply the strict analyzer to every report and preserve the matching server log:

```bash
python development/mmp_staging_benchmark/analyze_report.py \
  development/mmp_staging_benchmark/results/shared-c08-r1.json \
  --phase mmp-shared-no-mps \
  --server-log development/mmp_staging_benchmark/results/shared-c08-r1.log \
  --pod-evidence development/mmp_staging_benchmark/results/pod.json \
  --capability-report development/mmp_staging_benchmark/results/capability-no-mps.json \
  --expected-node "${BASELINE_NODE}" \
  --expected-gpu-uuid "${BASELINE_GPU_UUID}"
```

A passing single-report analyzer result is not a capacity claim. After both
repetitions at every point through the first failing boundary, certify the
curve. This requires two passes at and below capacity plus two failures at the
next allowed point; a split pair or right-censored curve fails:

```bash
python development/mmp_staging_benchmark/analyze_curve.py \
  development/mmp_staging_benchmark/results/shared-*-analysis.json \
  --matrix development/mmp_staging_benchmark/matrix.staging.json \
  --phase mmp-shared-no-mps \
  --output development/mmp_staging_benchmark/results/shared-curve.json
```

The exact head image is already built at the digest pinned by the matrix
(Cloud Build `f0bd00b6-f0d1-410a-8ba6-bf9f8e8ccfd0`). Its provenance records
the uploaded source tar rather than a Git URI, so use the in-image package
manifest and revision label as evidence but do not overclaim a cryptographic
Git-tree equivalence. The remaining first-run gates are: separate staging
authorization, dedicated 4 GiB `/dev/shm`, runtime MPS binaries, two local-only
staging workspace keys, and verification of the JPEG digest pinned by
`matrix.staging.json`.

The existing image predates the local matrix, renderers, analyzer, and runbook;
those are deliberately executed from the clean checked-out harness. Do not
claim they are embedded in that digest. The server code and original capability
probe are embedded; a future rebuild would be a separately labeled artifact.

## What this comparison does not answer

Keep these three topologies separate in summaries:

1. **MMP subprocess + auto-batching, MPS off**: this PR's shared/isolated/mixed
   HTTP-image matrices.
2. **The exact same MMP matrices + raw same-pod NVIDIA MPS**: changes only CUDA
   process scheduling; no CUDA IPC and no tenant hard boundary.
3. **Video POC D/E/F per-job process mode without MMP**: each video job owns
   decode, workflow, model, CUDA context, and publisher. It is measured with
   live MediaMTX streams and the video latency/continuity gates, not this JPEG
   HTTP harness.

The MMP image packages neither InferencePipeline nor the video processor. A
result here cannot establish end-to-end video capacity, NVDEC behavior, frame
freshness, watched-output latency, or failure containment of D/E/F. A later
video integration must preserve the D/E/F source/workflow/FPS/output matrix and
add MMP as its only changed dimension.
