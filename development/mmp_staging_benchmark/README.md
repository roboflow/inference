# Staging MMP capacity experiment

This directory packages the `feat/new-model-manager` inference server into a
staging-only GPU image and provides two tools:

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

python /opt/mmp-benchmark/run_concurrent_clients.py \
  --spec /opt/mmp-benchmark/spec.same-model.example.json \
  --image /data/dog.jpg \
  --output /results/same-model.json
```

For a mixed-model test, use `spec.mixed-model.example.json`. Every client has
its own tenant ID, API-key environment variable, model, instance, concurrency,
optional target FPS, and model parameters. `target_fps: 0` means closed-loop
maximum throughput.

The current MMP worker boundary is the routing key, not the tenant label or API
key. Two clients with the same `model_id` and empty `instance` intentionally
share one model subprocess and can be auto-batched together; this is what
`spec.same-model.example.json` measures. It is **not** cross-workspace process
isolation. Use `spec.same-model-isolated.example.json` to give each tenant a
distinct `instance`. The MMP then creates separate routing keys and separate
model subprocesses while loading the same weights. Comparing those two specs
with and without MPS measures the real isolation/VRAM/throughput tradeoff.

The runner refuses known production Roboflow hosts. Accepted targets are
localhost/private addresses, Kubernetes service names, and
`*.roboflow.one`/explicit staging hosts.

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

## First staging sequence

1. Start one dedicated L40S pod with `NVIDIA_MPS=0` and the 4 GiB `/dev/shm`.
2. Save the capability report, image digest, node, GPU UUID, driver, and cache
   state.
3. Run same-model shared-worker `concurrency = 1, 2, 4, 8, 12, 16, 24, 32`
   until the SLO knee, repeating each point after model warmup.
4. Run the same-weight isolated-instance spec, then mixed public
   detection/segmentation models and distinct private staging-workspace models.
5. Repeat exactly with `NVIDIA_MPS=1`; compare throughput, tail latency,
   fairness, worker failures, and VRAM.
6. Re-run the winning configurations as a long soak and inject one client
   cancellation/model-worker failure before considering video integration.

The first L40S run is blocked until (a) the image is built/pushed after GCP
auth is restored, (b) the dedicated pod has a memory-backed `/dev/shm`, (c)
the container runtime exposes MPS binaries, (d) at least one staging model API
key is available as a Secret, and (e) a test image is mounted or downloaded by
an explicit setup step.
