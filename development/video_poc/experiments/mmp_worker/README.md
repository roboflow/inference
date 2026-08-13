# Video worker with the new model manager

This experiment merges Inference PR #2251 into the video POC and explicitly
passes its legacy compatibility adapter to every `InferencePipeline`. The
normal worker remains unchanged unless `PROCESSOR_MODEL_MANAGER_MODE` is set.

## Isolation boundary

The experimental worker creates one bundled model manager per workspace:

- jobs in the same workspace share model subprocesses, weights, and batching;
- jobs in different workspaces use different in-memory routing caches and
  loaded model subprocesses;
- the parent video worker still owns claims, credentials, media pipelines,
  publishing, and heartbeats.

This is model-process isolation, not complete tenant-process isolation. It is
suitable for staging throughput, fairness, and failure-containment tests. It
must not be represented as a cross-tenant security boundary.

The pod filesystem and its downloaded-weight cache are still visible to every
process. A real tenant boundary requires workspace-scoped cache roots (or a
broker that authenticates each cache access), in addition to moving the full
pipeline and credentials out of the parent process.

## Build

Build from the repository root. Use a source SHA from this isolated merged
branch and an immutable output tag; deploy the resulting digest, never the tag.

```bash
gcloud builds submit \
  --project roboflow-staging \
  --config development/video_poc/experiments/mmp_worker/cloudbuild.yaml \
  --substitutions \
_SOURCE_SHA=REPLACE,_BASE_OUTPUT=us-central1-docker.pkg.dev/roboflow-staging/video-proc/video-inference-mmp:REPLACE,_WORKER_OUTPUT=us-central1-docker.pkg.dev/roboflow-staging/video-proc/video-processor-mmp:REPLACE \
  .
```

The build compiles the complete merged GPU Inference image before layering the
video worker. An overlay on the current video-worker image is not valid because
PR #2251 adds and changes several installed Python distributions.

## Staging runtime

Start with an exclusive L40S pod and the bundled subprocess backend. These
settings bound per-workspace shared memory while permitting small batches:

```text
PROCESSOR_MODEL_MANAGER_MODE=mmp-bundled-subprocess
LEGACY_MMP_ADAPTER_MODE=bundled
LEGACY_MMP_ADAPTER_BUNDLED_BACKEND=subprocess
INFERENCE_N_SLOTS=8
INFERENCE_INPUT_MB=32
INFERENCE_BATCH_MAX_SIZE=8
INFERENCE_BATCH_MAX_WAIT_MS=5
```

Eight 32 MB slots reserve roughly 256 MB of shared-memory payload capacity per
active workspace, excluding manager metadata and model memory. The standalone
one-workspace smoke Pod uses the staging deployment's proven 2 GiB
memory-backed `/dev/shm`; use 4 GiB before a multi-workspace load test.

`mmp-bundled-direct` is an explicit control variant. It exercises the new
manager and adapter without model subprocess isolation and should not be the
primary safety or fairness result.

## Initial A/B sequence

1. Build and record the base and worker image digests.
2. Run image-level adapter and worker lifecycle tests without a GPU.
3. Deploy one separate staging-only experimental worker; do not replace the
   normal ready pool.
4. Run one YOLOv8 Nano workflow and verify frames, output, model-manager status,
   and clean shutdown.
5. Compare same-workspace concurrency 1/2/4/8 against the legacy worker.
6. Compare two workspaces using the same model and verify separate model PIDs,
   caches, failure containment, throughput, latency, and fairness.
7. Only after bundled subprocess mode is sound, test raw CUDA MPS inside the
   same exclusive-GPU pod.

The external MMP process mode and Kubernetes GPU sharing are deliberately out
of scope for this first integration test.

## Standalone workflow smoke

`render_staging_local_job.py` produces a ConfigMap and a bounded, standalone
Pod in the Crusoe staging `video-proc` namespace. It never joins the ready pool
or calls the video-job service. The Pod runs Roboflow's public benchmark video
as a batch through the staging YOLOv8 Nano workflow, owns one L40S, has a
15-minute deadline, and does not mount a Kubernetes service-account token.

The renderer accepts only an immutable image digest in the
`roboflow-staging/video-proc` repository. It puts no API key in the rendered
manifest. Create an exact-run Secret containing only the staging workspace key
under `api-key`, then render with its name:

```bash
python development/video_poc/experiments/mmp_worker/render_staging_local_job.py \
  --image us-central1-docker.pkg.dev/roboflow-staging/video-proc/video-processor-mmp@sha256:REPLACE \
  --run-id smoke-001 \
  --workspace rf-inference-benchmark \
  --api-key-secret video-mmp-smoke-001 \
  > /tmp/video-mmp-smoke-001.json
```

Applying the Secret or rendered manifest is a staging cluster write and
requires explicit operator approval. After apply, inspect the exact Pod's logs
and `/status`; require a completed job, frames greater than zero, manager mode
`mmp-bundled-subprocess`, and one active manager domain. Delete the exact Pod,
ConfigMap, and API-key Secret after collecting evidence.

## Validated staging smoke

Run `smoke-886488932` completed on Crusoe staging with worker image
`video-processor-mmp@sha256:c6ad147dd30897874dc3a5dda4fc97345ab9f4220d15405139bf76fde415b1cd`
built by Cloud Build `984175ce-625d-42aa-9f93-ba691d1006b1`.

- the YOLOv8 Nano workflow completed all 538 frames with zero drops;
- 538 frames were decoded, inferred, and rendered at 15.38 delivered FPS;
- time to first result was 12.99 seconds, including source download and cold
  model loading;
- status reported one `mmp-bundled-subprocess` workspace manager domain;
- the worker was PID 1 and the loaded model ran in subprocess PID 169 using
  892 MiB of GPU memory;
- the Pod, ConfigMap, and short-lived API-key Secret were removed after the
  result was collected.

The batch decoder ran ahead of inference, so its frame-capture-to-result
latency histogram accumulated queue residence and is not an inference-latency
measurement. Use the controlled-FPS stream corpus for latency comparisons.

Two harness assumptions were corrected during the run: the final runtime image
does not include repository test videos, so the smoke now uses the published
Roboflow fixture; decoded frames require more than 12 MB per shared-memory
slot, so the tested bounded value is 32 MB.

## Staging same-workspace A/B

The production-shaped stream matrix ran on one staging L40S using the same
immutable worker image as the smoke. Each stream replayed the public 3840x2160
vehicles fixture in real time through a pod-local MediaMTX relay, ran YOLOv8
Nano (`microsoft-coco-obj-det/8`), targeted 5 FPS, and did not publish output.
Each result is a fixed roughly 60-second frame delta after every stream had
produced a first result. All points used one workspace and the same pod resource
shape; only the model-manager backend and concurrency changed.

| Backend | Streams | Aggregate FPS | Versus legacy | GPU memory | First result |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 1 | 4.380 | baseline | 890 MiB | 13.17 s |
| subprocess MMP | 1 | 4.509 | +2.9% | 892 MiB | 18.11 s |
| legacy | 2 | 8.982 | baseline | 1266 MiB | 12.51 s |
| subprocess MMP | 2 | 8.815 | -1.9% | 890 MiB | 18.41-18.45 s |
| legacy | 4 | 18.053 | baseline | 2016 MiB | 12.43-13.09 s |
| subprocess MMP | 4 | 13.305 | -26.3% | 890 MiB | 18.74-18.83 s |
| direct control | 4 | 17.992 | -0.3% | 962 MiB | 13.00-13.03 s |
| legacy | 8 | 33.134 | baseline | 3516 MiB | 13.12-15.11 s |
| subprocess MMP | 8 | 11.363 | -65.7% | 890 MiB | 19.38-19.63 s |
| direct control | 8 | 33.070 | -0.2% | 1078 MiB | 14.08-14.10 s |

Legacy and the direct control were evenly shared at c4 and c8. Subprocess MMP
was reasonably even at c1/c2, became about 20% uneven at c4, and lost aggregate
throughput at c8 instead of merely dividing a fixed ceiling. The direct control
uses the new manager and compatibility adapter but keeps the model in the
worker process. Its legacy-equivalent throughput and much smaller memory
footprint show that same-workspace model reuse works; the adapter and manager
lookup are not the throughput bottleneck.

The bottleneck is the subprocess transport of decoded 4K frames. One BGR frame
is about 24.9 MB, so eight 5 FPS inputs imply roughly 1 GB/s of raw input payload
before accounting for copies, synchronization, and model work. The current
standalone ndarray path performs several full-frame CPU copies: `np.save` into
a parent-side `BytesIO`, a copy into the SHM slot, then child-side `bytes(mv)`,
another `BytesIO`, and `np.load`. Shared memory removes socket payload transfer,
but this path does not provide zero-copy ndarray handoff.

### 640p isolation control

The c4/c8 matrix was repeated after one pod-local fixture producer downscaled
the source to 640 pixels high before MediaMTX and worker decode. All jobs in a
point subscribed to that one local RTSP path. This changes the relay topology
from the 4K run, so only comparisons among the 640p rows are causal.

| Backend | Streams | Aggregate FPS | Versus 640p legacy | GPU memory | First result |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 4 | 16.812 | baseline | 2016 MiB | 7.85-8.65 s |
| subprocess MMP | 4 | 17.067 | +1.5% | 890 MiB | 12.65-12.66 s |
| direct control | 4 | 16.952 | +0.8% | 962 MiB | 6.73-6.74 s |
| legacy | 8 | 33.852 | baseline | 3516 MiB | 6.86-10.18 s |
| subprocess MMP | 8 | 33.097 | -2.2% | 890 MiB | 13.22-13.25 s |
| direct control | 8 | 33.829 | -0.1% | 1078 MiB | 7.42-7.47 s |

Subprocess MMP recovered from 26.3% and 65.7% below legacy at 4K to within
2.2% of legacy at 640p, with even per-stream delivery. This confirms that the
model subprocess and shared model can sustain eight simple YOLO streams; the
full-resolution ndarray marshaling is the observed throughput limiter. Cold
first-result time remains roughly four to six seconds slower in subprocess
mode and needs separate optimization.

The next implementation experiment should avoid serializing full decoded
frames. Prefer a typed SHM descriptor for an already-owned frame buffer, or do
resize/letterbox preprocessing in the pipeline process and pass a model-sized
tensor/buffer handle. Add slot-wait, marshal-copy, child-unmarshal, batch-size,
batch-wait, and GPU-utilization telemetry before tuning batch parameters.

Do not replace the legacy staging pool with subprocess MMP from this result.
The direct control is a useful throughput/model-reuse option, but it has no
model subprocess boundary and is not a tenant security or failure-isolation
result. Multi-workspace separation, subprocess failure containment, MPS, and a
long soak remain untested. The standalone Pods, ConfigMaps, and short-lived
API-key Secret used by this matrix were deleted after evidence collection.

## Recommended process boundary

For the production design, isolate each processing job (one source plus one
workflow execution) rather than treating the model subprocess as the complete
job boundary. The source connector/relay remains independent. Within a worker
pod, a source-ingest owner can decode once into a bounded shared frame ring;
one process per workflow consumes immutable frame references, performs workflow
preprocessing, and calls a workspace-scoped shared model service using compact
model inputs or buffer descriptors. This preserves multiple workflows per
source without reconnecting and decoding the same stream for every workflow.

A per-job process is useful for GIL avoidance, crash containment, cancellation,
resource accounting, and keeping one workflow's Python state out of another.
It is not by itself a hard cross-tenant security boundary when processes share
a pod filesystem, UID, secrets, and GPU. Initial production packing should be
workspace-affine: never place jobs from different workspaces in the same worker
pod/model-manager domain. Stronger cross-workspace isolation requires separate
pods plus an appropriate GPU-sharing boundary; L40S has no MIG support, and raw
MPS is a throughput mechanism rather than a security boundary.
