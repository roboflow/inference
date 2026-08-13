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
