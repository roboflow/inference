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
INFERENCE_INPUT_MB=12
INFERENCE_BATCH_MAX_SIZE=8
INFERENCE_BATCH_MAX_WAIT_MS=5
```

Eight 12 MB slots reserve roughly 96 MB of shared-memory payload capacity per
active workspace, excluding manager metadata and model memory. The pod must
have a memory-backed `/dev/shm`; 4 GiB is the initial staging target.

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
