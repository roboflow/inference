# Video processor runtime

This directory contains the warm video-job worker used by the hosted video
service. The supervisor claims jobs, maintains platform heartbeats, and exposes
the authenticated status/event surface. Each job can run either in that
supervisor or in its own spawned OS process.

This runtime is one part of a larger system. The durable project context that
originally lived with the broad video POC in inference
[#2616](https://github.com/roboflow/inference/pull/2616) is maintained here now:

- [ARCHITECTURE.md](ARCHITECTURE.md) — project intent, repository ownership,
  end-to-end control/media/result flows, process boundaries, and security.
- [DEPLOYMENT.md](DEPLOYMENT.md) — cell topology, ready-pool lifecycle, runtime
  configuration, rollout order, smoke gates, and rollback boundaries.
- [MULTI_CELL_SCALING_RFC.md](MULTI_CELL_SCALING_RFC.md) — the draft design for
  sticky source placement, multiple cells, dedicated capacity, and
  workload-aware admission.

PR #2616 remains the experiment notebook and evidence archive. This directory
is the source of truth for the deployable processor and its architecture.

The intended hosted topology is one spawned process per video job. The child
owns the decoder, `InferencePipeline`, workflow/model instances, CUDA context,
and direct MediaMTX output publisher. Pixels and tensors never cross IPC.
Image-redacted JSON workflow results return through a bounded latest-value
queue so `/events` and `/events/poll` preserve their browser contract without
letting a slow reader backpressure inference.

## Runtime switches

All switches are process-start settings and are reported in credential-free
runtime telemetry.

| Setting | Values | Default | Purpose |
|---|---|---|---|
| `PROCESSOR_JOB_EXECUTION_MODE` | `thread`, `process` | `thread` | Run pipelines in the supervisor or one spawned process per job. Hosted staging selects `process`. |
| `ENABLE_TENSOR_DATA_REPRESENTATION` | boolean | Inference runtime default | Preserve tensor-native workflow values. Set explicitly in deployment configuration. |
| `PROCESSOR_VIDEO_INGEST_MODE` | `pyav`, `gstreamer_cuda` | `pyav` | Select CPU/PyAV ingest or fail-loud GStreamer NVDEC ingest. |
| `USE_INFERENCE_MODELS` | boolean | Inference runtime default | Select the Inference 1.4 model implementations used by tensor-native workflows. |
| `WORKFLOWS_IMAGE_TENSOR_DEVICE` | device name | Inference runtime default | Select the workflow tensor device; hosted GPU workers use `cuda`. |
| `VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE` | boolean | Inference runtime default | Use demand-driven live-source backpressure. |

`gstreamer_cuda` requires `ENABLE_TENSOR_DATA_REPRESENTATION=true`; startup
fails rather than silently falling back to CPU decode. This makes PyAV/tensor
and NVDEC/tensor deployments differ only by the ingest switch.

## Images

`Dockerfile` builds the complete worker on an immutable Inference server base.
`Dockerfile.overlay` and the Cloud Build manifests support a thin worker-only
rebuild on an already-proven base. Always pass immutable base and source
revisions, publish an immutable digest, and deploy by digest rather than tag.

## Validation

The focused tests cover process lifecycle and crash containment, bounded JSON
result IPC, credentials, worker retirement, telemetry, file replay, PyAV
low-latency ingest, and fail-loud NVDEC selection. A deployment is not complete
until a real staging job proves both annotated preview output and advancing JSON
events, then cancels cleanly with `activeJobs=0`.
