# inference-server

HTTP server for model inference. Wraps `inference-model-manager` with FastAPI endpoints.

## Install

Requires Python 3.10–3.13. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install uv

# For development: install inference-models and inference-model-manager editable first
# uv pip install -e "../inference_models"
# uv pip install -e "../inference_model_manager"

# CPU (torch + ONNX)
uv pip install -e ".[torch-cpu,onnx-cpu]"

# CUDA 12.4
uv pip install -e ".[torch-cu124,onnx-cu12]"
```

Extras cascade through `inference-model-manager` to `inference-models`. The
`otel` extra installs the OpenTelemetry packages behind `OTEL_TRACING_ENABLED`.

## Quick start

```bash
python -m inference_server.app
```

Models load on first request via a direct in-process `ModelManager`.

## Run in Docker

Build from the **repo root** (the Dockerfile COPYs `inference_models`, `inference_model_manager`, `inference_server`, `workflows`):

```bash
docker build -f inference_server/docker/Dockerfile.cpu -t inference-server:cpu .
```

Run:

```bash
docker run --rm -it \
  -p 9001:9001 \
  inference-server:cpu
```

```bash
curl -X POST "http://localhost:9001/v2/models/infer?model_id=yolov8n-640" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: image/jpeg" \
  --data-binary @image.jpg
```

Model parameters can also be passed as query parameters or multipart text
fields; they are converted to the type `GET /v2/models/interface` declares.
Booleans accept `true`/`false`/`1`/`0`/`yes`/`no` (any case); list parameters take one
value per repeated key (`?classes=cat&classes=dog`), and a single value becomes
a one-element list. Nested lists and objects need a JSON body. Where a model
accepts `confidence=best` or `confidence=default`, those pass through; any other
confidence must be a number. A value that does
not match its declared type is answered with 400 `INVALID_PARAM`.

## Extension points

`inference-server` and `inference-model-manager` resolve additional
implementations via entry points, so a separate package can extend either
without a code change here:

| Entry-point group | Resolves | Selected via |
|---|---|---|
| `inference_server.gateway` | Alternative `resolve_gateway()` targets | `INFERENCE_GATEWAY` env var |
| `inference_model_manager.backends` | Alternative `ModelManager.load(backend=...)` implementations | `backend=` kwarg |
| `inference_model_manager.decoders` | Alternative image decoders | decoder name |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | HTTP bind address (`__main__` dev runner) |
| `PORT` | `9001` | HTTP port (`__main__` dev runner) |
| `NUM_WORKERS` | `1` | uvicorn worker processes (`__main__` dev runner) |
| `INFERENCE_GATEWAY` | `direct` | Gateway resolved by `gateway_resolver.resolve_gateway()` |
| `INFERENCE_PRELOAD_MODELS` | | Comma-separated model IDs loaded unpinned at server startup, two at a time; an entry may carry its own key as `model_id:api_key`. `/readiness` and `/v2/server/ready` report not-ready until the startup loads have finished, whether or not these unpinned loads succeeded |
| `PINNED_MODELS` | | Comma-separated model IDs loaded at startup like `INFERENCE_PRELOAD_MODELS` but pinned against eviction; same `model_id:api_key` form; a failed pinned load is retried every 15 s within `PINNED_PRELOAD_MAX_WAIT_S` while both readiness routes stay not-ready |
| `PINNED_PRELOAD_MAX_WAIT_S` | `300.0` | Hard cap in seconds on how long startup waits for each `PINNED_MODELS` load, retries included; every attempt's wait is cut to the remaining budget. Afterwards the failure is logged and the server reports ready without that model, and a later on-demand load of it is still pinned; a load still running in the model manager continues in the background and the model is served once it finishes. Keep it below the deployment's startup probe window |
| `PRELOAD_HF_IDS` | | Comma-separated OWLv2 Hugging Face IDs loaded unpinned at startup as `owlv2/<name>` (the part after the last `/`); readiness does not wait for them |
| `INFERENCE_DECODER` | `imagecodecs` | Image decoder of the serving workers. `imagecodecs` is CPU decode and matches the legacy server's predictions; `nvjpeg` is opt-in GPU decode and currently changes predictions relative to the legacy server, so parity-gated deployments must not set it |
| `INFERENCE_LOAD_WAIT_S` | `10.0` | Seconds `ensure_loaded()` waits before reporting a load timeout |
| `INFERENCE_INFER_TIMEOUT_S` | `30.0` (with `VLLM_PROXY_ENABLED`: `max(30.0, VLLM_REQUEST_TIMEOUT_S)`) | Per-request inference timeout |
| `INFERENCE_MAX_BODY_BYTES` | `100MB` | Max request body / aggregate URL-image size |
| `INFERENCE_MAX_IMAGES_PER_REQUEST` | `32` | Max images per request (body, multipart, or `?image=<url>` params) |
| `API_BASE_URL` | `https://api.roboflow.com` | Roboflow API for auth |
| `ENABLE_CONTROL_PLANE_ROUTES` | `false` | Enables model list/load/unload and server info/metrics routes; they accept any valid key without workspace scoping |
| `INFERENCE_MAX_ACTIVE_MODELS` | `8` | Max concurrently loaded models (direct gateway); LRU drain-unload on overflow, `<=0` unbounded |
| `INFERENCE_MEMORY_FREE_THRESHOLD` | `0` | Free-VRAM fraction below which loads evict LRU models first; `0` off |
| `PRELOAD_API_KEY` | `ROBOFLOW_API_KEY` | API key for `INFERENCE_PRELOAD_MODELS`, `PINNED_MODELS` and `PRELOAD_HF_IDS` startup loads when an entry carries no key of its own |
| `MAX_INFERENCE_MODELS_CACHE_SIZE_MB` | `-1` | Disk cache watchdog size cap; `-1` off |
| `INFERENCE_MODELS_CACHE_WATCHDOG_INTERVAL_MINUTES` | `60` | Disk cache watchdog interval |
| `ENABLE_CUDA_MEMORY_RECLAMATION_WATCHDOG` | `false` | Periodic CUDA cache reclamation daemon |
| `CUDA_MEMORY_RECLAMATION_WATCHDOG_INTERVAL_SECONDS` | `300` | Reclamation interval |

## Legacy and Workflows routes

`inference-server` also serves the legacy `roboflow-inference-server` HTTP
surface (`/model/*`, `/infer/*`, `/{workspace}/{model}` catch-all, and, when
the optional `workflows` extra is installed, `/workflows/*`). Install it with:

```bash
uv pip install -e ".[torch-cpu,onnx-cpu,workflows]"
```

`workflows` pulls in `roboflow-workflows`. Without it, `inference-server`
still runs (the legacy inference routes work), but the `/workflows/*` routes
are dropped at startup instead of registered.

| Variable | Default | Description |
|----------|---------|-------------|
| `LEGACY_ROUTES_ENABLED` | `true` | Enables the legacy model routes; workflows routes depend only on the installed `workflows` extra and `DISABLE_WORKFLOW_ENDPOINTS` |
| `LEGACY_ROUTE_ENABLED` | `true` | Enables the legacy catch-all model-inference route |
| `LEGACY_CONTROL_PLANE_ROUTES_ENABLED` | `true` | Enables `/model/*`, `/clear_cache`, `/start/*`; like the legacy server, these accept any caller with no per-workspace scoping — set to `false` on multi-tenant deployments |
| `DISABLE_WORKFLOW_ENDPOINTS` | `false` | Drops the `/workflows/*` routes even when `roboflow-workflows` is installed |
| `INFERENCE_LEGACY_LOAD_TIMEOUT_S` | `300.0` | Seconds a legacy route waits for a model to finish loading before returning 503 |
| `OFFLINE_MODE` | `false` | Skip the Roboflow registry auth/stat call and derive task type from the loaded model instead |
| `ALLOW_URL_INPUT` | `true` | Allow images to be fetched from a URL |
| `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM` | `false` | Allow images to be loaded from a local path |
| `ALLOW_ORIGINS` | `*` | Comma-separated CORS origins |
| `CORRELATION_ID_HEADER` | `X-Request-ID` | Request header read as the correlation id and echoed on every response; generated when absent. Honoured only with `API_LOGGING_ENABLED=true`; otherwise the header is always `X-Request-ID`, like the legacy server's library defaults |
| `API_LOGGING_ENABLED` | `false` | When `true`, the correlation id comes from `CORRELATION_ID_HEADER` and any non-empty value is accepted, and application logs are one JSON object per line carrying `CORRELATION_ID_LOG_KEY`, `execution_id`, `trace_id` and `span_id` when present; when `false`, it comes from `X-Request-ID`, a value that is not a UUID is replaced, and logs are plain lines |
| `LOG_LEVEL` | `WARNING` | Level of the `inference_server` logger; uvicorn's own loggers keep uvicorn's defaults |
| `STRUCTURED_API_LOGGING` | `false` | With `API_LOGGING_ENABLED=true`, replaces uvicorn's access line with one JSON object per response logged at `INFO` through the application logger (`method`, `path`, `status`, `duration_ms`, the correlation key and the legacy `processing_time`, `model_*`, `workflow_id`, `workspace_id`, `execution_id` and `trace_id` fields read from the response headers); health paths log at `DEBUG` |
| `CORRELATION_ID_LOG_KEY` | `request_id` | JSON log field carrying the correlation id with `API_LOGGING_ENABLED=true` |
| `OTEL_TRACING_ENABLED` | `false` | OpenTelemetry tracing: FastAPI and `requests` instrumentation, OTLP span export, `X-Trace-Id` on every response and `X-Force-Trace: true` to sample one request. Needs the `otel` extra; forced off under `OFFLINE_MODE` |
| `OTEL_SERVICE_NAME` | `inference-server` | `service.name` resource attribute; `service.instance.id` is `INFERENCE_SERVER_ID` or a per-process id |
| `OTEL_EXPORTER_PROTOCOL` | `grpc` | `grpc` or `http` OTLP transport |
| `OTEL_EXPORTER_ENDPOINT` | `localhost:4317` | OTLP collector `host:port` for spans |
| `OTEL_SAMPLING_RATE` | `1.0` | Root span sampling ratio; parent decisions are honoured |
| `OTEL_TRACE_EXPORT_INTERVAL_MS` | `5000` | Span batch export interval |
| `OTEL_METRICS_ENABLED` | `true` | With tracing enabled, exports the model load, inference, API-call and error metrics over OTLP; forced off under `OFFLINE_MODE` |
| `OTEL_METRIC_EXPORTER_ENDPOINT` | `OTEL_EXPORTER_ENDPOINT` | OTLP collector `host:port` for metrics when it differs from the span collector |
| `OTEL_METRIC_EXPORT_INTERVAL_MS` | `10000` | Metric export interval |
| `HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_WORKERS` | `16` | Thread-pool size backing workflow execution |
| `WORKFLOWS_MAX_CONCURRENT_STEPS` | `8` | Max concurrent steps per workflow run |
| `LANDING_DIR` | `<checkout>/inference/landing/out` | Directory of the exported legacy landing page served at `/`; the Docker images set it to `/app/landing` |
| `ENABLE_DASHBOARD` | `false` | Serves `/dashboard.html` instead of 404 |
| `ENABLE_BUILDER` | `false` | Mounts the Workflow Builder at `/build` |
| `BUILDER_ORIGIN` | `https://app.roboflow.com`, or `https://app.roboflow.one` when `PROJECT=roboflow-staging` | Origin allowed to call `/build/api/*` and `/workflows/*` from the browser |

`/` serves the legacy landing page from `LANDING_DIR`; its dashboard tab calls
`/metrics`, `/logs`, and `/inference_pipelines`. `/logs` answers with the recent
log records when `ENABLE_IN_MEMORY_LOGS=true` and 404 otherwise; `/metrics` is
not ported here and answers 404; `/inference_pipelines/*` is served when
`ENABLE_STREAM_API=true` (see below) and answers 404 otherwise.

`/build` serves the Workflow Builder; local workflows are stored under
`MODEL_CACHE_DIR/workflow/local` and run through `/workflows/run` with
workspace `local`; the model picker lists models loaded in the model manager,
models present in the `inference_models` cache (`INFERENCE_HOME/models-cache`),
and cached foundation models.

Authentication differs by surface: `/v2/*` keeps Bearer-token auth plus
`ENABLE_CONTROL_PLANE_ROUTES` gating. Legacy routes instead take an API key
from, in order, the `api_key` query parameter, the `Authorization: Bearer`
header, the request body, then the `API_KEY` / `ROBOFLOW_API_KEY` env vars,
and are authorised per model by the Roboflow registry rather than by a single
workspace check.

A few legacy behaviours are not (yet) available here:

- Inference pipelines (`/inference_pipelines/*`) are served when
  `ENABLE_STREAM_API=true`: the lifespan starts one `streamvision` stream
  manager process per uvicorn worker (`STREAM_API_PRELOADED_PROCESSES` idle
  pipeline processes, reached through `STREAM_MANAGER_HOST` /
  `STREAM_MANAGER_PORT` / `STREAM_MANAGER_OPERATIONS_TIMEOUT`) and stops it on
  shutdown; with `NUM_WORKERS` above 1 only the first manager can bind the port.
  The WebRTC worker routes (`/initialise_webrtc_worker`, `/webrtc/session/*`)
  are not ported; requests to these paths 404.
- Several routes and parameters that legacy accepted now return 501 instead
  of the real behaviour: `/sam3_3d/infer`; `/sam3/embed_image` with
  `SAM3_EXEC_MODE=remote` (concept and visual segmentation are proxied to the
  Roboflow API); `format=binary` on the SAM/SAM2/SAM3 segmentation routes is not implemented (embedding routes still return binary).
- Prediction visualization (`format=image`, `visualize_predictions`) uses the
  class colours the model manager reports, else the legacy default palette; the
  per-model colour mapping is not fetched from the Roboflow API.
- Usage reporting is ported for the legacy model routes (`/infer/*`, the core
  model routes and `/{dataset_id}/{version_id}`) and the workflow run routes
  (`/workflows/run`, `/infer/workflows`, `/{workspace_name}/workflows/{workflow_id}`,
  `/infer/workflows/{workspace_name}/{workflow_id}`): one `request` row per
  request, carrying the models it invoked and, for workflows, the steps and the
  custom Python blocks it ran, is posted to `TELEMETRY_API_USAGE_ENDPOINT_URL`;
  the per-model, `workflows` and `workflow_block` rows of the legacy server are
  not sent, and stream routes report nothing yet. Other
  telemetry side effects of the legacy server are not ported. Active learning is ported for
  `/infer/object_detection`, `/infer/instance_segmentation`,
  `/infer/classification` and `/{dataset_id}/{version_id}`; it needs the
  `workflows` extra and `ACTIVE_LEARNING_ENABLED=false` turns it off. The periodic pingback to
  `METRICS_URL` is ported for inferences served through the legacy routes and
  Workflows steps; `METRICS_ENABLED=false` turns it off.

## SAM3 video tracking routes

`/sam3/video/embed`, `/sam3/video/segment`, `/sam3/video/propagate` and
`/sam3/video/propagate/stream` track
point-prompted objects over a whole clip with the `sam3trackervideo` model
(`SAM3TrackerVideo`), offline rather than frame by frame:

1. `embed` decodes a clip (every decoded frame is a frame, numbered from 0) and
   keeps it in the worker under the caller's `video_id`, scoped to the API key.
   `encode_frames` starts encoding every frame in the background; it pauses
   while a `segment` or `propagate` call is waiting or running, so a call
   made while a clip is still being encoded is not held until the end.
   `keep_existing` keeps the clip the worker already has under that id (same
   number of frames and frame size) with the frames it has encoded, instead of
   replacing it: use it when an id always names the same clip, so asking again
   costs no encoding.
2. `segment` returns the masks on one frame of the objects clicked there: the
   mask `propagate` would give each on that frame from the same clicks. Use it
   for click previews. On an object's anchor (see below) it is one model call;
   on another frame the object is tracked from its anchor to that frame, about
   0.063 s per frame of distance on a 24 GB GPU once the frames are encoded
   (a frame with exclude points alone is empty and needs no tracking).
3. `propagate` tracks every clicked object over the clip: each object on its
   own, from its anchor forward to the end and then backward to the start, so
   clicks of different objects on the same frame never interfere.
   `frame_indices` limits which frames come back.
4. `propagate/stream` tracks the same way and answers as NDJSON: an `object`
   line with each object's masks as soon as it is tracked, a `heartbeat` line
   after every 15 s without another, and a last line that is `done`, or `error`
   with the status and body the call would have failed with. Use it behind a
   proxy that ends quiet responses, often after 60 to 100 s.

Clicks and what they do:

- An object's **anchor** is its first frame with an include point (`label` 1).
  It is segmented from its own clicks alone, and the object is tracked from
  there. Every object needs one: a request with an object that has exclude
  points only answers 400 `Objects [3] have no include point; every object
  needs one.`
- Clicks on any other frame are a **correction**. They are applied when
  tracking reaches that frame, on top of what the tracker remembers of the
  object, and the corrected frame then conditions the frames tracked after it
  (later frames for a click after the anchor, earlier frames for a click
  before it). The frames between the anchor and the click, and the frames on
  the other side of the anchor, keep the masks they have without the click.
- Exclude points alone (`label` 0) on a correction frame mark the object
  **absent** from that frame, for the case where the object has left the
  picture and the mask moved to something else. The mask on that frame is
  empty wherever the points are, and the frames tracked after it start from
  "not here". To trim a mask where the object is still in the picture, add an
  include point on the object on the same frame.
- Marking an object absent does not say it is gone for good: the tracker
  picks it up when it sees it again, and may pick up something else. A client
  that knows the object has left should drop the masks after that frame
  itself.
- The tracker attends to at most four conditioning frames, the ones nearest
  the frame it is tracking, so with more than three corrections on one side
  of the anchor the farthest ones stop being read there.

The HTTP server of the `inference` package exposes the same four routes with
the same contract, behind `SAM3_VIDEO_ENABLED=true` (off by default there). It
loads the tracker once per worker process on the first call and keeps it, so
it too needs one worker process (`NUM_WORKERS=1`) and sticky routing per clip.

Clips come in and decode through the same video input as
`/infer/action_recognition`: URL or base64, the same URL policy and
`MAX_VIDEO_DOWNLOAD_SIZE_MB`, a header probe, then one sequential decode.

Every segment and propagate request carries all the clicks; the server keeps
only the clip (preprocessed frames and per-frame encoder output). A call on a
clip that was never embedded or was evicted answers 400 `No video embedded`.
The clip lives in one worker's memory, so a deployment with several workers
needs the calls of one clip routed to the same worker.

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM3_VIDEO_MODEL_ID` | `sam3trackervideo` | Model the routes use when the request names none |
| `SAM3_VIDEO_CACHE_MAX_GB` | `16` | Host RAM, in GB, the cached clips of a worker may hold (fractions allowed). It sets how many frames the worker keeps over all clips and so the longest clip `embed` accepts; a longer one is refused with 413 before it is decoded. The least recently used clips are evicted to make room |

Cached clips live in host memory, not on the GPU, and
`SAM3_VIDEO_CACHE_MAX_GB` is the one setting that bounds them. The tracker
turns it into a number of frames, `floor(GB × 1e9 / bytes per cached frame)`,
and logs the result when the model loads. A cached frame is held as its
preprocessed pixels (6,096,384 bytes in bfloat16) until it is encoded
(`encode_frames`, or the first call that runs on it) and as its encoder output
(10,616,832 bytes) afterwards, never both; the larger figure is the bytes per
cached frame. Both figures are computed from the model's shapes and dtype, and
a unit test checks the encoder output figure against a really encoded frame.

| `SAM3_VIDEO_CACHE_MAX_GB` | Frames kept (computed) | Video at 6 fps | Video at 12 fps |
|---|---|---|---|
| `4` | 376 | 62 s | 31 s |
| `8` | 753 | 125 s | 62 s |
| `16` (default) | 1507 | 251 s | 125 s |
| `32` | 3014 | 502 s | 251 s |

The limit covers the cached frames only. Size the host as cache GB, plus the
memory an embed needs while it runs, plus the process itself:

- Embedding a clip forgets the clips it replaces before it builds the new one,
  so the cached frames stay within the limit during an embed. What an embed
  adds is the decoded clip at its native resolution, held from decoding until
  the embed returns (height × width × 3 bytes per frame of the clip being
  embedded: 1.8 MB at 1024×576, 6.2 MB at 1080p, computed), plus about 0.65 GB
  while frames are preprocessed, whatever the clip's length (16 frames at a
  time at about 40 MB each, computed from a measured 40 MB per frame). There is
  no separate limit on a clip's length, so the first part grows with the
  longest clip the cache admits. Embeds running at the same time each add
  their own.
- The process with the model loaded and one 60-frame clip took 1.1 to 2.1 GB
  of host RAM (measured; GPU memory peaked at 3.3 GB).

Measured on a 24 GB GPU with `SAM3_VIDEO_CACHE_MAX_GB=10.2` (960 frames), as
resident memory of the server; clips are 1024×576, encoded in the background.
"Before" is the first version of these routes, which kept each frame's pixels
and encoder output, limited the cache to two clips of up to 480 frames, and
preprocessed a clip in one call:

| State | Before | Now | Now, with `MALLOC_MMAP_THRESHOLD_=1048576` |
|---|---|---|---|
| Model loaded, one 60-frame clip not encoded | 1.8 to 2.0 GB | 2.1 GB | 1.1 GB |
| One 480-frame clip encoded, one 60-frame clip not | 12.5 GB | 9.2 GB | 6.6 GB |
| Cache full: two 480-frame clips | 21.4 to 22.3 GB | 14.0 to 15.1 GB | 11.4 GB |
| Cache full: sixteen 60-frame clips | | 15.8 GB | 11.6 GB |
| Cache full: one 960-frame clip | | 18.3 GB | 11.6 GB |
| Peak while a third 480-frame clip is embedded | 39.1 GB | 14.8 to 16.0 GB | 12.3 GB |
| Peak of one 270-frame embed above the level before it | 8.4 to 9.3 GB | 0.4 GB | |
| Peak of two 270-frame embeds at once above the level before | 18.0 GB | 2.1 GB | |

A full cache of 960 frames holds 10.2 GB by the computed figure (960 ×
10,616,832 bytes). What the "Now" column shows above that and the loaded
model is memory the C allocator keeps after decoded frames and other
short-lived buffers are freed: with the allocator's defaults it grew with the
embeds and encoding threads the worker had seen, up to about 6.4 GB. Setting
`MALLOC_MMAP_THRESHOLD_=1048576` in the server's environment makes glibc
return freed buffers of 1 MB or more to the operating system. With it the
full cache measured 11.4 to 11.6 GB in every state, which is the 10.2 GB of
frames, the 0.7 GB the process took with the model loaded and no clip, and
0.5 to 0.7 GB more (up to 0.7 MB per cached frame above the computed figure).
It cost about 1.5 s on a 480-frame embed (10.2 to 11.0 s instead of 8.7 to
8.9 s) and about 1 s on tracking one object over 480 frames (31.4 s instead of
30.3 s). Without it, allow about 6.4 GB on top of the cache limit, the loaded
model and the embed's own memory.

A worker that runs out of memory is killed with every model it serves, so
leave headroom.

`propagate` answers once the whole clip is tracked, which for long clips or
many objects takes longer than the default `INFERENCE_INFER_TIMEOUT_S` of 30 s;
raise it on servers that serve these routes. The stream makes one model call per
object, so there the timeout bounds one object. The routes need
`SAM3_EXEC_MODE=local`.

## Running the legacy integration suite against this image

The black-box suite in `tests/inference/integration_tests/` reaches the server only through `BASE_URL` and `PORT`, so it runs unchanged against `inference_server`. In CI pick `server-image: new` when dispatching `INTEGRATION TESTS - Inference Server CPU x86` or `Code Quality & Regression Tests - NVIDIA T4`. Locally, from the repository root:

```bash
docker build -t roboflow/inference-server-experimental:test -f inference_server/docker/Dockerfile.cpu .
PORT=9101 USE_INFERENCE_MODELS=true INFERENCE_SERVER_REPO=inference-server-experimental make start_test_docker_cpu
export API_KEY=... asl_instance_segmentation_API_KEY=... asl_poly_instance_seg_API_KEY=... bccd_favz3_API_KEY=... \
  bccd_i4nym_API_KEY=... cats_and_dogs_smnpl_API_KEY=... coins_xaz9i_API_KEY=... melee_API_KEY=... yolonas_test_API_KEY=...
USE_INFERENCE_MODELS=true PORT=9101 SKIP_LMM_TEST=True \
  python -m pytest tests/inference/integration_tests --ignore=tests/inference/integration_tests/test_video_processing_endpoints.py
make stop_test_docker
```

The nine key variables are the ones the CI workflows pass (`tests/inference/integration_tests/README.md` explains the `<project_slug>_API_KEY` convention). `USE_INFERENCE_MODELS=true` on the client side selects the `*_inference_models.json` expectation files. `test_video_processing_endpoints.py` targets `/inference_pipelines/*`, which this server serves only with `ENABLE_STREAM_API=true`. Failures elsewhere are parity findings; do not regenerate the expectation files from this server.
