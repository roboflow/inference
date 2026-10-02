# Legacy environment audit

Every environment variable name the legacy `inference` server reads, classified
against the names the new packages read. The mapping that results lives in one
module, `inference_server/inference_server/legacy_env.py`: a derived-defaults
step for the region-dependent legacy defaults (`ROBOFLOW_SERVICE_URLS`,
`DERIVED_URL_DEFAULTS`), `LEGACY_ENV_ALIASES` (legacy name -> new name),
`LEGACY_FLAG_ALIASES` (legacy boolean name -> new name, propagated only when
true) and `LEGACY_DEFAULTS` (new-stack name -> legacy default), applied in that
order. `apply_legacy_env()` writes `os.environ` only and runs before any
configuration module of `inference_server`, `inference_model_manager`,
`inference_models`, `roboflow_workflows` or `streamvision` is imported. Explicit
values always win.

Categories:

- **(a) shared** - the new stack reads the name under the same spelling with the
  same default. Nothing to do; the name stays where it is.
- **(b) alias** - the new stack reads the setting under another name. One row in
  `LEGACY_ENV_ALIASES`.
- **(c) default** - same spelling, different default. One row in `LEGACY_DEFAULTS`
  carrying the `inference/core/env.py` default, so self-hosted defaults equal
  legacy defaults. Hosted pools override through infra env, never through the
  module.
- **(d) legacy-only** - no new-stack behaviour reads it. Listed with a reason;
  never aliased, never defaulted.

## Method

Legacy side (all names as string literals):

```
grep -oh '"[A-Z][A-Z_0-9]*"' \
  inference/core/env.py \
  inference/models/vllm_proxy/config.py \
  inference/usage_tracking/config.py \
  inference/core/interfaces/streams_configuration.py \
  inference/core/interfaces/workflows_configuration.py \
  docker/config/cpu_http.py docker/config/gpu_http.py | sort -u
```

382 literals. `inference/usage_tracking/config.py` declares its names through
pydantic-settings with `env_prefix="telemetry_"`, so its seven `TELEMETRY_*` names
are added by hand (they are not string literals).

New side (the five configuration modules named by the task):

```
grep -oh '"[A-Z][A-Z_0-9]*"' \
  inference_server/inference_server/configuration.py \
  inference_server/inference_server/workflows/host.py \
  inference_model_manager/inference_model_manager/configuration.py \
  inference_models/inference_models/configuration.py \
  stream_vision/streamvision/stream/configuration.py | sort -u
```

316 literals. A second sweep over every `os.getenv` / `os.environ` read in the
five packages (`inference_server`, `inference_model_manager`, `inference_models`,
`workflows/roboflow_workflows`, `stream_vision/streamvision`) catches names read
outside a configuration module; it moves six legacy names from "not read" to (a):
`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `YOLO_OFFLINE`
(`inference_models/_offline.py:140-142`), `MODAL_WEB_ENDPOINT_URL`,
`MODAL_WS_ENDPOINT_URL`
(`roboflow_workflows/execution_engine/v1/dynamic_blocks/modal_executor.py:544,1162,1166`)
and `WORKFLOWS_PLUGINS`
(`roboflow_workflows/execution_engine/introspection/blocks_loader.py:41,603`).

Literals that are not environment names and are excluded from every table:
`L2CS` (a model version string, `env.py:253`), `WARNING` / `INFO` (log-level
defaults), `O_NOFOLLOW` (an `os` flag in `host.py`), and the commented-out
`SAM3_BPE_PATH`, `SAM3_CHECKPOINT_PATH`, `SAM3_REPO_PATH` (`env.py:884-887`).

Default comparison for every shared name was done by reading both definitions;
the line references below are the evidence.

## (b) Alias rows - `LEGACY_ENV_ALIASES`

| legacy name | new name | legacy evidence | new evidence | why they are the same setting |
|---|---|---|---|---|
| `MAX_ACTIVE_MODELS` | `INFERENCE_MAX_ACTIVE_MODELS` | `env.py:708` (`int`, default 8); consumed as the `WithFixedSizeCache(max_size=...)` LRU cap in `docker/config/cpu_http.py:55`, `gpu_http.py:55` | `inference_model_manager/configuration.py:28-30` (default 8): max concurrently loaded models, LRU drain-unload | Same LRU cap on loaded models, same default. |
| `PRELOAD_MODELS` | `INFERENCE_PRELOAD_MODELS` | `env.py:1338-1340` (comma list, default unset); loaded at startup in `inference/core/interfaces/http/http_api.py:2976-2990` | `inference_server/configuration.py:132,147-160` (`preload_model_ids()`, comma list); loaded unpinned in `app.py:66-99,132-142` | Same comma-separated startup preload list. Legacy also merges `PINNED_MODELS` into the preload set (`http_api.py:3021`); the new server does the same, see (a). |
| `MEMORY_FREE_THRESHOLD` | `INFERENCE_MEMORY_FREE_THRESHOLD` | `env.py:1425-1427` (`float`, default 0.0, 0 disables); compared as `free_memory / total_memory < MEMORY_FREE_THRESHOLD` in `inference/core/managers/decorators/fixed_size_cache.py:296,305` | `inference_model_manager/configuration.py:34-36` (default 0.0, 0 disables); compared as `free / total < threshold` in `model_manager.py:38-52` | Same fraction-of-free-GPU-memory threshold that triggers eviction before a load, same units, same default. |
| `API_KEY` | `ROBOFLOW_API_KEY` | `env.py:222-223`: `API_KEY = os.getenv("ROBOFLOW_API_KEY") or os.getenv("API_KEY")`, so `API_KEY` is a legacy synonym of `ROBOFLOW_API_KEY` with lower precedence | `inference_server/configuration.py:188-190` and `workflows/host.py:359` already read both; `inference_models/configuration.py:48` reads only `ROBOFLOW_API_KEY`, and `inference_models/models/auto_loaders/core.py:226-230` / `weights_providers/roboflow.py:241` fall back to it when a load carries no key | The alias copies `API_KEY` into `ROBOFLOW_API_KEY` only when the latter is unset, which is exactly the legacy precedence, and lets `inference_models` see the key. `inference_server` behaviour is unchanged (both spellings were already read). |
| `LEGACY_MMP_LOAD_WAIT_S` | `INFERENCE_LEGACY_LOAD_TIMEOUT_S` | `env.py:449-451` (default 600): "legacy clients expect the request to block until the model is loaded" | `inference_server/configuration.py:161-163` (default 300.0); the legacy-route load deadline in `legacy/bridge.py:183` and the pinned load timeout in `bridge.py:207` | Same budget: how long a legacy route blocks waiting for a load. Defaults differ (600 vs 300) but the names differ too, so this is an alias row, not a default row. |
| `LEGACY_MMP_INFER_TIMEOUT_S` | `INFERENCE_INFER_TIMEOUT_S` | `env.py:453-454` (default 300): "inference wait budget for the MMP adapter bridge" | `inference_server/configuration.py:34` (default 30.0); the per-inference `asyncio.wait_for` timeout in `gateway.py:441`, also part of the legacy sync budget in `legacy/bridge.py:399` | Same budget: how long one inference may run before the request fails. Defaults differ (300 vs 30) but the names differ too. |

### Flag alias rows - `LEGACY_FLAG_ALIASES`

A flag alias propagates only when the legacy value parses as true under the
legacy `str2bool` set (`true`, `1`, `yes`, `y`, `t`, case-insensitive), and then
writes the literal string `True`. A false or unparseable legacy value leaves the
new name unset.

| legacy name | new name | legacy evidence | new evidence | why a flag alias |
|---|---|---|---|---|
| `RUNS_ON_JETSON` | `RUNNING_ON_JETSON` | `env.py:1247-1249`: `str2bool(os.getenv("RUNS_ON_JETSON", os.getenv("RUNNING_ON_JETSON", "False")))` | `inference_models/configuration.py:96`; parsed with `str2bool` in `runtime_introspection/core.py:198-199`, but tested by raw-string truthiness in `models/glm_ocr/glm_ocr_hf.py:32` and `models/cosmos3/cosmos3_reasoner_hf.py:64` | Two spellings of the same Jetson flag. A verbatim copy of `RUNS_ON_JETSON=False` would be truthy for the two raw-string consumers, so only a true value propagates. Shipped Jetson images already set `RUNNING_ON_JETSON=True` (`docker/dockerfiles/Dockerfile.onnx.jetson.6.2.0:862`, `7.2.0:630`, `5.1.1:181`). |

### Derived defaults - `ROBOFLOW_SERVICE_URLS` / `DERIVED_URL_DEFAULTS`

Legacy resolves these defaults from the region and environment
(`inference/core/utils/regions.py:33-56` matrix, `:60-72` region, `:75-90`
environment); the new packages read fixed defaults, so a staging or EU
deployment that relied on the derivation would silently point at the us/prod
hosts. The step runs first, before the alias rows, and only fills names that are
unset.

| name | inputs | derivation |
|---|---|---|
| `ROBOFLOW_ENVIRONMENT` | `PROJECT` | Set only when `ROBOFLOW_ENVIRONMENT` is unset and `PROJECT` is set: `prod` for `roboflow-platform`, `staging` for any other project (`regions.py:83-90`). `inference_models/configuration.py:59-83` reads it for `ROBOFLOW_API_HOST`. |
| `API_BASE_URL` | `ROBOFLOW_REGION`, `ROBOFLOW_ENVIRONMENT` / `PROJECT` | `api` column of the matrix below (`env.py:195-198`); read by `inference_server/configuration.py:75`. |
| `BUILDER_ORIGIN` | `ROBOFLOW_REGION`, `ROBOFLOW_ENVIRONMENT` / `PROJECT` | `app` column of the matrix below (`env.py:823-826`); read by `inference_server/configuration.py:117-124`. |
| `ROBOFLOW_API_HOST` | `API_BASE_URL` (explicit or derived above) | Set from `API_BASE_URL` when unset, without a log line. Legacy fetches model weights from `API_BASE_URL` (`inference/core/roboflow_api.py:720`); `inference_models` fetches them from `ROBOFLOW_API_HOST` (`weights_providers/roboflow.py:306`, default from `ROBOFLOW_REGION` x `ROBOFLOW_ENVIRONMENT` in `configuration.py:75-83`). An explicit `ROBOFLOW_API_HOST` wins. |

Region is `ROBOFLOW_REGION` stripped and lower-cased, default `us`; an unknown
region falls back to `us` (`regions.py:65-72`). Environment is
`ROBOFLOW_ENVIRONMENT` when set (`prod` only when it equals `prod`, else
`staging`), otherwise derived from `PROJECT`, otherwise `prod`.

| region | environment | `api` (`API_BASE_URL`) | `app` (`BUILDER_ORIGIN`) |
|---|---|---|---|
| `us` | `prod` | `https://api.roboflow.com` | `https://app.roboflow.com` |
| `us` | `staging` | `https://api.roboflow.one` | `https://app.roboflow.one` |
| `eu` | `prod` | `https://api.roboflow.eu` | `https://app.roboflow.eu` |
| `eu` | `staging` | `https://api.roboflow-eu.one` | `https://app.roboflow-eu.one` |

## (c) Default rows - `LEGACY_DEFAULTS`

| name | legacy default | new default | legacy evidence | new evidence | consumer in the new server |
|---|---|---|---|---|---|
| `ALLOW_URL_TO_NON_GLOBAL_ADDRESSES` | `True` | `False` | `env.py:81-87` (default True "to preserve the legacy behaviour") | `inference_server/configuration.py:66-71` | `framework/input_parsers/url_fetch.py:145` |
| `MAX_IMAGE_URL_REDIRECTS` | `30` | `3` | `env.py:77-80` ("30 mirrors the historical `requests` default") | `inference_server/configuration.py:72` | `framework/input_parsers/url_fetch.py:194` |
| `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM` | `True` | `False` | `env.py:1319-1321` | `inference_server/configuration.py:158-160` | `workflows/host.py:645` |

All three are permissive legacy defaults on the input boundary. The rows exist
because the plan requires self-hosted defaults to equal legacy defaults; hosted
pools that want the stricter value set it in infra env.

### Seed entries not carried

The task seed listed `INFERENCE_MAX_IMAGES_PER_REQUEST: "0"` and
`INFERENCE_INFER_TIMEOUT_S: "0"`. Neither name exists in `inference/core/env.py`,
so neither is a default row under the ruling (same spelling, different default).
`INFERENCE_MAX_IMAGES_PER_REQUEST=0` would be safe (`image_limits.py:20-21` treats
`limit > 0` as the only enforcing case) but removes a request-count guard legacy
never had, which is an infra-env decision. `INFERENCE_INFER_TIMEOUT_S=0` is not
"unbounded": `gateway.py:441` passes it straight to `asyncio.wait_for`, so every
inference would time out immediately. Legacy parity for these two is recorded
under "New-stack-only names" below.

## Infra table (rows (b) and (c))

Pool values come from the hosting summary (plan section 2.1). `unset` means the
summary gives no value for that pool. The summary lists the cpu-direct env as the
GPU-pool env plus additions, so cpu-direct inherits the GPU-pool values; the vLLM
pool summary lists only `VLLM_PROXY_ENABLED`, `VLLM_BASE_URL` and
`HTTP_API_THREADPOOL_WORKERS`, and the dedicated deployments list nothing.

| legacy name | new name | legacy default | new default | value on GPU pools | cpu-direct | vLLM pools | dedicated |
|---|---|---|---|---|---|---|---|
| `MAX_ACTIVE_MODELS` | `INFERENCE_MAX_ACTIVE_MODELS` | `8` | `8` | `70` | `70` | `unset` | `unset` |
| `PRELOAD_MODELS` | `INFERENCE_PRELOAD_MODELS` | unset | unset | set on the LMM pool only (value not in the summary) | `unset` | `unset` | `unset` |
| `MEMORY_FREE_THRESHOLD` | `INFERENCE_MEMORY_FREE_THRESHOLD` | `0.0` | `0.0` | `unset` | `unset` | `unset` | `unset` |
| `API_KEY` | `ROBOFLOW_API_KEY` | unset | unset | `unset` | `unset` | `unset` | `unset` |
| `API_BASE_URL` | same | region x environment matrix | `https://api.roboflow.com` (module derives the matrix value) | `unset` | `unset` | `unset` | `unset` |
| `BUILDER_ORIGIN` | same | region x environment matrix | `PROJECT`-only, us hosts (module derives the matrix value) | `unset` | `unset` | `unset` | `unset` |
| `ROBOFLOW_ENVIRONMENT` | same (derived from `PROJECT`) | `prod` unless `PROJECT` differs from `roboflow-platform` | `prod` (`inference_models/configuration.py:59`) | `unset` | `unset` | `unset` | `unset` |
| `ROBOFLOW_API_HOST` | same (filled from `API_BASE_URL`) | not a legacy name | region x environment matrix (`inference_models/configuration.py:75-83`) | `unset` | `unset` | `unset` | `unset` |
| `RUNS_ON_JETSON` | `RUNNING_ON_JETSON` (flag alias) | `False` | unset (treated as not Jetson) | `unset` | `unset` | `unset` | `unset` |
| `LEGACY_MMP_LOAD_WAIT_S` | `INFERENCE_LEGACY_LOAD_TIMEOUT_S` | `600` | `300.0` | `unset` | `unset` | `unset` | `unset` |
| `LEGACY_MMP_INFER_TIMEOUT_S` | `INFERENCE_INFER_TIMEOUT_S` | `300` | `30.0` | `unset` | `unset` | `unset` | `unset` |
| `ALLOW_URL_TO_NON_GLOBAL_ADDRESSES` | same | `True` | `False` (module ships `True`) | `unset` | `unset` | `unset` | `unset` |
| `MAX_IMAGE_URL_REDIRECTS` | same | `30` | `3` (module ships `30`) | `unset` | `unset` | `unset` | `unset` |
| `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM` | same | `True` | `False` (module ships `True`) | `unset` | `unset` | `unset` | `unset` |

`PINNED_MODELS` and `PRELOAD_HF_IDS` are read under the same spelling, see (a).
Hosted names from the summary that are legacy-only today and therefore need an
infra decision rather than a mapping: `HTTP_API_THREADPOOL_WORKERS` (vLLM pools
set 128), `GCP_SERVERLESS`,
`ENFORCE_CREDITS_VERIFICATION`, `MODELS_CACHE_AUTH_*`,
`METRICS_ENABLED`, `ENABLE_PROMETHEUS`, `REDIS_*`, `LOAD_ENTERPRISE_BLOCKS`, `WEBRTC_*`,
`VLLM_PROXY_ENABLED`. Each is in the (d) table.

## (a) Shared names

Read under the same spelling with the same default. `env.py:N` is the legacy
definition; the last column is the new reader.

### Read by `inference_server/configuration.py`

| name | legacy | new | default |
|---|---|---|---|
| `ALLOW_URL_INPUT` | `env.py:49` | `configuration.py:157` (also `inference_models/configuration.py:207`) | `True` |
| `WHITELISTED_DESTINATIONS_FOR_URL_INPUT` | `env.py:54` | `configuration.py:59` (also `inference_models/configuration.py:215`) | unset |
| `BLACKLISTED_DESTINATIONS_FOR_URL_INPUT` | `env.py:61` | `configuration.py:63` (also `inference_models/configuration.py:223`) | unset |
| `ALLOW_NON_HTTPS_URL_INPUT` | `env.py:50` | `configuration.py:73` (also `inference_models/configuration.py:208`); applied on legacy-compatible routes and workflow image inputs (`legacy/common.py`), not on `/v2` | `False` |
| `ALLOW_URL_INPUT_WITHOUT_FQDN` | `env.py:51` | `configuration.py:76` (also `inference_models/configuration.py:211`); applied on legacy-compatible routes and workflow image inputs (`legacy/common.py`), not on `/v2` | `False` |
| `VALIDATE_IMAGE_URL_REDIRECTS` | `env.py:74` | `configuration.py:79`; on legacy-compatible routes and workflow image inputs redirect targets are checked against the URL rules only when it is on (`legacy/common.py`); `/v2` checks every redirect target regardless | `False` |
| `ALLOW_ORIGINS` | `env.py:191` | `configuration.py:187` | `*` |
| `API_BASE_URL` | `env.py:195-198` | `configuration.py:75` | `https://api.roboflow.com` for the us/prod case; the module derives the other cases, see note 1 |
| `ROBOFLOW_API_EXTRA_HEADERS` | `env.py:216` | `configuration.py:268` | unset |
| `ROBOFLOW_API_KEY` | `env.py:222-223` | `configuration.py:143,189`; `inference_models/configuration.py:48` | unset |
| `API_KEY` | `env.py:222-223` | `configuration.py:189`; `host.py:359` | unset; also aliased for `inference_models`, see (b) |
| `CLIP_VERSION_ID` | `env.py:239` | `configuration.py:174`; `host.py:295` | `ViT-B-16` |
| `PERCEPTION_ENCODER_VERSION_ID` | `env.py:245` | `configuration.py:175` | `PE-Core-L14-336` |
| `OWLV2_VERSION_ID` | `env.py:259` | `configuration.py:184` | `owlv2-large-patch14-ensemble` |
| `CLIP_MAX_BATCH_SIZE` | `env.py:292` | `configuration.py:173` | `8` |
| `CLASS_AGNOSTIC_NMS` | `env.py:295-299` | `configuration.py:185`; `streamvision/stream/configuration.py:29,34` | `False` |
| `CORE_MODELS_ENABLED` | `env.py:307` | `configuration.py:195` | `True` |
| `CORE_MODEL_CLIP_ENABLED` | `env.py:310` | `configuration.py:196` | `True` |
| `CORE_MODEL_PE_ENABLED` | `env.py:313` | `configuration.py:197`; `host.py:302` | `True` |
| `CORE_MODEL_SAM_ENABLED` | `env.py:316` | `configuration.py:198` | `True` |
| `CORE_MODEL_SAM2_ENABLED` | `env.py:317` | `configuration.py:199`; `host.py:296` | `True` |
| `CORE_MODEL_SAM3_ENABLED` | `env.py:318` | `configuration.py:200`; `host.py:299` | `True` |
| `CORE_MODEL_OWLV2_ENABLED` | `env.py:320` | `configuration.py:201` | `False` |
| `SAM3_MAX_PROMPT_BATCH_SIZE` | `env.py:323` | `configuration.py:180` | `16` |
| `SAM3_EXEC_MODE` | `env.py:324-329` | `configuration.py:232`; `host.py:114` | `local` |
| `SAM3_FINE_TUNED_MODELS_ENABLED` | `env.py:333-336` | `configuration.py:233-235` | `True` unless `SAM3_EXEC_MODE=remote` |
| `CORE_MODEL_GAZE_ENABLED` | `env.py:343` | `configuration.py:204`; `host.py:305` | `True` |
| `CORE_MODEL_DOCTR_ENABLED` | `env.py:355` | `configuration.py:205` | `True` |
| `CORE_MODEL_EASYOCR_ENABLED` | `env.py:358` | `configuration.py:208` | `True` |
| `CORE_MODEL_TROCR_ENABLED` | `env.py:361` | `configuration.py:211` | `True` |
| `CORE_MODEL_PPOCR_ENABLED` | `env.py:364` | `configuration.py:214` | `True` |
| `CORE_MODEL_GROUNDINGDINO_ENABLED` | `env.py:367` | `configuration.py:217` | `True` |
| `LMM_ENABLED` | `env.py:371` | `configuration.py:223`; `host.py:291` | `False` |
| `DEPTH_ESTIMATION_ENABLED` | `env.py:383` | `configuration.py:225`; `host.py:318` | `True` |
| `ACTION_RECOGNITION_ENABLED` | `env.py:384` | `configuration.py:229` | `True` |
| `MOONDREAM2_ENABLED` | `env.py:388` | `configuration.py:224`; `host.py:317` | `True` |
| `SAM3_3D_OBJECTS_ENABLED` | `env.py:394` | `configuration.py:228`; `host.py:309` | `False` |
| `CORE_MODEL_YOLO_WORLD_ENABLED` | `env.py:399` | `configuration.py:220` | `True` |
| `GET_MODEL_REGISTRY_ENABLED` | `env.py:650` | `configuration.py:192` | `True` |
| `LEGACY_ROUTE_ENABLED` | `env.py:666` | `configuration.py:147` | `True` |
| `SECURE_GATEWAY` | `env.py:672-675` | `configuration.py:266`; `host.py:64`; `inference_models/configuration.py:86` | unset |
| `MODEL_CACHE_DIR` | `env.py:795` | `configuration.py:267`; `inference_models/configuration.py:102` | `/tmp/cache` |
| `ENABLE_BUILDER` | `env.py:817` | `configuration.py:116` | `False` |
| `BUILDER_ORIGIN` | `env.py:823-826` | `configuration.py:117-124` | `https://app.roboflow.com` for the us/prod case; the module derives the other cases, see note 1 |
| `ENABLE_DASHBOARD` | `env.py:841` | `configuration.py:111` | `False` |
| `NUM_WORKERS` | `env.py:844` | `configuration.py:99` | `1` |
| `API_LOGGING_ENABLED` | `env.py:653` | `configuration.py:113`; `middlewares/correlation_id.py`: selects `CORRELATION_ID_HEADER` and accepts any non-empty id when true, `X-Request-ID` and UUID-only ids when false, like the legacy correlation library defaults; `logging_config.py`: JSON application logs when true | `False` |
| `STRUCTURED_API_LOGGING` | `env.py:657` | `configuration.py:118`; `logging_config.py`: with `API_LOGGING_ENABLED`, the structured access log middleware replaces uvicorn's access line, health paths at `DEBUG`, like the legacy `structured_access_log` middleware | `False` |
| `CORRELATION_ID_LOG_KEY` | `env.py:663` | `configuration.py:119`; `logging_config.py`: JSON field of the correlation id | `request_id` |
| `CORRELATION_ID_HEADER` | `env.py:660` | `configuration.py:102`; `middlewares/correlation_id.py`, honoured only with `API_LOGGING_ENABLED=true` | `X-Request-ID` |
| `PORT` | `env.py:852` | `configuration.py:98`; `app.py:354` | `9001` |
| `SAM_VERSION_ID` | `env.py:882` | `configuration.py:178` | `vit_h` |
| `SAM2_VERSION_ID` | `env.py:883` | `configuration.py:179` | `hiera_large` |
| `DISABLE_SAM3_LOGITS_CACHE` | `env.py:896` | `configuration.py:243` | `False` |
| `EASYOCR_VERSION_ID` | `env.py:899` | `configuration.py:183` | `english_g2` |
| `INFERENCE_SERVER_ID` | `env.py:902` | `configuration.py:191` | unset |
| `DISABLE_WORKFLOW_ENDPOINTS` | `env.py:1034` | `configuration.py:153` | `False` |
| `WORKFLOWS_MAX_CONCURRENT_STEPS` | `env.py:1095` | `configuration.py:246` | `8` |
| `ENABLE_WORKFLOWS_PROFILING` | `env.py:1291` | `configuration.py:252` | `False` |
| `WORKFLOWS_PROFILER_BUFFER_SIZE` | `env.py:1292` | `configuration.py:255` | `64` |
| `WORKFLOWS_DEFINITION_CACHE_EXPIRY` | `env.py:1293` | `configuration.py:258` | `900` |
| `ALLOW_WORKFLOWS_FONTS_DOWNLOAD` | `env.py:1324` | `configuration.py:261` | `True` |
| `ROBOFLOW_INTERNAL_SERVICE_SECRET` | `env.py:1331` | `configuration.py:270` | unset |
| `ROBOFLOW_INTERNAL_SERVICE_NAME` | `env.py:1332` | `configuration.py:269` | unset |
| `PRELOAD_API_KEY` | `env.py:1346` | `configuration.py:86-88,144`; `app.py:146` | unset; see note 2 |
| `PINNED_MODELS` | `env.py:1351` | `configuration.py:133,163-174`; loaded pinned in `app.py:66-99,133-142` | unset |
| `PRELOAD_HF_IDS` | `env.py:281` | `configuration.py:134,177-185`; loaded as `owlv2/<name>` in `hf_preload.py:27-52`, started by `app.py:134,143-151` | unset |
| `CONFIDENCE_LOWER_BOUND_OOM_PREVENTION` | `env.py:1478` | `configuration.py:170` | `0.01` |
| `HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_WORKERS` | `env.py:1651` | `configuration.py:249` | `16` |
| `OFFLINE_MODE` | `env.py:509` (owned by `inference_models._offline`) | `configuration.py:156`; `host.py:63`; `inference_models/configuration.py:109` | `False` |
| `LOG_LEVEL` | `env.py:705` | `configuration.py:117`; `logging_config.py`: level of the `inference_server` logger, like legacy's `inference` logger; uvicorn's loggers keep uvicorn's defaults; `inference_models` reads it separately, see its table | `WARNING` |
| `OTEL_TRACING_ENABLED` | `env.py:768` | `configuration.py:232,241-243` (forced off under `OFFLINE_MODE`); `telemetry.py` | `False` |
| `OTEL_SERVICE_NAME` | `env.py:769` | `configuration.py:233`; `telemetry.py` | `inference-server` |
| `OTEL_EXPORTER_PROTOCOL` | `env.py:770` | `configuration.py:234`; `telemetry.py` | `grpc` |
| `OTEL_EXPORTER_ENDPOINT` | `env.py:771` | `configuration.py:235`; `telemetry.py` | `localhost:4317` |
| `OTEL_SAMPLING_RATE` | `env.py:772` | `configuration.py:236`; `telemetry.py` | `1.0` |
| `OTEL_TRACE_EXPORT_INTERVAL_MS` | `env.py:773` | `configuration.py:237-239`; `telemetry.py` | `5000` |
| `OTEL_METRICS_ENABLED` | `env.py:774` | `configuration.py:240,241-243` (forced off under `OFFLINE_MODE`); `telemetry.py` | `True` |
| `OTEL_METRIC_EXPORTER_ENDPOINT` | `env.py:778` | `configuration.py:244`; `telemetry.py` (falls back to `OTEL_EXPORTER_ENDPOINT`) | unset |
| `OTEL_METRIC_EXPORT_INTERVAL_MS` | `env.py:779` | `configuration.py:245-247`; `telemetry.py` | `10000` |

### Read by `build_workflows_configuration` (`inference_server/workflows/host.py`)

| name | legacy | new | default |
|---|---|---|---|
| `PROJECT` | `env.py:42` | `host.py:127`; `configuration.py:121` | `roboflow-platform` |
| `ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES` | `env.py:97` | `host.py:177` | `True` |
| `ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES` | `env.py:103` | `host.py:174` | `True` |
| `POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES` | `env.py:110` | `host.py:180` | unset |
| `POSTGRESQL_WORKFLOWS_SINK_WHITELISTED_ADDRESSES` | `env.py:122` | `host.py:183` | unset |
| `KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS` | `env.py:136` | `host.py:186` | `True` |
| `KAFKA_WORKFLOWS_SINKS_WHITELISTED_BOOTSTRAP_SERVERS` | `env.py:150` | `host.py:190` | unset |
| `COSMOS3_ENABLED` | `env.py:373` | `host.py:321` | `True` |
| `QWEN_2_5_ENABLED` | `env.py:375` | `host.py:313` | `True` |
| `QWEN_3_ENABLED` | `env.py:377` | `host.py:314` | `True` |
| `QWEN_3_5_ENABLED` | `env.py:379` | `host.py:315` | `True` |
| `SMOLVLM2_ENABLED` | `env.py:386` | `host.py:316` | `True` |
| `FLORENCE2_ENABLED` | `env.py:392` | `host.py:312` | `True` |
| `GLM_OCR_ENABLED` | `env.py:396` | `host.py:322` | `True` |
| `INFERENCE_DEBUG_OUTPUT_DIR` | `env.py:804` | `host.py:364` | unset |
| `LOCAL_INFERENCE_API_URL` | `env.py:992` | `host.py:211` | `http://127.0.0.1:9001` |
| `HOSTED_DETECT_URL` | `env.py:993-1000` | `host.py:214-221` | by `PROJECT`, identical matrix |
| `HOSTED_INSTANCE_SEGMENTATION_URL` | `env.py:1001-1008` | `host.py:230-237` | by `PROJECT`, identical matrix |
| `HOSTED_SEMANTIC_SEGMENTATION_URL` | `env.py:1009-1016` | `host.py:238-245` | by `PROJECT`, identical matrix |
| `HOSTED_CLASSIFICATION_URL` | `env.py:1017-1024` | `host.py:222-229` | by `PROJECT`, identical matrix |
| `HOSTED_CORE_MODEL_URL` | `env.py:1025-1032` | `host.py:246-253` | by `PROJECT`, identical matrix |
| `WORKFLOWS_STEP_EXECUTION_MODE` | `env.py:1040-1094` | `host.py:65-98` (same offline / gateway rewrites) | `local` |
| `WORKFLOWS_REMOTE_API_TARGET` | `env.py:1043` | `host.py:68` | `hosted` |
| `WORKFLOWS_REMOTE_API_KEY_TRANSPORT` | `env.py:1052-1067` | `host.py:69-77` | `both` |
| `WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH` | `env.py:1096` | `host.py:151` | `4` |
| `WORKFLOWS_MAX_INNER_WORKFLOW_COUNT` | `env.py:1099` | `host.py:154` | `32` |
| `WORKFLOWS_INNER_WORKFLOW_REMOTE_TARGET` | `env.py:1102` | `host.py:260` | `https://serverless.roboflow.com` |
| `WORKFLOWS_INNER_WORKFLOW_REMOTE_DISPATCH_REQUEST_TIMEOUT` | `env.py:1106` | `host.py:264` | `300.0` |
| `WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE` | `env.py:1109` | `host.py:254` | `1` |
| `WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS` | `env.py:1112` | `host.py:257` | `8` |
| `ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS` | `env.py:1115` | `host.py:157` | `True` |
| `WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE` | `env.py:1120-1130` | `host.py:99-113` | `local` |
| `WEBEXEC_JPEG_QUALITY` | `env.py:1135` | `host.py:340` | `95` |
| `WEBEXEC_TRANSPORT` | `env.py:1140` | `host.py:341` | `http` |
| `WEBEXEC_WS_CONNECT_TIMEOUT_SECONDS` | `env.py:1144` | `host.py:342` | `30` |
| `WEBEXEC_WS_READ_TIMEOUT_SECONDS` | `env.py:1153` | `host.py:345` | `720` |
| `WEBEXEC_WS_FAIL_ON_SESSION_LOSS` | `env.py:1175` | `host.py:351` | `False` |
| `WEBEXEC_WS_CONNECTION_POOL_SIZE` | `env.py:1179` | `host.py:348` | `1` |
| `WEBEXEC_WS_IDLE_RELEASE_SECONDS` | `env.py:1190` | `host.py:354` | `120` |
| `WEBEXEC_MODAL_EXECUTOR_IDLE_TTL_SECONDS` | `env.py:1193` | `host.py:337` | `1800` |
| `MODAL_TOKEN_ID` | `env.py:1199,1203` | `host.py:143,325` (same quote stripping) | unset |
| `MODAL_TOKEN_SECRET` | `env.py:1200,1204` | `host.py:144,326` | unset |
| `MODAL_WORKSPACE_NAME` | `env.py:1205` | `host.py:329` | `roboflow` |
| `WEBEXEC_MODAL_APP_NAME` | `env.py:1208` | `host.py:336` | `webexec-{PROJECT}` |
| `MODAL_ALLOW_ANONYMOUS_EXECUTION` | `env.py:1212` | `host.py:330` | `False` |
| `MODAL_ANONYMOUS_WORKSPACE_NAME` | `env.py:1216` | `host.py:333` | `anonymous` |
| `WORKFLOWS_ENFORCE_DENSE_INSTANCE_MASKS` | `env.py:1274` | `host.py:204` | `False` |
| `WORKFLOWS_VLM_SEGMENTATION_MAX_POLYGON_VERTICES` | `env.py:1285` | `host.py:292` | `500` |
| `ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE` | `env.py:1313` | `host.py:161` | `True` |
| `ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES` | `env.py:1316` | `host.py:164` | `True` |
| `WORKFLOW_BLOCKS_WRITE_DIRECTORY` | `env.py:1327` | `host.py:167` | unset |
| `ROBOFLOW_API_REQUEST_TIMEOUT` | `env.py:1415-1417` | `host.py:420-421` | `120` |
| `WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT` | `env.py:1473` | `host.py:148` | `60.0` |
| `OPENAI_COMPATIBLE_ALLOWED_BASE_URLS` | `env.py:1665-1670` | `host.py:268-277` | `*` |
| `WORKFLOW_DISABLED_BLOCK_TYPES` | `env.py:1674-1680` | `host.py:168` | empty |
| `WORKFLOW_DISABLED_BLOCK_PATTERNS` | `env.py:1683-1691` | `host.py:171` | empty |
| `ENABLE_TENSOR_DATA_REPRESENTATION` | `env.py:1728-1731` | `host.py:129` | `False`; see note 3 |
| `WORKFLOWS_TENSOR_VISUALISATION_VALIDATE_OWNERS` | `env.py:1762` | `host.py:200` | `False` |
| `WORKFLOWS_IMAGE_TENSOR_DEVICE` | `env.py:1766-1789` | `host.py:196-199` | unset (auto) |
| `WORKFLOWS_SAM_VIDEO_MASK_REPRESENTATION` | `env.py:1809-1818` | `host.py:132-142` | `rle` |

### Read by `inference_model_manager/configuration.py`

| name | legacy | new | default |
|---|---|---|---|
| `OWLV2_IMAGE_CACHE_SIZE` | `env.py:262` | `configuration.py:69` | `10000` |
| `OWLV2_MODEL_CACHE_SIZE` | `env.py:265` | `configuration.py:68` | `100` |
| `OWLV2_CACHE_SEND_TO_CPU` | `env.py:268` | `configuration.py:70` | `True` |
| `MAX_INFERENCE_MODELS_CACHE_SIZE_MB` | `env.py:429` | `configuration.py:75` | `-1` |
| `INFERENCE_MODELS_CACHE_WATCHDOG_INTERVAL_MINUTES` | `env.py:432` | `configuration.py:78` | `60` |
| `ENABLE_CUDA_MEMORY_RECLAMATION_WATCHDOG` | `env.py:439` | `configuration.py:81` | `False` |
| `CUDA_MEMORY_RECLAMATION_WATCHDOG_INTERVAL_SECONDS` | `env.py:442` | `configuration.py:84` | `300` |
| `SAM_MAX_EMBEDDING_CACHE_SIZE` | `env.py:875` | `configuration.py:50` | `10` |
| `SAM2_MAX_EMBEDDING_CACHE_SIZE` | `env.py:877` | `configuration.py:53` | `100` |
| `SAM2_MAX_LOGITS_CACHE_SIZE` | `env.py:878` | `configuration.py:56` | `1000` |
| `SAM3_MAX_EMBEDDING_CACHE_SIZE` | `env.py:888` | `configuration.py:59` | `100` |
| `SAM3_MAX_LOGITS_CACHE_SIZE` | `env.py:892` | `configuration.py:62` | `1000` |
| `SAM3_INTERACTIVE_CACHE_SEND_TO_CPU` | `env.py:893` | `configuration.py:65` | `True` |

### Read by `inference_models/configuration.py`

| name | legacy | new | default |
|---|---|---|---|
| `ROBOFLOW_REGION` | `inference/core/utils/regions.py:65` via `env.py:45` | `configuration.py:68` | `us` |
| `LICENSE_SERVER` | `env.py:672,676-682` | `configuration.py:84,89-95` | unset (deprecated alias of `SECURE_GATEWAY` on both sides) |
| `LOG_LEVEL` | `env.py:705` | `configuration.py:129` | `WARNING` |
| `HF_HUB_CACHE` | `env.py:796` (required) | `configuration.py:104` (required) | none |
| `INFERENCE_HOME` | `env.py:803` (`setdefault` to `MODEL_CACHE_DIR`) | `configuration.py:101-103` (falls back to `MODEL_CACHE_DIR`, then `/tmp/cache`) | `MODEL_CACHE_DIR` |
| `ONNXRUNTIME_EXECUTION_PROVIDERS` | `env.py:846-849` (bracketed list) | `configuration.py:18-25` (brackets stripped) | same four providers |
| `SAM3_IMAGE_SIZE` | `env.py:886` | `configuration.py:137` | `1008` |
| `RUNNING_ON_JETSON` | `env.py:1248` (fallback spelling) | `configuration.py:96` | unset |
| `RFDETR_ONNX_MAX_RESOLUTION` | `env.py:1466-1469` | `configuration.py:37-42` | `1600`, `<= 0` disables on both sides |
| `DISABLED_INFERENCE_MODELS_BACKENDS` | `env.py:1703-1726` (validates entries) | `configuration.py:29-33` (no validation) | empty |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `YOLO_OFFLINE` | written by `env.py:515-517` under `OFFLINE_MODE` | written by `inference_models/_offline.py:140-142` under the same condition | not read by either side |

### Read by `streamvision/stream/configuration.py` (`ModelConfigDefaults`)

| name | legacy | new | default |
|---|---|---|---|
| `CONFIDENCE` | `env.py:302-304` | `configuration.py:30,35` | `0.4` |
| `IOU_THRESHOLD` | `env.py:609-611` | `configuration.py:31,36` | `0.3` |
| `MAX_CANDIDATES` | `env.py:718-720` | `configuration.py:32,37` | `3000` |
| `MAX_DETECTIONS` | `env.py:723-725` | `configuration.py:33,38` | `300` |

### Read by `roboflow_workflows` directly

| name | legacy | new | default |
|---|---|---|---|
| `WORKFLOWS_PLUGINS` | `env.py:1369-1399` (read, then rewritten to prepend `inference.*` plugin modules) | `blocks_loader.py:41,603` | empty; see note 4 |
| `MODAL_WEB_ENDPOINT_URL` | `env.py:1206` | `modal_executor.py:544,1166` | empty |
| `MODAL_WS_ENDPOINT_URL` | `env.py:1207` | `modal_executor.py:1162` | empty |

### Shared by ruling, no new-package reader today

The plan names `VLLM_*` and `TELEMETRY_*` as shared names that stay where they
are. No new package reads them yet (the hosting summary confirms
`VLLM_PROXY_ENABLED` is unread); the spelling is reserved for the tasks that add
the vLLM proxy and usage tracking.

| name | legacy | default |
|---|---|---|
| `VLLM_PROXY_ENABLED` | `inference/models/vllm_proxy/config.py:24`; `env.py:519` | `False` |
| `VLLM_BASE_URL` | `vllm_proxy/config.py:49` | `http://127.0.0.1:8000` |
| `VLLM_REQUEST_TIMEOUT_S` | `vllm_proxy/config.py:54` | `120` |
| `VLLM_MAX_LORA_RANK` | `vllm_proxy/config.py:59` | `64` |
| `VLLM_MAX_REGISTERED_ADAPTERS` | `vllm_proxy/config.py:65` | `64` |
| `VLLM_VISION_LORA_NORM_THRESHOLD` | `vllm_proxy/config.py:74` | `0.0` |
| `VLLM_DORA_POLICY` | `vllm_proxy/config.py:81` | `reject` |
| `VLLM_SERVED_BASE_VARIANT` | `vllm_proxy/config.py:85` | `qwen3_5-0.8b` |
| `VLLM_SERVED_BASE_NAME` | `vllm_proxy/config.py:93` | the served base variant |
| `VLLM_ADAPTER_KEY_TEMPLATE` | `vllm_proxy/config.py:97` | `base_model.model.model.language_model.layers.{suffix}` |
| `TELEMETRY_API_USAGE_ENDPOINT_URL` | `usage_tracking/config.py:16` | `{METRICS_COLLECTOR_BASE_URL}/usage/inference` |
| `TELEMETRY_API_PLAN_ENDPOINT_URL` | `usage_tracking/config.py:17` | `{METRICS_COLLECTOR_BASE_URL}/usage/plan` |
| `TELEMETRY_API_PLAN_CACHE_TTL_SECONDS` | `usage_tracking/config.py:18` | `86400` |
| `TELEMETRY_WEBRTC_PLANS_ENDPOINT_URL` | `usage_tracking/config.py:19` | `{METRICS_COLLECTOR_BASE_URL}/webrtc_plans` |
| `TELEMETRY_FLUSH_INTERVAL` | `usage_tracking/config.py:20` | `10` |
| `TELEMETRY_USE_PERSISTENT_QUEUE` | `usage_tracking/config.py:21` | `True` |
| `TELEMETRY_QUEUE_SIZE` | `usage_tracking/config.py:22` | `10` |

### Notes on (a) rows whose defaults are equal only in the common case

1. `API_BASE_URL` and `BUILDER_ORIGIN`: legacy resolves the default from
   `ROBOFLOW_REGION` and `ROBOFLOW_ENVIRONMENT` / `PROJECT`
   (`inference/core/utils/regions.py:33-56,75-90`), so `PROJECT=roboflow-staging`
   alone yields `https://api.roboflow.one` / `https://app.roboflow.one`, and
   `ROBOFLOW_REGION=eu` yields the `.eu` hosts. `inference_server/configuration.py:75`
   fixes `API_BASE_URL` to `https://api.roboflow.com`; `configuration.py:117-124`
   derives `BUILDER_ORIGIN` from `PROJECT` only (no EU). `inference_models`
   derives `ROBOFLOW_API_HOST` from `ROBOFLOW_REGION` and `ROBOFLOW_ENVIRONMENT`
   (`configuration.py:59-83`) and does not consult `PROJECT`. The
   derived-defaults step of `apply_legacy_env()` reproduces the legacy
   derivation for all three names and fills `ROBOFLOW_API_HOST` from the
   resulting `API_BASE_URL` (see "Derived defaults" above), so a staging or EU
   deployment configured the legacy way
   (`ROBOFLOW_REGION`, `PROJECT`) resolves the same hosts as before.
2. `PRELOAD_API_KEY`: legacy falls back to `API_KEY` (`env.py:1346`, itself
   `ROBOFLOW_API_KEY or API_KEY`). The new server falls back to
   `ROBOFLOW_API_KEY` (`configuration.py:86-88`). With the
   `API_KEY -> ROBOFLOW_API_KEY` alias the fallback chain matches legacy.
3. `ENABLE_TENSOR_DATA_REPRESENTATION`: legacy ANDs the flag with
   `USE_INFERENCE_MODELS` (`env.py:1728-1731`); the new stack is
   `inference_models`-only, so the AND is always true there. Default equal.
4. `WORKFLOWS_PLUGINS`: legacy always prepends
   `inference.roboflow_workflows_plugin.loader` (and, with
   `LOAD_ENTERPRISE_BLOCKS`, `inference.enterprise.workflows.enterprise_blocks.loader`)
   to the operator's list (`env.py:1366-1399`). Those modules live in the legacy
   package and cannot be imported by the new stack; the Roboflow-platform blocks
   they provide are outside this task.

## (d) Legacy-only names

No new package reads these. They never get an alias or a default row.

| name | legacy | reason |
|---|---|---|
| `ACTIVE_LEARNING_ENABLED` | `env.py:946` | legacy active-learning model manager; the new stack has no active learning |
| `ACTIVE_LEARNING_TAGS` | `env.py:949` | legacy active-learning model manager |
| `ALLOW_API_KEY_FROM_HEADERS` | `env.py:229` | legacy toggle for the Bearer-header fallback; the new server has no such toggle |
| `ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES` | `env.py:413` | legacy adapter flag forwarded as a `from_pretrained` argument; no new package reads the env |
| `ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES` | `env.py:410` | legacy adapter flag forwarded as a `from_pretrained` argument; no new package reads the env |
| `ALLOW_NUMPY_INPUT` | `env.py:48` | legacy pickled-numpy input type; the new server has no numpy input |
| `ALLOW_OFFLINE_MODEL_CACHE_AUTH_BYPASS` | `env.py:732` | legacy models-cache auth; unread by any new package (hosting summary) |
| `ALLOW_UNSAFE_GSTREAMER_PIPELINES` | `env.py:1242` | `StreamsConfiguration` field; no new package reads the env name today, spelling reserved |
| `API_DEBUG` | `env.py:219` | legacy debug flag with no consumer in the new stack |
| `API_PROXY_BASE_URL` | `env.py:199` | legacy weights-proxy URL; unread (hosting summary) |
| `ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN` | `env.py:1335` | legacy assume-identity headers; unread (hosting summary) |
| `ATOMIC_CACHE_WRITES_ENABLED` | `env.py:207` | legacy artifact cache internals; `inference_models` has its own cache settings |
| `AWS_ACCESS_KEY_ID` | `env.py:232` | legacy AWS-era setting |
| `AWS_SECRET_ACCESS_KEY` | `env.py:235` | legacy AWS-era setting |
| `CACHE_METADATA_LOCK_TIMEOUT` | `env.py:1453` | legacy model cache lock; `inference_models` uses `INFERENCE_MODELS_FILE_LOCK_ACQUIRE_TIMEOUT` with different semantics |
| `CELERY_LOG_LEVEL` | `env.py:989` | legacy celery; none in the new stack |
| `CORE_MODEL_BUCKET` | `env.py:927` | legacy AWS-era setting |
| `DEBUG_AIORTC_QUEUES` | `env.py:978` | `StreamsConfiguration` field; no new package reads the env name today |
| `DEBUG_WEBRTC_PROCESSING_LATENCY` | `env.py:979` | `StreamsConfiguration` field; no new package reads the env name today |
| `DEDICATED_DEPLOYMENT_ID` | `env.py:1329` | legacy dedicated-deployment auth; the new `auth.py` validates keys against `API_BASE_URL` only |
| `DEDICATED_DEPLOYMENT_WORKSPACE_URL` | `env.py:1225` | legacy dedicated-deployment auth |
| `DEVICE` | `env.py:1223` | steers four legacy HF models only (`inference/models/qwen25vl`, `easy_ocr`, `doctr`, `sam3_3d`); `inference_models` `DEFAULT_DEVICE` (`configuration.py:43`) steers every model, so an alias would change models legacy never touched |
| `DEVICE_ID` | `env.py:472` | legacy device management; unread (hosting summary) |
| `DISABLE_GSTREAMER_VIDEO_SOURCES` | `env.py:1254` | `StreamsConfiguration` field; no new package reads the env name today |
| `DISABLE_INFERENCE_CACHE` | `env.py:480` | legacy inference-result cache; none in the new stack |
| `DISABLE_NATIVE_STDERR_CAPTURE` | `env.py:1265` | `StreamsConfiguration` field; no new package reads the env name today |
| `DISABLE_PREPROC_AUTO_ORIENT` | `env.py:493` | legacy ORT preprocessing; `inference_models` preprocesses per model |
| `DISABLE_PREPROC_CONTRAST` | `env.py:496` | legacy ORT preprocessing |
| `DISABLE_PREPROC_GRAYSCALE` | `env.py:499` | legacy ORT preprocessing |
| `DISABLE_PREPROC_STATIC_CROP` | `env.py:502` | legacy ORT preprocessing |
| `DISABLE_SAM2_LOGITS_CACHE` | `env.py:879` | legacy SAM2 cache toggle; only mentioned in field descriptions in `inference_server/legacy/entities.py:1261,1267`, never read |
| `DISABLE_VERSION_CHECK` | `env.py:546` | legacy GitHub version check; none in the new stack |
| `DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS` | `env.py:1037` | legacy `describe_workload` routes; not in the new server |
| `DISK_CACHE_CLEANUP` | `env.py:1424` | legacy artifact cache internals |
| `DOCKER_SOCKET_PATH` | `env.py:1289` | legacy docker introspection |
| `ELASTICACHE_ENDPOINT` | `env.py:551` | legacy AWS-era setting |
| `ENABLE_BYTE_TRACK` | `env.py:561` | legacy stream-mode setting |
| `ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING` | `env.py:974` | `StreamsConfiguration` field; no new package reads the env name today |
| `ENABLE_HTTPS` | `env.py:587` | legacy uvicorn TLS; the new entrypoints (Task 5.7) own TLS |
| `ENABLE_IN_MEMORY_LOGS` | `env.py:838` | legacy in-memory log buffer |
| `ENABLE_PROMETHEUS` | `env.py:563` | legacy metrics; unread (hosting summary) |
| `ENABLE_STREAM_API` | `env.py:1241` | legacy stream-manager process; unread (hosting summary) |
| `ENFORCE_CREDITS_VERIFICATION` | `env.py:642` | unread (hosting summary) |
| `ENFORCE_FPS` | `env.py:574` | legacy stream-mode setting |
| `FIX_BATCH_SIZE` | `env.py:580` | legacy ORT batch padding |
| `GAZE_MAX_BATCH_SIZE` | `env.py:286` | gaze was removed; legacy keeps a 410 stub |
| `GAZE_VERSION_ID` | `env.py:253` | gaze was removed |
| `GCP_SERVERLESS` | `env.py:626` | `host.py:283` hardcodes `False`; unread (hosting summary) |
| `HOST` | `env.py:583` | legacy bind host; `app.py:358` hardcodes `0.0.0.0`, which is the legacy default |
| `HOT_MODELS_QUEUE_LOCK_ACQUIRE_TIMEOUT` | `env.py:1455` | legacy model manager lock |
| `HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_ENABLED` | `env.py:1648` | the new server always uses the shared pool (`app.py:80-82`) |
| `HTTP_API_THREADPOOL_WORKERS` | `env.py:1657` | anyio thread-pool size; unread (hosting summary); vLLM pools set 128 |
| `HUGGINGFACE_TOKEN` | `env.py:1222` | legacy HF token plumbing; no new package reads it |
| `IGNORE_MODEL_DEPENDENCIES_WARNINGS` | `env.py:35` | legacy warning filter |
| `INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE` | `env.py:955` | `StreamsConfiguration` field; no new package reads the env name today |
| `INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY` | `env.py:958` | `StreamsConfiguration` field; no new package reads the env name today |
| `INFERENCE_WARNINGS_DISABLED` | `env.py:27` | legacy warning filter |
| `INFER_BUCKET` | `env.py:937` | legacy AWS-era setting |
| `INTERNAL_WEIGHTS_URL_SUFFIX` | `env.py:203` | legacy weights-proxy suffix; unread (hosting summary) |
| `IP_BROADCAST_ADDR` | `env.py:614` | legacy device-mode setting |
| `IP_BROADCAST_PORT` | `env.py:617` | legacy device-mode setting |
| `JSON_RESPONSE` | `env.py:620` | legacy device-mode setting |
| `LAMBDA` | `env.py:623` | `host.py:284` hardcodes `False`; unread (hosting summary) |
| `LEGACY_MMP_ADAPTER_BUNDLED_BACKEND` | `env.py:467` | adapter transport switch; the new server replaces the adapter |
| `LEGACY_MMP_ADAPTER_ENABLED` | `env.py:447` | adapter switch; the new server replaces the adapter |
| `LEGACY_MMP_ADAPTER_MODE` | `env.py:462` | adapter transport switch |
| `LOAD_ENTERPRISE_BLOCKS` | `env.py:1355` | expands to an `inference.*` plugin module; unread (hosting summary) |
| `MAX_BATCH_SIZE` | `env.py:711-715` | legacy ORT batch chunking / padding (`inference/core/models/roboflow.py:875`, `object_detection_base.py:235-247`); `inference_models` batches per model |
| `MAX_FPS` | `env.py:575` | legacy stream-mode setting |
| `MAX_VIDEO_DOWNLOAD_SIZE_MB` | `env.py:417` | legacy action-recognition video download; no new package reads it |
| `MAX_VIDEO_DURATION_SECONDS` | `env.py:427` | legacy action-recognition video download |
| `MD5_VERIFICATION_ENABLED` | `env.py:205` | legacy artifact cache internals |
| `MEMORY_CACHE_EXPIRE_INTERVAL` | `env.py:728` | legacy memory cache |
| `METLO_KEY` | `env.py:924` | legacy AWS-era setting |
| `METRICS_COLLECTOR_BASE_URL` | `env.py:210` | feeds the `TELEMETRY_*` defaults; usage tracking is a later task |
| `METRICS_ENABLED` | `env.py:784` | legacy metrics; unread (hosting summary) |
| `METRICS_INCLUDE_SOURCE_LABELS` | `env.py:569` | legacy Prometheus labels |
| `METRICS_INTERVAL` | `env.py:789` | legacy metrics |
| `METRICS_URL` | `env.py:792` | legacy metrics |
| `MODELS_CACHE_AUTH_CACHE_MAX_SIZE` | `env.py:763` | legacy models-cache auth; unread (hosting summary) |
| `MODELS_CACHE_AUTH_CACHE_TTL` | `env.py:760` | legacy models-cache auth; unread (hosting summary) |
| `MODELS_CACHE_AUTH_ENABLED` | `env.py:731` | legacy models-cache auth; unread (hosting summary) |
| `MODEL_ID` | `env.py:807` | legacy device-mode setting |
| `MODEL_LOCK_ACQUIRE_TIMEOUT` | `env.py:1454` | legacy model manager lock |
| `MODEL_MONITORING_CACHE_BACKEND` | `env.py:486` | legacy pingback cache selection |
| `MODEL_VALIDATION_DISABLED` | `env.py:1220` | legacy model validation |
| `MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST` | `env.py:167` | `roboflow_workflows/configuration.py:77` has the field but `build_workflows_configuration` does not read the env; the field default (`True`) equals the legacy default, so only an operator override is lost |
| `MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS` | `env.py:179` | `roboflow_workflows/configuration.py:78` has the field but `build_workflows_configuration` does not read the env; default (`None`) equals legacy |
| `NOTEBOOK_ENABLED` | `env.py:829` | legacy jupyter route |
| `NOTEBOOK_PASSWORD` | `env.py:832` | legacy jupyter route |
| `NOTEBOOK_PORT` | `env.py:835` | legacy jupyter route |
| `NUM_CELERY_WORKERS` | `env.py:988` | legacy celery |
| `NUM_PARALLEL_TASKS` | `env.py:952` | legacy async model manager |
| `ORT_TENSORRT_CACHE_PATH` | `env.py:918` (written, not read) | legacy ORT TensorRT cache; no new package reads it |
| `OWLV2_COMPILE_MODEL` | `env.py:274` | legacy OWLv2 implementation knob |
| `OWLV2_CPU_IMAGE_CACHE_SIZE` | `env.py:271` | legacy OWLv2 implementation knob |
| `PALIGEMMA_ENABLED` | `env.py:390` | legacy model-route gate; `roboflow_workflows` `ModelsConfiguration` has no such field |
| `PALIGEMMA_VERSION_ID` | `env.py:237` | legacy model-route setting |
| `PROFILE` | `env.py:855` | legacy profiler flag |
| `QWEN_3_8_ENABLED` | `env.py:381` | legacy model-route gate; no `ModelsConfiguration` field |
| `REDIS_HOST` | `env.py:858` | unread (hosting summary) |
| `REDIS_PORT` | `env.py:861` | unread (hosting summary) |
| `REDIS_SSL` | `env.py:862` | unread (hosting summary) |
| `REDIS_TIMEOUT` | `env.py:863` | unread (hosting summary) |
| `REQUIRED_ONNX_PROVIDERS` | `env.py:866` | legacy ORT provider assertion |
| `RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API` | `env.py:1406` | legacy API client; `inference_models` uses `API_CALLS_MAX_TRIES` / `IDEMPOTENT_API_REQUEST_CODES_TO_RETRY` with different semantics |
| `ROBOFLOW_API_VERIFY_SSL` | `env.py:1422` | legacy API client |
| `ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN` | `env.py:1333` | legacy assume-identity headers; unread (hosting summary) |
| `ROBOFLOW_SERVER_UUID` | `env.py:869` | the new server generates `SERVER_ID` per process (`configuration.py:275`) |
| `ROBOFLOW_SERVICE_SECRET` | `env.py:872` | unread (hosting summary) |
| `SAM3_MAX_DETECTIONS` | `env.py:891` | legacy `concept_segment` cap; no new package reads it |
| `SECURE_GATEWAY_HEALTH_CHECK_TIMEOUT` | `env.py:700` | legacy gateway health route; not in the new server |
| `SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED` | `env.py:693` | legacy gateway health route |
| `SINGLE_TENANT_WORKFLOW_CACHE` | `env.py:1299` | legacy definition cache mode; the new server has only the in-memory TTL cache |
| `SSL_CA_CERTS` | `env.py:606` | legacy uvicorn TLS |
| `SSL_CERTFILE` | `env.py:596` | legacy uvicorn TLS |
| `SSL_KEYFILE` | `env.py:599` | legacy uvicorn TLS |
| `SSL_KEYFILE_PASSWORD` | `env.py:602` | legacy uvicorn TLS |
| `STREAM_API_PRELOADED_PROCESSES` | `env.py:1245` | legacy stream-manager process; unread |
| `STREAM_ID` | `env.py:905` | legacy device-mode setting |
| `STREAM_MANAGER_MAX_ACTIVE_PIPELINES` | `env.py:1447` | `StreamsConfiguration` field; no new package reads the env name today |
| `STREAM_MANAGER_MAX_RAM_MB` | `env.py:1431` | `StreamsConfiguration` field; no new package reads the env name today |
| `STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE` | `env.py:1438` | `StreamsConfiguration` field; no new package reads the env name today |
| `STUB_CACHE_SIZE` | `env.py:953` | legacy stub model cache |
| `TAGS` | `env.py:912` | legacy device management |
| `TENSORRT_CACHE_PATH` | `env.py:915` | legacy ORT TensorRT cache |
| `TINY_CACHE` | `env.py:289` | legacy inference-result cache |
| `TRANSIENT_ROBOFLOW_API_ERRORS` | `env.py:1401` | legacy API client |
| `TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES` | `env.py:1409` | legacy API client |
| `TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL` | `env.py:1412` | legacy API client |
| `USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS` | `env.py:1296` | legacy definition file cache; not in the new server |
| `USE_INFERENCE_MODELS` | `env.py:407` | legacy adapter switch; the new stack is `inference_models`-only |
| `USE_PYTORCH_FOR_PREPROCESSING` | `env.py:475` | legacy ORT preprocessing |
| `VERSION_CHECK_MODE` | `env.py:921` | legacy version check |
| `VIDEO_DOWNLOAD_TIMEOUT_SECONDS` | `env.py:421` | legacy action-recognition video download |
| `VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE` | `env.py:1748` | `StreamsConfiguration` field; no new package reads the env name today |
| `VIDEO_SOURCE_ADAPTIVE_MODE_READER_PACE_TOLERANCE` | `env.py:965` | `StreamsConfiguration` field |
| `VIDEO_SOURCE_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE` | `env.py:962` | `StreamsConfiguration` field |
| `VIDEO_SOURCE_BUFFER_SIZE` | `env.py:1798` | `StreamsConfiguration` field |
| `VIDEO_SOURCE_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW` | `env.py:971` | `StreamsConfiguration` field |
| `VIDEO_SOURCE_MINIMUM_ADAPTIVE_MODE_SAMPLES` | `env.py:968` | `StreamsConfiguration` field |
| `WEBEXEC_INFERENCE_VERSION` | `env.py:1209` | legacy webexec version pin; no `ModalConfiguration` field |
| `WEBRTC_DATA_CHANNEL_ACK_WINDOW` | `env.py:1629` | legacy WebRTC; the new server has no WebRTC |
| `WEBRTC_DATA_CHANNEL_BUFFER_DRAINING_DELAY` | `env.py:1617` | legacy WebRTC |
| `WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT` | `env.py:1620` | legacy WebRTC |
| `WEBRTC_GZIP_PREVIEW_FRAME_COMPRESSION` | `env.py:1638` | legacy WebRTC |
| `WEBRTC_MJPEG_ALLOW_NON_GLOBAL_ADDRESSES` | `env.py:984` | legacy WebRTC |
| `WEBRTC_MODAL_APP_NAME` | `env.py:1504` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_ENFORCE_REGION` | `env.py:1585` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS` | `env.py:1535` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT` | `env.py:1521` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_FUNCTION_GPU` | `env.py:1525` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_FUNCTION_MAX_INPUTS` | `env.py:1527` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_FUNCTION_MAX_TIME_LIMIT` | `env.py:1516` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS` | `env.py:1532` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW` | `env.py:1539` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_FUNCTION_TIME_LIMIT` | `env.py:1512` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_GCP_SECRET_NAME` | `env.py:1551` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_IMAGE_NAME` | `env.py:1542` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_IMAGE_TAG` | `env.py:1545` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_MIN_CPU_CORES` | `env.py:1556` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_MIN_RAM_MB` | `env.py:1560` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_MODELS_PRELOAD_API_KEY` | `env.py:1552` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_PRELOAD_HF_IDS` | `env.py:1554` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_PRELOAD_MODELS` | `env.py:1553` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_PUBLIC_STUN_SERVERS` | `env.py:1563` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_REQUIRED_REGION` | `env.py:1588` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_RESPONSE_TIMEOUT` | `env.py:1508` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_ROBOFLOW_INTERNAL_SERVICE_NAME` | `env.py:1546` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_ROUTING_REGION` | `env.py:1591` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_RTSP_PLACEHOLDER` | `env.py:1549` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_RTSP_PLACEHOLDER_URL` | `env.py:1550` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_SHUTDOWN_RESERVE` | `env.py:1520` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_TOKEN_ID` | `env.py:1485` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_TOKEN_SECRET` | `env.py:1486` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_USAGE_QUOTA_ENABLED` | `env.py:1579` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_VOLUME_NAME` | `env.py:1589` | legacy WebRTC Modal worker |
| `WEBRTC_MODAL_WATCHDOG_TIMEMOUT` | `env.py:1510` | legacy WebRTC Modal worker |
| `WEBRTC_PREVIEW_FRAME_JPEG_QUALITY` | `env.py:1644` | legacy WebRTC |
| `WEBRTC_REALTIME_PROCESSING` | `env.py:982` | `StreamsConfiguration` field; no new package reads the env name today |
| `WEBRTC_SESSION_HEARTBEAT_INTERVAL_SECONDS` | `env.py:1613` | legacy WebRTC |
| `WEBRTC_SESSION_HEARTBEAT_URL` | `env.py:1608` | legacy WebRTC |
| `WEBRTC_WORKER_ENABLED` | `env.py:1482` | legacy WebRTC; unread (hosting summary) |
| `WEBRTC_WORKSPACE_STREAM_QUOTA` | `env.py:1601` | legacy WebRTC |
| `WEBRTC_WORKSPACE_STREAM_QUOTA_ENABLED` | `env.py:1598` | legacy WebRTC |
| `WEBRTC_WORKSPACE_STREAM_TTL_SECONDS` | `env.py:1603` | legacy WebRTC |
| `WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING` | `env.py:646` | not present in `roboflow_workflows` |
| `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT` | `env.py:1228` | legacy dedicated-deployment auth |

## New-stack-only names that change behaviour relative to legacy

Not part of the legacy name set, so outside the four categories, but recorded
because legacy parity depends on them and they are infra-env decisions:

| name | new default | legacy behaviour | note |
|---|---|---|---|
| `INFERENCE_MAX_IMAGES_PER_REQUEST` | `32` (`configuration.py:46-48`) | no per-request image-count cap | `0` disables the cap (`image_limits.py:20-21`) |
| `INFERENCE_INFER_TIMEOUT_S` | `30.0` (`configuration.py:34`) | plain legacy server: no per-inference timeout; adapter: 300 s | `0` is not "unbounded" (`gateway.py:441`); set a large value if parity is wanted; the `LEGACY_MMP_INFER_TIMEOUT_S` alias covers adapter deployments |
| `INFERENCE_LEGACY_LOAD_TIMEOUT_S` | `300.0` (`configuration.py:161-163`) | plain legacy server: blocks until loaded; adapter: 600 s | the `LEGACY_MMP_LOAD_WAIT_S` alias covers adapter deployments |
| `INFERENCE_MAX_BODY_BYTES` | 100 MiB (`configuration.py:39-41`) | no body cap | |
| `ENABLE_CONTROL_PLANE_ROUTES` | `False` (`configuration.py:82-84`) | legacy control-plane routes always registered (gated by `LEGACY_CONTROL_PLANE_ROUTES_ENABLED` on the legacy paths only) | |
