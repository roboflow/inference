# Changelog

This is the canonical changelog for the `roboflow-workflows` package, including
blocks, dependencies, and execution-engine behavior. Add engine behavior changes
under `## Unreleased` → `### Execution engine`; one entry is sufficient.

Package releases record their bundled execution-engine compatibility version,
which can remain unchanged across package releases. The package version and
engine version are separate; workflow and block compatibility use the engine
version. The mapping starts with package `0.2.2` below.

Earlier engine changes and migration guidance remain in the
[historical execution-engine changelog](https://docs.roboflow.com/workflows/developer-guide/developer-guide/execution-engine-changelog).
See the [engine release rule](../.cursor/rules/execution-engine-version-changelog.mdc)
for contributor and maintainer responsibilities.

## Unreleased

### Execution engine

- `VideoMetadata` carries an optional `total_frames`: the frame count of the
  source video file, or `None` for live streams, files that don't report it,
  realtime WebRTC uploads, and after a source restart or initial seek.
  Blocks can compute playback progress as `frame_number / total_frames` (e.g. a
  timeline overlay). Image inputs may include it under `video_metadata`.

### Added

- `roboflow_workflows.http_contract` (`describe`, `entities`, `errors`) — the HTTP surface `inference-server` imports to expose workflow routes.
- Optional `original_image` and `mask_metadata` on instance-segmentation response
  DTOs retain the input dimensions and output-to-input coordinate mapping for
  opted-in HTTP responses. Response `image`, boxes, polygons and RLE masks all
  use the selected mask grid. Existing workflow blocks do not opt in and retain
  their image-space contract. The `image` field continues to accept a single image
  descriptor or a list for multi-image responses.

- `ModelsConfiguration.use_inference_models`, defaulting to `True`. Hosts using
  the legacy backend must set it to `False` to retain non-tensor local legacy
  decoding.
  The server sets it from `inference.core.env.USE_INFERENCE_MODELS`.

- MQTT Writer v2 (`roboflow_enterprise/mqtt_writer_sink@v2`): v1 plus an optional `fire_and_forget` (default `False`) and outage hardening; v1 is unchanged.
  - With `fire_and_forget` a run queues the message and returns without waiting for the send or the broker's acknowledgement, and after the first run it never waits for a reconnect, so a disconnected broker no longer stalls each frame by up to `timeout`. If the first connection attempt fails, the client is kept and retries in the background instead of reconnecting synchronously on every frame. Success means queued, not delivered.
  - While the broker is disconnected a QoS 0 message is dropped and a QoS 1/2 message is queued and sent after reconnect, both reported with `error_status` set; these repeating failures are logged when they start or change and once on recovery, not on every run.
  - The step declares the `fire_and_forget_hides_persistence_failures` and `connection_and_state_rebuilt_per_request` restrictions when the switch is on (over the HTTP API every request builds a fresh block, so each request is a first run and a message still queued when the request ends is dropped), and reports a selector-driven switch as unknown.
  - In either mode the block refuses a new publish once 1000 QoS 1/2 messages await acknowledgement or 1000 packets wait to be sent (paho buffers both without limit by default), and reports it as a queue-full error; after a reconnect up to about 2000 packets can be buffered briefly. The limits count messages, not bytes.
  - The background reconnect backs off exponentially: the first retry waits half of `timeout` (at least 0.1 s), each further one doubles that, up to twice `timeout` (at least 1 s), plus the time each attempt takes, so a tiny `timeout` cannot retry a dead broker in a busy loop.
  - Selector-supplied `retain`, `fail_fast` and `fire_and_forget` are coerced the way the manifest validates them (for example `"false"` is False; v1 treats any non-empty string as True), and so are `port` and `qos` (for example `"1883.0"`, which v1 rejects); booleans are still rejected as a port or QoS.
  - While the broker stays unreachable or keeps answering "unavailable", the background reconnect loop logs the first failure as an error and later attempts at debug level until a connection succeeds, instead of an error about once per second.
  - Closing the block (when a pipeline stops) is bounded even when it races a background reconnect: a connection the broker accepts after the close began is disconnected at once, and the close never waits for acknowledgements or for the broker; messages not yet acknowledged are dropped, though packets already queued may still be sent before the disconnect. A close during an in-flight reconnect first waits for that connection attempt (TCP bounded by `timeout`, a TLS handshake by the 15 s keepalive, DNS by the OS resolver).

### Changed

- `execution_engine.v1.executor.core` no longer imports the opt-in `inference_models.utils.performance` profiler; step execution timing is unaffected, only the instrumentation is gone.
- Instance segmentation `@v1`–`@v4` preserve image-resolution masks when using
  the local `inference_models` backend. Non-tensor local legacy execution
  continues to honor `mask_decode_mode` and `tradeoff_factor` when
  `USE_INFERENCE_MODELS=False`.
  Remote calls always request `accurate` / `1.0`, since the local backend setting
  does not identify the remote server. Remote legacy workflows that previously
  requested fast/tradeoff masks now receive accurate masks. `@v4` retains its
  RLE response format.
- The new instance-segmentation `@v5` block is deferred to a separate PR together
  with reduced-grid Supervision conversion and downstream block support.
  **Full tensor support will come in a separate PR.** Existing tensor variants
  remain pinned to `accurate` / `1.0`.
- `PlatformClient.register_image_at_roboflow` takes an optional `annotation_follows`
  (default `False`). Dataset Upload `@v1` sets it when an annotation upload follows,
  so a host that retries uploads can fail an upload whose retry reports a duplicate
  instead of silently dropping the annotation.

### Fixed

- Tracker blocks log the missing-FPS fallback only when creating a tracker for a video, including tensor variants.

## `0.2.4-post1`

Bundled execution engine: `1.16.1`.

### Added

- Anthropic Claude block (`anthropic_claude@v5`): `claude-haiku-5-5` model option.
- Mistral AI block (`roboflow_core/mistral_vlm@v1`): runs Mistral Large 4 (`mistralai/mistral-large-4-0`) via OpenRouter with the vlm-exam request contract (image-first user message, reasoning off by default with a `high` option, `max_tokens` unset by default) and in-block decoding of detections and classifications.
- VLM detection box format `xyxy_0_999`: `box_2d` integers normalized to 0-999, Mistral's documented grounding convention. Used by the Mistral AI block and selectable as `detection_format` on the OpenRouter block (`openrouter@v3`).
- The shared OpenRouter executor accepts `max_tokens=None`: the direct path omits the parameter so the provider default applies, the Roboflow-proxied path sends the proxy ceiling (16384) because the proxy requires the field and otherwise applies a 500-token default.

## `0.2.4`

Bundled execution engine: `1.16.1`.

### Added

- Embedding Model (`roboflow_core/embedding_model@v1`): extract feature vectors
  before the final linear layer or logits before Softmax/Sigmoid from existing
  single-label and multi-label ResNet, ViT and DINOv3 classifiers, including
  pretrained ResNet aliases. List and tensor variants return embeddings compatible
  with Cosine Similarity and metadata identifying their embedding space.
- Local tensor-mode Embedding Model execution retains materialized tensor images
  and embedding tensors through the model-provider boundary. Remote responses and
  final JSON outputs remain serialized vectors.
- Anthropic Claude block (`anthropic_claude@v5`): `claude-sonnet-5-5` model option.
- OpenAI block (`open_ai@v7`): `gpt-6.1-sol` model option (`low`-`max` reasoning effort; object detection and instance segmentation reuse the GPT-6 prompts).

### Changed

- Carries forward the `0.2.2` model catalog: Anthropic Claude v5 lists `claude-opus-5-5` (Claude Opus 5.5, 128000 max output tokens) and the temperature warning names Opus 5.x; OpenAI v7 lists `gpt-6-sol` and `gpt-6-luna` (reasoning effort `none` through `max`, structured absolute detection prompts).
- `prototypes.platform_errors`: `RoboflowAPINotAuthorizedError`, `RoboflowAPINotNotFoundError`, `RoboflowAPITimeoutError` and `RoboflowAPIConnectionError`, for hosts to raise and translate platform request failures. Names and bases match the `inference` server classes, which now re-export them; `RoboflowAPINotAuthorizedError` is not a `RoboflowAPIForbiddenError`.
- Requires `supervision>=0.30.6,<0.31.0` (was `>=0.29.0,<0.30.0`). `supervision` 0.30 depends on `av`, `pydeprecate` and `defusedxml`, so installing this package now installs them too.
- Time in Zone (`v1`, `v2`, `v3`, tensor variants included) follows `supervision` 0.30 and rounds a detection's anchor to the nearest pixel instead of rounding it up. A detection whose anchor lies within one pixel of a zone edge can change zone membership: an anchor at `x=10.4` was outside a zone ending at `x=10` and is now inside.
- Detections Stitch with `overlap_filtering_strategy` set to `nmm` follows `supervision` 0.30 and compares instance segmentation masks at their own resolution instead of scaled to 640 pixels. Which detections merge can change where the scaled masks used to land on the other side of the IoU threshold. With NumPy data the merge is slower on large frames, because that time is spent inside `supervision`: 200 masks on a 3840x2160 frame took about 29 s with `supervision` 0.30.6 and about 1 s with 0.29.1 on the same CPU. With tensor data the block counts overlaps only where mask bounding boxes intersect and takes about as long as before.
- Predictions whose polygon has fewer than 3 points are dropped before `supervision` parses the response, in model blocks, OCR blocks and detections passed as workflow inputs. The result is unchanged: the prediction is left out and the others keep their masks. `supervision` 0.30 on its own would keep it as a box and remove the masks of every detection in the response.

### Fixed

- Detections Stitch (`roboflow_core/detections_stitch@v1`): segmentation masks are stitched as crop-scoped compact masks (`supervision.CompactMask`) and only the detections that survive overlap filtering are materialised at reference resolution. Previously every crop mask was first re-allocated as a full-size dense array, merged, then filtered, which needed about `N x H x W` bytes before any filtering: a 1080p frame sliced 12 ways with ~25 masks per slice took ~3 GiB inside this block and OOM-killed an 8 GiB video worker. Outputs keep the same boxes, order and dense masks, with one numerical correction: mask IoU is now computed from exact pixel counts instead of `float32` arithmetic, so a pair whose IoU equals `iou_threshold` is decided by the documented comparison. With `nms`, IoU equal to the threshold no longer suppresses (30 of 100 pixels at `0.3` used to round to `0.30000001` and drop a detection); with `nmm`, IoU equal to the threshold now merges (70 of 100 pixels at `0.7` used to round to `0.69999999` and keep both). A mix of crops with and without masks now raises a clear `ValueError` instead of failing inside `Detections.merge`.

- CLIP v1 and CLIP Comparison v1/v2 blocks, including tensor variants, now report
  a model's text-context-length validation error as `RuntimeInputError`, allowing
  workflow HTTP requests to return 400 instead of 500. Other model input errors
  retain their existing handling.

- Inner Workflow block no longer imports `fastapi`, which only the `enterprise` extra installs; its `background_tasks` argument is typed with `BackgroundTaskScheduler`. `roboflow_workflows.execution_engine.core` now imports without `fastapi`.

### Execution engine

- Dependency preloading distinguishes classification, feature-vector and logits
  registrations for the same model ID, including runtime-selected IDs, and verifies
  the corresponding capability-specific cache entries. Existing workflows need
  no migration.

## `0.2.3`

Bundled execution engine: `1.16.0`.

### Fixed

- Blur Visualization: instance segmentation predictions are now blurred in the shape of each mask, as the block's description always said, instead of as a rectangle covering the bounding box. The blur also covers mask pixels that reach past the box. Predictions without masks (object detection, keypoints) are blurred exactly as before. Masks can leave thin edges such as hair or fingers uncovered where the box used to hide them; set the new `padding` to widen the blur.
- Keypoint Visualization draws partial skeletons at their true joints. Keypoints below the model's confidence threshold are left out of predictions and the rest were padded at the end, so every keypoint after a missing one shifted position: bones joined the wrong joints, and a frame where nobody had all 17 COCO keypoints drew no bones at all. Keypoints are now placed by their keypoint class id, and COCO skeletons that are missing trailing joints are still recognised. Keypoints without class ids, or with ids that are negative or too large for the keypoint slot or padding limits, are drawn as before. The tensor-native path (`ENABLE_TENSOR_DATA_REPRESENTATION`) places keypoints the same way when they are rebuilt from a remotely executed model step, a rollup or a dynamic block, so both representations render alike.
- MQTT Writer: a broker that refuses the connection (bad user name or password, not authorised, unacceptable protocol version) is now reported in the outputs with the broker's reason, and the client stops reconnecting instead of retrying the same credentials about once a second, which tripped brokers' authentication rate limiting. The refusal is reported as soon as the broker answers rather than after `timeout`, and every later run repeats it without touching the broker; fix the configuration and restart the pipeline. A broker answering "unavailable" keeps retrying in the background.
- MQTT Reader: declares its workload - external requests and messages buffered between runs, no dependent models or projects - and gives its two restrictions stable codes (`unavailable_on_hosted_platform`, `connection_and_state_rebuilt_per_request`) in both the editor and the actual declaration. Runtime behaviour and the editor payload are unchanged.
- SAM2 video, SAM3 video and action recognition blocks, tensor variants included, now declare a hard restriction against remote step execution. This matches how they already behave at runtime; execution is unchanged.
- Preserve image dimensions and parent lineage when Template Matching returns no detections.
- `actual_restrictions_of()` / `get_actual_restrictions()`: every entry is validated and rebuilt from its validated projection, with axes as new lists of enum members in authored order, configuration as a new `dict` (or `None`), and the authored note kept. A malformed declaration, or an exception while evaluating the host view, yields only a sanitized `declaration_failed` problem with no exception text or values; previously the host view could raise, for example on a configuration given as a list of pairs. The `RuntimeRestriction` constructor and `to_dict()` are unchanged. The `discover_dependent_resources()` annotation now admits `Discovery[DependentResource]`.
- `EngineConfiguration` validates `custom_python_execution_mode`: only the exact values `local` and `modal` are accepted, and any other value raises `WorkflowEnvironmentConfigurationError` at construction. Previously an unknown value was silently treated as local execution. `modal` stays valid when local custom Python is disabled. The inference server still strips, lowercases and validates the value before building the configuration, so servers are unaffected.
- LMM (`roboflow_core/lmm@v1`) and LMM for classification (`roboflow_core/lmm_for_classification@v1`) no longer declare `hosted_endpoint_disabled_by_flag` for `LMM_ENABLED`, in either the actual or the editor restrictions. Both blocks call OpenAI directly and never reach a Roboflow LMM endpoint, so that flag does not gate them. Runtime behaviour is unchanged.
- SAM3 (`roboflow_core/sam3@v1`, `@v2`, `@v3`, numpy and tensor): a missing (`null`) `model_id` that the step would use is reported as an incomplete resource declaration with an `invalid_resource_identifier` problem, instead of a complete empty one, so the workload model inventory is incomplete too. Runtime behaviour is unchanged.
- BoT-SORT tracker (`roboflow_core/trackers_botsort@v1`, numpy and tensor): the editor restrictions (`get_restrictions()`) now include the stateful-video and still-image caveats, matching SORT, OC-SORT and ByteTrack. Actual workload restrictions are unchanged.
- Continue If (`roboflow_core/continue_if@v1`): a positive `stop_delay` keeps grace-period state in the block instance, so the workload report declares `cooldown_timer_resets_on_stateless_http` (soft; hosted serverless and dedicated deployment; any step execution mode). The default `stop_delay=0` declares nothing. A selector-fed value declares the caveat conservatively and marks restrictions incomplete. The editor view always shows the cooldown caveat, because it cannot see `stop_delay`. Field validation and runtime behaviour are unchanged.
- Camera Focus v1 and Contours v1 declare `visualization` next to `image_analysis`. Camera Focus v2 declares `visualization` when `show_zebra_warnings`, `show_hud`, `show_focus_peaking` or `show_center_marker` is enabled, or `grid_overlay` is not the string `"None"`; with all four flags off and `grid_overlay: "None"` (no grid) it declares only `image_analysis`.
- PostgreSQL sink (`roboflow_core/postgresql_sink@v1`): declares the hard `unavailable_on_hosted_platform` restriction (target runtime `hosted_serverless`) for every `fire_and_forget` value, including an unresolved selector, and in the editor `get_restrictions()`. This matches the existing runtime guard; database and runtime behaviour are unchanged. The conditional fire-and-forget caveat is unchanged.
- Microsoft SQL Server sink (`roboflow_core/microsoft_sql_server_sink@v1`), Event Writer (`roboflow_enterprise/event_writer_sink@v1`) and OPC UA Writer (`roboflow_enterprise/opc_writer_sink@v1`): a literal `fire_and_forget: true` (the default) now declares the soft `fire_and_forget_hides_persistence_failures` restriction for `inference_pipeline`; `false` omits it; a selector makes restrictions incomplete with an `unresolved_selector` problem. OPC UA Writer keeps its cooldown caveat in every case. Editor restrictions and runtime behaviour are unchanged.

### Added

- Blur Visualization `padding`: blurs extra pixels around each detection, growing bounding boxes on every side and segmentation masks outward. Defaults to `0`, which keeps the previous coverage.
- MQTT Reader enterprise block (`roboflow_enterprise/mqtt_reader@v1`): subscribes to an MQTT topic and returns one message per run as raw text and parsed JSON, with `read_mode` selecting the newest message (`latest`) or the next unread one (`sequential`). Retained messages are returned on the first run; the subscription is restored after a reconnect, and a 15 s keepalive reports a silently lost broker within about 25 s. Live subscriptions need an `InferencePipeline`; over the HTTP API each request only sees retained messages. A broker that refuses the connection (bad user name or password, not authorised, rejected client id) is reported with the broker's reason as soon as it answers, and the client stops reconnecting; a broker answering "unavailable" keeps retrying. After the first subscription a run never waits for the broker: while the connection is down, runs return the last message with `error_status` set instead of an empty message, buffered messages first; the outage is logged once at its start and once at recovery.
- Persistent sessions for the MQTT Reader: an optional `client_id` (unique per broker, for example the pipeline name from a workflow input) connects with that id and a non-clean session, so the broker keeps the subscription and queues QoS 1/2 messages while the block is away and delivers the backlog when the same id reconnects, after a pipeline or process restart included. Requires `qos` 1 or 2 (a run with `client_id` and `qos` 0 reports an error); the publisher must also publish at QoS 1 or higher. Meant for `InferencePipeline`s; `sequential` works through the backlog one message per run. Left empty, the behaviour is unchanged.
- TLS for the MQTT Reader and MQTT Writer: an `encryption` dropdown (`none`, default, or `tls`) encrypts the connection and verifies the broker's certificate against the system trust store; `ca_certificate_path` (shown for TLS only) points at a PEM bundle for a private CA and requires `ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=True`. The port is not switched automatically and verification cannot be disabled.
- Operator policy for the MQTT Reader and MQTT Writer broker address: `MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS` (comma-separated `host[:port]` allowlist, no DNS) and `MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST` (default `True`; when `False` the workflow's host and port are ignored and the first allowlist entry is used). Defaults preserve existing behaviour.
- `roboflow_workflows.http_contract` (`describe`, `entities`, `errors`) — the HTTP surface `inference-server` imports to expose workflow routes.

### Changed

- Widened `zxing-cpp` from `~=2.2.0` to `>=2.2.0,<=2.3.0` so installs can pick 2.3.0: 2.2.0 ships no Python 3.13 wheels, so installs on 3.13 compiled it from source.

### Execution engine

- Add compile-time workload introspection with graph structure, dimensionality, model usage, resources and conditional restrictions. Each entity `type` carries its contract version (e.g. `workflow_introspection_v1`); there is no separate `schema_version`.
- Report unresolved resource identities explicitly and include streaming-video models without changing how they load.
- Align static and actual restriction codes, and correct stateful-block and industrial-sink caveats without changing the editor payload's serialization shape. Declaration corrections listed under Fixed change which editor entries some blocks list.
- **Run-scoped thread pool for executor-less runs** — runs that receive no
  host-provided executor (standalone `roboflow-workflows`, SDK and embedded
  usage) now reuse one run-scoped thread pool across step waves instead of
  constructing a `ThreadPoolExecutor` per wave, lowering latency on multi-wave
  chains; host-provided executors and scheduling are unchanged. No migration
  is needed.
- Reject cyclic saved inner-workflow references during resolution with a composition
  error instead of `RecursionError`. Enforce nesting depth and total inner-workflow
  count limits during reference expansion, before fetching or expanding children
  beyond those limits. Valid repeated references and remote dispatch are unchanged;
  no migration is required.
- Workload introspection now enforces the requested engine version as a minimum,
  like `ExecutionEngine.init`. A newer requested minor or patch version than the
  installed engine raises `NotSupportedExecutionEngineError` before compilation,
  inner-workflow fetches or model metadata lookups. Supported requests are unchanged.
- The duplicate dynamic block warning now names the skipped and the retained
  definitions by position (for example
  `steps[2].workflow_definition.dynamic_blocks_definitions[1]`) instead of logging
  the request-provided `block_type`. Which definition is kept is unchanged.

---

## `0.2.2`

Bundled execution engine: `1.15.2`.

### Added
- OpenAI block (`open_ai@v7`): `gpt-6-sol` and `gpt-6-luna` model options.
- Anthropic Claude block (`anthropic_claude@v5`): `claude-opus-5-5` model option.

---

## `0.2.1`

### Fixed

- Expose native Qwen 3.8 VL 27B in Qwen VLM blocks v2–v4, including model discovery and thinking support.
- SpaceXAI object detection encodes PNG at compression level 9. OpenCV's default compression put large frames over xAI's 25MB upload limit.

### Added

- SpaceXAI block (`spacexai@v3`): `grok-4.7` model option with `low`/`medium`/`high`/`xhigh` reasoning effort. Detection reuses the Grok 4.5/4.6 prompt.

---

## `0.2.0`

### Added

- Python 3.13 support (`requires-python` is now `>=3.10,<3.14`).
- OpenAI v7 detection and instance-segmentation accept optional `output_classes`, keeping visual
  prompts in `classes` while constraining and decoding model output with stable labels.
---

## `0.2.0`

### Added

- `roboflow_workflows.http_contract`: HTTP request/response entities and framework-free describe/error helpers, moved from `inference.core` (which now re-exports them).

---

## `0.1.2`

### Fixed

- Webhook sink: SSRF hardening (destination validation, DNS pinning, redirect and proxy refusal) now applies only when `ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES=false`. The default restores the previous plain `requests` transport, so environment `HTTP_PROXY` / `HTTPS_PROXY` work again.

---

## `0.1.1`

### Fixed

- Accept Python string class names from NumPy object arrays in detection-property expressions, avoiding `.item()` errors after custom Python blocks.

### Added

- Kafka Consumer and Kafka Producer enterprise blocks; `confluent-kafka` and `aws-msk-iam-sasl-signer-python` join the `enterprise` extra.

### Changed

- Dropped unused dependencies: `opencv-contrib-python`, `requests-toolbelt`, and test extras `pytest-asyncio`, `pytest-timeout`, `aioresponses`.

---

## `0.1.0`

- Initial release: Workflows execution engine and block library extracted from `inference`.
