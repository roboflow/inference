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

## `0.2.15`

Bundled execution engine: `1.17.1`.

### Execution engine

- **Inner Workflow literal bindings keep their type.** A literal bound in an
  Inner Workflow step's `parameter_bindings` (for example `0.3`, `true`, a list
  or an object) now reaches the child workflow's steps with that type when the
  child references the input as a whole `$inputs.<name>` value. Inlining used
  to stringify every binding, so a child boolean input bound to `true` became
  `"True"` and a number became `"0.3"`, which differed from the same value
  supplied through the child's `default_value`. A literal embedded in a larger
  string (`"prefix-$inputs.x"`) is still interpolated as text. Compilation now
  rejects a non-string literal bound to a child output selector with
  `InnerWorkflowInvalidStepEntryError` instead of producing an invalid selector
  later. Workflows that worked around the cast by converting inside the child
  keep working; no migration is needed.

### Breaking

- Require `openai>=3.32.0,<4.0.0` (it was `>=1.17.0,<2.0.0`). 3.32 is the first release whose `client.decisions.create` accepts `prompt_cache_options`. Environments that cap `openai` below 3, as the current litellm, crewai and llama-index-llms-openai releases do, can no longer install `roboflow-workflows` alongside them.
  - openai 3 replaced HTTPX with HTTPX2. The blocks that call models through the SDK (OpenAI, OpenAI-Compatible, and the OpenRouter- and xAI-based blocks) now verify TLS certificates against the operating-system trust store instead of `certifi`; `SSL_CERT_FILE` and `SSL_CERT_DIR` still apply.
  - The OpenAI-Compatible block builds its client with `DefaultHttpx2Client` and parses base URLs with `httpx2.URL`. It still refuses redirects, and its redirect test intercepts the HTTPX2 transport.
  - Migration: install `roboflow-workflows` in an environment that allows `openai` 3, and on a host that verified TLS through a patched `certifi` bundle, point `SSL_CERT_FILE` at that bundle.

### Added

- OpenAI block (`open_ai@v7`): add GPT-6.1 Terra (`gpt-6.1-terra`) with
  `none` through `max` reasoning effort and the existing structured absolute-pixel
  detection prompt.
- Add Muse Spark Open to `meta_vlm@v3` via OpenRouter, with minimal, low,
  medium, high, and xhigh reasoning and the existing normalized 0–1000
  detection format.

- SAM 3 Interactive (`roboflow_core/sam3_interactive@v1`): optional `output_format`, `polygons` (default, output unchanged) or `rle`, with the same values as SAM 3 v3. With `rle` the block asks SAM 3 for RLE masks on every execution path and returns one prediction per prompt that keeps the model's mask in `rle_mask` (decoded into `mask` for downstream blocks), declared as `rle_instance_segmentation_prediction` first, so serialized outputs carry `rle_mask` instead of polygon `points`. A server that answers polygons anyway has them rasterised into the prompt's mask. The tensor-native variant already carried RLE masks; with `rle` it declares and serializes them as RLE, and, like the NumPy block, keeps one mask per prompt for a polygon answer, drops empty masks and reports COCO box extents (a one-pixel mask is a 1x1 box). `ModelsProvider.run_sam2_segmentation` gains `format`, which the block sends only for `rle`.
- Vision Events (`roboflow_core/roboflow_vision_events@v1`) and Vision Event Bundle (`roboflow_core/vision_event_bundle@v1`) blocks gain an optional `value_base_units` input: what the event is worth, in base units of the use case's currency (e.g. cents for USD). The currency always comes from the use case. The value is sent as `valueBaseUnits` to the Vision Events API and in the bundle `payload.json`, and as `value_base_units` to the local event store. A typed value or workflow input must be a whole number (`true`/`false` are rejected). A value from another step is rounded to whole base units; one that is not a finite number or falls outside ±10,000,000,000 is dropped with a warning and the event is still recorded. Left empty, the use case's default value applies. `operator_feedback` events never carry a value, so the blocks omit it for them.

### Changed

- Remove duplicated model lists and counts from `meta_vlm@v3` descriptions;
  available models are defined by the registry-backed dropdown.

### Fixed

- Workflows' standalone image codec now reports non-ASCII base64 payloads as
  `WorkflowImageLoadError` instead of allowing a raw `ValueError` to escape.

- `roboflow_core/asset_library_attributes@v1` normalizes metadata keys before sending them: characters outside `[A-Za-z0-9_.]` become `_`, and a leading `.` gets a `_` prefix. If two keys of one image normalize to the same key (for example `traffic light` and `traffic_light`), the block raises an error instead of letting one value overwrite the other. Before, one invalid key (for example a class name with a space) made the API reject every attribute for that image.

- Root-coordinate recovery (`sv_detections_to_root_coordinates`, applied to workflow outputs by default) re-encodes `rle_mask` from the re-anchored dense masks. A block that ran on a crop and returns RLE with dense masks, such as SAM 3 v3 with `output_format: rle`, used to serialize root-coordinate boxes next to an RLE still sized and placed in the crop. RLE-only predictions (`mask=None`) are unchanged, as in `scale_sv_detections`.

- Video-state blocks now accept an image from a step output as well as from a Workflow input, so they keep working inside an embedded `roboflow_core/inner_workflow@v1` whose `image` is bound to a parent step (for example a preprocessing block). Inlining replaces the child's `$inputs.image` with that step selector, and the merged definition used to fail with `Could not parse workflow definition` even though each Workflow compiled on its own. Affected blocks: `time_in_zone@v2`/`@v3`, `velocity@v1`, `line_counter@v2`, `path_deviation_analytics@v2`, `detection_event_log@v1`, `byte_tracker@v2`, `cache_get@v1`, `cache_set@v1`, `delta_filter@v1` and `rate_limiter@v1` (`video_reference_image`). The blocks read `video_metadata` from the image, which step outputs built with `WorkflowImageData.copy_and_replace` keep, so per-video state is unchanged.
- Absolute and relative static crops (`absolute_static_crop@v1`, `relative_statoic_crop@v1`) now give a crop the same `video_identifier` on every frame (`<video> | crop: <block>:<x_min>,<y_min>,<x_max>,<y_max>`). They used to append the per-frame crop ID, so a stateful block fed by a static crop (Cache Get/Set, trackers, analytics) saw each frame as a new video: the cache missed on every frame and state kept growing with one namespace per frame. Crop `parent_id` values are still unique per frame. `WorkflowImageData.create_crop` and `create_crop_from_tensor` take an optional `video_crop_identifier` for this; other callers are unchanged.
- Camera Calibration (`camera-calibration@v1`) keeps the input image's video metadata and root-ancestor metadata on its output. It used to drop them, so a rate limiter using the calibrated frame as `video_reference_image` saw every frame at time zero and suppressed its downstream steps after the first frame.

## `0.2.14`

Bundled execution engine: `1.17.0`.

### Changed

- Security maintenance release, no block or engine change. The Workflows lockfile is refreshed for three advisories in transitive dependencies: `certifi` 2026.7.22 -> 2026.10.23 (GHSA-6q48-6c7w-9gw2, bundled emSign root CAs whose intermediate CA keys were compromised), `GitPython` 3.1.62 -> 3.2.1 (GHSA-83vg-56qc-22m7, remote code execution through a submodule `.git` path) and `setuptools` 84.0.0 -> 84.2.1 (GHSA-xmj3-9gf7-h52w, Windows path escape during legacy wheel installation). Declared dependency ranges are unchanged.

## `0.2.13`

Bundled execution engine: `1.17.0`.

### Changed

- Require the stable `inference-models>=0.39.5` and `inference-sdk>=1.8.1`
  releases, and refresh the Workflows lockfile to match these minimum versions.

### Execution engine

- Update the engine compatibility version to `1.17.0` and align the integration
  and hosted-platform version assertions with the published `0.2.13rc1` package.

- Workflow syntax errors no longer repeat the raw input of the step that failed
  validation. The rendered `pydantic` error (the `inner_error_message` of an HTTP
  `WorkflowSyntaxError` response, and the traceback the server logs) used to
  include each failing step's literal field values, secrets such as a `password`
  or credentials in a URL included. The same holds wherever a block manifest is
  validated directly, such as the input and output discovery behind
  `/workflows/describe_interface`: `WorkflowBlockManifest` now sets
  `hide_input_in_errors`, and every block inherits it. Error messages, locations
  and `blocks_errors` are unchanged.
- When a block fails to initialise during compilation, the blocks initialised
  before it are closed (each failure to close is logged) before the error is
  raised, instead of keeping their threads or connections until garbage
  collection.
- `ExecutionEngine.close()` releases what the workflow's blocks hold between runs:
  it calls `close()` on every initialised block that defines one (for example the
  background connections of the OPC UA Reader, MQTT Reader and Kafka Consumer). A
  block that fails to close is logged and does not stop the others; repeated calls
  are safe. `streamvision` calls it when an `InferencePipeline` is joined, so a
  stopped pipeline that is still referenced no longer keeps those connections
  until it is garbage collected.

### Added

- OPC UA Reader enterprise block (`roboflow_enterprise/opc_reader@v1`): subscribes to one variable on an OPC UA server and returns its latest `value` on every run, with `is_new`, `error_status` and `error_message`. The variable is located like in the OPC UA Writer (`hierarchical` browse path or `direct` string NodeId, which also covers the Roboflow edge OPC UA server's `Folder.Tag` string NodeIds; pass its namespace by URI, since the index differs between servers).
  - The first run connects, reads the value and creates an OPC UA subscription (`publishing_interval_ms`, default 100); later runs return the latest pushed value from memory and never wait for the server, so the block suits 30-60 fps video. Only the first run waits, up to `timeout` (at most 60 s), and it settles only once the subscription has delivered its initial value. `publishing_interval_ms` is bounded to 10–60000.
  - While the connection is down, runs repeat the last value with `error_status` set and the block reconnects and re-subscribes in the background with exponential backoff; the outage is logged once at its start and once on recovery. A value whose OPC UA quality is not Good is returned with `error_status` set and the status name. A crashed server is reported after up to `timeout` plus about 2 s (about 4 s with the defaults): the OPC UA client's 1 s watchdog, its request timeout, and the block's 1 s subscription check. A server that keeps dropping sessions or subscriptions is re-approached with exponential backoff (0.5 s doubling to 10 s), which only starts over once a connection has stayed up for 10 s.
  - Credentials go in `user_name` and `password` (a `url` containing `user:password@` is rejected, since the `url` field is not masked in the editor like `password`; use a secret selector for `password` to keep it out of the workflow definition). A `url` that does not validate is never echoed: errors use fixed messages, and logs show only the scheme, host and port. Rejected credentials are reported and not retried, and the block's thread then ends; an unknown namespace or node is retried every 10 s; a server at its session limit is retried after the OPC UA Writer's jittered session-exhaustion backoff. Sessions request the writer's 60 s session timeout (`OPC_SESSION_TIMEOUT_SECONDS`) and are closed when the pipeline running the workflow is joined after it ends, when the block is closed or garbage collected, or when the interpreter exits.
  - Each block instance holds its own session, separate from the OPC UA Writer's pool. Over the HTTP API every request builds a fresh instance, so each request opens a session, reads the current value and closes it. Not available on the Roboflow hosted platform.

### Fixed

- OPC UA Reader waits for the subscription's final server-revised publishing interval when awaiting its initial notification, including a second revision during `ModifySubscription`, so a server that revises the interval upward does not cause repeated reconnects before the first value arrives. The first workflow run still returns within its configured `timeout` while the subscription continues waiting in the background.
- Buffer block outputs retain their per-frame list contents when later frames update the buffer, including batched workflows and padded histories.
- `DetectionsPropertyExtract` reads Supervision tracker IDs from their native field, so `tracker_id` extraction works on tracked detections in NumPy mode. Properties that live alongside each detection (`tracker_id`, `time_in_zone`, Bounding Rectangle `width` and the other non built-in properties) return an empty list on empty detections in both NumPy and tensor modes instead of raising. Requesting such a property on nonempty detections that do not carry it now raises an operation error naming the property and asking for the upstream block that supplies it; NumPy mode previously raised a bare `KeyError`, and tensor mode now uses the same message.
- Detections Stitch (`roboflow_core/detections_stitch@v1`) with `overlap_filtering_strategy` set to `none` no longer encodes masks as compact masks and decodes them again: every mask survives, so each crop's masks are copied straight into the single dense output array (only each mask's bounding box is written). Outputs are unchanged. Measured with `supervision` 0.30.6, median block latency against the compact-mask stitch: 300 masks from twelve 640x640 crops onto 1920x1012 19 ms -> 12 ms; 48 salt-and-pepper masks from four 256x256 crops onto 640x480 14 ms -> 0.7 ms; 24 masks from four 100x100 crops onto 300x200 0.5 ms -> 0.2 ms. Peak memory stays at the size of the output. Crops whose masks already arrive as `CompactMask` are decoded one crop at a time on every filtering strategy, instead of all up front.

## `0.2.12`

Bundled execution engine: `1.16.1`.

### Changed

- `aiohttp~=3.14.4`: requires the aiohttp release that fixes GHSA-2g87-pp6c-29x5 (unbounded memory/CPU use) and GHSA-5wr8-gcqf-gqf3 (no initial request/header timeout). aiohttp reached the package only through `inference-sdk` and `twilio`, which still allow 3.14.3.

## `0.2.11`

Bundled execution engine: `1.16.1`.

### Added

- Anomaly Detection Model block (`roboflow_core/anomaly_detection_model@v1`): runs a PatchCore or FoundAD model trained on a Roboflow anomaly detection project (served on the classification route). Outputs the classification-style `predictions` (plus `anomaly_score`, `anomaly_threshold`, `anomaly_scale`, `is_anomalous`), the raw `anomaly_score`, the `anomaly_threshold` in effect, the `anomaly_decision_threshold` (the raw score at which the anomalous confidence equals `confidence`), the `is_anomalous` verdict and the `anomaly_map` (a float32 `numpy_array` at the network input resolution; `None` when `include_anomaly_map` is off). `confidence` (default 0.5, the calibrated decision) is the anomalous confidence an image must reach; `anomaly_threshold` overrides the threshold saved with the model and re-decides with the same rule. `ModelsProvider.run_classification` gains `include_anomaly_map`.
- Anomaly Heatmap Visualization block (`roboflow_core/anomaly_heatmap_visualization@v1`): stretches an anomaly map onto its image (bilinear) and blends it as a heatmap on a ramp anchored at its `anomaly_threshold` input, meant to be wired to the model block's `anomaly_decision_threshold`: `t = max(v, 0) / (max(v, 0) + anchor)`; an anchor at or below zero saturates every positive value. `color_map` is the Roboflow app's ramp (`ROBOFLOW`) or an OpenCV map (`JET`, `TURBO`, `INFERNO`, `VIRIDIS`, `HOT`); `opacity` and `copy_image` as the other visualization blocks.
- `roboflow_workflows.http_contract` (`describe`, `entities`, `errors`) — the HTTP surface `inference-server` imports to expose workflow routes.

### Changed

- `inference-sdk>=1.8.0`: the hosted (API v0) request carries `include_anomaly_map` from 1.8.0, which the Anomaly Detection Model block needs in REMOTE execution. The anomaly blocks need `inference-models` 0.39.5 (`anomaly_scale`).

## `0.2.10`

Bundled execution engine: `1.16.1`.

### Fixed

- Dataset upload v1 (`roboflow_core/roboflow_dataset_upload@v1`) compiles again. Its `images` field listed `image` as the first validation alias, so the JSON schema named the property `image` while the manifest attribute is `images`; compilation raised `BlockInterfaceError` and `/describe_interface` raised `AttributeError`. The alias order is now `images`, `image`: both keys still validate, and the block schema lists `images`, as dataset upload v2 does.

## `0.2.9`

### Execution engine

- Prevent supporting metadata and workflow validation-model caches from retaining dynamically generated custom Python manifests after their compiled workflows are evicted. Static manifest caching and compiled-workflow reuse are preserved; no workflow changes are required.
- Workflows with custom Python blocks executed locally are cached again after compilation (as before 1.6.1), fixing per-request memory growth and the throughput drop on HTTP hosts; module-level state of a local custom Python block is again shared across requests that use the same definition in one process. Modal execution and unresolved inner-workflow references still compile per request.
- The compilation cache no longer raises `KeyError` when evicting a key that was cached twice.

### Added

- The nine Roboflow platform blocks are core blocks of this package again: dataset upload v1 and v2, custom metadata, model monitoring inference aggregator, vision events, vision event bundle, asset library attributes, visual search and visual search classifier. They live in `core_steps.sinks.roboflow` and `core_steps.integrations.roboflow`, so a standalone install now lists 248 blocks. Inference 1.6.1 to 1.7.2 shipped them as the `inference.roboflow_workflows_plugin` plugin; that plugin is now an empty compatibility stub.
- `RoboflowPlatformClient` gains the operations these blocks call: `get_roboflow_workspace`, `add_custom_metadata`, `register_image_at_roboflow`, `annotate_image_at_roboflow`, `update_image_metadata_at_roboflow`, `batch_update_image_metadata_at_roboflow`, `search_project_images_at_roboflow`, `send_inference_results_to_model_monitoring`, `get_device_id`, `get_server_version` and `get_system_info`. The offline client refuses the API operations. A host can register a fallback client for these blocks with `set_platform_blocks_client_factory`, used when a block gets no client, the offline client, or a client that does not implement these operations itself.
- Vision events sends its cloud image upload and event request only through a client that implements these operations. With the offline client it returns `error_status` with the offline refusal and sends nothing; writing to a local event store is unaffected.

### Changed

- `/workflows/blocks/describe` lists these blocks under their names from before the plugin, `inference.core.workflows.core_steps.{sinks,integrations}.roboflow.…`, and in their original positions within the core list, instead of `inference.roboflow_workflows_plugin.…` at the end.
- A disable pattern on the canonical module path, such as `roboflow_workflows.core_steps.sinks`, now also disables these blocks, like any other core block. Patterns written against either historic path keep matching the same blocks.
- OpenAI v7: optional `service_tier` (`auto`, `default`, `flex`, `priority`, `ultrafast`), sent with direct OpenAI requests and forwarded to the Roboflow proxy for `rf_key:` keys. Unset keeps OpenAI's default tier.
- `PlatformClient.register_image_at_roboflow` takes an optional `annotation_follows`
  (default `False`). Dataset Upload `@v1` sets it when an annotation upload follows,
  so a host that retries uploads can fail an upload whose retry reports a duplicate
  instead of silently dropping the annotation.


## `0.2.8`

Bundled execution engine: `1.16.1`.

### Added

- Google Gemini v6: Argon-only instance segmentation with normalized YX polygons, detached components, holes, and RLE mask output.

## `0.2.7`

Bundled execution engine: `1.16.1`.

### Added

- Meta block (`roboflow_core/meta_vlm@v3`): Muse Wonder (`meta/muse-wonder` via OpenRouter), reasoning levels `minimal` through `xhigh`, with the same flat 0-1000 `named_0_1000` detection decoding as the Spark models.

### Changed

- Meta block (`roboflow_core/meta_vlm@v3`) defaults `model_version` to Muse Wonder instead of Muse Spark 1.3. Workflows that omit `model_version` now call Muse Wonder.

## `0.2.6`

Bundled execution engine: `1.16.1`.

### Execution engine

- `VideoMetadata` carries an optional `total_frames`: the frame count of the
  source video file, or `None` for live streams, files that don't report it,
  realtime WebRTC uploads, and after a source restart or initial seek.
  Blocks can compute playback progress as `frame_number / total_frames` (e.g. a
  timeline overlay). Image inputs may include it under `video_metadata`.

### Changed

- V-JEPA streaming samples accept frames up to 50 ms after the requested source index, capped at half a sample interval. Predictions use the intended sample clock. Larger gaps skip affected windows with an error.
- Action Recognition Model rejects missing samples only when the model declares that it requires regular sampling. Cosmos retains its previous behavior for FPS-limited streams.
- Action Recognition Model uses explicit confidence and observed-duration capabilities for each model family.
- Inference hosts can set `VJEPA2_1_ENABLED=False` to block V-JEPA loading through the Action Recognition Model block before weights load.
- Action-recognition response schemas retain frame indices, class fields, and optional confidence for HTTP clients and Workflows.
- Action Recognition Model ignores confidence overrides for unscored models, including Cosmos, instead of rejecting them.
- Action Recognition Model keeps streaming sample counts and span durations within the model's recorded window at fractional source frame rates.
- Action Recognition Model applies an optional model-owned frame transform before buffering. V-JEPA retains resized RGB uint8 frames; Cosmos keeps its existing resize behavior.
- Action Recognition Model accepts `confidence="best"` for model-eval thresholds and `"default"` for the package default, like detection blocks. Numeric overrides and unscored Cosmos behavior stay unchanged.
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

### Added

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

### Fixed

- Action Recognition Model preserves repeated samples when V-JEPA requests a higher sampling rate than the source FPS. Complete low-FPS streams run inference while genuinely missing source frames still invalidate affected windows.
- SAM3 v1/v2/v3 remote SDK execution, with NumPy or tensor predictions, now honors `WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS` across the input batch while sending one image per HTTP request. Each concurrency-sized group is converted before dispatching the next, bounding response buffering and stopping later groups after an HTTP failure. Results retain input order, formats and class mapping; empty batches issue no requests. Local execution and the inference-proxy transport are unchanged.
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
