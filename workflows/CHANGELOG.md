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

- **V2 explicit processing reset of idle sessions and live runs (opt-in)** — `ExecutionSession.assess_update(plan)` tells, without constructing anything or writing state, whether an update preserves the graph, needs a reset or is unsupported, listing separately what rules out a preserving update (`preserve_blocked_by`) and what rules out a reset (`reset_blocked_by`), and what a reset would replace and keep (steps, operators, handler sessions, controls, managed state and per-step resource sources). `prepare_update(plan, reset=True)` and `update(plan, reset=True)` construct every step and handler session again in a fresh resolver: block-local state restarts, `Factory` values are created again, caller values are passed again. Engine-owned managed state starts fresh with declared defaults; replacement assessments distinguish closing the previous engine-owned service from leaving previous caller-owned state untouched. A caller's service is kept with its values or explicitly replaced by an isolated one (another backend or namespace), and is never cleared or closed. Resets that would change state defaults on a kept service, or split a source from the state it shares, are refused with named reasons. The control panel object stays: controls declared alike keep their values at the commit, others start at their defaults. Each change now reports `requires` (`preserve`, `reset`, `unsupported`), in the same words as `PlanDiff.kind`; `breaking` is read from it, older `PlanChange(..., breaking)` calls keep working, and a preserving update never resets. Results, sessions and receipts report a `processing_version` next to `graph_version`; receipts carry the reset's `receipt.cleanup`, a read-only `updates.Cleanup` (`state`: `pending`, `done` or `failed`; `errors`, `finished_at`, `wait(timeout)`) that cannot close, cancel or finish the engine's retirement, and `receipt.cleanup_failures`, for idle and active resets alike. Changed sources, inputs and recording remain unsupported. Applied candidates no longer keep their instances alive. `ActiveRun.apply_update` applies a reset candidate live: operators and reactions are built before the cut, then steps, operators, reactions, stage gates and domain progress are replaced while sources, readers and workers stay open; operator ordinals continue per name, ended sources are told to the new operators and partial windows are dropped, never flushed. The replaced reactions refuse later signals. Old processing closes in a documented order (run reactions, run operators, session reactions, engine-owned state) on a background thread that is started before the cut; a reset that cannot start one is refused with nothing changed. While a retirement is pending, the session refuses another reset, idle or in this or a later run; preserving updates still apply. `receipt.cleanup_failures` lists every cleanup failure known so far, also of a cleanup that finished after the call returned. The new reactions refuse signals until the resume. The new reaction runtime gets no second `started` event, and the assessment names the handlers subscribed to it (`handlers_on_started`); state machines initialize on their first event. The reset boundary is host-quiescent, not device-quiescent: the engine waits for declared futures and adds no device synchronization. `Block` documents that a reset constructs a new instance while the old one may still run, and drops the old one without a close or `reset_state()` call. Active receipts add raw `called_at`, `cut_at`, `drained_at`, `resumed_at` stamps and the durations `admission_pause_seconds`, `drain_after_cut_seconds`, `call_to_resume_seconds` and `run_build_seconds`; `drained_seconds` and `paused_seconds` keep their earlier meaning, from the start of the boundary reservation, so they include its lock waits. Resets are refused when a retained source and the new processing would hold different values of one resource (`source_resource_shared`, `source_resource_overridden`), compared by the values themselves under full resolver precedence and including handler-plan steps, so one object passed under two keys (`log` and `demo.log`) counts as shared; when a replacement state on another backend object has the current namespace (`state_isolation_unknown`); or when a replacement state uses the backend of the engine-owned state, which the reset closes (`state_backend_closes`). A live run that records (`recording_run`) or whose sources all ended (`sources_ended`) rules a reset out at assessment and preparation; the idle session can reset once the run finished.

- **V2 additive graph updates (opt-in)** — Compare compiled plans and prepare compatible additions while retaining existing block instances, local and managed state, controls and session resources. Idle sessions use `ExecutionSession.apply_update` or `update`; running active sessions use `ActiveRun.apply_update` to pause admission, drain admitted work, publish the new graph and resume with sources and temporal buffers retained. New prunable consumers can extend an unchanged control’s closure. Only added steps are constructed; incompatible edits are rejected, and results carry a `graph_version` separately from the control version. Active updates support handler patches and bounded waiting, and reject recording runs. Open passive pipelines must close before an update. Closing a busy session is rejected; closed sessions reject execution and updates. Examples 29 and 30 demonstrate retained state, new output demand, temporal windows, timeout/retry and rejection.

- **V2 source identity and observation time** — Active sources can read their declared name from `Source.source_name` before `open()`. Source authors can import `engine_observation()` and `ENGINE_CLOCK_ID` from `v2.sources` to timestamp individual members of a collected batch; the previous `v2.active.execution` imports remain available. The source lifecycle documentation now distinguishes construction on the caller thread from `open`, `read` and `close` on the reader thread.

- **V2 causal reset of shared static steps** — In a pipelined active run with several sources or operators, a `reset_on_enable` member that reads only workflow inputs runs in every source's route. A newer pulse from one source could reset it while an older pulse from another source was still upstream; the older call then ran against the reset instance. The pulse that resets such a member now first waits until all work admitted under older control versions has finished. Only pulses that reach such a member before its reset wait; ordinary updates, `keep_ticking` controls and source-bound members do not wait. A cancelled or failed run no longer raises a `TypeError` from the reset guard's abort check.

- **V2 tensor serialization** — Recording and native JSON outputs now accept singleton and empty tensor views with non-unit strides, including one-detection model results. Dtypes, shapes and values are preserved; existing recording and wire formats are unchanged.

- **V2 capture and retrospective analysis (opt-in)** — An active V2 definition can declare a root `recording` of selected output groups into a local file directory, and one `retrospective` stage that reads it later. The engine writes each delivered group in order during delivery, before user handlers. It keeps named fields, layouts, filtered entries, original source/temporal context and pulse identity. Images, tensors and native predictions are stored as binary tensors and decoded on the CPU. `Catalogue(codecs=...)` registers codecs for new payload types; a payload without a codec, a changed group schema, an existing directory and an unfinished recording fail explicitly. Workflow replay requires the lossless `block` policy; `latest` is rejected before replay starts. A workflow retrospective reads recorded fields as `$sources.<group>.<field>` through `plan.retrospective.start(...)`. Every replay has new block instances and fresh managed state, and never constructs or calls capture-side sources or blocks. Python retrospectives (`run_python`) iterate the whole recording lazily and repeatably, or receive one call per saved chunk, and register their own result files. Completed and gracefully stopped recordings are replayable; recording, failed and cancelled ones are rejected. Example 20 demonstrates capture, threshold replays, windows, two aligned sources, both Python modes and the error cases on synthetic CPU data. Definitions without these sections, and V1, are unchanged.

- **V2 events, reactions and managed state (opt-in)** — Blocks can declare events and emit them with source and temporal context. Separately compiled handler workflows run synchronously or through bounded asynchronous queues, with visible delivery, drop, failure and snapshot counters. Active runs accept external signals and expose workflow-defined state machines, including handler-selected transitions protected against outdated decisions. Graceful stop drains admitted event cascades; cancellation discards waiting work. Injectable global and per-source state provides atomic single-key operations in memory or through the optional Redis backend (`roboflow-workflows[redis]`), with caller-controlled sharing and explicit backend failures. Examples 17–19 demonstrate the behavior with two synthetic cameras, interactive acknowledgement/reset and real Redis clients in multiple processes. V1 remains unchanged; workflows without these features do not create reaction workers or state connections.

- **V2 payload lifetime and execution overhead** — Remove reference cycles created while building entries and projecting results. Completed runs no longer retain images and predictions through these helpers until cyclic garbage collection. Avoid a second traversal of resolved plain inputs and validate selectors without per-call selector objects; runtime mapping checks use the standard ABC. Validation, output metadata and serial/pipelined behavior are preserved.

- **V2 bounded pipelining (opt-in)** — `PipelineOptions(max_in_flight=...)` lets different runs or pulses of one V2 session execute at different stages at once, while serial execution stays the default and the reference. Every step, phase, operator push and group delivery admits one call at a time, in order per source (or per passive pipeline). So a stateful block sees its calls in order, while frame 1 can run phase A as frame 0 runs phase B. `phase_overlap = False` on a block or implementation holds one gate for its whole call. Active runs take `session.start(..., pipeline=PipelineOptions(...))` with a per-source `block` (lossless, bounded read-ahead) or `latest` (newest pending value only, dropped values counted) overload policy, plus `cancel()`. Passive sessions take `session.pipeline(options=...)`: a context manager whose `submit` returns a `Future` once an idle worker accepts the run, raising `PipelineFullError` when none is free in time. While a pipeline is open, `session.run` and a second pipeline raise. A failed submission aborts the pipeline; runs waiting behind it fail with `PipelineAbortedError`, and running block calls finish. Observer and error-handler callbacks of a pipelined run are serialized; `current_pulse_run_id()` attributes them. Counters report retained and executing work, stage waits, drops and result age. They count work items, never bytes or device memory. In an active definition, mutating a workflow input in place is now reported (warning by default, `MutationConflictError` with `mutation_conflicts="error"`), because every pulse shares it; serial behavior is unchanged. Development examples compare serial and pipelined ResNet-18 runs on CPU and MPS; no CUDA behavior or speedup is claimed.

- **V2 block phases and implementation selection** — Blocks can declare private acyclic phase graphs alongside explicit `run()` methods. Opt-in phase execution uses the existing passive, nested and active execution paths, with invocation-local intermediates and phase-attributed failures. Class-owned alternatives are selected at compile time from a declared target; plans explain the choice and resolve only the selected constructor resources. Introspection lists implementations and phases without loading models. Phase execution is serial in this version; ordinary blocks and V1 defaults are preserved. Runnable tensor-native ResNet-18 examples exercise CPU and optional MPS execution.

- **V2 alignment, windows and temporal blocks** — Active V2 definitions can declare root-level `operators`: `v2/align@v1` pairs samples of several inputs on a named clock (nearest within a tolerance, `drop`/`partial` for missing inputs, as named fields or an `[N]` batch) and `v2/window@v1` collects successive samples into one T axis, with a held parent-level reference and `drop`/`emit` for partial windows. Operator outputs (`$operators.<name>.<port>`) run through the same graph with their own pulses, causes, gates, output groups and nested workflows. Every operator input has an explicit retention bound; exceeding it, incompatible clocks and non-increasing timestamps are reported with operator attribution instead of forcing matches. Windows accept only stationary layouts without an existing T. Block outputs declare `first`, `last` or `selected` context policies; `Selected`/`Selection` choose members by full logical index, and a selection stays a collection even with one member. New V2 blocks: `v2/static_crop`, `v2/best_frame` and `v2/top_k_brightest`; `v2/mosaic` takes the last member's timestamp. Passive definitions accept explicit timestamped `[N,T]` inputs and reject operators. Operators are root declarations only in this version. Existing V1 behavior is unchanged.

- **V2 finite sources and output groups** — Add class-owned source declarations and an opt-in active execution path alongside passive invocation. Independent sources deliver named output groups with source identity, pulse identity and indexed timestamps; one shared graph preserves block state, nested workflows and conditional execution. Finite runs support bounded admission, cooperative stop/drain and attributed failures. Unaligned cross-source consumers are rejected pending explicit synchronization operators. Existing V1 behavior and passive V2 definitions remain unchanged.

- **V2 tensor-native media** — The opt-in V2 catalogue accepts tensor-backed images and all eleven native prediction/tensor kinds. Crop and resize operations preserve image identity and spatial transforms through nested workflows; prediction outputs can retain own coordinates or restore workflow-root coordinates with anisotropic scaling. Faithful serialization preserves native dtype, shape and provenance, including empty values; wildcard outputs handle native carriers, and the legacy `numpy_array` label accepts tensor-native depth values. Composite mosaics retain source-tile provenance and reject ambiguous restoration to contributing roots. V2 image block authors use `ImageData.tensor_image` and explicit pixel export at visualization boundaries; existing RGB NumPy workflow inputs remain accepted. Existing V1 behavior, defaults and dependencies are unchanged.

- **Opt-in Workflows V2 sequential engine** — Add class-owned block parameters and configuration selectors, pure compilation, persistent execution sessions, nested workflows, indexed batching and conditional execution. Selected values retain identity and are checked against kinds, shared field constraints and custom validators; literal normalization happens during compilation. Nested input boundaries retain kind checks and codecs; whole-child gates also govern forwarded outputs and their consumers. Configured outputs retain their declared kind validators and codecs. Resources support scoped factories, including constructor defaults. Dynamic blocks support session-shared state (explicitly shareable across sessions), execution context and input/output representation policies; `tensor_native` requires a capable policy. V2 has its own catalogue and structural/workload introspection. Development examples compare sequential V1/V2 behavior and exercise CPU image workflows. Select it through `roboflow_workflows.execution_engine.v2`; existing V1 defaults, discovery and execution remain unchanged. The earlier V2 draft's detached registry and `inputs`/`config` syntax are replaced by class-owned parameters and flat steps.

### Added

- Anthropic Claude block (`anthropic_claude@v5`): `claude-sonnet-5-5` model option.
### Changed

- Carries forward the `0.2.2` model catalog: Anthropic Claude v5 lists `claude-opus-5-5` (Claude Opus 5.5, 128000 max output tokens) and the temperature warning names Opus 5.x; OpenAI v7 lists `gpt-6-sol` and `gpt-6-luna` (reasoning effort `none` through `max`, structured absolute detection prompts).
- `prototypes.platform_errors`: `RoboflowAPINotAuthorizedError`, `RoboflowAPINotNotFoundError`, `RoboflowAPITimeoutError` and `RoboflowAPIConnectionError`, for hosts to raise and translate platform request failures. Names and bases match the `inference` server classes, which now re-export them; `RoboflowAPINotAuthorizedError` is not a `RoboflowAPIForbiddenError`.

### Fixed

- Inner Workflow block no longer imports `fastapi`, which only the `enterprise` extra installs; its `background_tasks` argument is typed with `BackgroundTaskScheduler`. `roboflow_workflows.execution_engine.core` now imports without `fastapi`.

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
