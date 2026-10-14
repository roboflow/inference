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

- **V2 annotation singleton columns** — Normalize noncanonical singleton inner strides before packing prediction columns as bytes, preserving exact integer class IDs and allowing single-detection visualization outputs.

- **V2 visualization consolidation** — Replace the five prototype detection painters with `accurate` and `fast` implementations of `roboflow_core/bounding_box_visualization@v1` and `roboflow_core/label_visualization@v2`. Accurate rendering remains the default. Fast boxes use approximate tensor outlines; fast labels cache opaque sprites and use reference rendering for other styles. Both retain canonical parameters, image metadata and copy ownership. Prototype type identifiers and the public drawing intermediate are removed; old prototype workflows must use the canonical blocks. Batch amortization, fused painting and input mutation remain separate follow-up work.

- **V2 model package layout** — Move `blocks.language` to `blocks.models.vision_language_models` and group the seven `blocks.models.roboflow_*` model families under `blocks.models.roboflow_platform`, dropping the `roboflow_` prefix from each Python family package. Public `roboflow_core/...@vX` block identifiers and parameters are unchanged.

- **V2 block definitions are explicit** — Block parameters, output declarations and ordinary execution now live in their family/version module. Shared model, geometry, drawing, state and transport algorithms remain shared. Concrete `Params` classes replace thin declaration bases. Some schema titles and direct Pydantic validation headers therefore change to `Params`; obsolete inherited descriptions are removed. Parameter validation and workflow compiler diagnostics are preserved.

- **V2 model configuration per call** — Model IDs, package IDs and configuration options now have one value for the entire image batch of a block invocation. Images and per-image user prompts remain batched; list-valued configuration such as class filters is one list for the call. Scalar input selectors can choose a different model or configuration on the next call. Per-image configuration selectors are rejected during compilation; use separate model steps for different models. A selected `None` model ID raises an error; it no longer skips an image. Direct empty image batches do not load or call a model. Custom instance-segmentation subclasses receive a call-local `ModelInput` in `_segmentations(network_input, raw_output)`, replacing `ModelGroup` and its per-row grouping fields.

- **V2 catalogue names and versioned packages** — Use `roboflow_core/...@vX` identifiers and organize block implementations by category, family and version. Ported blocks use their originating V1 family/version; native V2 blocks have distinct versioned identities. Matching identifiers do not imply identical V1/V2 parameters or runtime behavior. V1 is unchanged.

- **V2 mask consensus bounds** — Mask-derived consensus boxes use half-open pixel extents, preserving single-pixel masks and the full width and height of downstream crops. Continuous box coordinates are unchanged.

- **V2 nearest-neighbor coordinate checks** — Reject known conflicting image identities, crop offsets, resize scales or camera sources before computing distances. Distances, cutoffs and ties remain in the shared local pixel frame; callers align predictions upstream. Missing metadata remains unknown.

- **V2 provider error redaction** — Stability AI adapters redact provider-echoed API keys before shortening error diagnostics and discard original secret-bearing exception chains.

- **V2 retained default discovery (opt-in)** — Export and register the existing retained image, geometry, code, embedding/depth/OCR, generation, annotation, sink, notification and Secret-store blocks through the public catalogue. Construction retains caller-owned models and transports; service and device qualification remain separate. Retired families stay excluded.

- **V2 final active tail discovery (opt-in)** — Register the existing Qwen3.5 VLM, ten active language providers, seven analytics/motion blocks, ONVIF and three decorative drawing blocks through the public catalogue and package exports. Language outputs and input-only Secret values have explicit kind registration. Buffer maps to the existing `roboflow_core/window@v1` operator. Models and industrial transports remain caller-owned; optional SDKs and services are acquired only by explicit active operations. Real services/models/devices, supported-runtime qualification and retained family limits remain separate. V1 behavior and defaults are unchanged.

- **V2 platform metadata, search and events (opt-in)** — Add custom metadata, visual search, visual search classification, vision events and event bundles to default V2 discovery and class exports. Hosts supply explicit ports; compilation and schema discovery create no providers. Metadata and search retain their documented synchronous/sequential baseline. Examples 94 and 97 use fake ports; real services and release builds require separate qualification.

- **V2 drawing discovery (opt-in)** — Register trace/heatmap and polygon-zone/line-counter/reference-path visualization blocks in `create_catalogue()` and expose their classes from the V2 block packages. Whole-frame history is bounded per source; counter drawings share current producer geometry validation and per-image count selectors. See development examples 73 and 74. V1 remains unchanged.

- **V2 segmentation discovery (opt-in)** — Register `roboflow_core/mask_edge_snap@v1`, `roboflow_core/segment_anything@v1`, `roboflow_core/sam3@v3`, `roboflow_core/sam3_interactive@v1`, `roboflow_core/segment_anything_2_video@v1` and `roboflow_core/sam3_video@v1` in `create_catalogue()` and expose their block classes through the V2 block packages. SAM models come from the caller-owned `models` resource; hosted execution is deferred. Video steps require an explicit `max_session_frames` policy; frame/source limits do not bound bytes or objects per frame, and tracker identities belong to an episode. SAM3 concept peak-memory bounds and real-backend/device qualification remain open. See development examples 83, 86 and 107.

- **V2 Roboflow platform discovery (opt-in)** — Register asset attributes, dataset upload and model monitoring in `create_catalogue()` and export their three block classes. Hosts supply explicit ports and credentials; active calls without the required port fail with actionable guidance. Examples 95–96 use local fake ports; release builds and real platform services require separate qualification.

- **V2 local VLM and direct OCR discovery (opt-in)** — Add opt-in Florence-2, Moondream2, Cosmos 3 Edge, current unified Qwen v4 and direct Google Vision OCR adapters. Florence exposes decoded text and explicit grounding coordinate modes while retaining auto/V1 defaults. Local models use host-owned cache; provider transports and credential boundaries stay explicit. Real-model/provider/device qualification is separate. Examples 109 and 110.

- **V2 Action recognition discovery** — Register `roboflow_core/roboflow_action_recognition_model@v1` in the opt-in catalogue and export `ActionRecognitionBlock`. Caller-loaded local models classify bounded clips; see development example 106.

### Execution engine

- **V2 overlap analysis and crop rollup (opt-in)** — `create_catalogue()` registers `roboflow_core/overlap_analysis@v1` and `roboflow_core/detections_list_rollup@v1`. Overlap reports intersection/reference-area records and follows the reference entry metadata; callers supply comparable coordinates because no alignment is performed. Masks use their longest contour at an explicit CPU reference boundary, with selective reference-mask requests. Crop rollup maps child predictions into the parent frame, merges boxes, dense/RLE masks, views or native keypoints, preserves empty-family layout and validates child alignment and lineage. Known crop lineage overrides V1's integer-offset fallback; nested masks use the parent canvas. The float32 IoU prefilter can differ from latest V1 at higher-precision or threshold boundaries. Examples 57 and 58 document the contracts and CPU boundaries; no speedup is claimed.

- **V2 catalogue: tracking, shapes, measurements and image processing (opt-in)** — `create_catalogue()` adds 27 ports of latest V1 blocks: `trackers` library variants (`roboflow_core/trackers_sort@v1`, `roboflow_core/trackers_bytetrack@v1`, `roboflow_core/trackers_ocsort@v1`, `roboflow_core/trackers_botsort@v1`); track consumers (`roboflow_core/track_class_lock@v1`, `roboflow_core/stabilize_detections@v1`, `roboflow_core/velocity@v1`); `roboflow_core/image_stack@v1`; circle, corner, dot, ellipse, triangle and halo (`roboflow_core/halo_visualization@v2`) painters; distance, size, mask-area and pixel-colour measurements; grayscale, blur, threshold, morphology and contrast blocks. The retired `roboflow_core/byte_tracker@v3` identity is excluded from default V2 discovery; use `roboflow_core/trackers_bytetrack@v1`. V1 remains unchanged; README 61 compares the trackers. Deliberate differences from V1: annotation `color_axis: TRACK` draws an unchanged image for a frame with zero rows; size rejects masks whose raster differs from the prediction's image and malformed reference polygons; distance and mask-area calibration must be finite; halo clips saturated pixels instead of wrapping. CPU tensors use V1's OpenCV for blur, morphology, adaptive threshold and halo blur; other devices copy to the host where V1's algorithm requires it.

- **V2 custom Python representation policy** — The default representation policy accepts dynamic blocks declaring `tensor_compatibility: tensor_native` and passes V2 images and native predictions unchanged; no NumPy or Supervision conversion. `allow_local_code=True` is still required. A host policy derived from the base `RepresentationPolicy` still rejects `tensor_native` unless it overrides `check_compatibility`; the error names the policy and the remedy. Example 66.

- **V2 selected-parameter constraints** — Preserve bounds declared on selector alternatives, including selectors inside containers. A selected value cannot bypass those bounds through a permissive literal alternative; valid selected payloads keep their identity.

- **V2 configuration errors and source history** — Reject a bare selector during compilation when a block requires a list or dictionary of selectors, with an example of the accepted form. Window diagnostics explain how passive workflow inputs declare a stationary axis. Frame delay exposes a literal `max_sources` limit (default 256); exceeding it raises `FrameHistoryCapacityError` before the rejected call copies data or changes history, instead of silently evicting another source; earlier calls in a failed batch may already have advanced their histories. Reset clears the retained histories. Cache descriptions explain default-source sharing, the missing-value sentinel and the absence of automatic expiry. Examples 50–56 document these rules.

- **V2 routing, history and prediction comparison (opt-in)** — The standard catalogue adds switch, delta and rate-limiter gates; image copying and frame delay; object-cache get/set; detection consensus and difference; and embedding change/outlier sampling. Stateful blocks derive source identity from the relevant input even when another selected parameter has no metadata. Cache blocks share one lazily created cache per session; callers can supply a cache explicitly, and a processing reset recreates only the session-owned default. Direct image copies preserve device, spatial metadata and CUDA readiness. Blocks declare retained inputs/outputs so mutation diagnostics include buffers kept across calls. Difference retains V1's greedy matching and documents its float32 threshold boundary; sampling defines constant/zero-vector behavior and finite-window rank limits. Examples 50–56 exercise these capabilities.

- **V2 passive history (opt-in)** — A source-free session can retain operator history across calls when each operator declares `passive_max_emissions = 1`, reads one upstream domain, and uses a supported flush policy (`Window.partial="drop"`). Emitted work continues in the submission's existing worker team. Warming outputs are `None`; a gate filtering an emitted collection produces `[]`. Row counts follow selected output lineage. Cancellation consistently aborts admitted submissions; failed history requires processing reset. Example 55 demonstrates copying reusable image buffers before retaining them.

- **V2 CUDA label patch selection** — Label painting weighs extra patch dispatches against copying a larger enclosing region. The portable heuristic now groups the tested medium-density layout on CUDA, trading some Thor latency in that layout for a larger Orin improvement. CPU/MPS retain area-based selection; pixels and public block contracts are unchanged.

- **V2 query and formatter clarification** — Static query definitions validate at workflow compilation without being recompiled for each selected payload; selected template placeholders still validate at invocation time. Classification property descriptions distinguish single-label numbers from multi-label lists and name `top_class_confidence_single` for numeric gates. CSV discovery and examples explain gated final rows and constant-only columns; templates document how present `None` renders. No V1 behavior changes.

- **V2 catalogue: V1 block ports (opt-in)** — `create_catalogue()` of `roboflow_workflows.v2.blocks` registers ports of the latest V1 blocks, plus a native semantic painter: crops, slicing and preprocessing (`roboflow_core/absolute_static_crop@v1`, `roboflow_core/relative_statoic_crop@v1`, `roboflow_core/dynamic_crop@v1`, `roboflow_core/image_slicer@v2`, `roboflow_core/image_preprocessing@v1`); classification, multi-label classification, keypoint, instance and semantic segmentation models beside object detection; region and parent mapping (`geometry`); prediction filters and transformations (`transforms`); detection combine and merge (`combine`); the UQL gate `roboflow_core/continue_if@v1`; formatters (property definition, expression, string template, CSV, first non-empty value, current time, JSON parser); and the `annotations` painters (nine V1 visualization ports and `roboflow_core/semantic_segmentation_visualization@v1` for native class grids). The existing 21 types keep their classes and contracts. Where a prototype and a port are related, both stay registered under different types: `roboflow_core/continue_if_boolean@v1` takes a ready boolean and `roboflow_core/continue_if@v1` a UQL condition; `roboflow_core/simple_box_visualization@v1` and `roboflow_core/simple_label_visualization@v1` remain fast prototypes beside the ports `roboflow_core/bounding_box_visualization@v1` and `roboflow_core/label_visualization@v2`. Their class descriptions now say which is which. `roboflow_workflows.v2.blocks.LabelVisualizationBlock` is still the prototype; the port is exported there as `AnnotationLabelVisualizationBlock`. The package also exports every newly registered block class, `TIMESTAMP_KIND` (output kind of `roboflow_core/current_time@v1`, registered with that block) and the gate's `HoldCapacityError` and `ObservationOrderError`; `control` and `formatters` export them too, and `formatters.BLOCKS` lists all seven formatters. Building the catalogue loads no model and creates no resource; model blocks still read the host's `models` resource. Examples 42–49 use `create_catalogue()` and add only their own stand-in blocks. The generic engine, catalogue duplicate rules and V1 registration are unchanged.

- **V2 native data foundations (opt-in)** — `FrameRef` describes the image a prediction was made on (id, size, parent and workflow-root mapping, mosaic sources, video metadata) without reading pixels; `FrameRef.prediction_metadata()` adds a versioned `workflows_v2_frame` record next to the legacy keys, and `frame_of` reads it back and rejects records that disagree with the legacy keys. `PredictionRows` is an opt-in carrier of a native prediction plus aligned `(N, ...)` tensor columns (kind `prediction_rows`); one row selection applies to both. `native_semantic_segmentation` carries the unchanged `SemanticSegmentationResult` (label grid and winning confidence); the per-class RLE `semantic_segmentation_prediction` is unchanged. `NATIVE_KINDS` now lists every native carrier, so it also includes `native_semantic_segmentation`, which has no V1 tensor-native counterpart. A semantic grid of a cropped or resized image refuses `root`/`parent` output conversion instead of losing data. `create_catalogue()` registers both kinds, and the `*` media wildcard serializes, decodes and converts both through their own kinds. Row selection no longer imports V1 `tensor_native`. Recording codecs `v2/PredictionRows@1` and `v2/SemanticSegmentationResult@1` are added. Example 38 demonstrates frames, rows and semantic results.

- **V2 model cache and detector selection (opt-in)** — `ModelCache` loads each distinct `ModelRequest(model_id, model_package_id)` once, with concurrent identical misses sharing one load; a failed load is not cached and the next call retries. `auto_model_loader(**options)` imports `inference_models.AutoModel` lazily on first load. A host can still pass a plain mapping of loaded models; it never loads. `roboflow_core/roboflow_object_detection_model@v3` takes `model_id` and an optional `model_package_id`, each a literal or a string selector; each non-empty batch uses one model request and one `pre_process`/`forward`/`post_process` sequence, preserving image order. Its predictions now carry the `FrameRef.prediction_metadata()` frame record (source, video time and root geometry) besides the existing model metadata, class names and prediction type.

- **V2 mask providers (opt-in)** — `roboflow_workflows.v2.data.masks` adds `GridMaskProvider`, `DenseMaskProvider` and `RLEMaskProvider` behind `mask_provider_of`. A caller asks for binary instance masks the same way for low-resolution score grids, dense masks and COCO RLE; each answer states whether it is the representation's reference result or a labelled approximation. `SegmentationView.select` now uses the shared row selection: it copies selected row-metadata dicts, shares tensors for an identity selection and validates detections with the object detection kind. Example 40 demonstrates the providers.

- **V2 query foundation (internal)** — `roboflow_workflows.v2.blocks.query` validates existing UQL JSON once and evaluates it on plain Python runtime values. Literals that need prediction, image, frame-metadata or zone adapters fail at compile time with an "adapter pending" error. No filter, transformation or expression block uses it yet. Example 41 demonstrates validation and evaluation.

- **V2 package layout and alpha import migration** — Opt-in V2 code lives in `roboflow_workflows.v2`: `engine` (compilation, execution, sources, operators, reactions, state, recording and updates), `data` (batches, kinds, images, native predictions, masks and `ContractError`), `blocks` and `recording_codecs`. The temporary `roboflow_workflows.execution_engine.v2` namespace is removed, including its forwarding modules; imports through it now fail. Use `from roboflow_workflows.v2.engine import ...` for engine APIs, `roboflow_workflows.v2.data` for data APIs and `roboflow_workflows.v2.blocks` for packaged blocks. Discover the packaged catalogue with `Catalogue.from_modules(["roboflow_workflows.v2.blocks"])`. `data` imports no engine runtime, `engine` imports no block family, and importing `roboflow_workflows.v2` loads neither. `ContractError` is defined in `roboflow_workflows.v2.data.errors`; `engine.errors` re-exports the same class. Classes and V2 logger names use canonical modules. Block contracts and their digests are unchanged; V1 behavior, imports and defaults are unchanged.

- **V2 implementation contributions (opt-in)** — A host can add implementations to an existing V2 block without changing its class, its catalogue or any workflow definition. `Catalogue.with_contributions([ImplementationContribution(...)])` returns a new catalogue whose description of the block lists the added `Implementation` classes before its own, in the host's order. A contribution names the block's canonical type and its `contract_digest`, a versioned digest (`v1:sha256:...`) of the parameter schema and selectors, outputs, effects and events that ignores documentation, `Params` and nested model class names, aliases, metadata and module paths. A different contract, digest version, alias, an implementation that restates the contract, narrows `run()`, requests `managed_state`, repeats a name or serves a quality label the block's own implementations do not serve is rejected; `Catalogue.contribution_problem` reports the reason without raising. Selection by target capabilities and quality labels, plan pinning and selected-only resource resolution are unchanged. Each added `ImplementationSpec` records an `ImplementationOrigin` (provider, version, contract), shown by `describe()` only for contributed implementations, so plan comparison requires a reset when the provider or its version changes, and also when the block's implementation list changes without changing its selection; re-applying identical contributions preserves the graph. `Catalogue.merge` keeps one block class once when its implementation lists match and now rejects different lists instead of silently keeping the first. A contributed implementation must match the block's recorded parity criterion and its readiness, lifetime and mutation rules (`contributions` module documentation). Example 37 adds a host-supplied box painter to `roboflow_core/simple_box_visualization@v1` with pixel-identical output. Catalogues without contributions, their descriptions and V1 are unchanged.

- **V2 parallel workflow branches (opt-in)** — `PipelineOptions.max_workflow_parallelism` adds reusable worker teams for independent ready branches and their joins, including nested workflows. Its default is 1; the engine owns `max_in_flight * max_workflow_parallelism` execution threads, including team leaders. Pipelined temporal operators admit downstream emissions separately, allowing progress with `max_in_flight=1`; pending emissions participate in drain, updates and cancellation. A worker that finishes a pulse runs the next waiting one itself, so a chained emission costs no extra dispatcher round trip. Older control versions run first, then deeper pending operator emissions, while preserving order within each domain. The keyword-only `max_pending_operator_pulses` guard defaults to 4096 pending emissions and fails the active run with an attributed error on overflow; it bounds a count, not retained bytes. Compilation includes output readers in conflicting-mutation diagnostics and logs successful compilation warnings once per plan scope. Conflicts remain warning-only by default and warnings add no ordering; strict rejection is opt-in. Example 36 demonstrates branches, operators, live updates, drain and cancellation. V1 and serial execution defaults are unchanged.

- The V2 ByteTrack block accepts an explicit nominal `frame_rate` for live sources whose decoder reports no finite positive fps. Without an override it uses source metadata. The value scales the lost-track buffer; it is not processing throughput and does not age tracks for frames dropped upstream.

- V2 live-update assessment compares nested handler and retrospective plans against equivalent catalogues by their runtime contracts. Recompiling an unchanged capture/replay workflow no longer falsely requires a reset or fails as unsupported. Changed runtime kind validators and providers remain conservative; recording codecs cannot be changed or removed across an update, while unrelated additions are allowed. Live changes during recording remain unsupported.

- The opt-in V2 catalogue includes batched, phased object detection with caller-loaded models, per-source ByteTrack with an explicit source bound, separate box/label painters and a fused detection painter. Drawing can share prepared detections; in-place labels declare their mutation. Painters draw with the V1 tensor primitives on CPU and CUDA; a host can add other implementations through catalogue contributions. Painter steps take no `backend` parameter; a definition that still sets it fails compilation with a message naming the removed field. Painters join pending work on the caller CUDA stream and bound sprite caches across changing image widths. These blocks do not download or own models; legacy block registration and defaults are unchanged.

- The opt-in V2 Python API supports execution sessions as context managers, exposes host-facing execution/resource/update errors and the update receipt, and documents explicit video-source catalogue discovery. Session exit keeps the existing rule that active runs and pipelines must finish first; caller-owned model and service resources remain caller-owned. Legacy engine selection is unchanged.

- **V2 explicit processing reset of idle sessions and live runs (opt-in)** — `ExecutionSession.assess_update(plan)` tells, without constructing anything or writing state, whether an update preserves the graph, needs a reset or is unsupported, listing separately what rules out a preserving update (`preserve_blocked_by`) and what rules out a reset (`reset_blocked_by`), and what a reset would replace and keep (steps, operators, handler sessions, controls, managed state and per-step resource sources). `prepare_update(plan, reset=True)` and `update(plan, reset=True)` construct every step and handler session again in a fresh resolver: block-local state restarts, `Factory` values are created again, caller values are passed again. Engine-owned managed state starts fresh with declared defaults; replacement assessments distinguish closing the previous engine-owned service from leaving previous caller-owned state untouched. A caller's service is kept with its values or explicitly replaced by an isolated one (another backend or namespace), and is never cleared or closed. Resets that would change state defaults on a kept service, or split a source from the state it shares, are refused with named reasons. The control panel object stays: controls declared alike keep their values at the commit, others start at their defaults. Each change now reports `requires` (`preserve`, `reset`, `unsupported`), in the same words as `PlanDiff.kind`; `breaking` is read from it, older `PlanChange(..., breaking)` calls keep working, and a preserving update never resets. Results, sessions and receipts report a `processing_version` next to `graph_version`; receipts carry the reset's `receipt.cleanup`, a read-only `updates.Cleanup` (`state`: `pending`, `done` or `failed`; `errors`, `finished_at`, `wait(timeout)`) that cannot close, cancel or finish the engine's retirement, and `receipt.cleanup_failures`, for idle and active resets alike. Changed sources, inputs and recording remain unsupported. Applied candidates no longer keep their instances alive. `ActiveRun.apply_update` applies a reset candidate live: operators and reactions are built before the cut, then steps, operators, reactions, stage gates and domain progress are replaced while sources, readers and workers stay open; operator ordinals continue per name, ended sources are told to the new operators and partial windows are dropped, never flushed. The replaced reactions refuse later signals. Old processing closes in a documented order (run reactions, run operators, session reactions, engine-owned state) on a background thread that is started before the cut; a reset that cannot start one is refused with nothing changed. While a retirement is pending, the session refuses another reset, idle or in this or a later run; preserving updates still apply. `receipt.cleanup_failures` lists every cleanup failure known so far, also of a cleanup that finished after the call returned. The new reactions refuse signals until the resume. The new reaction runtime gets no second `started` event, and the assessment names the handlers subscribed to it (`handlers_on_started`); state machines initialize on their first event. The reset boundary is host-quiescent, not device-quiescent: the engine waits for declared futures and adds no device synchronization. `Block` documents that a reset constructs a new instance while the old one may still run, and drops the old one without a close or `reset_state()` call. Active receipts add raw `called_at`, `cut_at`, `drained_at`, `resumed_at` stamps and the durations `admission_pause_seconds`, `drain_after_cut_seconds`, `call_to_resume_seconds` and `run_build_seconds`; `drained_seconds` and `paused_seconds` keep their earlier meaning, from the start of the boundary reservation, so they include its lock waits. Resets are refused when a retained source and the new processing would hold different values of one resource (`source_resource_shared`, `source_resource_overridden`), compared by the values themselves under full resolver precedence and including handler-plan steps, so one object passed under two keys (`log` and `demo.log`) counts as shared; when a replacement state on another backend object has the current namespace (`state_isolation_unknown`); or when a replacement state uses the backend of the engine-owned state, which the reset closes (`state_backend_closes`). A live run that records (`recording_run`) or whose sources all ended (`sources_ended`) rules a reset out at assessment and preparation; the idle session can reset once the run finished.

- **V2 additive graph updates (opt-in)** — Compare compiled plans and prepare compatible additions while retaining existing block instances, local and managed state, controls and session resources. Idle sessions use `ExecutionSession.apply_update` or `update`; running active sessions use `ActiveRun.apply_update` to pause admission, drain admitted work, publish the new graph and resume with sources and temporal buffers retained. New prunable consumers can extend an unchanged control’s closure. Only added steps are constructed; incompatible edits are rejected, and results carry a `graph_version` separately from the control version. Active updates support handler patches and bounded waiting, and reject recording runs. Open passive pipelines must close before an update. Closing a busy session is rejected; closed sessions reject execution and updates. Examples 29 and 30 demonstrate retained state, new output demand, temporal windows, timeout/retry and rejection.

- **V2 source identity and observation time** — Active sources can read their declared name from `Source.source_name` before `open()`. Source authors can import `engine_observation()` and `ENGINE_CLOCK_ID` from `roboflow_workflows.v2.engine.sources` to timestamp individual members of a collected batch; the previous `roboflow_workflows.v2.engine.active.execution` imports remain available. The source lifecycle documentation now distinguishes construction on the caller thread from `open`, `read` and `close` on the reader thread.

- **V2 causal reset of shared static steps** — In a pipelined active run with several sources or operators, a `reset_on_enable` member that reads only workflow inputs runs in every source's route. A newer pulse from one source could reset it while an older pulse from another source was still upstream; the older call then ran against the reset instance. The pulse that resets such a member now first waits until all work admitted under older control versions has finished. Only pulses that reach such a member before its reset wait; ordinary updates, `keep_ticking` controls and source-bound members do not wait. A cancelled or failed run no longer raises a `TypeError` from the reset guard's abort check.

- **V2 tensor serialization** — Recording and native JSON outputs now accept singleton and empty tensor views with non-unit strides, including one-detection model results. Dtypes, shapes and values are preserved; existing recording and wire formats are unchanged.

- **V2 capture and retrospective analysis (opt-in)** — An active V2 definition can declare a root `recording` of selected output groups into a local file directory, and one `retrospective` stage that reads it later. The engine writes each delivered group in order during delivery, before user handlers. It keeps named fields, layouts, filtered entries, original source/temporal context and pulse identity. Images, tensors and native predictions are stored as binary tensors and decoded on the CPU. `Catalogue(codecs=...)` registers codecs for new payload types; a payload without a codec, a changed group schema, an existing directory and an unfinished recording fail explicitly. Workflow replay requires the lossless `block` policy; `latest` is rejected before replay starts. A workflow retrospective reads recorded fields as `$sources.<group>.<field>` through `plan.retrospective.start(...)`. Every replay has new block instances and fresh managed state, and never constructs or calls capture-side sources or blocks. Python retrospectives (`run_python`) iterate the whole recording lazily and repeatably, or receive one call per saved chunk, and register their own result files. Completed and gracefully stopped recordings are replayable; recording, failed and cancelled ones are rejected. Example 20 demonstrates capture, threshold replays, windows, two aligned sources, both Python modes and the error cases on synthetic CPU data. Definitions without these sections, and V1, are unchanged.

- **V2 events, reactions and managed state (opt-in)** — Blocks can declare events and emit them with source and temporal context. Separately compiled handler workflows run synchronously or through bounded asynchronous queues, with visible delivery, drop, failure and snapshot counters. Active runs accept external signals and expose workflow-defined state machines, including handler-selected transitions protected against outdated decisions. Graceful stop drains admitted event cascades; cancellation discards waiting work. Injectable global and per-source state provides atomic single-key operations in memory or through the optional Redis backend (`roboflow-workflows[redis]`), with caller-controlled sharing and explicit backend failures. Examples 17–19 demonstrate the behavior with two synthetic cameras, interactive acknowledgement/reset and real Redis clients in multiple processes. V1 remains unchanged; workflows without these features do not create reaction workers or state connections.

- **V2 payload lifetime and execution overhead** — Remove reference cycles created while building entries and projecting results. Completed runs no longer retain images and predictions through these helpers until cyclic garbage collection. Avoid a second traversal of resolved plain inputs and validate selectors without per-call selector objects; runtime mapping checks use the standard ABC. Validation, output metadata and serial/pipelined behavior are preserved.

- **V2 bounded pipelining (opt-in)** — `PipelineOptions(max_in_flight=...)` lets different runs or pulses of one V2 session execute at different stages at once, while serial execution stays the default and the reference. Every step, phase, operator push and group delivery admits one call at a time, in order per source (or per passive pipeline). So a stateful block sees its calls in order, while frame 1 can run phase A as frame 0 runs phase B. `phase_overlap = False` on a block or implementation holds one gate for its whole call. Active runs take `session.start(..., pipeline=PipelineOptions(...))` with a per-source `block` (lossless, bounded read-ahead) or `latest` (newest pending value only, dropped values counted) overload policy, plus `cancel()`. Passive sessions take `session.pipeline(options=...)`: a context manager whose `submit` returns a `Future` once an idle worker accepts the run, raising `PipelineFullError` when none is free in time. While a pipeline is open, `session.run` and a second pipeline raise. A failed submission aborts the pipeline; runs waiting behind it fail with `PipelineAbortedError`, and running block calls finish. Observer and error-handler callbacks of a pipelined run are serialized; `current_pulse_run_id()` attributes them. Counters report retained and executing work, stage waits, drops and result age. They count work items, never bytes or device memory. In an active definition, mutating a workflow input in place is now reported (warning by default, `MutationConflictError` with `mutation_conflicts="error"`), because every pulse shares it; serial behavior is unchanged. Development examples compare serial and pipelined ResNet-18 runs on CPU and MPS; no CUDA behavior or speedup is claimed.

- **V2 block phases and implementation selection** — Blocks can declare private acyclic phase graphs alongside explicit `run()` methods. Opt-in phase execution uses the existing passive, nested and active execution paths, with invocation-local intermediates and phase-attributed failures. Class-owned alternatives are selected at compile time from a declared target; plans explain the choice and resolve only the selected constructor resources. Introspection lists implementations and phases without loading models. Phase execution is serial in this version; ordinary blocks and V1 defaults are preserved. Runnable tensor-native ResNet-18 examples exercise CPU and optional MPS execution.

- **V2 alignment, windows and temporal blocks** — Active V2 definitions can declare root-level `operators`: `roboflow_core/align@v1` pairs samples of several inputs on a named clock (nearest within a tolerance, `drop`/`partial` for missing inputs, as named fields or an `[N]` batch) and `roboflow_core/window@v1` collects successive samples into one T axis, with a held parent-level reference and `drop`/`emit` for partial windows. Operator outputs (`$operators.<name>.<port>`) run through the same graph with their own pulses, causes, gates, output groups and nested workflows. Every operator input has an explicit retention bound; exceeding it, incompatible clocks and non-increasing timestamps are reported with operator attribution instead of forcing matches. Windows accept only stationary layouts without an existing T. Block outputs declare `first`, `last` or `selected` context policies; `Selected`/`Selection` choose members by full logical index, and a selection stays a collection even with one member. New V2 blocks: `roboflow_core/static_crop@v1`, `roboflow_core/best_frame@v1` and `roboflow_core/top_k_brightest@v1`; `roboflow_core/mosaic@v1` takes the last member's timestamp. Passive definitions accept explicit timestamped `[N,T]` inputs and reject operators. Operators are root declarations only in this version. Existing V1 behavior is unchanged.

- **V2 finite sources and output groups** — Add class-owned source declarations and an opt-in active execution path alongside passive invocation. Independent sources deliver named output groups with source identity, pulse identity and indexed timestamps; one shared graph preserves block state, nested workflows and conditional execution. Finite runs support bounded admission, cooperative stop/drain and attributed failures. Unaligned cross-source consumers are rejected pending explicit synchronization operators. Existing V1 behavior and passive V2 definitions remain unchanged.

- **V2 tensor-native media** — The opt-in V2 catalogue accepts tensor-backed images and all eleven native prediction/tensor kinds. Crop and resize operations preserve image identity and spatial transforms through nested workflows; prediction outputs can retain own coordinates or restore workflow-root coordinates with anisotropic scaling. Faithful serialization preserves native dtype, shape and provenance, including empty values; wildcard outputs handle native carriers, and the legacy `numpy_array` label accepts tensor-native depth values. Composite mosaics retain source-tile provenance and reject ambiguous restoration to contributing roots. V2 image block authors use `ImageData.tensor_image` and explicit pixel export at visualization boundaries; existing RGB NumPy workflow inputs remain accepted. Existing V1 behavior, defaults and dependencies are unchanged.

- **Opt-in Workflows V2 sequential engine** — Add class-owned block parameters and configuration selectors, pure compilation, persistent execution sessions, nested workflows, indexed batching and conditional execution. Selected values retain identity and are checked against kinds, shared field constraints and custom validators; literal normalization happens during compilation. Nested input boundaries retain kind checks and codecs; whole-child gates also govern forwarded outputs and their consumers. Configured outputs retain their declared kind validators and codecs. Resources support scoped factories, including constructor defaults. Dynamic blocks support session-shared state (explicitly shareable across sessions), execution context and input/output representation policies; `tensor_native` requires a capable policy. V2 has its own catalogue and structural/workload introspection. Development examples compare sequential V1/V2 behavior and exercise CPU image workflows. Select it through `roboflow_workflows.v2.engine`; existing V1 defaults, discovery and execution remain unchanged. The earlier V2 draft's detached registry and `inputs`/`config` syntax are replaced by class-owned parameters and flat steps.

### Added

- The nine Roboflow platform blocks are core blocks of this package again: dataset upload v1 and v2, custom metadata, model monitoring inference aggregator, vision events, vision event bundle, asset library attributes, visual search and visual search classifier. They live in `core_steps.sinks.roboflow` and `core_steps.integrations.roboflow`, so a standalone install now lists 248 blocks. Inference 1.6.1 to 1.7.2 shipped them as the `inference.roboflow_workflows_plugin` plugin; that plugin is now an empty compatibility stub.
- `RoboflowPlatformClient` gains the operations these blocks call: `get_roboflow_workspace`, `add_custom_metadata`, `register_image_at_roboflow`, `annotate_image_at_roboflow`, `update_image_metadata_at_roboflow`, `batch_update_image_metadata_at_roboflow`, `search_project_images_at_roboflow`, `send_inference_results_to_model_monitoring`, `get_device_id`, `get_server_version` and `get_system_info`. The offline client refuses the API operations. A host can register a fallback client for these blocks with `set_platform_blocks_client_factory`, used when a block gets no client, the offline client, or a client that does not implement these operations itself.
- Vision events sends its cloud image upload and event request only through a client that implements these operations. With the offline client it returns `error_status` with the offline refusal and sends nothing; writing to a local event store is unaffected.

### Changed

- `/workflows/blocks/describe` lists these blocks under their names from before the plugin, `inference.core.workflows.core_steps.{sinks,integrations}.roboflow.…`, and in their original positions within the core list, instead of `inference.roboflow_workflows_plugin.…` at the end.
- A disable pattern on the canonical module path, such as `roboflow_workflows.core_steps.sinks`, now also disables these blocks, like any other core block. Patterns written against either historic path keep matching the same blocks.
- OpenAI v7: optional `service_tier` (`auto`, `default`, `flex`, `priority`, `ultrafast`), sent with direct OpenAI requests and forwarded to the Roboflow proxy for `rf_key:` keys. Unset keeps OpenAI's default tier.


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
