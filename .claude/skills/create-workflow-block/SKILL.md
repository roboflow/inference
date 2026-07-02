---
name: create-workflow-block
description: Author a new Roboflow Workflows block — a `roboflow_core/{family}@v{N}` step under `inference/core/workflows/core_steps/**` (models, transforms, sinks, visualizations, flow control). Covers the compiler/execution-engine mental model, the manifest + `run()` contract, kinds & selectors, batch/dimensionality, loader/serializer registration, versioning, and a per-category map of ~210 existing blocks to pattern-match. Trigger when the user asks to "create a workflow block", "add a workflow block", "expose X as a workflow step", or wire a model/transform/sink/visualization into Workflows.
---

# Creating a Workflow block

The playbook for authoring a new Roboflow **Workflows block** — a
`roboflow_core/{family}@v{N}` step under `inference/core/workflows/core_steps/`.

**How to use this skill.** Read *What a Workflow is* and *Anatomy of a block* for the
basics, then jump to the *Block category* that matches what you are building for its
mental model, author nuances, and canonical example blocks to pattern-match. This
skill is a **map**: the authoritative depth lives in `docs/workflows/**` — each
section points at the exact doc and example block to read. Always match an existing
sibling block before writing a line. Reviewer-side counterpart: `review-workflows-blocks`.

## What a Workflow is (mental model)

A Workflow is a JSON **definition** (inputs + steps + outputs) written in the Workflows language. It is not run directly — a **Compiler** parses it against the pool of installed blocks, and an **Execution Engine** (EE) runs the resulting DAG. As a block author you never touch the EE internals; you write a Python class the EE instantiates and calls.

- **Definition → Compiler → Execution Engine.** Each definition declares an EE version (`version: 1.x` means `>=1.x,<2.0.0`). A **step** is one instance of a **block** (`{"type": "...@v1", "name": "my_step", ...}`); step inputs are either literals or **selectors** (`$inputs.<name>`, `$steps.<step>.<output>`) that reference data resolved at runtime. Concepts: `docs/workflows/understanding.md`, `docs/workflows/workflow_execution.md`, `docs/workflows/definitions.md`.
- **Data flows as typed `kinds`.** A step output has a `kind` (e.g. `image`, `object_detection_prediction`); a downstream input declaring the same kind is assumed compatible — this gives compile-time verification without runtime checks. See `docs/workflows/kinds/index.md`.
- **Batch orientation & dimensionality.** The EE is batch-oriented: it fans data through steps as batches, and each datapoint sits at a **dimensionality level**. A block can keep, increase (e.g. crop-per-detection), or decrease that level — this is the single most important concept for non-trivial blocks. See `docs/workflows/workflow_execution.md` and the dimensionality section of `docs/workflows/create_workflow_block.md`.
- Compiler/EE roles: `docs/workflows/workflows_compiler.md`, `docs/workflows/workflows_execution_engine.md`.

## Anatomy of a block

A block is **two classes in a `vN.py` module**: a `WorkflowBlockManifest` (the schema/prototype for a step declaration) and a `WorkflowBlock` (the logic). Primary guide: `docs/workflows/create_workflow_block.md`. Confirmed real shape: `inference/core/workflows/core_steps/transformations/detection_offset/v1.py`.

**Manifest** — a `pydantic` model subclassing `WorkflowBlockManifest`:

- `type: Literal["roboflow_core/<family>@v1", "<Alias>"]` — the **discriminator** the Compiler uses to pick this manifest when parsing a step. The `Literal` may carry a legacy alias as a second value (see `detection_offset/v1.py:94`).
- **Inputs** are ordinary fields. Use `Selector(kind=[...])` for data references and `Union[<literal type>, Selector(kind=[...])]` when a value may be hardcoded or selected (e.g. `Union[PositiveInt, Selector(kind=[INTEGER_KIND])]`). Wrap each with pydantic `Field(description=..., examples=...)` — descriptions/examples power the UI. `model_config = ConfigDict(json_schema_extra={...})` carries UI metadata (name, block_type, icon).
- `@classmethod describe_outputs() -> List[OutputDefinition]` — one `OutputDefinition(name=..., kind=[...])` per output the `run()` dict must supply. For manifest-dependent outputs, return a single `name="*", kind=[WILDCARD_KIND]` and add instance method `get_actual_outputs()`.
- `@classmethod get_execution_engine_compatibility() -> Optional[str]` — semver range, e.g. `">=1.3.0,<2.0.0"`; gates loading.
- **Batch/dimensionality hooks** (all classmethods, opt-in): `get_parameters_accepting_batches()` → names of params delivered as `Batch[...]`; `get_parameters_accepting_batches_and_scalars()` → mixed params; `get_output_dimensionality_offset()` → `+1`/`-1` when the block changes nesting level.

**Block** — subclass `WorkflowBlock`:

- `@classmethod get_manifest() -> Type[WorkflowBlockManifest]` returns the manifest class.
- `def run(self, ...) -> BlockResult` — the EE calls this with kwargs matching manifest input names. Image inputs arrive as `WorkflowImageData` (use `.numpy_image`); scalar params arrive as plain values; batch params (if declared) arrive as `Batch[...]`. Return a dict `{output_name: value}` (or, for batch/dimensionality-increasing blocks, a **list** of such dicts, one per element — a `None` entry drops that datapoint downstream). `__init__` may hold reusable state (persists across `run()` calls, e.g. per-frame video).
- Imports come from `inference.core.workflows.prototypes.block` (`WorkflowBlock`, `WorkflowBlockManifest`, `BlockResult`) and `.execution_engine.entities.base` (`Batch`, `OutputDefinition`, `WorkflowImageData`).

**Registration.** A block is invisible until its class is returned from the plugin's `load_blocks()`. For `roboflow_core`, add the import + list entry in `inference/core/workflows/core_steps/loader.py` (see `DetectionOffsetBlockV1` at lines 512 / 815). Custom **kinds** and their (de)serializers are exposed via `load_kinds()` / `KINDS_SERIALIZERS` in the plugin `__init__.py` — see `docs/workflows/blocks_bundling.md`.

## Kinds & selectors (the type system)

**Kinds** are the Workflows type system (`docs/workflows/kinds/index.md`). Each kind pairs a **semantic name** (`image`, `point`), a **Python representation** blocks receive (e.g. `object_detection_prediction` → `sv.Detections`), and an optional **serialized representation** for the wire. There is no polymorphism — express alternatives as a **union**, i.e. `Selector(kind=[A_KIND, B_KIND])` (see the three-kind predictions input in `detection_offset/v1.py:95-100`).

- **Where kinds live:** defined as `Kind(...)` constants in `inference/core/workflows/execution_engine/entities/types.py` (e.g. `IMAGE_KIND`, `OBJECT_DETECTION_PREDICTION_KIND`, `FLOAT_ZERO_TO_ONE_KIND`, `WILDCARD_KIND`). **Where documented:** one page per kind under `docs/workflows/kinds/` (autogenerated index in `kinds/index.md`).
- **Selector vs literal.** `Selector(kind=[...])` accepts only runtime references (`$inputs.*` / `$steps.*.*`); a bare Python type accepts only hardcoded values; `Union[type, Selector(kind=[...])]` accepts both. `StepSelector` (`$steps.<step>`, no output) marks a **flow-control** block.
- **Batch vs non-batch kinds.** A kind name is orthogonal to batching — whether a param arrives as a scalar or `Batch[...]` is decided by `get_parameters_accepting_batches()`, not the kind. (Historically `Batch[X]` vs `X` were separate kinds; unified in inference `0.18.0` — see the warning in `kinds/index.md`.) Deeper representation notes: `docs/workflows/internal_data_types.md`.

## Versioning & bundling

Rules from `docs/workflows/versioning.md` and `docs/workflows/blocks_bundling.md`:

- **Bug-fix in place; anything else is a new version.** Only patch the existing `vN.py` for bug fixes. Behavioral/interface changes create `v(N+1).py` in a new module under the block package — **stability over DRY**, code duplication is accepted and blocks should stay independent.
- **Type identifiers & aliases.** Convention `{plugin}/{block_family}@v{X}` (e.g. `roboflow_core/detection_offset@v1`). The `type` `Literal` may list a legacy alias (`"DetectionOffset"`) so old definitions keep parsing.
- **EE compatibility.** Every manifest returns a semver range from `get_execution_engine_compatibility()`; if a block needs a feature added in `1.3.7`, declare `">=1.3.7,<2.0.0"`. A definition's `version: 1.1.0` resolves to `>=1.1.0,<2.0.0`. History: `docs/workflows/execution_engine_changelog.md`.
- **Plugin layout & the `__init__.py` requirement.** A plugin is a Python package: `{plugin}/{block_name}/v1.py` per block, plus a main `__init__.py` exposing `load_blocks()` (required), optionally `load_kinds()`, `REGISTERED_INITIALIZERS` (default block-constructor params), and `KINDS_SERIALIZERS`/`KINDS_DESERIALIZERS`. External plugins load via `WORKFLOWS_PLUGINS="plugin_a,plugin_b"`. Forgetting to register in `load_blocks()` (or `core_steps/loader.py` for core) means the block does not exist to the EE.

## Going deep (assets)

Curated index — read the doc for the matching need:

- `docs/workflows/create_workflow_block.md` — PRIMARY. Full walkthrough: manifest → block → registration, plus advanced batch inputs, flow-control blocks, nested/named selectors, and input/output dimensionality vs `run()` signature.
- `docs/workflows/custom_python_code_blocks.md` — when authoring an in-definition Python block instead of a packaged plugin block.
- `docs/workflows/inner_workflow_design.md` — when your block runs a nested/sub-workflow.
- `docs/workflows/testing.md` — how to write unit + integration tests (also `create_workflow_block.md` integration-test template).
- `docs/workflows/blocks_connections.md` — how kinds drive which steps can connect, and UI connection semantics.
- `docs/workflows/workflows_execution_engine.md` + `docs/workflows/workflow_execution.md` — EE internals, batch fan-out, dimensionality, empty/conditional datapoints.
- `docs/workflows/internal_data_types.md` — `WorkflowImageData`, `Batch`, and the concrete Python types behind each kind.
- `docs/workflows/kinds/index.md` (+ per-kind pages under `docs/workflows/kinds/`) — the type catalog; `docs/workflows/schema_api.md` for how manifests export to UI schema.
- `docs/workflows/blocks_bundling.md` + `docs/workflows/versioning.md` — plugin packaging, loaders, (de)serializers, version lifecycle.
- `docs/workflows/batch_processing/` and `docs/workflows/video_processing/` — batch-heavy and video/stateful (`__init__` state, per-frame) block patterns.
- `docs/workflows/execution_engine_changelog.md` — which EE version introduced a feature (pin `get_execution_engine_compatibility()` accordingly).

## Block categories

Blocks live under `inference/core/workflows/core_steps/<category>/`. The 16 categories
(~210 versioned blocks) — the sections below group them into 8 maps:

| Category (dir) | ~blocks | What it is |
| --- | --: | --- |
| `models` | 75 | model inference — detection / segmentation / keypoints / classification / OCR + foundation / VLM / LMM |
| `visualizations` | 28 | draw predictions onto images |
| `transformations` | 24 | crop / offset / filter / perspective / stitch / slice detections & images |
| `classical_cv` | 23 | non-DL CV — SIFT, template match, contours, thresholds, color, QR / barcode |
| `sinks` | 15 | side-effects — email / webhook / slack / file / dataset-upload / monitoring |
| `analytics` | 11 | stateful video analytics — line counter, time-in-zone, path deviation |
| `formatters` | 10 | reshape data — expression, property, json, csv, vlm-as-detector |
| `fusion` | 8 | combine / reconcile predictions — consensus, dimension collapse |
| `flow_control` | 5 | branch / gate execution — continue_if, rate_limiter, delta_filter |
| `trackers` | 4 | multi-object tracking — byte_tracker, botsort |
| `sampling` / `cache` / `secrets_providers` / `math` / `integrations` | 1–2 ea | utilities |

### Model blocks — Roboflow-trained & OCR

**What / when.** Reach here when the block's job is to *run a model and emit predictions*: a Roboflow-trained model (object detection, instance/semantic segmentation, keypoint detection, single/multi-label classification) selected by `model_id`, or a text-extraction model (DocTR OCR core model, Google Vision OCR). These are the canonical "producer" blocks — everything downstream (visualizations, filters, UQL) consumes what they output.

**Mental model.** Every block in this category is a thin, *stateless* adapter over the model backend, all cut from one template:
- Manifest subclasses `WorkflowBlockManifest`; `block_type: "model"`, `ui_manifest.section: "model"`. Roboflow-trained blocks take `images: Selector(kind=[IMAGE_KIND])` (via `ImageInputField`) + `model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str]` (via `RoboflowModelField`) plus per-task knobs (`confidence`, `iou_threshold`, `class_filter`, NMS params, `active_learning_*`).
- `images` is **batch-oriented**: `get_parameters_accepting_batches` returns `["images"]`, so `run` receives `Batch[WorkflowImageData]` and returns a `BlockResult` list — one dict per input image (no dimensionality change; batch in, batch out).
- Init params are always `["model_manager", "api_key", "step_execution_mode"]`; `run` dispatches on `StepExecutionMode` to `run_locally` (via `self._model_manager` / `load_core_model` for core models) vs `run_remotely` (via `InferenceHTTPClient` + `WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_*`). Both funnel into a shared `_post_process_result`.
- Detection-family blocks produce a typed detections **kind** (`OBJECT_DETECTION_PREDICTION_KIND`, `INSTANCE_SEGMENTATION_PREDICTION_KIND`, `KEYPOINT_DETECTION_PREDICTION_KIND`, `SEMANTIC_SEGMENTATION_PREDICTION_KIND`) carried as `sv.Detections`; classification produces `CLASSIFICATION_PREDICTION_KIND` (a dict, not `sv.Detections`); OCR produces a `STRING_KIND` `result` (+ boxes as `OBJECT_DETECTION_PREDICTION_KIND`).

**Nuances an author MUST know.**
- **Shared post-processing pipeline, not ad-hoc.** Detection blocks convert raw inference dicts to `sv.Detections` and enrich them through a fixed util chain: `convert_inference_detections_batch_to_sv_detections` → `attach_prediction_type_info_to_sv_detections_batch` → `filter_out_unwanted_classes_from_sv_detections_batch` → `attach_parents_coordinates_to_batch_of_sv_detections` (see `object_detection/v2.py` `_post_process_result`). Reuse these; don't hand-roll `sv.Detections` construction or coordinate attachment.
- **Coordinate systems + parent metadata are load-bearing.** `attach_parents_coordinates_to_batch_of_sv_detections` is what lets downstream crops/visualizations re-project boxes onto the original frame. Classification carries this instead as `parent_id`/`root_parent_id` keys pulled from `image.parent_metadata` / `image.workflow_root_ancestor_metadata` (`multi_class_classification/v1.py` `_post_process_result`). Omitting it silently breaks stitching back to the source image.
- **Output-key contract is exact and typed.** `describe_outputs` names must match the keys returned by `run`. OCR encodes this explicitly with an `EXPECTED_OUTPUT_KEYS` set fed to `post_process_ocr_result` (`ocr/v1.py`), and note the *output name* differs from the constant in some blocks (`OutputDefinition(name=INFERENCE_ID_KEY, ...)` in keypoint v2 vs literal `"inference_id"` in OD v2) — be deliberate about the string.
- **sv.Detections data density differs by task.** Keypoint blocks call an *extra* `add_inference_keypoints_to_sv_detections` step to fold keypoints into `detections.data` after the standard conversion (`keypoint_detection/v2.py`), and instance/semantic segmentation carry masks. If you introduce a new per-detection field, it must round-trip through the kind's serializer or it will be dropped on REMOTE / API boundaries.
- **Model selection & variants.** Roboflow blocks gate on `get_compatible_task_types` (e.g. `["object-detection"]`); OCR/core-model blocks advertise `get_supported_model_variants` (e.g. `["doctr/default"]`) and load via `load_core_model(core_model="doctr")` rather than a user `model_id`. Segmentation exposes task-specific knobs (`mask_decode_mode`, `tradeoff_factor`, `enforce_dense_masks_in_inference_models`) that must be threaded into both the local request and the `InferenceConfiguration`.
- **LOCAL vs REMOTE parity.** The two paths hit different backends and the REMOTE path must mirror every knob into `InferenceConfiguration` (compare `confidence` → `confidence_threshold` renaming in OD v2 `run_remotely`). A knob added to only one path is a correctness bug, not a style nit. `WORKFLOWS_REMOTE_API_TARGET == "hosted"` also forces `client.select_api_v0()`.
- **These blocks are stateless** — no per-video state or eviction here (that lives in tracker/video blocks). Statefulness is deliberately kept out; keep new model blocks stateless too.

**Example blocks to pattern-match.**
- `inference/core/workflows/core_steps/models/roboflow/object_detection/v2.py` — the reference template for the whole category: manifest fields, LOCAL/REMOTE split, the full `sv.Detections` post-process chain. Read this first. (`object_detection/v3.py` shows the newer `confidence: Literal["best","default","custom"]` variant selection.)
- `inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v2.py` — read for **mask-carrying detections** and task-specific decode knobs (`mask_decode_mode`, `tradeoff_factor`, `enforce_dense_masks_in_inference_models`).
- `inference/core/workflows/core_steps/models/roboflow/keypoint_detection/v2.py` — read for **augmenting `sv.Detections` with extra structured data** via `add_inference_keypoints_to_sv_detections`.
- `inference/core/workflows/core_steps/models/roboflow/multi_class_classification/v1.py` — read for the **non-`sv.Detections` output shape** (dict prediction + manual `parent_id`/`root_parent_id`); pair with `multi_label_classification/v1.py` for the label vs multi-label variant.
- `inference/core/workflows/core_steps/models/foundation/ocr/v1.py` — read for a **core-model** block: `load_core_model`, `EXPECTED_OUTPUT_KEYS` contract, `STRING_KIND` result + detection boxes. Contrast with `foundation/google_vision_ocr/v1.py` (external API, `SECRET_KIND` key, `rf_key:` proxy, `AirGappedAvailability`). Both commonly feed `transformations/stitch_ocr_detections/v1.py` to reassemble text from cropped regions.

Docs (assets, don't copy): `docs/workflows/blocks/object_detection_model.md`, `instance_segmentation_model.md`, `keypoint_detection_model.md`, `single_label_classification_model.md`, `multi_label_classification_model.md`, `ocr_model.md`, `google_vision_ocr.md`, `stitch_ocr_detections.md`.

### Foundation & VLM/LMM model blocks

**What / when:** Wrappers around generic ("foundation") vision or vision-language models — CLIP/SAM2/YOLO-World/depth/OCR core models, and prompt-driven VLM/LLM blocks (Gemini, Claude, GPT, Florence-2, LLaMA-vision, Qwen, gaze). Reach here when your block *invokes a model* rather than transforming data, and the model is either a Roboflow core model or a third-party LLM API.

**Mental model.** Every block subclasses `WorkflowBlock` with a `BlockManifest(WorkflowBlockManifest)`, `block_type="model"`, and `ui_manifest.section` = `"model"` (or `"video"`). Two archetypes dominate:

- **Core-model blocks** (CLIP, SAM2, YOLO-World, depth, gaze) init with `["model_manager", "api_key", "step_execution_mode"]` and split `run()` into `run_locally()` (via `load_core_model` + `self._model_manager.infer_from_request_sync`) vs `run_remotely()` (via `InferenceHTTPClient`) keyed on `StepExecutionMode`. See `models/foundation/clip/v1.py` (simplest) and `models/foundation/segment_anything2/v1.py`.
- **VLM/LLM API blocks** (Gemini, Claude, OpenAI) init with just `["model_manager", "api_key"]`, have **no** local/remote split (they always call an external API), and expose a `task_type` enum that drives which fields are required. See `models/foundation/google_gemini/v3.py`, `models/foundation/anthropic_claude/v3.py`.

Kinds consumed: `IMAGE_KIND` (+ `STRING_KIND` for prompts/text). Kinds produced vary widely — `EMBEDDING_KIND` (clip), `INSTANCE_SEGMENTATION_PREDICTION_KIND` (SAM2), `OBJECT_DETECTION_PREDICTION_KIND` (yolo_world), `LANGUAGE_MODEL_OUTPUT_KIND` (VLMs), `NUMPY_ARRAY_KIND` (depth), `KEYPOINT_DETECTION_PREDICTION_KIND` (gaze). Batch behavior: image-consuming blocks declare `get_parameters_accepting_batches()` and receive `Batch[WorkflowImageData]`, returning one dict per element (`clip/v1.py` is the exception — scalar, no batch decl). Most are **stateless**; the video trackers are the notable stateful ones.

**Nuances an author MUST know.**

- **Output-key contract is the block's public API.** `describe_outputs()` names must exactly match keys in the returned dict. VLM blocks emit a single `"output"`/`"raw_output"` string of `LANGUAGE_MODEL_OUTPUT_KIND` plus `"classes"` — they deliberately *do not* parse into detections themselves. The `task_type` field carries a `recommended_parsers`/`recommended_parser` map in `json_schema_extra` pointing downstream to `roboflow_core/vlm_as_detector@v2`, `vlm_as_classifier@v2`, `json_parser@v1` (`anthropic_claude/v3.py:200-205`). Mirror this — don't bolt parsing onto the model block.
- **Conditional-required fields via `relevant_for` + `model_validator`.** `task_type` gates `prompt`/`classes`/`output_structure`; each field carries `json_schema_extra={"relevant_for": {...}}` for the UI *and* a `@model_validator(mode="after")` enforces it server-side (`google_gemini/v3.py:341-358`). Both are needed — UI hint alone doesn't validate.
- **Coordinate systems + prompt forwarding.** SAM2 accepts upstream `boxes` (OD/seg/keypoint kinds) as prompts, converts `xyxy → center+wh` per box, and **forwards the prompt box's `class_name`/`detection_id` onto the output mask**; unprompted, it invents `"foreground"`/integer ids (`segment_anything2/v1.py:266-285,487-492`). Note the 6-element `sv.Detections` iteration `for xyxy, _, confidence, class_id, _, bbox_data in boxes_for_image` — that unpack shape (`xyxy, mask, confidence, class_id, tracker_id, data`) is the `sv.Detections` contract (the 7-tuple form belongs to the tensor-native `NativeDetections` path).
- **Post-process sv.Detections consistently.** Detection-producing blocks end with the trio `convert_inference_detections_batch_to_sv_detections` → `attach_prediction_type_info_to_sv_detections_batch` → `attach_parents_coordinates_to_batch_of_sv_detections` (`segment_anything2/v1.py:459-473`). Skipping the parent-coords attach breaks coordinate re-anchoring for cropped/tiled inputs.
- **Statefulness ⇒ LOCAL-only + per-video eviction.** `segment_anything2_video/v1.py` holds `self._sessions: Dict[str, VideoSessionBookkeeping]` keyed on `video_metadata.video_identifier`, and **raises `NotImplementedError` for any non-LOCAL mode** (:282-283) because remote execution would fan frames across processes and destroy temporal memory. Switching `model_id` clears all sessions (:270). Stateful blocks also register `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION` / `STILL_IMAGE_INPUT_SOFT_RESTRICTION` in `get_restrictions()`.
- **Env-gated core models need `RuntimeRestriction`s.** Blocks behind a feature flag (`CORE_MODEL_SAM2_ENABLED`, `FLORENCE2_ENABLED`, `CORE_MODEL_GAZE_ENABLED`) add a `Severity.HARD` restriction for `HOSTED_SERVERLESS`+`REMOTE` (endpoint returns 404) and for `SELF_HOSTED_CPU`+`LOCAL` when CUDA is required (`segment_anything2/v1.py:151-174`).
- **API-key kinds & managed keys.** VLM blocks accept `Selector(kind=[STRING_KIND, SECRET_KIND, ROBOFLOW_MANAGED_KEY])` with `private=True` and default `"rf_key:account"` (`anthropic_claude/v3.py:246-252`); they also set `get_air_gapped_availability(available=False, reason="requires_internet")`.
- **Deprecation stays as a live block.** Removed features (gaze) keep the manifest with `"deprecated": True` and raise `FeatureDeprecatedError` at `run()` rather than being deleted (`gaze/v1.py:31-56`) — matches the MediaPipe-deprecation scope for gaze/cog_vlm.

**Example blocks to pattern-match (simplest → representative):**

- `models/foundation/clip/v1.py` — read for the **minimal core-model shape**: `str`-or-`WorkflowImageData` input, LOCAL/REMOTE split, an `LRUCache` for text embeddings, single scalar `EMBEDDING_KIND` output.
- `models/foundation/depth_estimation/v1.py` — read for a **numpy-tensor output** (`NUMPY_ARRAY_KIND`) alongside a passthrough `IMAGE_KIND`; canonical "non-detection dense output" example.
- `models/foundation/yolo_world/v1.py` — read for **class-prompted zero-shot OD**: dynamic `classes` list → `OBJECT_DETECTION_PREDICTION_KIND`; contrast with SAM2's box-prompted flow.
- `models/foundation/segment_anything2/v1.py` — read for the **full detection block**: batch handling, upstream-boxes-as-prompts with class/id forwarding, the sv.Detections post-process trio, env-gated `RuntimeRestriction`s.
- `models/foundation/google_gemini/v3.py` (peer: `anthropic_claude/v3.py`) — read for the **VLM/LLM archetype**: `task_type` enum, `relevant_for`+validator conditional fields, `recommended_parsers` handoff, `LANGUAGE_MODEL_OUTPUT_KIND` output, concurrency via `run_in_parallel`.
- `models/foundation/segment_anything2_video/v1.py` — read for the **stateful video tracker**: per-`video_identifier` session dict, prompt-vs-track gating, LOCAL-only enforcement, `_streaming_video_common.py` helpers, model-switch eviction.

Docs to reference (not copy): `docs/workflows/blocks/clip_embedding_model.md`, `.../segment_anything2_model.md`, `.../yolo_world_model.md`, `.../google_gemini.md`, `.../anthropic_claude.md`, `.../florence2_model.md`, `.../gaze_detection.md`.

### Visualization blocks

Blocks that render annotations onto an image (boxes, masks, labels, keypoints, traces, blur/pixelate, background fills) and return a new image. Reach for this category whenever the block's job is "take an image (+ usually predictions) and produce an annotated image" — almost always a thin wrapper over a `supervision` annotator.

**Mental model.** Every block subclasses a shared hierarchy in `inference/core/workflows/core_steps/visualizations/common/base.py`:
- `VisualizationManifest` / `VisualizationBlock` — the root. The manifest always declares an `image: Selector(kind=[IMAGE_KIND])` input (aliased `image`/`images`) plus a `copy_image: bool` toggle, and hard-codes a single output named `"image"` (the `OUTPUT_IMAGE_KEY` constant) of `IMAGE_KIND`. The block must implement `get_manifest`, `getAnnotator`, and `run`.
- `PredictionsVisualizationManifest` / `PredictionsVisualizationBlock` — adds a `predictions` input accepting the four detection kinds (object detection, instance segmentation, keypoint, RLE instance segmentation). Use for anything that annotates detections but doesn't need color config (`blur/v1.py`, `background_color/v1.py`).
- `ColorableVisualizationManifest` / `ColorableVisualizationBlock` in `common/base_colorable.py` — the most common base. Adds `color_palette` / `palette_size` / `custom_colors` / `color_axis` and a `getPalette(...)` helper resolving names → `sv.ColorPalette`. Most annotators (bounding box, label, mask, halo, trace, corner, dot, ...) build on this.

Shared shape: operate per-image (batch handled by the engine — no `get_parameters_accepting_batches`), stateless w.r.t. workflow data, but each block instance keeps an in-memory `self.annotatorCache` dict keyed on the styling params so the `sv` annotator object is reused across frames (`getAnnotator` lazily fills it). `run` almost always ends `return {OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(origin_image_data=image, numpy_image=annotated)}`. Not all blocks in the dir are colorable/prediction-driven: `model_comparison/v1.py` and `keypoint/v1.py` extend `VisualizationManifest` directly (keypoint redeclares its own `predictions` restricted to `KEYPOINT_DETECTION_PREDICTION_KIND`).

**Nuances an author MUST know.**
- **Fixed output key + no custom kinds.** The output is always `"image"`/`IMAGE_KIND` via `describe_outputs` in the base — don't rename it, and downstream stacking depends on it. These blocks consume existing kinds and produce `IMAGE_KIND`, so you generally never introduce/serialize a new data kind here.
- **`copy_image` mutation semantics.** When `copy_image=False` the annotator writes into `image.numpy_image` in place (`bounding_box/v1.py:166`). Every block sets a UI `warnings` entry flagging that this mutates the shared input if other blocks read it — replicate that manifest warning.
- **Statefulness is the annotator cache, and it bites under REMOTE.** `trace/v1.py` stores per-track position history *inside* the cached `sv.TraceAnnotator`. It declares a `RuntimeRestriction` (SOFT) via `get_restrictions()` (`trace/v1.py:139`) warning that on stateless/multi-replica REMOTE HTTP runtimes successive frames hit different workers, so traces reset/split — use local `InferencePipeline` execution. Any block relying on cross-frame annotator state must do the same.
- **Precondition checks on prediction contents.** Detection-driven blocks must guard for missing data density: `trace/v1.py:183` raises if `predictions.tracker_id is None`; `label/v1.py:248` short-circuits on `len(predictions) == 0`; label pulls optional `sv.Detections.data` keys (e.g. `time_in_zone`, `AREA_KEY_IN_SV_DETECTIONS`) with N/A fallbacks rather than assuming presence.
- **Coordinate system = pixel space of the passed image.** `sv` annotators draw in `predictions.xyxy` pixel coords; label derives dimensions/centroids straight from `predictions.xyxy[i]` (`label/v1.py:298-307`). Predictions must already be in the coordinate frame of the `image` input (crops carry their own frame), so don't feed parent-image detections onto a crop image without re-anchoring.
- **`getAnnotator` cache key hygiene.** The cache key is a `_`-join of every styling param that changes the annotator (see `bounding_box/v1.py:123`). Omitting a param from the key means stale annotators are reused when that param changes — include every config field that affects the annotator.
- **Compatibility pin.** Colorable manifests pin `get_execution_engine_compatibility() -> ">=1.3.0,<2.0.0"`; keep the same pin on new blocks in this category.

**Example blocks to pattern-match** (simplest → most representative):
- `inference/core/workflows/core_steps/visualizations/blur/v1.py` — simplest prediction-driven, non-colorable block (`PredictionsVisualizationManifest`); read for the minimal `getAnnotator`-cache + `copy_and_replace` skeleton.
- `inference/core/workflows/core_steps/visualizations/background_color/v1.py` — same base as blur but with a color string + opacity; read for consuming masks/detections without the full colorable palette machinery.
- `inference/core/workflows/core_steps/visualizations/bounding_box/v1.py` — the canonical `ColorableVisualizationBlock`; read for the full palette/`color_axis` wiring, cache key construction, and the standard `run` shape. Pattern-match this first for a new colorable annotator.
- `inference/core/workflows/core_steps/visualizations/label/v1.py` — most feature-rich colorable block; read for branching label content off `sv.Detections` fields/`.data` keys, empty-predictions guard, and pixel-space dimension math.
- `inference/core/workflows/core_steps/visualizations/trace/v1.py` — the stateful outlier; read for `RuntimeRestriction`/`get_restrictions()` and how per-frame annotator state interacts with REMOTE execution (contrast with the stateless `bounding_box` sibling).
- `inference/core/workflows/core_steps/visualizations/keypoint/v1.py` — extends `VisualizationManifest` directly (not colorable/`PredictionsVisualization`) with its own keypoint-only `predictions` selector; read when your block needs a bespoke prediction kind or annotator-mode switch (`edge`/`vertex`/`vertex_label`).

Per-block docs live at `docs/workflows/blocks/<name>_visualization.md` (e.g. `bounding_box_visualization.md`, `label_visualization.md`, `trace_visualization.md`) — reference these for user-facing param descriptions rather than restating them.

### Transformation & fusion blocks

**What this is / when to reach for it.** Blocks under `core_steps/transformations/**` reshape geometry or spatial data in-place (crop images, offset/filter/reproject boxes, slice, stabilize), while `core_steps/fusion/**` blocks combine or reduce multiple predictions/batch elements into one (consensus voting, class replacement, stitching crops back, collapsing a batch dimension). Reach here whenever you're moving `sv.Detections` between coordinate spaces or merging/reducing batches — not producing predictions from a model.

**Mental model.** Almost every block subclasses the plain `WorkflowBlock`/`WorkflowBlockManifest` (from `inference.core.workflows.prototypes.block`), sets `block_type` to `"transformation"` or `"fusion"`, consumes/produces the detection kinds (`OBJECT_DETECTION_PREDICTION_KIND`, `INSTANCE_SEGMENTATION_PREDICTION_KIND`, `KEYPOINT_DETECTION_PREDICTION_KIND`) and/or `IMAGE_KIND`, and threads work over `sv.Detections`. The distinguishing axis is **dimensionality**, declared via manifest classmethods:
- **1:1, same dim** — process each batch element independently, return a list aligned to the input `Batch`. Declare batch inputs with `get_parameters_accepting_batches()`. See `detection_offset/v1.py`, `detections_filter/v1.py`.
- **+1 (fan-out)** — one input element → many outputs. `get_output_dimensionality_offset() -> 1`; `run` returns a list-of-lists. See `dynamic_crop/v1.py`, `image_slicer/v2.py`.
- **-1 (collapse)** — nested batch → flat. `get_output_dimensionality_offset() -> -1`. See `dimension_collapse/v1.py`.
- **mixed-dim fusion** — inputs live at different nesting levels; align with `get_input_dimensionality_offsets()` + `get_dimensionality_reference_property()`. See `detections_stitch/v1.py` (predictions +1 vs reference_image) and `detections_classes_replacement/v1.py` (detections at 0, classifications at +1).

Most blocks are **stateless** (pure function of inputs). The video family is **stateful** — instance attributes hold per-`video_identifier` caches (e.g. `stabilize_detections/v1.py`).

**Nuances an author MUST know.**
- **Output-key contract is load-bearing.** Downstream blocks bind to your exact output name (`crops`, `predictions`, `slices`, `tracked_detections`). `dimension_collapse/v1.py` outputs `output`, not `predictions`. Keep `describe_outputs()` names and the `run` dict keys identical, and keep them stable across versions.
- **Coordinate systems must move together.** When you crop/offset/stitch, every geometry field has to be re-based in lockstep: `xyxy`, `mask`, keypoints (`KEYPOINTS_XY_KEY_IN_SV_DETECTIONS`), polygons (`POLYGON_KEY_IN_SV_DETECTIONS`), and OBB corners (`ORIENTED_BOX_COORDINATES`). `dynamic_crop/v1.py` (`crop_image`) and `detections_stitch/v1.py` (`move_detections`) both explicitly translate all five — forgetting OBB corners leaves them in the parent frame next to crop-local `xyxy`. Use `WorkflowImageData.create_crop(...)` so parent-coordinate metadata is attached for the inverse stitch.
- **`sv.Detections` data-density / metadata keys.** Blocks depend on keys other blocks planted: `dynamic_crop` requires `DETECTION_ID_KEY` and raises if absent; `detections_stitch` requires `PARENT_COORDINATES_KEY` (and rejects scaled detections via `SCALING_RELATIVE_TO_PARENT_KEY`); `detection_offset` rewrites `PARENT_ID_KEY`/`DETECTION_ID_KEY` (and carries the known-bug TODO of breaking parent coords — issue #380). When you mutate boxes, regenerate `DETECTION_ID_KEY` (fresh `uuid4`) and set `PARENT_ID_KEY` to the source id, exactly as `detections_classes_replacement/v1.py` does.
- **Statefulness + REMOTE execution.** Stateful video blocks must key state by `image.video_metadata.video_identifier` (a `Dict[str, ...]` per video, see `stabilize_detections/v1.py.__init__`) and evict entries when tracks age out (window-based deletion of `cached_detections`). Declare runtime restrictions — `get_restrictions()` returning `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION` + `STILL_IMAGE_INPUT_SOFT_RESTRICTION` — so the engine warns on stateless HTTP / single-image use where the cache never accumulates.
- **Fusion aggregation semantics are explicit knobs, not magic.** `detections_stitch` exposes `overlap_filtering_strategy` (none/nms/nmm) + `iou_threshold`; `detections_consensus/v1.py` takes a `List[Selector(...)]` (`predictions_batches`) — a variadic list of prediction sources — plus separate aggregation modes for presence-confidence, merge-confidence, coordinates, and masks. If your fusion merges overlapping boxes, surface the strategy rather than hard-coding NMS.
- **Empty / None handling.** Fusion inputs are frequently absent; set `accepts_empty_values() -> True` and branch (`detections_classes_replacement` returns `sv.Detections.empty()` / `{"predictions": None}`). Fan-out blocks emit `{"crops": None, "predictions": None}` placeholders for zero-area crops (`dynamic_crop`) to keep batch alignment.
- **New data kinds → serialization.** If a block introduces a kind not already round-trippable, register a serializer; existing detection/image kinds are already handled, so reuse them unless you truly need a new one.

**Example blocks to pattern-match (simplest → most representative).**
- `transformations/detection_offset/v1.py` — simplest 1:1 same-dim detections transform; read for the `Batch[sv.Detections]` → list return shape, box clipping to `image_dimensions`, and parent/detection-id rewiring.
- `fusion/dimension_collapse/v1.py` — the minimal -1 block; read for `get_output_dimensionality_offset() -> -1` + `get_parameters_enforcing_auto_batch_casting()` and a two-line `run`.
- `transformations/dynamic_crop/v1.py` — canonical +1 fan-out; read for `create_crop`, translating all geometry into crop space, and per-detection None placeholders. Pairs with `image_slicer/v2.py` (SAHI tiling, same +1 pattern).
- `fusion/detections_stitch/v1.py` — the inverse of crop/slice; read for mixed-dim fusion (`get_input_dimensionality_offsets` + `get_dimensionality_reference_property`), reading `PARENT_COORDINATES_KEY` to move boxes/masks/OBB back to the original frame, and nms/nmm merge. This is the other half of the `dynamic_crop`/`image_slicer` → model → `detections_stitch` loop.
- `fusion/detections_classes_replacement/v1.py` — canonical two-stage fusion joining detections (dim 0) with per-crop classifications (dim +1) via `PARENT_ID_KEY`, with positional fallback for raw string/OCR inputs; read for cross-dim matching and class/confidence/id rewrite.
- `transformations/stabilize_detections/v1.py` — the reference stateful video block; read for per-`video_identifier` caches, Kalman/EMA smoothing, gap-fill eviction, and `get_restrictions()`.
- `fusion/detections_consensus/v1.py` — multi-source voting; read for the `List[Selector(...)]` variadic input and multiple independent aggregation-mode fields plus extra outputs (`object_present`, `presence_confidence`).

### Classical CV blocks

**When to reach for it.** Non-model, OpenCV/NumPy-based image operations: preprocessing (grayscale, blur, threshold, morphology, contrast), classical detectors/matchers (SIFT, template matching, contours), and image-level measurements/analytics (dominant color, pixel color count, camera focus, size/distance measurement). Live under `core_steps/classical_cv/**` (note: QR/barcode detection actually live under `core_steps/models/third_party/**`, not here). Reach for these when you need a deterministic, dependency-light transform or a cheap heuristic rather than a learned model.

**Mental model.** Every block is the standard `WorkflowBlockManifest` + `WorkflowBlock` pair — nothing category-specific in the base classes. They are **non-batch, single-image-in / N-things-out, and stateless**: no block in this dir overrides `accepts_batch_input`, declares `WorkflowVideoMetadata`, or offsets dimensionality (verified — the grep for those hooks returns nothing). `run` receives one `WorkflowImageData`, reads `image.numpy_image` (BGR), and returns a dict keyed by output name. Two output archetypes:
- **Image-producing** (grayscale, contours, sift, camera_focus): emit a new image via `WorkflowImageData.copy_and_replace(origin_image_data=image, numpy_image=...)` under the shared `OUTPUT_IMAGE_KEY` (imported from `visualizations/common/base.py`), preserving parent metadata/coordinates.
- **Data-producing** (dominant_color, pixel_color_count, size_measurement, sift_comparison): emit scalars/lists/detections under custom output keys, often typed with **classical-CV-specific kinds** (`RGB_COLOR_KIND`, `CONTOURS_KIND`, `IMAGE_KEYPOINTS_KIND`, `NUMPY_ARRAY_KIND`, `LIST_OF_VALUES_KIND`).

All carry the `json_schema_extra` block-type `"classical_computer_vision"` (except size/distance measurement which are tagged `"transformation"`) and `ui_manifest.section: "classical_cv"` with `opencv: True`. All pin `get_execution_engine_compatibility` to `>=1.3.0,<2.0.0`.

**Nuances an author MUST know.**
- **Custom data kinds need serializers to survive REMOTE / API boundaries.** `RGB_COLOR_KIND` has both a serializer and deserializer registered in `loader.py` (`deserialize_rgb_color_kind`, line 753). But `CONTOURS_KIND`, `IMAGE_KEYPOINTS_KIND`, and the `NUMPY_ARRAY_KIND` SIFT descriptors have **no dedicated entry** in `KINDS_SERIALIZERS`/`KINDS_DESERIALIZERS` — they fall through to `serialize_wildcard_kind`. If you invent a new classical-CV data kind, you must register a serializer/deserializer pair in `loader.py` or it won't cleanly cross the JSON boundary. Cite `contours/v1.py` (`hierarchy` is a raw cv2 `np.ndarray`) and `sift/v1.py` (`keypoints` are hand-serialized to dicts precisely because `cv2.KeyPoint` isn't JSON-safe).
- **`sv.Detections` you build by hand must be fully populated.** `template_matching/v1.py` shows the contract: after constructing `sv.Detections(xyxy=..., confidence=..., class_id=..., data={CLASS_NAME_DATA_FIELD: ...})`, you must attach `PARENT_ID_KEY`, `PREDICTION_TYPE_KEY`, `DETECTION_ID_KEY`, `IMAGE_DIMENSIONS_KEY`, then call `attach_parents_coordinates_to_sv_detections(...)` — downstream blocks (crop, visualize, filter) depend on this metadata density. Return `sv.Detections.empty()` on zero matches, never `None`.
- **Coordinate systems & color order are on you.** OpenCV is BGR; RGB-facing blocks reverse channels explicitly (`pixel_color_count/v1.py`'s `convert_color_to_bgr_tuple` does `color[::-1]`; `dominant_color/v1.py` returns `reversed(dominant_color)`). Detectors that "work on grayscale" convert internally (`cv2.cvtColor(..., COLOR_BGR2GRAY)`) rather than requiring a grayscale input.
- **Output-key contract must match `describe_outputs` exactly.** Use the module-level `OUTPUT_KEY`/`OUTPUT_IMAGE_KEY` constants consistently across `describe_outputs` and the `run` return dict (see `size_measurement/v1.py`'s `OUTPUT_KEY = "dimensions"`).
- **Polymorphic inputs are common.** `sift_comparison/v2.py` accepts either `IMAGE_KIND` or `NUMPY_ARRAY_KIND` per input (`isinstance(input_1, WorkflowImageData)` branch) and returns a full output dict even on the early-exit path (<2 descriptors) — every declared output key must appear in **every** return branch or the engine complains.
- **Measurement blocks consume predictions, not images.** `size_measurement`/`distance_measurement` take `OBJECT_DETECTION_PREDICTION_KIND`/`INSTANCE_SEGMENTATION_PREDICTION_KIND` (plus `LIST_OF_VALUES_KIND` polygons) and prefer mask contours over bbox when a mask is present (`get_detection_dimensions`), so they sit downstream of models — reference-object scaling, not raw pixels.

**Example blocks to pattern-match (simplest → representative).**
- `classical_cv/convert_grayscale/v1.py` — the minimal image-in/image-out template: read `numpy_image`, one `cv2` call, `copy_and_replace`, return under `OUTPUT_IMAGE_KEY`. Read this first for the skeleton. Doc: `docs/workflows/blocks/image_convert_grayscale.md`.
- `classical_cv/pixel_color_count/v1.py` — simplest data-out block; read for `RGB_COLOR_KIND` input parsing (hex/tuple/string) and RGB↔BGR conversion. Doc: `docs/workflows/blocks/pixel_color_count.md`.
- `classical_cv/contours/v1.py` — read for emitting a **new data kind** (`CONTOURS_KIND`) alongside an image, plus a raw-`np.ndarray` `hierarchy` output that rides the wildcard serializer. Doc: `docs/workflows/blocks/image_contours.md`.
- `classical_cv/template_matching/v1.py` — the canonical "hand-build `sv.Detections`" block: full metadata attachment, NMS, `attach_parents_coordinates_to_sv_detections`, empty-case handling. Read this before producing any detections from classical CV. Doc: `docs/workflows/blocks/template_matching.md`.
- `classical_cv/sift/v1.py` + `classical_cv/sift_comparison/v2.py` — the paired feature-extraction / feature-consumption idiom: `sift` emits `IMAGE_KEYPOINTS_KIND` (dict-serialized) + `NUMPY_ARRAY_KIND` descriptors; `sift_comparison@v2` shows polymorphic image-or-descriptor inputs, all-branches-return outputs, and optional visualization outputs. Docs: `docs/workflows/blocks/sift.md`, `docs/workflows/blocks/sift_comparison.md`.
- `classical_cv/camera_focus/v2.py` — most elaborate: image + optional detections in, per-bbox `LIST_OF_VALUES_KIND` out, and an identity-preserving optimization (`if result_image is image.numpy_image: output = image`) to avoid a needless copy when no overlays are enabled. Read for the "optional detections drive per-region output" pattern shared with `size_measurement`. Doc: `docs/workflows/blocks/camera_focus.md`.

### Analytics, tracker & sampling blocks

**What / when.** Reach here for the *stateful, per-video* blocks: object trackers (assign persistent `tracker_id`), analytics that consume tracked detections to derive temporal metrics (time-in-zone, velocity, line crossings, path deviation), aggregators that fold a data stream into periodic summaries, and embedding-based sampling/anomaly gates. If a block needs to remember something from a previous frame, it lives here.

**Mental model.** Almost every block in this category is a plain `WorkflowBlock` whose `run(...)` is called *once per frame* and which stashes cross-frame state in instance attributes keyed by `image.video_metadata.video_identifier` — see the `Dict[str, ...]` fields in `VelocityBlockV1.__init__` and `TimeInZoneBlockV3.__init__`. Two shared patterns dominate:
- **Trackers** inherit `TrackerBlockBase` (`.../trackers/_base.py`), which owns the per-`video_identifier` tracker dict + `InstanceCache`, filters immature tracks (`tracker_id == -1`), and emits the fixed 3-output contract `tracked_detections` / `new_instances` / `already_seen_instances` via `tracker_describe_outputs()`. A concrete tracker only implements `_create_tracker` (and optionally `_tracker_update`).
- **Analytics** blocks take `image` (for `video_metadata`) + tracked `sv.Detections`, mutate/annotate the detections, and hand them back. They accept `OBJECT_DETECTION` / `INSTANCE_SEGMENTATION` (velocity/trackers also `KEYPOINT_DETECTION`) prediction kinds and return the same kinds — data flows through, enriched.

Batch/dimensionality: these are single-image (dimensionality 1), no batch fan-out — they operate frame-at-a-time so state stays coherent. The exception is `DataAggregator` (`.../analytics/data_aggregator/v1.py`), a *reducer*: it returns empty outputs most frames and emits only at interval boundaries.

**Nuances an author MUST know.**
- **Statefulness + REMOTE execution is the #1 gotcha.** Per-video state lives in process memory, so every block here returns `[STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION, STILL_IMAGE_INPUT_SOFT_RESTRICTION]` from `get_restrictions()`. On stateless/multi-replica HTTP runtimes with `StepExecutionMode.REMOTE`, successive frames may hit different workers, state resets, and tracking/counting/aggregation output is meaningless — see the preset note in `prototypes/block.py`. Declare these restrictions on any new stateful block.
- **Tracked input is a hard precondition.** Analytics blocks require `detections.tracker_id is not None` and raise `ValueError` otherwise (`TimeInZoneBlockV3.run`, `VelocityBlockV1.run`). Author your block to fail loud, and expect it to sit *downstream of a tracker*.
- **fps / timestamp duality.** Time math must branch on source: `frame_number / fps` for `metadata.comes_from_video_file`, else `metadata.frame_timestamp.timestamp()` (identical branch in time_in_zone v3 and velocity). Trackers default fps to 30 with a warning when absent (`TrackerBlockBase._run_tracker`).
- **New detection data-kinds must be serialized.** Analytics blocks write metrics into `detections.data[...]` under named constants (`TIME_IN_ZONE_KEY_IN_SV_DETECTIONS`, `VELOCITY_KEY_IN_SV_DETECTIONS`, `SPEED_KEY_IN_SV_DETECTIONS` in `execution_engine/constants.py`). Each such key needs a matching branch in `core_steps/common/serializers.py` (see the `if TIME_IN_ZONE_KEY... in data` / `if VELOCITY_KEY... in data` blocks) or it will not survive JSON serialization on the sink side. Adding a metric = constant + write + serializer entry.
- **Per-video eviction / cache bounds.** Unbounded per-`video_identifier` dicts leak across long-lived streams. Follow the FIFO patterns: `InstanceCache` (deque `maxlen`) in the tracker base, and `time_in_zone` v3's `OrderedDict` zone cache with `popitem(last=False)` at `ZONE_CACHE_SIZE`.
- **Coordinate system / anchors.** Zone and line blocks work in *pixel space* of the frame and use `sv.Position` triggering anchors (e.g. `CENTER`, `BOTTOM_CENTER`) to decide membership; velocity/perspective is uncalibrated pixels unless `pixels_per_meter` is set. Zone inputs are validated for nesting depth (`ensure_zone_is_list_of_polygons` / `calculate_nesting_depth`).
- **Tracker params are init-only.** `_create_tracker` runs once per video; changing thresholds on later frames has no effect (documented in `_run_tracker`). Trackers only associate on bboxes but index back into the full `sv.Detections`, preserving masks/keypoints/custom data.
- **Dynamic outputs.** `DataAggregator` declares `OutputDefinition(name="*")` and computes real names/kinds at runtime via `get_actual_outputs()` (`{variable}_{mode}`) — the pattern to copy when output shape depends on config.

**Example blocks to pattern-match (simplest → most representative).**
- `.../trackers/_base.py` — read first: the canonical stateful, per-`video_identifier`, 3-output tracker contract + `InstanceCache` eviction.
- `.../trackers/bytetrack/v1.py` — minimal concrete tracker: manifest + `_create_tracker` only. Contrast with `.../trackers/botsort/v1.py`, which overrides `_tracker_update` to feed frame pixels for camera-motion compensation (the reason `_tracker_update` exists).
- `.../analytics/velocity/v1.py` — cleanest "annotate tracked detections with a new metric": tracker_id guard, fps/timestamp branch, writes 4 keys into `detections.data`.
- `.../analytics/time_in_zone/v3.py` — the fullest analytics reference: multi-zone polygon logic, `sv.Position` anchors, per-track entry-time state, and a bounded `OrderedDict` zone cache with FIFO eviction.
- `.../analytics/data_aggregator/v1.py` — the *reducer* exception: interval-gated emission, empty outputs between intervals, dynamic `get_actual_outputs()`, UQL `data_operations`.
- `.../sampling/identify_outliers/v1.py` — non-detection state: consumes `EMBEDDING_KIND`, keeps a sliding window + vMF stats, emits `is_outlier` / `percentile` / `warming_up`; sibling `.../sampling/identify_changes/v1.py` mirrors the shape.

### Sink, notification, secret & cache blocks

**What this category is / when to reach for it.** These are the *terminal / side-effecting* blocks — they push workflow results out to the world (email, Slack, SMS, webhook, local file, Roboflow dataset/monitoring/metadata) or manage runtime state (secrets from env, in-memory cache). Reach for this category when the block's job is a **side effect**, not producing a data kind for downstream steps.

**Mental model.** Every block still subclasses `WorkflowBlock` + `WorkflowBlockManifest` (`inference/core/workflows/prototypes/block.py`) — there is no special "SinkBlock" base. What unifies them is *convention*, not inheritance:
- **`block_type` in `ui_manifest` json_schema_extra** tags the category: `"sink"` (email/webhook/slack/twilio/local_file/dataset_upload/custom_metadata/model_monitoring), `"secrets_provider"` (environment_secrets_store), `"fusion"` (cache_get/cache_set — cache is *not* tagged "sink").
- **Standard sink output contract**: sinks return a fixed dict, almost always `{"error_status": BOOLEAN_KIND, "message": STRING_KIND}` and (for throttled ones) `throttling_status: BOOLEAN_KIND` — see `describe_outputs` in `sinks/webhook/v1.py`. Sinks emit *status*, never data kinds meant to feed compute.
- **The "notification sink" shape** (email/webhook/slack/twilio) is a near-copy-paste family: `disable_sink` early-exit → cooldown check → build a `partial()` handler → dispatch via `BackgroundTasks` / `ThreadPoolExecutor` when `fire_and_forget`, else call sync and return real `error_status`. Compare `run()` in `sinks/webhook/v1.py:386` and `sinks/email_notification/v2.py:456` — structurally identical.
- **Message templating + UQL**: notification bodies use `{{ $parameters.x }}` placeholders filled from a `message_parameters` selector dict, optionally transformed per-key via `message_parameters_operations` (UQL op chains, `build_operations_chain`). See `format_email_message`/`execute_operations_on_parameters` in `email_notification/v2.py` and `webhook/v1.py:467`.
- **Batch/dimensionality**: notification sinks are scalar (one message per run). The *Roboflow* sinks are the batch-oriented outliers — `dataset_upload/v2.py` declares `get_parameters_accepting_batches() -> ["images","predictions","image_name"]` and its `run` takes `Batch[WorkflowImageData]` / `Batch[Union[sv.Detections, dict]]`, iterating with `zip`.
- **Statefulness**: several are stateful and therefore fragile on HTTP — cooldown timestamps (`self._last_notification_fired` in email/webhook), in-memory prediction aggregation (`model_monitoring_inference_aggregator/v1.py`), and the process-global `WorkflowMemoryCache` behind cache_get/set.

**Nuances an author MUST know.**
- **Cooldown is video-only, and you must declare it.** The `cooldown_seconds` timer lives in instance state, so it silently no-ops behind a stateless/multi-replica HTTP server. Notification sinks attach `COOLDOWN_HTTP_SOFT_RESTRICTION` via `get_restrictions()` (`email_notification/v2.py:430`, `webhook/v1.py:362`) to surface this to the runtime. If your sink throttles, declare the restriction.
- **`fire_and_forget=True` destroys your `error_status`.** When dispatched to a background task/thread the block returns `error_status=False` unconditionally — the docstrings explicitly tell users to set `fire_and_forget=False` for debugging (`email_notification/v2.py` LONG_DESCRIPTION). Design outputs knowing async == blind.
- **Secrets: use `SECRET_KIND` and `private=True`.** Password/token fields must accept `Selector(kind=[STRING_KIND, SECRET_KIND])` and set `private=True` so they aren't persisted in the workflow definition — see `sender_email_password` in `email_notification/v2.py:315` and `slack_token` in `slack/notification/v1.py`. The `environment_secrets_store/v1.py` provider is the *source* of `SECRET_KIND`: it declares `describe_outputs -> [OutputDefinition(name="*")]` (wildcard) but resolves concrete lowercased outputs at runtime via `get_actual_outputs()` — the pattern for a **dynamic-output** block.
- **REMOTE execution + statefulness = hard restriction.** Cache and model-monitoring blocks raise `NotImplementedError` outside LOCAL mode; they encode this with a `RuntimeRestriction(severity=Severity.HARD, applies_to_step_execution_modes=[StepExecutionMode.REMOTE])` (`cache/cache_set/v1.py:132`) and re-check `self._step_execution_mode is not StepExecutionMode.LOCAL` inside `run()`. Env-secrets similarly hard-restricts `HOSTED_SERVERLESS`/`DEDICATED_DEPLOYMENT` runtimes gated on `ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES`.
- **Cache namespacing + eviction is per-video, via image metadata.** cache_get/set derive the namespace from `image.video_metadata.video_identifier or "default"` and clean up in `__del__` via `WorkflowMemoryCache.clear_namespace(self.namespace)` (`cache_set/v1.py:163`). If you write a stateful per-stream block, mirror this: key state by `video_identifier`, evict on destruct. cache values are `WILDCARD_KIND` in/out (pass-through), and a miss returns `False`, not `None`.
- **sv.Detections data density — `inference_id` lives in `.data`.** `custom_metadata/v1.py` pulls `inference_ids = predictions.data.get(INFERENCE_ID_KEY, [])` for `sv.Detections` but `predictions[INFERENCE_ID_KEY]` for a classification dict — two different access paths for the two prediction shapes, and it errors if none are found. Any sink that needs per-inference identity must know detections carry it in the `.data` dict, not as a top-level field.
- **Image attachments must be serialized by the sink itself.** There's no automatic `WorkflowImageData` → wire conversion. Each sink hand-rolls it: email JPEG-encodes via `encode_image_to_jpeg_bytes` and base64s for the Roboflow proxy, or builds inline-CID MIME parts for SMTP (`email_notification/v2.py:726` `serialize_image_data` / `_send_email_using_smtp_server_v2`). Attachment fields accept `Selector(kind=[STRING_KIND, BYTES_KIND, IMAGE_KIND])`.
- **Roboflow-managed vs. custom-credential branching.** email_notification v2 exposes an `email_provider` dropdown; SMTP fields are conditionally shown via `json_schema_extra.relevant_for` and validated at runtime (returns `error_status=True` if required SMTP fields missing rather than raising). This `relevant_for` UI-conditioning pattern is how a sink offers "use Roboflow / bring your own" without separate blocks.

**Example blocks to pattern-match** (simplest → most representative):
- `inference/core/workflows/core_steps/cache/cache_set/v1.py` (+ `cache_get/v1.py`) — read for the **stateful, LOCAL-only, video-namespaced** minimal block: `RuntimeRestriction` HARD on REMOTE, `WILDCARD_KIND` pass-through, `__del__` eviction.
- `inference/core/workflows/core_steps/secrets_providers/environment_secrets_store/v1.py` — read for a **dynamic-output secrets provider**: `describe_outputs` wildcard + `get_actual_outputs()`, env-gated runtime restriction, `SECRET_KIND`.
- `inference/core/workflows/core_steps/sinks/webhook/v1.py` — read for the **canonical scalar notification sink**: disable→cooldown→`partial`→fire-and-forget dispatch, UQL `_operations` on payload dicts, `error_status/throttling_status/message` outputs.
- `inference/core/workflows/core_steps/sinks/email_notification/v2.py` — read for **secrets + `relevant_for` provider branching + image/attachment serialization + HTML templating**; the richest notification example.
- `inference/core/workflows/core_steps/sinks/roboflow/dataset_upload/v2.py` — read for a **batch-oriented sink**: `get_parameters_accepting_batches`, `Batch[...]` params, `zip` iteration, sampling/quota + `persist_predictions`.
- `inference/core/workflows/core_steps/sinks/roboflow/model_monitoring_inference_aggregator/v1.py` — read for **in-memory stateful aggregation over time** (`frequency`, `PredictionsAggregator`, per-run flush) and the paired REMOTE hard-restriction; contrast with `custom_metadata/v1.py` for how `inference_id` is extracted from `sv.Detections.data`.

`local_file/v1.py` (write CSV/JSON/txt, append-log rotation) and `slack/notification/v1.py` / `twilio/sms/v2.py` are the remaining notification-family siblings — pattern-match webhook/email first, then diff.

### Formatter, flow-control & misc blocks

**What / when.** The grab-bag of non-model blocks: reshape/extract data (formatters), gate or throttle execution (flow-control), and small analytical utilities (math, sampling). Reach here when you're routing/branching a workflow, turning one kind into another (VLM text → detections, dict → CSV), extracting a scalar for a downstream `Continue If`, or doing per-frame stateful bookkeeping.

**Mental model.** Every block subclasses `WorkflowBlock` + `WorkflowBlockManifest` from `inference/core/workflows/prototypes/block.py` (same as every other category). Two sub-shapes:

- **Formatters / math** — pure functions. Consume some kind(s), produce a transformed kind. Most declare a single generic output (`OutputDefinition(name="output")`, no `kind=` → wildcard) and lean on the UQL operations engine (`build_operations_chain` / `build_eval_function`) rather than hand-written logic. See `formatters/property_definition/v1.py` and `formatters/expression/v1.py`.
- **Flow-control** — return `FlowControl` (from `execution_engine/v1/entities`), NOT a data dict. `describe_outputs` returns `[]`; the block's job is to emit `FlowControl(mode="select_step", context=next_steps)` or `FlowControl(mode="terminate_branch")`. `block_type="flow_control"` and they take a `next_steps: List[StepSelector]`. See `flow_control/continue_if/v1.py`.

Statefulness splits the category: formatters/math are stateless; flow-control and sampling blocks (`rate_limiter`, `delta_filter`, `identify_outliers`, `continue_if` with `stop_delay`) carry `__init__` instance state across `run` calls.

**Nuances an author MUST know.**

- **Flow-control returns `FlowControl`, not outputs.** `describe_outputs()` MUST be `[]`. Terminating a branch is `FlowControl(mode="terminate_branch")`; continuing is `mode="select_step", context=next_steps`. Guard the empty-`next_steps` case (`continue_if/v1.py:154`).
- **Statefulness is broken behind HTTP.** Instance state (`self._last_executed_at`, `self.cache`, `self.start_time`) only persists in a long-lived video pipeline; behind stateless HTTP each request is a fresh instance so throttling/delta silently no-op. Declare this via `get_restrictions()` returning `RuntimeRestriction` constants — `COOLDOWN_HTTP_SOFT_RESTRICTION` (`rate_limiter/v1.py:122`), `STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION` + `STILL_IMAGE_INPUT_SOFT_RESTRICTION` (`delta_filter/v1.py:106`).
- **Per-video state must be keyed + evicted.** `delta_filter` caches by `image.video_metadata.video_identifier` so multiple streams don't cross-contaminate (`delta_filter/v1.py:128`); `rate_limiter` derives video-time from `1/fps * frame_number` instead of wall-clock so throttling is correct under faster-than-realtime playback (`rate_limiter/v1.py:145`). If you keep per-stream state, key it and think about growth.
- **Merging conditional branches needs `accepts_empty_values`.** By default a block is skipped when any input is `None`. `first_non_empty_or_default/v1.py:120` overrides `accepts_empty_values() -> True` so it actually runs on the `None`s it's meant to coalesce — and its `data` param arrives as a `Batch[Any]` to iterate.
- **Dynamic output keys.** `json_parser` declares a wildcard `OutputDefinition(name="*")` in `describe_outputs`, then resolves real keys per-config via `get_actual_outputs(self)` from `self.expected_fields` (`json_parser/v1.py:139-152`). Use this pattern when output keys depend on manifest config; `error_status` is a reserved key both here and in `vlm_as_detector`.
- **Batch/scalar + dimensionality collapse.** `csv_formatter` opts one param into batch delivery via `get_parameters_accepting_batches_and_scalars() -> ["columns_data"]`, then aggregates a whole batch into ONE csv string — returning `None` for all rows but the last (`csv/v1.py:126,160-165`). This is the idiom for a batch→scalar reducer.
- **VLM→detections is a manual kind constructor.** `vlm_as_detector/v2.py` builds an `sv.Detections` by hand: scales normalized 0–1 coords to pixels via `image.numpy_image.shape[:2]`, maps class names→ids, stuffs `IMAGE_DIMENSIONS_KEY`/`DETECTION_ID_KEY`/`INFERENCE_ID_KEY`/`PREDICTION_TYPE_KEY` into `.data`, and finishes with `attach_parents_coordinates_to_sv_detections(...)` so crop-offset coordinates stay correct (`vlm_as_detector/v2.py:349,375,388`). If you emit `OBJECT_DETECTION_PREDICTION_KIND`, you owe the same coordinate-system + metadata contract.
- **Give real `kind=` on typed outputs.** Analytical blocks pin their output kind so the graph validates downstream: `cosine_similarity/v1.py:105` → `FLOAT_KIND`; `identify_outliers/v1.py` → `BOOLEAN_KIND`/`FLOAT_ZERO_TO_ONE_KIND` from `EMBEDDING_KIND` inputs. Only fall back to a bare wildcard `output` when the type genuinely depends on user operations (expression, property_definition).

**Example blocks to pattern-match (simplest → representative).**

- `math/cosine_similarity/v1.py` — read for the minimal pure formatter: two typed inputs, one `FLOAT_KIND` output, no state, no UQL. Start here.
- `formatters/property_definition/v1.py` — read for delegating all logic to the UQL operations chain (`build_operations_chain`) and emitting a generic wildcard `output`. Its sibling `formatters/expression/v1.py` layers `build_eval_function` on top for switch/case results.
- `flow_control/continue_if/v1.py` — read for the canonical flow-control shape: `FlowControl` returns, empty `describe_outputs`, `next_steps`, plus optional `stop_delay` state. Pair with `flow_control/rate_limiter/v1.py` (video-time throttling + `RuntimeRestriction`) and `flow_control/delta_filter/v1.py` (per-`video_identifier` cache) for the stateful variants.
- `formatters/first_non_empty_or_default/v1.py` — read for `accepts_empty_values()` + `Batch[Any]` iteration (branch-merge semantics).
- `formatters/json_parser/v1.py` — read for config-driven dynamic outputs (`get_actual_outputs` + wildcard `"*"`). `formatters/csv/v1.py` is the batch→scalar reducer counterpart (`get_parameters_accepting_batches_and_scalars`).
- `formatters/vlm_as_detector/v2.py` — read as the most involved: hand-constructing an `sv.Detections`, coordinate scaling, metadata keys, and `attach_parents_coordinates_to_sv_detections`. The reference for producing a detection kind from raw text.

## Minimal new-block checklist

- [ ] New package dir with `__init__.py` + `vN.py`; two classes (Manifest + Block).
- [ ] `type` Literal `roboflow_core/{family}@v{N}` (+ legacy alias if replacing one).
- [ ] Inputs are `Selector(kind=[...])` / `Union[literal, Selector(...)]` with `Field(description=, examples=)`.
- [ ] `describe_outputs()` matches every key `run()` emits — on every path (empty / error branches included).
- [ ] `get_execution_engine_compatibility()` set to the minimum EE version you rely on.
- [ ] Batch / dimensionality hooks declared if the block batches inputs or changes nesting level.
- [ ] Registered in `core_steps/loader.py` (`load_blocks()`); new kinds in `types.py` + `load_kinds()` + (de)serializers.
- [ ] Stateful block raises `NotImplementedError` on `StepExecutionMode.REMOTE`.
- [ ] Unit test under `tests/workflows/unit_tests/core_steps/...` + integration test under `tests/workflows/integration_tests/execution/...`.
- [ ] Version bump `inference/core/version.py`; bump `EXECUTION_ENGINE_V1_VERSION` + `docs/workflows/execution_engine_changelog.md` ONLY if the EE itself changed.
- [ ] `docs/workflows/blocks/<block>.md` for a user-facing block.

Reviewer's checklist for block PRs: `review-workflows-blocks`.
