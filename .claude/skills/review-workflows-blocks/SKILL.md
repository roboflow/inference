---
name: review-workflows-blocks
description: Review guidance for PRs changing inference/core/workflows/core_steps/** (Workflow blocks: manifests, run(), loader registration, kinds, serializers) — enforces block-authoring standards, versioning/compatibility contracts, and required companions (loader wiring, kinds, tests, version bump).
---

# Reviewing workflows-blocks changes

## Scope
Triggers when a PR touches:
- `inference/core/workflows/core_steps/**` — the Roboflow Core plugin blocks (block packages `{category}/{block}/v{N}.py`, `common/`, `loader.py`).
- `inference/core/workflows/execution_engine/entities/types.py` — only when a NEW **kind** is added for a block (kind lives here; review it alongside the block).

OUT of scope (other skills own these): the Execution Engine internals under `inference/core/workflows/execution_engine/v1/**` (compiler, executor, output_constructor); model backends under `inference_models/`; the tensor-data-representation `_tensor` sibling pivot on `mvp/new-inference-pipeline` (that has its own review surface). This skill covers the **stable numpy/`sv.Detections` blocks on `main`**.

## What this surface is
Workflow blocks are the user-facing building units of Roboflow Workflows. Each block is a package with per-version modules (`v1.py`, `v2.py`, …). A block module contains two classes and a set of contract methods (see `docs/workflows/create_workflow_block.md`):

- `BlockManifest(WorkflowBlockManifest)` — pydantic schema. Declares:
  - `model_config = ConfigDict(json_schema_extra={...})` with `name`, `version`, `short_description`, `long_description`, `license`, `block_type`, and `ui_manifest`.
  - `type: Literal["roboflow_core/{family}@v{N}", <legacy aliases>]` — the block identifier namespace. Legacy string aliases (e.g. `"DynamicCrop"`, `"Crop"`) are kept for backward compat.
  - Input fields typed as `Selector(kind=[...])` (or `Union[Selector(...), literal_type]`) with `Field(title=, description=, examples=)`.
  - `describe_outputs() -> List[OutputDefinition]` — the declared output keys + their kinds.
  - `get_execution_engine_compatibility() -> Optional[str]` — a version range like `">=1.3.0,<2.0.0"`.
  - Optionally `get_parameters_accepting_batches()`, `get_output_dimensionality_offset()`, `get_actual_outputs()` (dynamic outputs).
- `{Block}V{N}(WorkflowBlock)` — `get_manifest()` returns the manifest; `run(...)` returns a `BlockResult`.

**Invariants a reviewer must protect:**
1. **Output contract**: every dict `run()` emits must contain EXACTLY the keys declared by `describe_outputs()` — on every path, including empty/zero-area/error branches (#2346).
2. **Versioning**: bug-fixes patch the existing version in place; behavior changes / new inputs-outputs go into a NEW `vN.py` module — even at the cost of code duplication (`docs/workflows/versioning.md`, `docs/workflows/blocks_bundling.md`). Never break `type` identifiers or legacy aliases.
3. **`sv.Detections` data contract**: values in `detections.data[...]` must be proper N-d numpy arrays (not ragged object-dtype), and any block-produced data key that must survive serialization needs matching serializer/deserializer entries.
4. **EE compatibility**: a block using an EE feature must declare a compatible range; the Execution Engine version is a single source of truth (`EXECUTION_ENGINE_V1_VERSION`, `inference/core/workflows/execution_engine/v1/core.py`).

## Standards enforced here
- **Block identifier + legacy aliases**: `type` Literal uses `roboflow_core/{family}@v{N}` and retains any pre-existing alias strings. Evidence: `dynamic_crop/v1.py` (`Literal["roboflow_core/dynamic_crop@v1", "DynamicCrop", "Crop"]`); convention in `docs/workflows/versioning.md`.
- **New block = full package**: new dir with `__init__.py` + `v1.py`, registered in `loader.py` `load_blocks()`, plus unit test AND integration test. Evidence: #2362 (overlap_analysis added `__init__.py`, `v1.py`, loader import, new kind, `tests/.../unit_tests/fusion/test_overlap_analysis.py`, `tests/.../integration_tests/execution/test_workflow_with_overlap_analysis_block.py`).
- **Missing `__init__.py` breaks docs generation / packaging** — every block package (including nested) must have one. Evidence: #2351 (added `trackers/botsort/__init__.py`), #830 ("Add init.py to fix docs generation"). `tests/workflows/unit_tests/core_steps/test_init_files.py` guards this.
- **New kind is defined in `types.py` with docs** and returned from `load_kinds()` in `loader.py`; serialization goes through `KINDS_SERIALIZERS`/`KINDS_DESERIALIZERS` (loader.py ~L711–725). Evidence: #2362 (`DETECTIONS_OVERLAPS_KIND` + `_KIND_DOCS`), #804 (`INFERENCE_ID_KIND` added as separate semantic kind across blocks).
- **Bug-fix stays in the existing version**: small, surgical, in-place; do NOT silently change output shape/behavior of a shipped version. Evidence: #2346, #1339, #1368, #1596.
- **New input fields keep backward compat**: add with a `default=` and update `describe_outputs()`/`run()` signature; behavior-changing additions warrant a new version. Watch for `default=None` misuse — #658 removed `default=None` from manifests where it was wrong. Evidence: #2384 (`enforce_dense_masks_in_inference_models` added in-place to existing v1/v2/v3, defaulted for backward compat).
- **`sv.Detections.data` must be dense numpy arrays**: object-dtype ragged arrays break supervision indexing/`is_data_equal`. Pad to uniform shape. Evidence: #2170 (keypoints padding), #1368 (velocity data must be `np.array` elements).
- **Serializer/deserializer parity for new data keys**: adding a data key that must round-trip requires entries in `common/serializers.py` + `common/deserializers.py`. Evidence: #1368 (velocity/speed keys), #1156 (timestamp serialization).
- **`describe_outputs()` must match every `run()` emission** including empty branches (`return []`, `{"x": None, "y": None}`). Evidence: #2346, #1341 ("return empty result when zone not found").
- **Rich `Field` metadata**: `title`, `description`, `examples`, and `json_schema_extra` (e.g. `relevant_for`, `always_visible`) drive the FE app + docs. Evidence: `dynamic_crop/v1.py` fields; #623/#704 docs enhancements; #2058 (property options propagated to FE).
- **Model version deprecation**: when a hosted model version is deprecated, update the block default and enum, do not just delete. Evidence: #1668/#1669 (Claude 3.5→4.5), #500 (GPT-4→GPT-4o), #2395 (Gemini default 2.5-flash).
- **`get_execution_engine_compatibility()` set to the minimum EE version the block relies on** (e.g. `">=1.3.0,<2.0.0"`). Evidence: `dynamic_crop/v1.py`; `docs/workflows/versioning.md`.

## Required companions
Block the PR if a functional change lacks these:
- **Version bump** in `inference/core/version.py` — bumped on essentially every functional block change. Evidence: #2351 (1.2.8→1.2.9), #2384 (→1.2.12), #1596.
- **Loader registration**: new block imported and added to `load_blocks()` in `inference/core/workflows/core_steps/loader.py`; new kind added to `load_kinds()`; new serializable data-kind added to `KINDS_SERIALIZERS`/`KINDS_DESERIALIZERS` (loader.py). Evidence: #2362, #804.
- **Tests**: unit test under `tests/workflows/unit_tests/core_steps/{category}/` AND, for new/changed blocks, an integration test under `tests/workflows/integration_tests/execution/test_workflow_with_*_block.py`. Evidence: #2362, #718, #928/#715.
- **Kind docs**: new kind carries a `*_KIND_DOCS` string in `types.py` (rendered into `docs/workflows/kinds/`). Evidence: #804, #2362.
- **Execution Engine version + changelog**: ONLY when the block requires an EE behavior change — bump `EXECUTION_ENGINE_V1_VERSION` in `inference/core/workflows/execution_engine/v1/core.py` and add an entry to `docs/workflows/execution_engine_changelog.md`. Most block PRs do NOT touch these; a plain block change bumping the EE version is a red flag. Evidence: #2527 (EE 1.12.0 + changelog + version-assert tests).
- **`__init__.py`** in every new (and nested) block package. Evidence: #2351, #830.

## Common pitfalls & past regressions
- **#2346** — `dynamic_crop` zero-area branch emitted only `{"crops": None}`, missing the `predictions` key it declared. → Every emitted dict must carry all `describe_outputs()` keys on every path.
- **#2170** — keypoints stored as `dtype="object"` ragged arrays broke supervision indexing. → Reject object-dtype `sv.Detections.data`; require dense padded arrays.
- **#1368** — velocity block wrote python lists / non-`np.array` elements into `detections.data`. → data values must be numpy arrays; check serializer parity.
- **#1339** — empty segmentation masks serialized incorrectly / raised; also SAM2 masks as points/lines (#697). → Check mask→polygon handles 0/1/degenerate contours (returns `None`, duplicates points).
- **#1341 / #972** — dynamic-zones returning nothing / perspective-correction not accepting dynamic-zones output. → Verify empty-result branches and cross-block kind compatibility (output kind of A ⊆ accepted kinds of B).
- **#2351 / #830** — missing `__init__.py` broke docs generation and block import. → New package must have `__init__.py`.
- **#718** — VLM foundation blocks failed on batch inference. → Check batch handling (`get_parameters_accepting_batches`, `Batch[...]` in `run`) for model blocks.
- **#658** — `default=None` left in manifests where the field was required. → Question `default=None` on non-optional inputs.
- **#814 / #811 / #1868** — repeated label/line visualization regressions. → Visualization blocks are fragile across supervision versions (see #1725/#1951 handling sv 0.27.0 + internal imports); verify against the pinned supervision API, not internals.
- **#2527 (revert)** — tracklet-recognition block reverted; the revert also rolled back `EXECUTION_ENGINE_V1_VERSION`. → Reverts of blocks that bumped the EE version must roll the version back and re-assert it in tests.
- **#1112 / #1161 / #1162 (reverts)** — automated "speed-up" PRs to query-language ops were reverted for changing behavior. → Be skeptical of micro-optimizations to UQL/query ops; correctness over speed.

## Review checklist
1. **Version bump present** in `inference/core/version.py` for any functional change (else block).
2. **Versioning discipline**: is this a bug-fix (patch existing `vN.py`) or a behavior change (new `vN.py`)? Confirm no shipped `type` identifier / legacy alias was removed or repurposed.
3. **Output contract**: enumerate `describe_outputs()` keys; confirm EVERY `run()`/helper return dict (happy, empty, error) has exactly those keys.
4. **Loader wiring**: new block in `load_blocks()`; new kind in `load_kinds()` + `types.py` (+ `*_KIND_DOCS`); serializable data-kind in `KINDS_SERIALIZERS`/`KINDS_DESERIALIZERS`.
5. **`__init__.py`** exists in every new/nested block package.
6. **Tests**: unit test added/updated; integration test for new or output-changing blocks.
7. **`sv.Detections.data`**: any new data key uses dense numpy arrays (no `dtype="object"` ragged) and has serializer/deserializer parity if it must round-trip.
8. **Field metadata**: new inputs have `title`/`description`/`examples`; `default=` correct (no stray `default=None` on required fields); `json_schema_extra` (`relevant_for`, etc.) sensible.
9. **EE compatibility**: `get_execution_engine_compatibility()` reflects features used. If the PR bumps `EXECUTION_ENGINE_V1_VERSION`, require a changelog entry + version-assert test updates; if it does NOT need EE changes, it must not touch that version.
10. **Cross-block kinds**: output kinds of upstream match input kinds accepted downstream (esp. detections/segmentation/keypoint interplay).
11. **Model-version changes** update defaults/enums (not silent deletion) and keep old identifiers loadable.
12. **Batch handling** for model/foundation blocks: `run` iterates `Batch[...]` correctly and declares `get_parameters_accepting_batches()`.

## Key files & entry points
- `inference/core/workflows/core_steps/loader.py` — `load_blocks()` (~L810), `load_kinds()` (~L1033), `KINDS_SERIALIZERS`/`KINDS_DESERIALIZERS` (~L711), `REGISTERED_INITIALIZERS` (~L699).
- `inference/core/workflows/core_steps/{category}/{block}/vN.py` — block modules (manifest + block class).
- `inference/core/workflows/core_steps/common/{serializers,deserializers,utils}.py` — `sv.Detections` <-> wire conversion, keypoints/mask helpers.
- `inference/core/workflows/execution_engine/entities/types.py` — kind definitions + `*_KIND_DOCS`.
- `inference/core/workflows/execution_engine/entities/base.py` — `WorkflowImageData`, `Batch`, `OutputDefinition`.
- `inference/core/workflows/prototypes/block.py` — `WorkflowBlock`, `WorkflowBlockManifest`, `BlockResult`.
- `inference/core/version.py` — package version (bump gate).
- `inference/core/workflows/execution_engine/v1/core.py` — `EXECUTION_ENGINE_V1_VERSION`.
- Docs: `docs/workflows/create_workflow_block.md`, `versioning.md`, `blocks_bundling.md`, `execution_engine_changelog.md`, `docs/workflows/kinds/`.
- Tests: `tests/workflows/unit_tests/core_steps/**`, `tests/workflows/integration_tests/execution/**`, `test_init_files.py`.

## Reference PRs
- [#2362](https://github.com/roboflow/inference/pull/2362) — feat: new fusion block + new kind + both test tiers (canonical new-block shape).
- [#2346](https://github.com/roboflow/inference/pull/2346) — bugfix: missing declared output key on empty branch (output-contract).
- [#2170](https://github.com/roboflow/inference/pull/2170) — bugfix: object-dtype ragged keypoint arrays break supervision.
- [#1368](https://github.com/roboflow/inference/pull/1368) — bugfix: `sv.Detections.data` must hold numpy arrays + serializer parity.
- [#2384](https://github.com/roboflow/inference/pull/2384) — feat: backward-compat input field added in-place to existing v1/v2/v3 + version bump.
- [#804](https://github.com/roboflow/inference/pull/804) — feat: new `INFERENCE_ID_KIND` across blocks (kinds discipline).
- [#565](https://github.com/roboflow/inference/pull/565) — refactor: Workflows block versioning foundation.
- [#2351](https://github.com/roboflow/inference/pull/2351) — bugfix: missing `__init__.py` (packaging/docs).
- [#2527](https://github.com/roboflow/inference/pull/2527) — revert: block removal + EE version rollback + test re-assert.
- [#658](https://github.com/roboflow/inference/pull/658) — bugfix: stray `default=None` in manifests.
- [#718](https://github.com/roboflow/inference/pull/718) — bugfix: VLM blocks failing on batch inference.
- [#1339](https://github.com/roboflow/inference/pull/1339) — bugfix: empty-segmentation serialization edge cases.

## Related topic skills

When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):

- `review-topic-workflow-state-management`
- `review-topic-local-vs-remote-execution`
- `review-topic-prediction-integrity`
- `review-topic-backward-compat-and-versioning`
- `review-topic-test-hygiene`
- `create-workflow-block`
