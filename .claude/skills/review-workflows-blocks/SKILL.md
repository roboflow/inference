---
name: review-workflows-blocks
description: Review PRs touching inference/core/workflows/core_steps/** (Workflow blocks — {category}/{block}/vN.py, common/, common/query_language/**, loader.py) or inference/core/workflows/execution_engine/entities/types.py (new kind). Diff signals — new block package, new/changed vN.py, new type Literal or legacy alias, describe_outputs()/run() changes, load_blocks()/load_kinds()/KINDS_SERIALIZERS edits, new *_KIND, sv.Detections.data writes, serializers/deserializers changes.
---

# Reviewing workflows-blocks changes

## Scope
Triggers when a PR touches:
- `inference/core/workflows/core_steps/**` — Roboflow Core plugin blocks (`{category}/{block}/v{N}.py`, `common/`, `common/query_language/**`, `loader.py`).
- `inference/core/workflows/execution_engine/entities/types.py` — only when a NEW **kind** is added for a block (review it alongside the block).

OUT of scope (other skills own these): Execution Engine internals under `inference/core/workflows/execution_engine/v1/**` (compiler, executor, output_constructor — `review-workflows-execution-engine`); model backends under `inference_models/`; the tensor-data-representation `_tensor` sibling pivot on `mvp/new-inference-pipeline`. This skill covers the **stable numpy/`sv.Detections` blocks on `main`**. UQL / `common/query_language/**` has no dedicated skill, so it IS reviewed here (see the query-language standard below).

## Review checklist
Severity: **BLOCK** = must fix before merge; **FLAG** = raise it; **NIT** = optional. Each item maps to one canonical rule in `## Standards` below.

1. **BLOCK — Output contract**: enumerate `describe_outputs()` keys; confirm EVERY `run()`/helper return dict (happy, empty, error, zero-area) carries exactly those keys (#2346).
2. **BLOCK — Versioning discipline**: bug-fix patches the existing `vN.py`; behavior change / new inputs-outputs goes in a NEW `vN.py`. Confirm no shipped `type` identifier or legacy alias was removed or repurposed.
3. **BLOCK — Loader wiring**: new block imported into `load_blocks()`; new kind into `load_kinds()` + defined in `types.py` with `*_KIND_DOCS`; serializable data-kind into `KINDS_SERIALIZERS`/`KINDS_DESERIALIZERS`.
4. **BLOCK — `__init__.py`** exists in every new/nested block package (guarded by `test_init_files.py`).
5. **BLOCK — `sv.Detections.data` density**: any new data key holds dense N-d numpy arrays (no `dtype="object"` ragged) and has serializer/deserializer parity if it must round-trip.
6. **FLAG — Tests**: unit test under `tests/workflows/unit_tests/core_steps/{category}/`; integration test under `tests/workflows/integration_tests/execution/test_workflow_with_*_block.py` for new or output-changing blocks.
7. **FLAG — EE compatibility floor**: actively verify `get_execution_engine_compatibility()` against the EE capabilities the block actually uses — contributors habitually leave the copy-pasted default range unexamined. Derive the true minimum from `docs/workflows/execution_engine_changelog.md` and require the floor raised when the block relies on a capability introduced after the declared minimum. If a capability the block relies on exists only under `## Unreleased` (including one added in the same PR), the EE version must be placed and bumped so the block can gate on it — see the EE-compatibility standard.
8. **FLAG — EE behavior change**: if the block forces an EE behavior change, require a user-facing entry under `## Unreleased` in `docs/workflows/execution_engine_changelog.md` and surface the maintainer notice **Execution Engine requires a version bump for release** — never ask the contributor to bump `EXECUTION_ENGINE_V1_VERSION` or update version-assert tests (maintainers do both in the release PR), except in the unreleased-dependency case above. Outside that exception, a contributor PR touching that constant is a red flag.
9. **FLAG — Cross-block kinds**: output kinds upstream ⊆ input kinds accepted downstream (esp. detections/segmentation/keypoint interplay).
10. **FLAG — Model-version changes** update defaults/enums (not silent deletion) and keep old identifiers loadable.
11. **FLAG — Batch handling** for model/foundation blocks: `run` iterates `Batch[...]` correctly and declares `get_parameters_accepting_batches()`.
12. **FLAG — UQL correctness over speed**: be skeptical of micro-optimizations to `common/query_language/**` ops that change behavior (#1112/#1161/#1162 were reverted).
13. **NIT — Field metadata**: new inputs have `title`/`description`/`examples`; `default=` correct (no stray `default=None` on required fields); `json_schema_extra` (`relevant_for`, `always_visible`) sensible.

### Not blocking
- Do NOT demand an `inference/core/version.py` bump — inference releases are versioned separately from feature/bugfix PRs.
- Do NOT demand a new `vN.py` for a pure in-place bug-fix; surgical patches to a shipped version are correct (#2346, #1339, #1368).
- Never demand an `EXECUTION_ENGINE_V1_VERSION` bump from a contributor — maintainers bump it at release. Sole exception: a block gating on unreleased EE capabilities (see the EE-compatibility standard), where the bump must be maintainer-coordinated before the block ships. The `## Unreleased` EE changelog entry is required only when the block forces an EE behavior change; most block changes don't.
- Do NOT demand serializer/deserializer entries for a data key that is purely intra-block and never crosses the wire.
- Do NOT demand hand-written `docs/workflows/blocks/<block>.md` — those are generated from `Field(description=, examples=)` metadata.

## Standards
The one canonical statement of each rule. Past regressions are cited inline; see `## Reference PRs`.

- **Output contract**: every dict `run()` emits contains EXACTLY the keys declared by `describe_outputs()`, on every path including empty/zero-area/error branches (`return []`, `{"x": None}`) (#2346 — `dynamic_crop` zero-area branch dropped `predictions`; #1341 — empty-zone branch).
- **Versioning**: bug-fixes patch the existing version in place; behavior changes / new inputs-outputs go into a NEW `vN.py` module — even at the cost of code duplication (`docs/workflows/versioning.md`, `blocks_bundling.md`). Never break `type` identifiers or legacy aliases; e.g. `dynamic_crop/v1.py` keeps `Literal["roboflow_core/dynamic_crop@v1", "DynamicCrop", "Crop"]`. Versioning foundation: #565.
- **New block = full package**: new dir with `__init__.py` + `v1.py`, imported and added to `load_blocks()` in `loader.py`, plus a unit test AND an integration test (#2362 — overlap_analysis added all of these + a new kind).
- **`__init__.py` in every package** (including nested): its absence breaks docs generation and block import; guarded by `test_init_files.py` (#2351, #830).
- **New kind discipline**: define the kind in `types.py` with a `*_KIND_DOCS` string, return it from `load_kinds()`, and route serialization through `KINDS_SERIALIZERS`/`KINDS_DESERIALIZERS` (#2362 — `DETECTIONS_OVERLAPS_KIND`; #804 — `INFERENCE_ID_KIND` as a distinct semantic kind across blocks).
- **`sv.Detections.data` must be dense numpy arrays**: object-dtype ragged arrays break supervision indexing / `is_data_equal`; pad to uniform shape (#2170 — keypoints; #1368 — velocity elements must be `np.array`).
- **Serializer/deserializer parity for round-tripping data keys**: a data key that must survive serialization needs matching entries in `common/serializers.py` + `common/deserializers.py` (#1368 — velocity/speed; #1156 — timestamps).
- **New input fields keep backward compat**: add with a `default=` and update `describe_outputs()`/`run()`; behavior-changing additions warrant a new version. Question `default=None` on non-optional inputs (#2384 — added in-place across v1/v2/v3 with defaults; #658 — removed wrong `default=None`).
- **EE compatibility**: `get_execution_engine_compatibility()` declares the minimum EE version the block relies on (e.g. `">=1.3.0,<2.0.0"`); the EE version is a single source of truth in `EXECUTION_ENGINE_V1_VERSION`. The floor is a contract, not boilerplate: derive it from `docs/workflows/execution_engine_changelog.md` — a block needing a capability added in `1.3.7` declares `">=1.3.7,<2.0.0"`, not the copy-pasted default. A block that forces an EE behavior change makes an EE version bump mandatory at release: the contributor adds the `## Unreleased` changelog entry only; maintainers apply the bump + final changelog heading + version-assert test updates in the release PR (#2527 — revert rolled back both the block and the EE version and re-asserted it in tests). **Unreleased-dependency exception**: when the block relies on a capability still under `## Unreleased`, there is no released floor to declare — the EE version must be placed (the `## Unreleased` entries get their final `vX.Y.Z` heading) and `EXECUTION_ENGINE_V1_VERSION` bumped with mirrored version-assert tests, maintainer-coordinated, in or before this PR, so the block's range can reference it. This is the one case where the bump cannot be deferred to a release PR.
- **Model-version deprecation**: when a hosted model version is deprecated, update the block default and enum; do not just delete, keep old identifiers loadable (#1668/#1669 — Claude 3.5→4.5; #500 — GPT-4→GPT-4o; #2395 — Gemini default).
- **Batch handling for model/foundation blocks**: `run` iterates `Batch[...]` and declares `get_parameters_accepting_batches()` (#718 — VLM blocks failed on batch inference).
- **Segmentation/mask edge cases**: mask→polygon must handle 0/1/degenerate contours (returns `None`, duplicates points); empty masks must serialize (#1339; #697 — SAM2 masks as points/lines).
- **Visualization blocks are fragile across supervision versions**: verify against the pinned supervision public API, not internals (#814/#811/#1868 — label/line regressions; #1725/#1951 — sv 0.27.0 handling).
- **Query language (UQL)**: `common/query_language/**` ops must preserve exact behavior; reject micro-optimizations that alter results (#1112/#1161/#1162 — automated speed-up PRs reverted).
- **Rich `Field` metadata**: `title`, `description`, `examples`, and `json_schema_extra` (`relevant_for`, `always_visible`) drive the FE app; block docs are generated from these (#623/#704; #2058 — property options propagated to FE).

## Required companions
Block a functional change that lacks these (canonical rules above; conditions here):
- **Loader registration** — new block in `load_blocks()`; new kind in `load_kinds()` + `types.py`; new serializable data-kind in `KINDS_SERIALIZERS`/`KINDS_DESERIALIZERS`, all in `loader.py` (#2362, #804).
- **Tests** — unit test always; integration test for new or output-changing blocks (#2362, #718).
- **`__init__.py`** — in every new/nested block package (#2351, #830).
- **EE changelog** — ONLY when the block forces an EE behavior change: add an entry under `## Unreleased` in `docs/workflows/execution_engine_changelog.md`; maintainers bump `EXECUTION_ENGINE_V1_VERSION` at release time.

## Key files & entry points
- `inference/core/workflows/core_steps/loader.py` — `load_blocks()`, `load_kinds()`, `KINDS_SERIALIZERS`/`KINDS_DESERIALIZERS`, `REGISTERED_INITIALIZERS`.
- `inference/core/workflows/core_steps/{category}/{block}/vN.py` — block modules (manifest + block class).
- `inference/core/workflows/core_steps/common/{serializers,deserializers,utils}.py` — `sv.Detections` <-> wire conversion, keypoints/mask helpers.
- `inference/core/workflows/core_steps/common/query_language/**` — UQL operations/evaluation engine.
- `inference/core/workflows/execution_engine/entities/types.py` — `Kind` definitions + `*_KIND_DOCS`.
- `inference/core/workflows/execution_engine/entities/base.py` — `WorkflowImageData`, `Batch`, `OutputDefinition`.
- `inference/core/workflows/prototypes/block.py` — `WorkflowBlock`, `WorkflowBlockManifest`, `BlockResult`.
- `inference/core/workflows/execution_engine/v1/core.py` — `EXECUTION_ENGINE_V1_VERSION`.
- Docs: `docs/workflows/create_workflow_block.md`, `versioning.md`, `blocks_bundling.md`, `execution_engine_changelog.md`.
- Tests: `tests/workflows/unit_tests/core_steps/**`, `tests/workflows/integration_tests/execution/**`, `test_init_files.py`.

## Reference PRs
- [#2362](https://github.com/roboflow/inference/pull/2362) — new fusion block + new kind + both test tiers (canonical new-block shape).
- [#2346](https://github.com/roboflow/inference/pull/2346) — missing declared output key on empty branch (output-contract).
- [#2170](https://github.com/roboflow/inference/pull/2170) — object-dtype ragged keypoint arrays break supervision.
- [#1368](https://github.com/roboflow/inference/pull/1368) — `sv.Detections.data` must hold numpy arrays + serializer parity.
- [#2384](https://github.com/roboflow/inference/pull/2384) — backward-compat input field added in-place to v1/v2/v3 with defaults.
- [#804](https://github.com/roboflow/inference/pull/804) — new `INFERENCE_ID_KIND` across blocks (kinds discipline).
- [#565](https://github.com/roboflow/inference/pull/565) — Workflows block versioning foundation.
- [#2351](https://github.com/roboflow/inference/pull/2351) — missing `__init__.py` (packaging/docs).
- [#2527](https://github.com/roboflow/inference/pull/2527) — revert: block removal + EE version rollback + test re-assert.
- [#658](https://github.com/roboflow/inference/pull/658) — stray `default=None` in manifests.
- [#718](https://github.com/roboflow/inference/pull/718) — VLM blocks failing on batch inference.
- [#1339](https://github.com/roboflow/inference/pull/1339) — empty-segmentation serialization edge cases.

## Related topic skills
When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):
- `review-topic-workflow-state-management`
- `review-topic-local-vs-remote-execution`
- `review-topic-prediction-integrity`
- `review-topic-backward-compat-and-versioning`
- `review-topic-test-hygiene`
- `create-workflow-block`
