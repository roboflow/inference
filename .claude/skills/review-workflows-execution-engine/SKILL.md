---
name: review-workflows-execution-engine
description: Review guidance for PRs changing inference/core/workflows/execution_engine/** and workflows core (entities/errors/prototypes, NOT core_steps) — enforces compiler/executor contracts, EE versioning + changelog companions, lineage/dimensionality invariants, and future-resolution/serialization safety.
---

# Reviewing workflows-execution-engine changes

## Scope
Trigger this skill when a PR changes any of:
- `inference/core/workflows/execution_engine/**` — the compiler, executor, introspection, profiling, dynamic blocks, inner workflows, entities.
- `inference/core/workflows/errors.py`, `inference/core/workflows/entities/**`, `inference/core/workflows/prototypes/block.py` — the workflow-core contracts the engine depends on.
- `docs/workflows/execution_engine_changelog.md`, `docs/workflows/versioning.md`, `docs/workflows/workflows_compiler.md`, `docs/workflows/workflow_execution.md`.

**Out of scope** (other skills own these): individual block implementations under `inference/core/workflows/core_steps/**` (block-authoring skill), and the query-language (`core_steps/common/query_language/**`). A `core_steps` change only enters this skill's scope when it *forces* an engine behavior change (then EE versioning rules apply — see PR #2106 which bumped EE + touched query-language together).

## What this surface is
The Execution Engine (EE) compiles a Workflow Definition (JSON) into a graph and runs it. Two phases:
- **Compile** (`v1/compiler/core.py`): normalize inner-workflow refs → collect+dedupe dynamic blocks → parse → validate spec → build execution graph (lineage, dimensionality, control-flow edges). Results cached (`v1/compiler/cache.py`).
- **Run** (`v1/executor/core.py`): assemble runtime params → validate → schedule steps via injectable `ThreadPoolExecutor` → assemble step inputs (`execution_data_manager/step_input_assembler.py`) → construct outputs (`output_constructor.py`).

Contracts/invariants a reviewer MUST protect:
- **Public engine API** — `ExecutionEngine.init(...)` / `.run(...)` / `.flush_stream_pipeline(...)` signatures in `execution_engine/core.py` and `entities/engine.py` (`BaseExecutionEngine`). Adding a run param means adding it to the ABC, `ExecutionEngine`, and `ExecutionEngineV1` in lockstep (all three edited together in PR #2475/#1640).
- **Versioned engine registry** — `REGISTERED_ENGINES` keyed by `EXECUTION_ENGINE_V1_VERSION` (`execution_engine/core.py`). A Workflow's `version: X.Y.Z` resolves to `>=X.Y.Z,<(X+1).0.0`; single version per major must match (`_select_execution_engine`).
- **Input types** are frozen public schema: `WorkflowImage`, `WorkflowVideoMetadata`, `WorkflowParameter`, `WorkflowBatchInput` (`entities/base.py`). `WorkflowImage`/`WorkflowVideoMetadata` are deprecated but supported until EE v2 — do not remove.
- **Kinds serializers/deserializers** decouple internal data from the wire (changelog v1.3.0); do not couple a kind to an internal format inside the engine.
- **`WorkflowError` hierarchy** (`errors.py`): compiler errors subclass `WorkflowCompilerError`; runtime subclass `WorkflowExecutionEngineError`. `ClientCausedStepExecutionError` carries an HTTP `status_code` and maps to 4xx (changelog v1.7.0) — regressing a client error back to `StepExecutionError` reintroduces HTTP 500s.

## Standards enforced here
- **Any behavior change to compile or run MUST bump `EXECUTION_ENGINE_V1_VERSION`** in `inference/core/workflows/execution_engine/v1/core.py` AND add a top section to `docs/workflows/execution_engine_changelog.md` — this is codified in `.cursor/rules/execution-engine-version-changelog.mdc`. Patch for bug fixes, minor for new capability, major for breaking (rare). Evidence: PR #2383 (bump-only), #2106, #1640, #1504 all pair version + changelog.
- **Changelog entry format is fixed**: `## Execution Engine \`vX.Y.Z\` | inference \`vA.B.C\`` where `A.B.C` is the current `inference/core/version.py` `__version__`, followed by user-facing "What changed" bullets (behavior, not file lists). See existing entries; cursor rule spells it out.
- **Defensive parsing of external JSON**: never index dict keys blindly — check presence *and* type. PR #1122 changed `KEY in property_definition` → `isinstance(property_definition.get(KEY), dict)` in `schema_parser.py`; PR #2380's dynamic-block collector treats malformed entries as pass-through for downstream validation.
- **`None` vs falsy is a real bug class**: use `is not None` / `len(x) > 0`, never truthiness, for indices/batches/collections. PR #777 fixed `if non_compound_indices:` → `if non_compound_indices is not None:` and `if batch_indices:` → `if len(batch_indices) > 0` (empty-detections two-stage workflow). PR #2348 guards empty `class_name`/`words` with explicit `len(...) == 0`.
- **Shared future-resolution helpers**: future detection/resolution lives in `v1/executor/utils.py` (`contains_future`, `resolve_futures`, `maybe_resolve_futures`). Do NOT re-implement per-module recursive `Future`/`Batch`/list/tuple/dict walkers — PR #2486 collapsed duplicated copies in `step_input_assembler.py` and `output_constructor.py` into these helpers. New future-carrying paths call the shared helpers with a `context=` string.
- **Errors carry structured references**: raise `WorkflowError` subclasses with `public_message` + `context` (`"phase | sub_phase"`); attach `block_id`/`block_type`/`WorkflowBlockError`/`BlockTraceback` where the surface supports it (PR #995, #504, #1331 add block/output references and serialisation error logs). Never leak raw tracebacks as `public_message`.
- **Injectable executor, not per-run pools**: the engine accepts a `ThreadPoolExecutor` and must respect `max_concurrent_steps` even when the pool is larger (changelog v1.5.0, PR #1014). Do not create a `ThreadPoolExecutor` inside `run(...)`.
- **Dimensionality / lineage semantics** are load-bearing: auto-batch-casting, `get_output_dimensionality_offset`, `get_parameters_enforcing_auto_batch_casting`, `get_parameters_accepting_batches[_and_scalars]` (changelog v1.6.0). Control-flow lineage lets control-flow-gated steps run without data lineage (PR #2106, changelog v1.8.0). Changes here must preserve results for pre-existing workflows.
- **Inner-workflow inlining is rewrite-order sensitive**: child step renames must run before parameter-binding substitution so a parent selector isn't captured by a same-named child step (PR #2352). Dynamic blocks from nested inner workflows are collected depth-first and deduped by `manifest.block_type`, first occurrence wins (PR #2380).
- **Serialization must handle empty/non-batch outputs**: empty outputs and non-batch outputs both serialize (PR #815, changelog v1.4.0 fixed the regression where non-batch outputs were skipped).

## Required companions
Block the PR if a behavior-changing EE PR is missing:
1. **Version bump** — `EXECUTION_ENGINE_V1_VERSION` in `inference/core/workflows/execution_engine/v1/core.py`.
2. **Changelog** — new top section in `docs/workflows/execution_engine_changelog.md` (exact heading format above).
3. **Version-assertion tests updated** — hardcoded EE version in `tests/inference/integration_tests/test_workflow_endpoints.py` (`test_get_versions_of_execution_engine`) and `tests/inference/hosted_platform_tests/test_workflows.py` (both edited in PR #2383, #2106, #645).
4. **Tests** — unit tests under `tests/workflows/unit_tests/execution_engine/{compiler,executor,inner_workflow,introspection,...}` and an integration test under `tests/workflows/integration_tests/execution/` reproducing the fixed scenario (PR #777, #2352, #645 all ship a scenario test).
5. **`inference/core/version.py`** `__version__` bump when the change ships as a release (paired in #2106, #1640, #1504, #585) — cross-check it matches the `inference \`vA.B.C\`` in the new changelog heading.
Exemptions (per cursor rule): comment/type-only refactors, formatting, and pure test/doc mirroring with **no** behavior change do not require a version bump.

## Common pitfalls & past regressions
- **#2486** — `Future` leaked into workflow outputs when a `Batch`-wrapped future was constructed; fix centralized future walking. Check: any new value that can hold a `Future` is passed through the shared resolvers before it reaches a client/output.
- **#2352** — inner+outer workflows with the **same step name** and the outer step provided in bindings: wrong rewrite renamed the parent selector. Check inlining substitution order (renames before input bindings).
- **#2380** — custom Python blocks defined only on a nested inner workflow failed to compile; dynamic blocks must be collected from every nested level and hoisted to root before `compile_dynamic_blocks`.
- **#777** — two-stage model + `continue_if` crashed when the primary model detected nothing (empty batch/`None` indices treated as falsy). Check `is not None` / `len(...)` on indices and batch lists.
- **#1122** — OpenAPI/pydantic BC: `additionalProperties: false` (a bool, not a dict) broke schema parsing. Check dict-key access uses `isinstance(..., dict)`.
- **#1379** — `WorkflowParameter.default_value` rejected `dict` defaults (JSON object). Union of allowed default types must include container types actually passable as JSON.
- **#815** — empty and non-batch outputs were dropped during serialization. Check output constructor serializes all output shapes.
- **#645** — lineage + conditional execution mismatch; came with EE version bump and branching-lineage integration test.
- **#585** — execution order bug in the executor scheduling utils. Verify ordering guarantees when touching `executor/utils.py` / `flow_coordinator.py`.
- **#2475** — deferring parent→root coordinate conversion out of the output constructor to the dispatch thread is a perf/parity trap: prediction parity must be preserved when `serialize_results=False` and futures are resolved late (`coordinates_system: parent` outputs).

## Review checklist
1. Does the diff change compile or run **behavior**? If yes: confirm `EXECUTION_ENGINE_V1_VERSION` bumped + changelog section added (correct heading + inference version) + version-assertion tests updated. If no bump, confirm it's genuinely comment/type/format/test-only.
2. If a `run(...)`/`init(...)` param or the engine surface changed: is it mirrored across `BaseExecutionEngine` (ABC), `ExecutionEngine`, and `ExecutionEngineV1`?
3. Any dict-key access on external/JSON input (`schema_parser.py`, definition parsing): presence **and** type checked (`isinstance(..., dict/list)`)?
4. Any index/batch/collection emptiness check using truthiness that should be `is not None` / `len(...) > 0`?
5. Any code walking values that may contain `Future`: does it use `v1/executor/utils.py` helpers rather than a new bespoke recursor?
6. Inner-workflow changes: substitution order (renames before bindings) and depth-first dynamic-block collection/dedupe preserved?
7. Errors raised as `WorkflowError` subclasses with `public_message` + `context`; client-caused failures use `ClientCausedStepExecutionError` with a `status_code` (not `StepExecutionError` → HTTP 500)?
8. New/changed compile-or-run path covered by a unit test in the matching `tests/workflows/unit_tests/execution_engine/*` dir and a reproducing integration test under `tests/workflows/integration_tests/execution/`?
9. Does the change preserve results for pre-existing workflows (lineage/dimensionality/serialization)? Any breaking behavior must be flagged in the changelog with a warning admonition.
10. No `ThreadPoolExecutor` created inside `run`; `max_concurrent_steps` still honored.
11. Deprecated inputs/selectors (`WorkflowImage`, `WorkflowVideoMetadata`, old `*Selector` annotations) not removed before EE v2.

## Key files & entry points
- `execution_engine/core.py` — version registry + engine dispatch (`retrieve_requested_execution_engine_version`, `_select_execution_engine`).
- `execution_engine/v1/core.py` — `EXECUTION_ENGINE_V1_VERSION`, `ExecutionEngineV1`, step-error-handler registry.
- `execution_engine/v1/compiler/core.py` — compile pipeline; `cache.py`, `graph_constructor.py`, `steps_initialiser.py`, `syntactic_parser.py`, `validator.py`.
- `execution_engine/v1/executor/core.py`, `output_constructor.py`, `execution_data_manager/step_input_assembler.py`, `executor/utils.py` (future helpers), `flow_coordinator.py`.
- `execution_engine/v1/inner_workflow/{inline.py,dynamic_blocks_collection.py,compiler_bridge.py}`.
- `execution_engine/entities/{engine.py,base.py,types.py}`, `errors.py`, `introspection/schema_parser.py`.
- `docs/workflows/execution_engine_changelog.md`, `.cursor/rules/execution-engine-version-changelog.mdc`.

## Reference PRs
- [#2486](https://github.com/roboflow/inference/pull/2486) — bugfix: centralize `Future` resolution helpers; fix Batch-wrapped future leak.
- [#2380](https://github.com/roboflow/inference/pull/2380) — bugfix: collect+dedupe dynamic blocks from nested inner workflows.
- [#2352](https://github.com/roboflow/inference/pull/2352) — bugfix: inner/outer same-step-name rewrite order (renames before bindings).
- [#2348](https://github.com/roboflow/inference/pull/2348) — bugfix: defensive guard against empty class/segment names in introspection.
- [#2383](https://github.com/roboflow/inference/pull/2383) — chore: canonical EE version bump + changelog + version-test update.
- [#2106](https://github.com/roboflow/inference/pull/2106) — feature: control-flow lineage (with EE + version bump).
- [#1640](https://github.com/roboflow/inference/pull/1640) — feature: custom step error handlers (registry + ABC/engine surface).
- [#1504](https://github.com/roboflow/inference/pull/1504) — feature: singular elements pushed into batch inputs.
- [#1122](https://github.com/roboflow/inference/pull/1122) — bugfix: pydantic/OpenAPI BC in schema parser (type-checked dict access).
- [#777](https://github.com/roboflow/inference/pull/777) — bugfix: two-stage + continue-if on empty detections (`is not None` / `len`).
- [#1014](https://github.com/roboflow/inference/pull/1014) — refactor: injectable ThreadPoolExecutor.
- [#815](https://github.com/roboflow/inference/pull/815) — bugfix: serialize empty / non-batch outputs.

## Related topic skills

When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):

- `review-topic-local-vs-remote-execution`
- `review-topic-workflow-state-management`
- `review-topic-backward-compat-and-versioning`
- `review-topic-concurrency-and-resource-safety`
- `review-topic-test-hygiene`
