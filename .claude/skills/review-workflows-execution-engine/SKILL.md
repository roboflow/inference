---
name: review-workflows-execution-engine
description: Review guidance for PRs touching inference/core/workflows/execution_engine/**, inference/core/workflows/errors.py, inference/core/workflows/prototypes/block.py, docs/workflows/execution_engine_changelog.md, or .cursor/rules/execution-engine-version-changelog.mdc. Diff signals: EXECUTION_ENGINE_V1_VERSION, REGISTERED_ENGINES, COMPILATION_CACHE, contains_future/resolve_futures, ClientCausedStepExecutionError, get_output_dimensionality_offset. NOT core_steps block implementations.
---

# Reviewing workflows-execution-engine changes

## Scope
Trigger when a PR changes any of:
- `inference/core/workflows/execution_engine/**` — compiler, executor, introspection, profiling, inner-workflows, dynamic blocks, engine entities (`execution_engine/entities/{engine.py,base.py,types.py}`).
- `inference/core/workflows/errors.py`, `inference/core/workflows/prototypes/block.py` — the workflow-core contracts the engine depends on.
- `docs/workflows/execution_engine_changelog.md`, `docs/workflows/versioning.md`, `docs/workflows/workflows_compiler.md`, `docs/workflows/workflow_execution.md`.

**Out of scope** (other skills own these): block implementations under `inference/core/workflows/core_steps/**` (block-authoring skill) and the query-language (`core_steps/common/query_language/**`). A `core_steps` change enters scope only when it *forces* an engine behavior change (then EE versioning rules apply — e.g. #2106 bumped EE + touched query-language together).

Note: there is no `inference/core/workflows/entities/` directory — the engine entities live under `execution_engine/entities/`.

## Review checklist
Severity-tagged. Verify each against the linked Standard before raising.

- **BLOCK** — Diff changes compile or run **behavior** but has no user-facing entry under `## Unreleased` in `docs/workflows/execution_engine_changelog.md`. Do not ask the contributor to bump a version. (Standard: Versioning.)
- **BLOCK** — Compile behavior depends on an input that is NOT part of the `COMPILATION_CACHE` hash key (`workflow_definition` + `execution_engine_version` only). Stale graphs get served on a cache hit. (Standard: Compilation cache.)
- **BLOCK** — Dict-key access on external/JSON input (`introspection/schema_parser.py`, definition parsing) checks presence but not type — must be `isinstance(..., dict/list)`. (Standard: Defensive parsing; broke in #1122.)
- **BLOCK** — Emptiness check on an index/batch/collection uses truthiness instead of `is not None` / `len(...) > 0`. (Standard: None-vs-falsy; broke in #777.)
- **BLOCK** — A client-caused failure is raised as `StepExecutionError` instead of `ClientCausedStepExecutionError` (with `status_code`), regressing a 4xx back to HTTP 500. (Standard: Structured errors.)
- **BLOCK** — A value that can hold a `Future` reaches a client/output without passing through the shared resolvers in `executor/utils.py`. (Standard: Future resolution; broke in #2486.)
- **FLAG** — `init(...)`/`run(...)` signature changed but not mirrored in lockstep. (Standard: Public engine API.)
- **FLAG** — Missing scenario test for a fixed bug. Version-assertion tests change only in the maintainer release PR. (Standard: Required companions.)
- **FLAG** — Inner-workflow change reorders substitution (renames must precede bindings) or drops depth-first dynamic-block collection/dedupe. (Standard: Inner-workflow inlining; #2352, #2380.)
- **FLAG** — A `ThreadPoolExecutor` is created inside `run(...)`, or `max_concurrent_steps` is no longer honored. (Standard: Injectable executor.)
- **FLAG** — Serialization change drops empty or non-batch outputs. (Standard: Serialization; broke in #815.)
- **FLAG** — Dimensionality/lineage change (`get_output_dimensionality_offset`, `get_parameters_accepting_batches[_and_scalars]`, control-flow lineage) does not preserve results for pre-existing workflows. (Standard: Dimensionality/lineage.)
- **NIT** — Deprecated inputs/selectors (`WorkflowImage`, `WorkflowVideoMetadata`, old `*Selector` annotations) removed before EE v2.

### Not blocking
- Comment/type-only refactors, formatting, and pure test/doc mirroring with **no** behavior change do NOT require a changelog entry or maintainer release notice (explicit exemption in `.cursor/rules/execution-engine-version-changelog.mdc`).
- A pure bug fix to a scenario already covered by an integration test does not require a *new* test file — an assertion added to the existing scenario is fine.
- Never demand an `EXECUTION_ENGINE_V1_VERSION` or `inference/core/version.py` bump from a contributor — maintainers handle both at release time.
- Do not demand `flush_stream_pipeline` mirroring on the ABC — it is intentionally not part of `BaseExecutionEngine` (see Public engine API).

## Standards

**Versioning.** Any behavior change to compile or run MUST add user-facing behavior bullets (not file lists) under `## Unreleased` in `docs/workflows/execution_engine_changelog.md`. Contributors do not choose or change a version. At release time, maintainers choose the bump (patch = bug fix, minor = new capability, major = breaking and rare), update `EXECUTION_ENGINE_V1_VERSION` and mirrored assertions, and replace `## Unreleased` with the final `## Execution Engine \`vX.Y.Z\` | inference \`vA.B.C\`` heading. Codified in `.cursor/rules/execution-engine-version-changelog.mdc`.

**Compilation cache.** `COMPILATION_CACHE` (a `BasicWorkflowsCache`, defined in `compiler/core.py`, class in `compiler/cache.py`) memoizes the compiled graph. Its hash key is built from exactly two hash functions: `workflow_definition` (`json.dumps(sort_keys=True)`) and `execution_engine_version`. Any new compile-time input that changes the resulting graph MUST be added to the cache's `hash_functions`, or a cache hit will serve a stale graph. The cache is size-bounded (`cache_size=256`, LRU-style eviction via `_keys_buffer`) — do not swap in an unbounded dict. Dynamic-blocks-allowed is re-validated on every hit (`ensure_dynamic_blocks_allowed`); preserve that so a cached graph cannot bypass the guard.

**Public engine API.** `init(...)` and `run(...)` are the ABC contract on `BaseExecutionEngine` (`execution_engine/entities/engine.py`). Adding a param to either means editing `BaseExecutionEngine`, `ExecutionEngine` (`execution_engine/core.py`), and `ExecutionEngineV1` (`execution_engine/v1/core.py`) in lockstep (#2475, #1640). `flush_stream_pipeline` exists only on `ExecutionEngine` and `ExecutionEngineV1`, NOT on the ABC — do not treat it as part of the three-way lockstep.

**Versioned engine registry.** `REGISTERED_ENGINES` (keyed by `EXECUTION_ENGINE_V1_VERSION`, `execution_engine/core.py`) resolves a Workflow's `version: X.Y.Z` to `>=X.Y.Z,<(X+1).0.0` via `retrieve_requested_execution_engine_version` / `_select_execution_engine`; a single matching version per major is required.

**Frozen input schema.** `WorkflowImage`, `WorkflowVideoMetadata`, `WorkflowParameter`, `WorkflowBatchInput` (`execution_engine/entities/base.py`) are public schema. `WorkflowImage`/`WorkflowVideoMetadata` are deprecated but supported until EE v2 — do not remove. `WorkflowParameter.default_value` must accept container defaults (dict/list) passable as JSON (#1379).

**Kinds serializers/deserializers** decouple internal data from the wire (changelog v1.3.0) — do not couple a kind to an internal format inside the engine.

**Structured errors.** Raise `WorkflowError` subclasses with `public_message` + `context` (`"phase | sub_phase"`); compiler errors subclass `WorkflowCompilerError`, runtime errors subclass `WorkflowExecutionEngineError` (all in `errors.py`). `ClientCausedStepExecutionError` carries an HTTP `status_code` and maps to 4xx (changelog v1.7.0); a plain `StepExecutionError` becomes HTTP 500. Attach `block_id`/`block_type`/`WorkflowBlockError`/`BlockTraceback` where the surface supports it (#995, #504, #1331). Never leak a raw traceback as `public_message`.

**Defensive parsing.** Never index dict keys from external JSON blindly — check presence *and* type. `isinstance(property_definition.get(KEY), dict)`, not `KEY in property_definition` (#1122, `introspection/schema_parser.py`); the dynamic-block collector treats malformed entries as pass-through for downstream validation (#2380).

**None-vs-falsy.** Use `is not None` / `len(x) > 0` for indices/batches/collections, never truthiness — `if non_compound_indices is not None:` and `if len(batch_indices) > 0`, and explicit `len(...) == 0` for empty `class_name`/`words` (#777, #2348).

**Future resolution.** Detection/resolution helpers live in `executor/utils.py` (`contains_future`, `resolve_futures`, `maybe_resolve_futures`). Do NOT re-implement per-module recursive `Future`/`Batch`/list/tuple/dict walkers — #2486 collapsed duplicated copies from `execution_data_manager/step_input_assembler.py` and `output_constructor.py` into these. New future-carrying paths call the shared helpers with a `context=` string. (#2475: deferring parent→root coordinate conversion to the dispatch thread is a parity trap — predictions with `coordinates_system: parent` must match when `serialize_results=False` and futures resolve late.)

**Injectable executor.** The engine accepts a `ThreadPoolExecutor` and must respect `max_concurrent_steps` (`executor/core.py`) even when the pool is larger (changelog v1.5.0, #1014). Do not create a `ThreadPoolExecutor` inside `run(...)`.

**Dimensionality / lineage.** Auto-batch-casting and `get_output_dimensionality_offset`, `get_parameters_enforcing_auto_batch_casting`, `get_parameters_accepting_batches[_and_scalars]` (`prototypes/block.py`) are load-bearing (changelog v1.6.0). Control-flow lineage lets control-flow-gated steps run without data lineage (#2106, changelog v1.8.0). Changes here must preserve results for pre-existing workflows; verify scheduling order when touching `executor/utils.py` / `executor/flow_coordinator.py` (#585).

**Inner-workflow inlining** is rewrite-order sensitive: child step renames must run before parameter-binding substitution so a parent selector isn't captured by a same-named child step (#2352, `inner_workflow/inline.py`). Dynamic blocks from nested inner workflows are collected depth-first and deduped by `manifest.block_type`, first occurrence wins (#2380, `inner_workflow/dynamic_blocks_collection.py`).

**Serialization** must handle empty AND non-batch outputs — both serialize (#815, changelog v1.4.0).

## Required companions
Block a behavior-changing EE PR that is missing:
1. **Changelog** — a user-facing entry under `## Unreleased` in `docs/workflows/execution_engine_changelog.md`.
2. **Tests** — unit test in the matching `tests/workflows/unit_tests/execution_engine/{compiler,executor,inner_workflow,introspection,profiling,...}` dir AND an integration test under `tests/workflows/integration_tests/execution/` reproducing the fixed scenario (#777, #2352, #645).

(See `### Not blocking` for version-bump and exemption carve-outs.)

## Release notice

For every compile/run behavior change, add a non-blocking top-level review notice
addressed to maintainers: **Execution Engine requires a version bump for
release**. This notice is required even when the contributor supplied the
changelog entry. Maintainers choose the version, update
`EXECUTION_ENGINE_V1_VERSION` and mirrored version assertions, move the entries
into the final version section, and leave a fresh `## Unreleased` section for
subsequent contributions.

## Key files & entry points
- `execution_engine/core.py` — `REGISTERED_ENGINES`, `retrieve_requested_execution_engine_version`, `_select_execution_engine`, `flush_stream_pipeline`.
- `execution_engine/v1/core.py` — `EXECUTION_ENGINE_V1_VERSION`, `ExecutionEngineV1`, `REGISTERED_STEP_ERROR_HANDLERS`.
- `execution_engine/v1/compiler/core.py` (`COMPILATION_CACHE`, `compile_workflow_graph`), `compiler/cache.py` (`BasicWorkflowsCache`), `graph_constructor.py`, `steps_initialiser.py`, `syntactic_parser.py`, `validator.py`.
- `execution_engine/v1/executor/core.py`, `output_constructor.py`, `execution_data_manager/step_input_assembler.py`, `executor/utils.py` (future helpers), `executor/flow_coordinator.py`.
- `execution_engine/v1/inner_workflow/{inline.py,dynamic_blocks_collection.py,compiler_bridge.py}`.
- `execution_engine/entities/{engine.py,base.py,types.py}`, `errors.py`, `introspection/schema_parser.py`.
- `.cursor/rules/execution-engine-version-changelog.mdc`.

## Reference PRs
- [#2486](https://github.com/roboflow/inference/pull/2486) — centralize `Future` resolution helpers; fix Batch-wrapped future leak.
- [#2475](https://github.com/roboflow/inference/pull/2475) — deferred parent→root coordinate conversion; run-param lockstep.
- [#2383](https://github.com/roboflow/inference/pull/2383) — canonical EE version bump + changelog + version-test update.
- [#2380](https://github.com/roboflow/inference/pull/2380) — collect+dedupe dynamic blocks from nested inner workflows.
- [#2352](https://github.com/roboflow/inference/pull/2352) — inner/outer same-step-name rewrite order (renames before bindings).
- [#2348](https://github.com/roboflow/inference/pull/2348) — guard empty class/segment names in introspection.
- [#2106](https://github.com/roboflow/inference/pull/2106) — control-flow lineage (EE + version bump).
- [#1640](https://github.com/roboflow/inference/pull/1640) — custom step error handlers (registry + ABC/engine surface).
- [#1504](https://github.com/roboflow/inference/pull/1504) — singular elements pushed into batch inputs.
- [#1379](https://github.com/roboflow/inference/pull/1379) — `WorkflowParameter.default_value` accepts container defaults.
- [#1122](https://github.com/roboflow/inference/pull/1122) — pydantic/OpenAPI BC in schema parser (type-checked dict access).
- [#1014](https://github.com/roboflow/inference/pull/1014) — injectable ThreadPoolExecutor.
- [#815](https://github.com/roboflow/inference/pull/815) — serialize empty / non-batch outputs.
- [#777](https://github.com/roboflow/inference/pull/777) — two-stage + continue-if on empty detections (`is not None` / `len`).
- [#645](https://github.com/roboflow/inference/pull/645) — lineage + conditional-execution branching-lineage test.
- [#585](https://github.com/roboflow/inference/pull/585) — executor scheduling order bug.

## Related topic skills
Load the matching topic skill when the PR also exhibits its concern (see each skill's `description` for the trigger):
- `review-topic-local-vs-remote-execution`
- `review-topic-workflow-state-management`
- `review-topic-backward-compat-and-versioning`
- `review-topic-concurrency-and-resource-safety`
- `review-topic-test-hygiene`
