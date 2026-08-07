# Execution Engine Changelog

Below you can find the changelog for Execution Engine.

## Unreleased
**What changed**

Add user-facing compile or execution behavior changes here. Maintainers replace
this heading with the Execution Engine and inference versions when releasing.

## Execution Engine `v1.15.0` | inference `1.3.9`

* **Remote HTTP 501 errors preserve their status without exposing internal URLs** —
  When a remotely executed Workflow step returns HTTP 501, the Execution Engine now
  surfaces a client-caused Workflow error with the same status and API message instead
  of a generic HTTP 500 containing the internal producer URL.

## Execution Engine `v1.14.0` | inference `1.3.8`

**What changed**

* **Output serialization accepts kinds declared as strings** — When an output kind was
  declared as a plain string (e.g. `"string"`) instead of a `Kind` object, output
  serialization crashed with `TypeError: unhashable type: 'list'`, surfacing as HTTP 500
  for the whole Workflow run. Such kinds are now resolved by name, so the matching
  serializer is applied (or the raw value is passed through when no serializer is
  registered for that kind).

* **Blocks can declare their dependent resources** — `WorkflowBlockManifest` gains
  an instance method `discover_dependent_resources() -> Optional[List[DependentResource]]`
  that lets a parsed step declare the external resources its execution will use,
  so callers (platform, preloading and auth pre-flight tooling) can enumerate
  them statically from a workflow definition. The envelope is regulated by the
  Execution Engine: resource types `roboflow_platform_model`,
  `roboflow_platform_project` and `third_party_model`, each with a typed,
  serializable (pydantic) metadata entity. Platform-model entries additionally
  state the nature of usage: `required_action` (`access` — the model entity only
  needs to be reachable on the platform, vs `execution` — the model is executed)
  and, for execution, `execution_location` (`local` / `remote` /
  `environment_defined` when the locality is decided at runtime by
  `WORKFLOWS_STEP_EXECUTION_MODE`). Blocks governed by their own locality
  override dictate it in place — the SAM3 image blocks declare nothing when
  `SAM3_EXEC_MODE=remote` (proxy execution ignores the configured model id
  and runs a fixed SAM3 server-side) and `environment_defined` otherwise. Returning `None` (the default) means the
  block does not declare its dependencies — distinct from `[]`, which declares
  that no external resources are needed. Field values that are workflow
  selectors (`$inputs.<name>` / `$steps.<name>.<property>`) are reported
  verbatim; each metadata entity exposes `requires_runtime_resolution()` to
  tell such references apart from concrete identifiers. Declarations whose
  final id is synthesized from the field value (family prefixes like
  `clip/<version>`, catalog lookups) additionally attach a non-serializable
  `model_id_resolver` callable that turns the substituted input value into
  the executed id — excluded from serialization, JSON schema and equality.
  All core blocks that reference models, Roboflow projects or third-party
  hosted models implement the method, and each implementation mirrors the
  model identifier that `run()` actually loads (including ids synthesized
  from version fields, e.g. `clip/<version>`). Project declarations follow
  their enabling controls, decided from static manifest values: Roboflow
  model blocks drop the active-learning target when `disable_active_learning`
  is literally `True` (the default), and dataset-upload blocks declare
  nothing when `disable_sink` is literally `True`; selector-fed controls
  keep the conservative may-need declaration. Introspection stops at the
  `active_learning_target_dataset` property — no project is derived from the
  model id when active learning is enabled without an explicit target. Blocks that load their model
  weights outside the model manager (the SAM2/SAM3 video trackers, which use
  `AutoModel.from_pretrained`) deliberately do not implement the method for
  now — their dependencies stay undeclared (`None`).

* **Dynamic (custom python) blocks report unknown dependencies** — manifests
  synthesized for dynamic blocks return `None` from
  `discover_dependent_resources()`: the python body is opaque to static
  analysis, so "unknown" is the only honest answer.

* **Opt-in pre-loading of declared Roboflow models at engine init** —
  `ExecutionEngine.init(...)` accepts a new optional parameter
  `dependencies_pre_init` (default `None`): a list of dependent-resource type
  names to pre-load, with `roboflow_platform_model` as the only supported
  value for now. When enabled, the engine deduces the declared dependencies of
  all compiled steps (`deduce_blocks_dependencies` in the compiler utils) and
  registers every concrete Roboflow platform model declared for execution in
  the model manager (both taken from `init_parameters`, as is the API key)
  during `init()` — before any run, for predictable first-inference latency.
  Declarations that reference `$inputs.<name>` cannot be loaded at init; on
  the **first** `run()` only — after runtime-input validation, so an invalid
  request neither consumes the single attempt nor triggers downloads — the
  engine resolves them against the provided
  runtime parameters (with input defaults applied), applies the declaration's
  `model_id_resolver` when attached (so e.g. a substituted CLIP version
  pre-loads `clip/<version>`, exactly the id execution uses), and registers
  the models whose identifiers became concrete. A resolver returning `None`
  declares the substituted value statically unresolvable (e.g. Qwen's
  fine-tuned sentinel label, whose final id depends on another input) — the
  dependency is skipped and resolves at execution time; a submitted input
  value the resolver cannot handle (e.g. an unknown catalog label) raises
  `RuntimeInputError`. Registration mirrors each block's actual loader:
  declarations may carry non-serializable `model_registration_kwargs`
  (e.g. `endpoint_type=CORE_MODEL` for CLIP / OCR / SAM2 / YOLO-World-style
  core models, matching `load_core_model()`). After each pre-loading pass the
  engine verifies the registered models are still present in the model
  manager and logs a warning when a size/memory-bounded manager evicted some
  of them (they lazily re-load at execution time). Pre-loading honours the effective step
  execution mode (explicit `step_execution_mode` init parameter, or the
  `WORKFLOWS_STEP_EXECUTION_MODE` default): `environment_defined` declarations
  are pre-loaded only when steps execute locally, `local` declarations always
  are, and access-only declarations (no weights pulled), remote execution and
  `$steps.…`-fed identifiers are never pre-loaded.
  `InferencePipeline.init_with_workflow(...)` exposes this as the opt-in
  `workflows_dependencies_pre_init` parameter (default `None` — no
  pre-loading) — video processing benefits most from predictable startup.


## Execution Engine `v1.13.0` | inference `v1.3.7`

**What changed**

* **Offline mode rejects remote Workflow step execution** — When `OFFLINE_MODE`
  is enabled, the compiler rejects `StepExecutionMode.REMOTE` during step
  initialisation (`WorkflowEnvironmentConfigurationError`) so Workflows cannot
  open remote inference clients without network access. Local step execution
  continues to work against warmed caches.

* **Proper model access failure status codes** - Model-access failures raised while loading local workflow models now preserve HTTP
  402, 403, and 423 statuses instead of surfacing as generic HTTP 500 errors.

---
template: redirect.html
redirect_url: https://docs.roboflow.com/workflows/developer-guide/developer-guide/execution-engine-changelog
---
