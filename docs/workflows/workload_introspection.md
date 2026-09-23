# Workload introspection

Workload introspection answers one question about a Workflow definition: **what work does this definition describe?** It compiles the definition structurally — no block is initialised, no model is loaded or registered, no custom Python is evaluated, nothing is executed — and returns the compiled graph, the per-step declarations the blocks make about themselves, and an inventory of the models the definition refers to.

The answer is *portable*: it describes the definition, not the server that answered. The same definition returns the same graph, step declarations and model references on a hosted deployment, on a Jetson and in an air-gapped container. The one exception is the optional model-metadata enrichment in the inventory (`summary.models.items[].metadata`): whether it is populated depends on the answering server's `USE_INFERENCE_MODELS` gate, on the credentials in the request and on the server's access to the model registry. Consumers (a cost estimator, a scheduler, a capacity planner) apply their own hardware assumptions to the portable part.

## Endpoints

| Endpoint | Body | Purpose |
| --- | --- | --- |
| `POST /workflows/describe_workload` | `{"api_key": "...", "specification": {...}}` | describe an inline definition |
| `POST /{workspace_name}/workflows/{workflow_id}/describe_workload` | `{"api_key": "...", "use_cache": true, "workflow_version_id": null}` | describe a saved definition |

Both accept the api key in the request body or as an `Authorization: Bearer <key>` header, exactly like `/workflows/describe_interface`; a request with neither is rejected with HTTP 400 and the same error envelope. Both are removed together with the other Workflow endpoints when `DISABLE_WORKFLOW_ENDPOINTS=True`. The saved-definition route forwards `use_cache` and `workflow_version_id` to the ordinary saved-definition lookup.

The Python entry point is `roboflow_workflows.execution_engine.introspection.workload.describe_workflow_workload(definition, init_parameters=None, execution_engine_version=None, model_metadata_provider=None)`. It lives in the standalone `roboflow-workflows` package and needs no inference server.

## Trying it from a notebook

The cell below describes a saved workflow and shows the complete response as an expandable tree. Set `BASE_URL` to an inference server that runs this feature (an image built from this repository or newer); the default assumes a local server on port 9001. The call inspects the definition only — it runs no inference, loads no model and sends no image. Whether the model inventory is enriched is decided by the server's `USE_INFERENCE_MODELS` setting; there is no request-body field for it.

```python
from getpass import getpass

import requests
from IPython.display import JSON, display

BASE_URL = "http://localhost:9001"  # a server running workload introspection
WORKSPACE_NAME = "<your workspace>"
WORKFLOW_ID = "<your workflow id>"

session = requests.Session()
session.headers["Authorization"] = f"Bearer {getpass('Roboflow API key: ')}"


def describe_workload(path: str, payload: dict) -> dict:
    response = session.post(f"{BASE_URL}{path}", json=payload, timeout=(10, 120))
    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code} from {path}: {response.text[:2000]}"
        )
    return response.json()


# Saved workflow: use_cache=False fetches the current definition from the platform.
description = describe_workload(
    f"/{WORKSPACE_NAME}/workflows/{WORKFLOW_ID}/describe_workload",
    {"use_cache": False},
)
display(JSON(description, expanded=False))
```

To describe a definition you already hold in memory, post it to the inline endpoint. `specification` is the raw definition dictionary (`version`, `inputs`, `steps`, `outputs`), not a whole request envelope:

```python
# specification = {"version": "1.0", "inputs": [...], "steps": [...], "outputs": [...]}
description = describe_workload("/workflows/describe_workload", {"specification": specification})
display(JSON(description, expanded=False))
```

## Response

The response is a `WorkflowIntrospection` document. Every entity object in it carries a defaulted `type` discriminator, at every nesting level, so another service can parse it without Python callables. Plain JSON maps that are not entities — `steps_by_dimensionality` (on the summary and on every model entry), a restriction's `configuration_equals` and a discovery problem's `details` — carry no `type`, and their contents are arbitrary JSON data. The full JSON Schema is published in the server's `/openapi.json` under `components.schemas.WorkflowIntrospection`, and in Python as `WorkflowIntrospection.model_json_schema()` from `roboflow_workflows.execution_engine.introspection.workload_entities`.

| Field | Meaning |
| --- | --- |
| `schema_version` | `"1"`. Bumped only for an incompatible change of this document. It is independent of the workflow-definition `version` and of `execution_engine_version`. |
| `execution_engine_version` | the Execution Engine version that compiled the definition. |
| `nodes` | one entry per input, step and output, with `kind` in `input` / `step` / `output`. Ids are canonical selectors (`$inputs.image`, `$steps.detection`, `$outputs.predictions`). Internal compiler nodes are never exposed. |
| `edges` | deduplicated `(source, target, kind)` triples, `kind` in `data` / `control`. The same node pair may carry both. This is connectivity, not a schedule. |
| `steps` | one `StepMetadata` per compiled step. |
| `summary` | the model inventory, `max_dimensionality` and the step histogram. |

### Step metadata

| Field | Meaning |
| --- | --- |
| `block_type` | the canonical manifest identifier including the version (`roboflow_core/dynamic_crop@v1`), never the alias a definition happened to use. |
| `input_dimensionality` | the depth of the step's **dimensionality-reference input** — the compiled reference lineage, not the deepest input the step consumes and not the executor's loop depth. See below. |
| `output_dimensionality` | the compiled **output** depth of the step: the depth of the data the step produces for its successors. |
| `accepts_batch_input` | the normalised manifest capability. It is not a promise that the block computes a batch simultaneously. |
| `resources` | `Discovery[DependentResource]` — the Roboflow models, Roboflow projects and third-party models the step needs. See [Resources](#resources). |
| `restrictions` | `Discovery[RestrictionMetadata]` — portable, conditional restrictions. See [Restrictions are conditional, not evaluated](#restrictions-are-conditional-not-evaluated). |
| `operations` | `Discovery[WorkOperation]` — what kind of work the block performs, from a fixed 25-member enum. |

#### Reading `input_dimensionality` and `output_dimensionality`

A manifest nominates one input property as its **dimensionality reference** (`get_dimensionality_reference_property()`); other properties may be declared to arrive one or more levels deeper (`get_input_dimensionality_offsets()`). `input_dimensionality` is the compiled depth of that *reference* property. The deeper side inputs are real, but they are **not** exposed by this document — by design, because the reference lineage is what determines the step's own position in the graph.

Four real blocks, all four values taken from an actual response:

| Step | What it reads | Reported |
| --- | --- | --- |
| `roboflow_core/dynamic_crop@v1` on a whole image | image at depth 1, detections at depth 1 | `in=1 out=2` — an **expansion**: one image in, a list of crops out |
| `roboflow_core/roboflow_classification_model@v2` on those crops | crops at depth 2 | `in=2 out=2` — an ordinary step one level deeper |
| `roboflow_core/detections_stitch@v1` | `reference_image` (the reference property) at depth 1, `predictions` at depth **2** | `in=1 out=1` — a **reducer**: it consumes depth‑2 data but sits at depth 1, and the depth‑2 side input is not visible in the response |
| `roboflow_core/detections_classes_replacement@v1` | detections at depth 1, classifications at depth 2 | `in=1 out=1` — same shape as above |

A block whose manifest declares an output dimensionality offset of −1 (an explicit collapse) reports `in=2 out=1`: its reference input really is one level deeper than its output.

The practical consequence for a consumer: `steps_by_dimensionality` places a stitch/reducer step at depth **1**, not at the depth of the batch it merges. If you need "how much data does this step touch", combine `input_dimensionality` with the `output_dimensionality` of its predecessors from `edges` — the document gives you both, but it never pre-computes that number for you.

### Discovery: known, known-absent and unknown

Every declaration is wrapped in a `Discovery`. A complete declaration with no items looks like this on the wire:

```json
{"type": "discovery", "items": [], "complete": true, "unknown_reasons": []}
```

* `complete: true` with an empty `items` means **known absence** — the block declares that it does none of this. It is a positive statement, not a gap in the data.
* `complete: false` means the list may be incomplete; `unknown_reasons` then carries at least one `DiscoveryProblem` object saying why.
* A complete result never carries reasons, and reasons never carry exception traces or secrets.

A block that declares nothing (a third-party plugin that has not been annotated) is reported as unknown, never as empty. Every block registered in this repository — core, enterprise and the server's own plugin — writes all three hooks explicitly in its manifest; the one exception is `roboflow_core/inner_workflow@v1`, which does not write the resources hook because its dispatched child may pull anything. For restrictions the hook is `get_actual_restrictions()`; see [How a block declares a restriction](#how-a-block-declares-a-restriction).

Writing a hook is not the same as knowing its answer. An explicitly written hook may still answer "unknown" or "partly known":

* **Audited unknown.** Some model blocks write the resources hook and return `None` on purpose, because their model has no identity they could declare truthfully. Four model families do this today: Google Vision OCR, Seg Preview, Stability AI inpainting and Stability AI outpainting. Their steps report `resources.complete: false` with `declaration_unavailable`. No id is guessed for them.
* **Conditional declarations.** Some blocks declare a resource only under a condition in their own configuration, for example an active-learning target project that matters only while active learning is enabled. A literal condition is evaluated by the block. When the condition itself comes from a selector (for example `disable_active_learning: "$inputs.flag"`), the block declares the resource it might use; the document does not model whether the condition will hold at run time.
* **Unknown identities.** A declared resource whose identity is a selector or a blank literal makes the step's resources incomplete. See [Resources](#resources).

### Discovery problems

Every entry of `unknown_reasons` is a `DiscoveryProblem`:

```json
{
  "type": "discovery_problem",
  "code": "unresolved_selector",
  "description": "Field `model_id` of the roboflow_platform_model declared by step `$steps.detection` is set by selector `$inputs.model`, which is only known at run time, so the resources are not fully known.",
  "details": {
    "node_id": "$steps.detection",
    "declaration": "resources",
    "field": "model_id",
    "selector": "$inputs.model",
    "resource_type": "roboflow_platform_model"
  }
}
```

* `code` is the machine-readable reason. The set is **closed** and owned by the Execution Engine; a new member is an Execution Engine change, never a producer's free choice.
* `description` is display text. Branch on `code` and read `details` — **never parse the description**. Its wording may change without a schema bump.
* `details` is an open JSON map understood **per code**. The conventions below are documented, not shape-enforced: a producer may add context to a problem without a schema change, so read the keys you know and ignore the rest.

Two problems are the same problem when their `code` and their `details` match; the wording plays no part. Reasons are deduplicated on that identity and ordered deterministically by it, so the same definition always produces the same list, in the same order, whatever order the steps were visited in. Distinct contexts stay distinct: two steps hitting the same unresolved selector, or one step with two unresolved fields, are two entries.

| `code` | Meaning |
| --- | --- |
| `declaration_unavailable` | the declaration is not available as a complete, portable statement. Three cases carry it: the block does not declare this domain at all (the hook returned `None`, e.g. an unannotated plugin); a block's restrictions come from the legacy `get_restrictions()` fallback (then `details.source` is `"get_restrictions"`); or a restriction's condition names configuration this process cannot evaluate (then `details.configuration_keys` lists those keys). Read `details` to tell them apart. |
| `declaration_failed` | the declaration hook raised, or answered with something that is not a valid declaration. Introspection stays non-fatal: the step is reported with an empty, incomplete declaration rather than failing the whole response. |
| `unresolved_selector` | a value the declaration depends on is a workflow selector, so it is only known at run time. |
| `invalid_resource_identifier` | a literal resource identifier names nothing (empty or whitespace-only). No identifier is ever fabricated. |
| `opaque_remote_workflow` | the step dispatches a child workflow to a remote server, which compiles it; nothing about the child is visible here. |
| `custom_python_internals_unknown` | the step runs user-supplied Python; what the code does beyond the declared items is not statically analysable. |

Details the built-in problems carry:

| Key | Present on | Meaning |
| --- | --- | --- |
| `node_id` | every built-in problem | the canonical `$steps.<name>` id of the step the problem belongs to. |
| `declaration` | every built-in problem | which declaration the problem is **about**: `resources`, `operations` or `restrictions`. A problem carried by the model inventory keeps the `resources` of the step it came from. |
| `block_type` | `declaration_unavailable`, `declaration_failed` | the canonical manifest identifier, where the reporter knows it. |
| `field` | `unresolved_selector`, `invalid_resource_identifier` | the manifest field whose value caused the problem, or — when `resource_type` is present — the resource metadata field. The identity fields reported are `model_id`, `provider` and `project_url`. |
| `selector` | `unresolved_selector` | the selector as written in the definition (`$inputs.model`, `$steps.a.b`). It is never resolved and never guessed from an input's default value. |
| `resource_type` | `unresolved_selector` on a resource identity field, `invalid_resource_identifier` | the values emitted are `roboflow_platform_model`, `third_party_model` and `roboflow_platform_project`. |
| `source` | `declaration_unavailable` on a restriction declaration | present, and equal to `"get_restrictions"`, when the answer came from the legacy `get_restrictions()` fallback — whether the block overrode that classmethod or inherited its empty default. It marks the fallback; absent otherwise. |
| `configuration_keys` | `declaration_unavailable` on a restriction declaration | the sorted, de-duplicated NAMES of the configuration keys the local (`ignore_environment_restrictions=False`) view could not evaluate. Never the host's values for them. |

The rows above describe what the built-in problems emit **today**. `resource_type` and `field` are general keys of the convention — a future producer may use them for another resource kind. Read the keys you know and ignore the rest; do not assume a key is present because the convention allows it.

`field` names where the problem was seen; it is not a lineage and does not claim to name the original input binding. An `invalid_resource_identifier` deliberately does **not** echo the invalid value back. Neither a description nor a details map ever contains raw exception text, a traceback, a credential or an authorization header — a block hook that raises with a secret in its message yields a plain `declaration_failed` problem.

A restriction-side example: the Kafka producer keeps the restriction it does know and explains the rest.

```json
{
  "type": "discovery_problem",
  "code": "unresolved_selector",
  "description": "Field `fire_and_forget` of step `$steps.producer` is set by selector `$inputs.wait_for_ack`, which is only known at run time, so the restrictions are not fully known.",
  "details": {
    "node_id": "$steps.producer",
    "declaration": "restrictions",
    "field": "fire_and_forget",
    "selector": "$inputs.wait_for_ack"
  }
}
```

### Resources

A step's `resources` discovery lists `DependentResource` entries: the external things the step needs. Each entry carries a `resource_type` and a typed `metadata` object whose `type` always agrees with `resource_type`:

| `resource_type` | `metadata.type` | Metadata fields |
| --- | --- | --- |
| `roboflow_platform_model` | `roboflow_platform_model` | `model_id`, `required_action`, `execution_location` |
| `roboflow_platform_project` | `roboflow_platform_project` | `project_url` |
| `third_party_model` | `third_party_model` | `provider`, `model_id` |

A Roboflow model reference declares what the step needs from the model:

* `required_action` is `execution` (the step runs the model: weights are pulled or inference is requested from a service) or `access` (the step only needs the model entity to be reachable on the platform, for example to attach monitoring metadata; nothing executes).
* `execution_location` is set only for `execution` and is `null` for `access`. Its values are `local` (always in-process), `remote` (always on a remote service) and `environment_defined`. Most model blocks declare `environment_defined`: the location is decided at runtime by the step-execution-mode configuration of whichever service or runtime eventually executes the definition. It is **not** a statement about the backend the inspecting server has selected — the document never reports that.

A model id or project url supplied through a selector (`$inputs.model`) is preserved verbatim in the per-step declaration and is never resolved by introspection.

#### Per-step completeness of resource identities

A step may know which resources it uses without knowing who they are. Each declared resource has identity fields:

| `resource_type` | Identity fields |
| --- | --- |
| `roboflow_platform_model` | `model_id` |
| `third_party_model` | `provider`, `model_id` |
| `roboflow_platform_project` | `project_url` |

For every identity field of every declared item:

* a selector value adds an `unresolved_selector` problem with `field`, `selector` and `resource_type`;
* a blank literal (empty or whitespace-only) adds an `invalid_resource_identifier` problem with `field` and `resource_type`, never the value.

Either problem makes the step's `resources` `complete: false`. The item itself stays in `items`, verbatim. Problems the block reported itself stay next to the new ones. Selectors are never resolved and no default id is guessed. So `StepMetadata.resources.complete == true` means every declared resource of the step is identified by a literal.

The model inventory is about **models** only (see [Model inventory](#model-inventory)):

* model identity problems (`roboflow_platform_model`, `third_party_model`) make `summary.models` incomplete too;
* project identity problems (`roboflow_platform_project`) keep the step incomplete but do **not** make `summary.models` incomplete: a project is not a model;
* every other step problem — an unavailable or failed declaration, an opaque remote workflow — still makes `summary.models` incomplete.

#### Declaring resources (block authors)

`discover_dependent_resources()` returns a plain list (a complete declaration), a `Discovery` (explicit completeness with reasons) or `None` (unknown). Return the configured id verbatim, selector included; introspection reports its completeness. Do not return a guessed default.

`roboflow_platform_model()` accepts a keyword-only `preloadable` argument, default `True`. It is an in-process aid for the Execution Engine, like `model_id_resolver` and `model_registration_kwargs`: it never appears in this document, in `to_dict()` or in the JSON schema. `preloadable=False` means "the generic Execution Engine model-manager preloader must not register this model". Use it for a block that loads and owns its model itself. It says nothing about whether the block loads weights or runs remotely. The streaming video blocks (SAM2 video, SAM3 video, action recognition) declare their model this way, with `required_action: "execution"` and `execution_location: "local"`. SAM3 video declares only the model its literal `tracking_mode` selects: `model_id` for `concept`, `visual_model_id` for `visual`.

Per-step `resources` and the inventory at `summary.models.items` answer different questions. The per-step declaration says what *this step* needs and whether it executes the model or only needs access to it. The inventory deduplicates model references across steps and may carry an optional enriched `metadata` block (`model_type`, `model_variant`, `task_type`) that per-step declarations never carry. See [Model inventory](#model-inventory).

### Restrictions are conditional, not evaluated

A `RestrictionMetadata` carries a `code`, a `severity` and a `when` condition; it is **not** filtered against the answering server's configuration. The condition ANDs across its axes (`runtimes`, `step_execution_modes`, `input_modes`, `configuration_equals`) and ORs within each populated axis; an axis left as `null` is unrestricted, and an axis is never an empty list. A custom Python step, for example, always reports

```json
{
  "type": "restriction",
  "code": "custom_python_execution_disabled",
  "severity": "hard",
  "when": {
    "type": "restriction_condition",
    "runtimes": null,
    "step_execution_modes": null,
    "input_modes": null,
    "configuration_equals": {
      "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": false,
      "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "local"
    }
  }
}
```

whether or not this server allows custom Python. The consumer decides what the restriction means for its own target configuration.

The builder asks the block for that portable view explicitly, with
`get_actual_restrictions(ignore_environment_restrictions=True)`, and projects each answer onto the DTO above. `schema_version` is unaffected: this document's restriction payload is exactly what it was.

### How a block declares a restriction

A block authors restrictions with ONE entity type, `RuntimeRestriction`, and two public methods return it. The state-loss caveats are the exception to sharing declarations: they are declared separately for each method on purpose (see [below](#state-loss-restrictions-editor-view-and-actual-view-differ-on-purpose)). Nothing here changes the editor: `get_restrictions()` returns what it always returned, in the same order, with the same notes and the same `to_dict()` payload.

| Method | Level | Returns | Filtered against this host? |
| --- | --- | --- | --- |
| `get_restrictions()` | classmethod | `List[RuntimeRestriction]` | yes — the legacy, editor-facing view, unchanged (state-loss caveats differ from the actual view on purpose, see below) |
| `get_actual_restrictions(*, ignore_environment_restrictions=False)` | instance | `Discovery[RuntimeRestriction]` | only when the flag is `False`, and only the configuration predicates |

`RuntimeRestriction` keeps `severity`, `note` and its three `applies_to_*` axes, and adds two defaulted fields at the end, so every existing constructor call is still valid:

* `code` — a stable machine-readable identifier authored from the restriction's meaning. It defaults to `generic_restriction`, which says "this caveat has no dedicated identifier". In the Python `Discovery[RuntimeRestriction]` the note is part of a restriction's identity, so two `generic_restriction` entries explaining different failure modes stay two entries. On the wire that distinction is gone by design: `RestrictionMetadata` carries no note, so two such entries project onto the same DTO and the document's existing de-duplication by `(code, severity, condition)` coalesces them into one.
* `applies_to_configuration` — a `key == value` map describing the TARGET deployment's configuration (e.g. `{"ENABLE_TENSOR_DATA_REPRESENTATION": true}`).

`to_dict()`, the editor payload, carries **neither** of the two new fields.

The wire DTO is derived, never authored twice: `restriction_metadata_of(restriction)` maps the code, the severity and the three axes plus `applies_to_configuration` onto a `RestrictionMetadata`, dropping the human note (the portable form has no `note` field).

#### State-loss restrictions: editor view and actual view differ on purpose

Some blocks keep state in their workflow block instance: trackers and other per-video analytics, frame / heat / trace history, cooldown and rate-limit timers, the S3 append-log buffer and the model-monitoring aggregation buffer. For these caveats the two methods intentionally differ:

| | `get_restrictions()` (editor) | `get_actual_restrictions()` (this document) |
| --- | --- | --- |
| code | unchanged (several custom declarations use `generic_restriction`) | unchanged (specific codes, e.g. `stateful_video_state_resets_on_stateless_http`) |
| severity, runtimes, input modes | unchanged | same values as the editor view |
| step execution modes | `["remote"]`, as before | `null` — applies to either mode |
| note | unchanged | explains state loss independently of the model execution mode |

Why the mode is dropped. `step_execution_modes` says where the Execution Engine runs a model step: `local` inside the engine's process, `remote` through an inference service. It does not say whether block state survives. A workflow request that builds a fresh workflow / block instance starts from empty state, even on the same CPU worker, whether its models run locally or remotely.

Who decides the lifecycle. Stable cross-frame behaviour needs the target to preserve the same step's state for the same logical stream across calls. That is a property of the target service that runs the workflow. The server answering this introspection call does not know it and does not evaluate it. Reusing some engine or process, request affinity, or CPU versus GPU hardware does not guarantee it by itself.

Current limitation. The runtime axis cannot fully tell a persistent embedded engine apart from a self-hosted server that builds fresh instances for every HTTP request. These caveats keep their existing `hosted_serverless` and `dedicated_deployment` scope, and a consumer evaluating the conditions as written should honour that runtime predicate. A self-hosted deployment that serves each request with fresh instances can still lose state the same way; recognising that case needs the consumer's own knowledge of its target's lifecycle, which this document does not encode.

A tracker, for example, reports:

```json
{
  "type": "restriction",
  "code": "stateful_video_state_resets_on_stateless_http",
  "severity": "soft",
  "when": {
    "type": "restriction_condition",
    "runtimes": ["dedicated_deployment", "hosted_serverless"],
    "step_execution_modes": null,
    "input_modes": ["video"],
    "configuration_equals": {}
  }
}
```

#### `ignore_environment_restrictions`

* `True` — the **portable** view, and what this document carries. No host evaluation at all: every applicable declaration is returned with its condition intact, for the downstream target to evaluate. It does **not** drop environment-dependent restrictions.
* `False` (the default) — the **host** view. Only the `applies_to_configuration` predicates are evaluated, against the configuration installed in this process (the parsed package configuration, never `os.environ`). An entry whose predicate definitively does not hold here is removed; the runtime, input-mode and step-execution-mode axes are never evaluated and never guessed, and no condition is ever stripped from a surviving entry.

Evaluation is three-valued and the predicates are ANDed. A key this package cannot evaluate only matters while the condition is still open: if another, evaluable predicate of the SAME restriction already does not hold, the restriction definitively does not apply here and is dropped with no problem reported. Otherwise the entry is kept AND the result becomes incomplete, with a `declaration_unavailable` problem listing the unevaluable key names (never their values).

#### Authoring

There is exactly one hook. A block overrides `get_actual_restrictions(self, *, ignore_environment_restrictions=False) -> Discovery[RuntimeRestriction]` and builds its declaration there, refined by the instance's own literal settings. A block that declares no restriction returns an empty complete discovery directly:

```python
def get_actual_restrictions(
    self, *, ignore_environment_restrictions: bool = False
) -> Discovery[RuntimeRestriction]:
    return Discovery[RuntimeRestriction](items=[], complete=True, unknown_reasons=[])
```

A block that does declare something passes the data through `actual_restrictions_of()`, which normalises it (`None` = unknown, a plain list = a complete declaration, a `Discovery` = explicit completeness with reasons) and applies the flag:

```python
def get_actual_restrictions(
    self, *, ignore_environment_restrictions: bool = False
) -> Discovery[RuntimeRestriction]:
    return actual_restrictions_of(
        declared=[STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION],
        node_id=f"$steps.{self.name}",
        ignore_environment_restrictions=ignore_environment_restrictions,
    )
```

`actual_restrictions_of()` takes restriction DATA, not a manifest: it calls nothing back and picks between no implementations. A plugin that subclasses a built-in can extend what it inherits by calling `super().get_actual_restrictions(ignore_environment_restrictions=...)` and adding to the result.

A declaration the portable contract cannot express — a blank or non-identifier `code`, an axis declared as an empty list, a blank configuration key — is reported as `declaration_failed` rather than published as a complete declaration the wire cannot carry. The `RuntimeRestriction` constructor itself stays permissive; the contract begins at the new API. A hook that raises is sanitised the same way at the workload-builder boundary: the exception text never reaches the response.

#### Compatibility

A block that does not override `get_actual_restrictions()` gets the default body, which simply calls `self.get_restrictions()` — the block's own legacy override through ordinary Python dispatch, or the inherited `[]` — and wraps whatever comes back. That fallback is **always incomplete**: a legacy getter may already have filtered its entries against the flags of the host that answered, and the inherited default declares nothing at all, so neither a short list nor an empty one proves absence. The items themselves are reported as declared, codes included — a legacy declaration may well carry a specific `code`. The reason carries `"source": "get_restrictions"`.

`ignore_environment_restrictions=True` does not change that verdict: the flag governs what this package evaluates, and it cannot undo filtering that happened inside the classmethod. With `False` the fallback still behaves like any other declaration — the `applies_to_configuration` predicates a legacy entry carries are evaluated against this host as usual.

A block whose class-level list IS complete and environment independent can say so, by wrapping it explicitly instead of taking the fallback:

```python
def get_actual_restrictions(
    self, *, ignore_environment_restrictions: bool = False
) -> Discovery[RuntimeRestriction]:
    return actual_restrictions_of(
        declared=list(self.get_restrictions()),
        node_id=f"$steps.{self.name}",
        ignore_environment_restrictions=ignore_environment_restrictions,
    )
```

### Model inventory

`summary.models` is a discovery whose `items` are `ModelSummary` entries: an inventory of model **references**, deduplicated by `(provider, model_id)`, with the referring step ids preserved and counted by the input depth of each referring step:

```json
{
  "type": "model_summary",
  "provider": "roboflow",
  "model_id": "my-project/3",
  "used_by_steps": ["$steps.crop_detection", "$steps.detection"],
  "steps_by_dimensionality": {"1": 1, "2": 1},
  "metadata": null,
  "metadata_status": "disabled"
}
```

* Two steps using one model id produce **one** entry with two `used_by_steps`, not two entries.
* `steps_by_dimensionality` is the per-model counterpart of the summary histogram: for every step in `used_by_steps`, its compiled **input** depth (the reference-lineage depth described above) is counted once. Here the model is referenced by `detection` at depth 1 and by `crop_detection` on the crops at depth 2. Keys are strings on the wire and sorted ascending, values are positive, depths with no referring step are omitted, and the values always sum to the length of `used_by_steps`. The field is always populated from compilation; it does not depend on the metadata gate.
* A step is counted **once** per model whatever it declares: a step that lists the same model twice (for example once with `required_action: "execution"` and once with `"access"`) is one entry in `used_by_steps` and one count in the histogram, while a step that references two different models contributes one count to each of the two entries. Different providers with the same model id remain separate entries with separate histograms.
* Roboflow platform references use the provider `roboflow`; an explicit third-party provider is preserved as declared. Roboflow *projects* are not model entries.
* A model id supplied through a selector (`$inputs.model`) is never invented into an id and never sent to a metadata lookup: it stays a per-step unresolved reference and makes the inventory incomplete with an `unresolved_selector` problem naming the step, the resource field and the selector. A blank literal id or provider yields an `invalid_resource_identifier` problem naming the field. A step whose resources are unknown or incomplete propagates its own problems into the inventory unchanged, with all the context they carry — except selector or blank problems about a project identity, which keep only the step incomplete (see [Per-step completeness of resource identities](#per-step-completeness-of-resource-identities)). Literal models of a partly known step are still listed. Unresolved and unknown references contribute nothing to any histogram; only literal, known references are counted, so an incomplete inventory carries exact counts for what it does list.
* Being listed is not a claim that the reference executes at runtime — a step may only need access to the model. The per-step `resources` entry keeps that distinction in `required_action`. Access-only references are inventoried and counted like every other reference: the histogram counts referring steps, not model calls.

`metadata_status` is `disabled`, `available` or `unavailable`:

| `USE_INFERENCE_MODELS` | Result |
| --- | --- |
| `False` | every entry is `disabled` with `metadata: null`, and **zero** metadata lookups happen. |
| `True` | the server resolves metadata once per unique model id (not per step) and maps `model_type`, `model_variant`, `task_type`. Resolution is not the same as a remote request: a resolution answered by the in-memory cache described below issues no call at all, so a request may make fewer remote calls than it has unique model ids. A failed or empty resolution yields `unavailable` without dropping the entry, and without making an otherwise complete inventory incomplete. |

A standalone package call with no metadata provider yields `unavailable` — the package makes no claim about any server flag.

#### How the lookup reaches the platform, and what that costs you

The server performs the lookup through the existing registry helper `get_model_metadata_from_inference_models_registry()` (`GET /models/v1/external/stat`). That call is **metadata only**: no weights are downloaded, no model is registered with the model manager, and the model-type resolution paths that would enforce this deployment's model support are deliberately not used. The helper is called with its **default** cache prefix, exactly like every other caller: introspection derives no *shared* cache key from credentials, so no credential-derived key reaches the shared cache (`inference.core.cache.cache` — Redis, with an in-process memory cache fallback when Redis is not configured or cannot be reached).

Repeat lookups are absorbed by an **in-memory cache inside the server process**, shared by every request and every provider instance. It holds at most 1000 entries (the least recently used entry is evicted at that bound) and expires them after `MODELS_CACHE_AUTH_CACHE_TTL` (default 15 minutes), the TTL the authorization cache already uses. Its key does contain the api key, and it never leaves the process. The key is an exact tuple, not a hash:

```
(api_key, model_id, authorised_workspace, MODELS_CACHE_AUTH_ENABLED)
```

* `api_key` as given, so "no key" (`None`) and an empty key are different entries.
* `authorised_workspace` is the workspace the call would send in the `x-assume-identity-authorised-workspace` header, resolved per call before the lookup — set only when the service access token is configured and the per-request workspace id is header-safe, and `None` otherwise. When it is `None`, the api key, model id and authorization mode still key the entry.
* `MODELS_CACHE_AUTH_ENABLED` is the authorization policy in force when the entry was written, so an answer obtained while enforcement was off can never be reused as an enforcement-on authorization success.

Only successful, usable metadata is cached. Exceptions, `unavailable` results and payloads whose every field is unknown are not stored, so the next request retries. The `USE_INFERENCE_MODELS`, provider and `OFFLINE_MODE` gates are all evaluated before the cache is consulted. The cache lives in one process: with several workers each keeps its own, and concurrent first-time lookups of the same key may issue more than one request — there is no request coalescing.

What the two authorization policies mean for isolation:

* **`MODELS_CACHE_AUTH_ENABLED=True`** — the helper does not read the shared cache, so an in-memory miss reaches the platform carrying that call's own key and identity headers; a hit reuses that answer for the TTL, for that key tuple only.
* **`MODELS_CACHE_AUTH_ENABLED=False`** (the default) — the helper keeps its existing policy for all of its callers: it reads its shared cache, which is keyed by model id alone. An in-memory miss can therefore be answered from an entry another caller populated for the same model id, without a fresh authorization. That entry pool is now the same one the model-resolution path uses, because both call the helper with the same default prefix; the cached value is the same metadata-only payload in either case. This is the helper's own policy, unchanged by workload introspection; **per-workspace isolation on a multi-tenant deployment relies on enabling `MODELS_CACHE_AUTH_ENABLED`.**

Two consequences of reusing that helper, which is shared with the loading paths and is not modified by this feature:

* **On hosted serverless the request carries the credits-verification header.** When `ENFORCE_CREDITS_VERIFICATION` is on, the helper adds `ENFORCE_CREDITS_VERIFICATION_HEADER` to every call, including this one. A workspace without credits therefore gets a platform error, which surfaces as `metadata_status: "unavailable"` for that entry — the inventory itself is unaffected. On the platform side an introspection lookup is indistinguishable from a model-load lookup.
* **Turning the metadata gate off also turns tensor mode off.** The server defines `ENABLE_TENSOR_DATA_REPRESENTATION` as the environment value **and** `USE_INFERENCE_MODELS`, so a deployment (or a test) that sets `USE_INFERENCE_MODELS=False` to avoid the lookup silently runs in numpy mode as well. This does not change the introspection response — it is identical in both representations — but it is worth knowing before reading a flag matrix.

### Dimensionality summary

`steps_by_dimensionality` counts every compiled step once, by its **input** depth (the reference-lineage depth described above); keys are strings on the wire, values are positive, dimensions with no steps are omitted, and the values sum to the number of steps. `max_dimensionality` is the deepest compiled input or output depth in the graph, including the input-to-output depth of a workflow with no steps at all.

Each `ModelSummary` carries a `steps_by_dimensionality` of its own with the same encoding, restricted to that model's `used_by_steps` (see [Model inventory](#model-inventory)). The per-model maps are not a partition of the summary map: a step that references two models appears in both of their histograms, and a step that references no model appears in neither.

#### Dimensionality is not multiplicity

Dimensionality is the nesting depth of the data, not the number of items. In `image -> dynamic crop -> classifier` the classifier sits at depth 2 whether the crop step yields 3 crops or 300.

* Counts per depth count **graph steps**, not inference items. `{"2": 1}` means one step works on depth-2 data. It does not mean one model call.
* A step at depth 2 that classifies N items may batch them into fewer model calls than N.
* The work under nested expansion depends on the actual child counts: their sum over parents at one level, and their products across levels. Input sizes and the number of detections decide those counts at run time; this document does not know them.

The document carries no item counts, cardinality estimate, work expression or cost score. A service that needs them supplies its own assumptions, for example target-service heuristics about typical detection counts.

## What this is not

This is compile-time introspection, not an estimator and not a profiler. The document deliberately contains **no**:

* cost score, latency estimate, weight or any other number derived from hardware assumptions;
* lineage identifiers, input bindings or origin paths;
* compute/IO classification of a step;
* execution groups, worker counts or a schedule;
* selected custom-Python backend, selected model backend or any other statement about how *this* server would run the definition.

Answering those is the job of the service that consumes this document.

## Example: detection, crop, classification

Request (`POST /workflows/describe_workload`):

```json
{
  "api_key": "<your key>",
  "specification": {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
      {"type": "roboflow_core/roboflow_object_detection_model@v3", "name": "detection",
       "images": "$inputs.image", "model_id": "my-project/3"},
      {"type": "roboflow_core/dynamic_crop@v1", "name": "crop",
       "images": "$inputs.image", "predictions": "$steps.detection.predictions"},
      {"type": "roboflow_core/roboflow_classification_model@v2", "name": "classification",
       "images": "$steps.crop.crops", "model_id": "my-other-project/1"}
    ],
    "outputs": [{"type": "JsonField", "name": "classes",
                 "selector": "$steps.classification.predictions"}]
  }
}
```

The complete response below shows the result with `USE_INFERENCE_MODELS=False` on the server (the same output the package compiler produces with a disabled metadata provider), so both inventory entries report `metadata_status: "disabled"` with `metadata: null` and no registry lookup happens. With the gate on, only those two fields of the inventory entries could differ; everything else is identical.

```json
{
  "type": "workflow_introspection",
  "schema_version": "1",
  "execution_engine_version": "1.15.2",
  "nodes": [
    {"type": "graph_node", "id": "$inputs.image", "kind": "input"},
    {"type": "graph_node", "id": "$steps.detection", "kind": "step"},
    {"type": "graph_node", "id": "$steps.crop", "kind": "step"},
    {"type": "graph_node", "id": "$steps.classification", "kind": "step"},
    {"type": "graph_node", "id": "$outputs.classes", "kind": "output"}
  ],
  "edges": [
    {"type": "graph_edge", "source": "$inputs.image", "target": "$steps.crop", "kind": "data"},
    {"type": "graph_edge", "source": "$inputs.image", "target": "$steps.detection", "kind": "data"},
    {"type": "graph_edge", "source": "$steps.classification", "target": "$outputs.classes", "kind": "data"},
    {"type": "graph_edge", "source": "$steps.crop", "target": "$steps.classification", "kind": "data"},
    {"type": "graph_edge", "source": "$steps.detection", "target": "$steps.crop", "kind": "data"}
  ],
  "steps": [
    {
      "type": "step_metadata",
      "node_id": "$steps.detection",
      "block_type": "roboflow_core/roboflow_object_detection_model@v3",
      "input_dimensionality": 1,
      "output_dimensionality": 1,
      "accepts_batch_input": true,
      "resources": {
        "type": "discovery",
        "items": [
          {
            "type": "dependent_resource",
            "resource_type": "roboflow_platform_model",
            "metadata": {
              "type": "roboflow_platform_model",
              "model_id": "my-project/3",
              "required_action": "execution",
              "execution_location": "environment_defined"
            }
          }
        ],
        "complete": true,
        "unknown_reasons": []
      },
      "restrictions": {
        "type": "discovery",
        "items": [],
        "complete": true,
        "unknown_reasons": []
      },
      "operations": {
        "type": "discovery",
        "items": ["model_inference"],
        "complete": true,
        "unknown_reasons": []
      }
    },
    {
      "type": "step_metadata",
      "node_id": "$steps.crop",
      "block_type": "roboflow_core/dynamic_crop@v1",
      "input_dimensionality": 1,
      "output_dimensionality": 2,
      "accepts_batch_input": true,
      "resources": {
        "type": "discovery",
        "items": [],
        "complete": true,
        "unknown_reasons": []
      },
      "restrictions": {
        "type": "discovery",
        "items": [],
        "complete": true,
        "unknown_reasons": []
      },
      "operations": {
        "type": "discovery",
        "items": ["image_crop"],
        "complete": true,
        "unknown_reasons": []
      }
    },
    {
      "type": "step_metadata",
      "node_id": "$steps.classification",
      "block_type": "roboflow_core/roboflow_classification_model@v2",
      "input_dimensionality": 2,
      "output_dimensionality": 2,
      "accepts_batch_input": true,
      "resources": {
        "type": "discovery",
        "items": [
          {
            "type": "dependent_resource",
            "resource_type": "roboflow_platform_model",
            "metadata": {
              "type": "roboflow_platform_model",
              "model_id": "my-other-project/1",
              "required_action": "execution",
              "execution_location": "environment_defined"
            }
          }
        ],
        "complete": true,
        "unknown_reasons": []
      },
      "restrictions": {
        "type": "discovery",
        "items": [],
        "complete": true,
        "unknown_reasons": []
      },
      "operations": {
        "type": "discovery",
        "items": ["model_inference"],
        "complete": true,
        "unknown_reasons": []
      }
    }
  ],
  "summary": {
    "type": "workflow_summary",
    "models": {
      "type": "discovery",
      "items": [
        {
          "type": "model_summary",
          "provider": "roboflow",
          "model_id": "my-other-project/1",
          "used_by_steps": ["$steps.classification"],
          "steps_by_dimensionality": {"2": 1},
          "metadata": null,
          "metadata_status": "disabled"
        },
        {
          "type": "model_summary",
          "provider": "roboflow",
          "model_id": "my-project/3",
          "used_by_steps": ["$steps.detection"],
          "steps_by_dimensionality": {"1": 1},
          "metadata": null,
          "metadata_status": "disabled"
        }
      ],
      "complete": true,
      "unknown_reasons": []
    },
    "max_dimensionality": 2,
    "steps_by_dimensionality": {"1": 2, "2": 1}
  }
}
```

What the response says:

* **Resources.** The detection step and the classification step each declare exactly one Roboflow model, with `required_action: "execution"` and `execution_location: "environment_defined"`: the model runs, and where it runs is left to the runtime that executes the definition. The crop step declares a complete, empty resource list: it is known to need no model, project or third-party model. That empty list is a known absence, not a missing or unknown declaration.
* **Restrictions.** All three steps declare complete, empty restriction lists. None of these blocks has a portable restriction, and the document says so explicitly rather than leaving the field out.
* **Inventory versus per-step declarations.** `summary.models.items` holds the same two model ids, deduplicated and mapped back to their steps. Because the metadata gate is off, the entries carry no enriched `metadata`; the per-step `resources` entries never carry it, whatever the gate.
* **Dimensionality.** The crop raises the dimension, so the classification step reads depth‑2 data: two steps at depth 1, one at depth 2, and the values sum to the three compiled steps. Per model, the detection model is referenced by one step at depth 1 (`{"1": 1}`) and the classification model by one step at depth 2 (`{"2": 1}`); each map sums to the length of its `used_by_steps`. The crop step references no model, so it is counted in the summary histogram only.

## Example: a step with populated restrictions

The Kafka producer sink is an enterprise block, so the server must be started with `LOAD_ENTERPRISE_BLOCKS=True` for this definition to compile. Introspection never connects to the broker: no Kafka cluster needs to exist at `localhost:9092`.

Request (`POST /workflows/describe_workload`):

```json
{
  "api_key": "<your key>",
  "specification": {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "message",
                "default_value": "metadata-only example"}],
    "steps": [
      {"type": "roboflow_enterprise/kafka_producer_sink@v1", "name": "producer",
       "bootstrap_servers": "localhost:9092", "topic": "events",
       "message": "$inputs.message", "fire_and_forget": true}
    ],
    "outputs": [{"type": "JsonField", "name": "error_status",
                 "selector": "$steps.producer.error_status"}]
  }
}
```

The complete response has the same envelope as the previous example (three nodes, two data edges, an empty complete model inventory, `max_dimensionality: 0` and `steps_by_dimensionality: {"0": 1}`). Only the single step entry is shown here. This is the complete `steps[0]` object extracted from that response, not a whole `WorkflowIntrospection` document:

```json
{
  "type": "step_metadata",
  "node_id": "$steps.producer",
  "block_type": "roboflow_enterprise/kafka_producer_sink@v1",
  "input_dimensionality": 0,
  "output_dimensionality": 0,
  "accepts_batch_input": false,
  "resources": {
    "type": "discovery",
    "items": [],
    "complete": true,
    "unknown_reasons": []
  },
  "restrictions": {
    "type": "discovery",
    "items": [
      {
        "type": "restriction",
        "code": "fire_and_forget_hides_persistence_failures",
        "severity": "soft",
        "when": {
          "type": "restriction_condition",
          "runtimes": ["inference_pipeline"],
          "step_execution_modes": null,
          "input_modes": null,
          "configuration_equals": {}
        }
      },
      {
        "type": "restriction",
        "code": "unavailable_on_hosted_platform",
        "severity": "hard",
        "when": {
          "type": "restriction_condition",
          "runtimes": ["hosted_serverless"],
          "step_execution_modes": null,
          "input_modes": null,
          "configuration_equals": {}
        }
      }
    ],
    "complete": true,
    "unknown_reasons": []
  },
  "operations": {
    "type": "discovery",
    "items": ["external_request"],
    "complete": true,
    "unknown_reasons": []
  }
}
```

What the step says:

* **A hard restriction on hosted serverless.** `unavailable_on_hosted_platform` applies when the target runtime is `hosted_serverless`; the other axes are `null`, so nothing else narrows it. The block does not publish from the hosted platform. The condition names the runtime; it is not the answering server's own `GCP_SERVERLESS` or `LAMBDA` flag.
* **A soft restriction on the inference pipeline.** `fire_and_forget_hides_persistence_failures` applies when the target runtime is `inference_pipeline`. The block still runs and returns the right output shape, but delivery failures are only logged, not returned.
* **Known-empty resources.** The producer needs no model, project or third-party model, and it declares that as a complete empty list.
* **How `fire_and_forget` shapes the declaration.** The block reads its own manifest value when declaring restrictions. A literal `true` (as here, and the default) yields both restrictions. A literal `false` yields only the hard hosted-platform restriction, because the run then waits for the broker's acknowledgement and the caveat does not exist. A selector such as `$inputs.wait_for_ack` cannot be resolved at compile time, so the block keeps the hard restriction it does know and returns `complete: false` with the `unresolved_selector` problem shown in [Discovery problems](#discovery-problems) instead of guessing.

## Errors

The routes reuse the existing error handling: a malformed definition, an unknown block, an invalid selector or an Execution Engine version outside `>=1.0.0,<2.0.0` returns the error the ordinary compilation path would return. A partially compiled or invented graph is never returned in place of an error.
