# Workload introspection

Workload introspection answers one question about a Workflow definition: **what work does this definition describe?** It compiles the definition structurally — no block is initialised, no model is loaded or registered, no custom Python is evaluated, nothing is executed — and returns the compiled graph, the per-step declarations the blocks make about themselves, and an inventory of the models the definition refers to.

The answer is *portable*: it describes the definition, not the server that answered. The same definition returns the same description on a hosted deployment, on a Jetson and in an air-gapped container. Consumers (a cost estimator, a scheduler, a capacity planner) apply their own hardware assumptions to it.

## Endpoints

| Endpoint | Body | Purpose |
| --- | --- | --- |
| `POST /workflows/describe_workload` | `{"api_key": "...", "specification": {...}}` | describe an inline definition |
| `POST /{workspace_name}/workflows/{workflow_id}/describe_workload` | `{"api_key": "...", "use_cache": true, "workflow_version_id": null}` | describe a saved definition |

Both accept the api key in the request body or as an `Authorization: Bearer <key>` header, exactly like `/workflows/describe_interface`; a request with neither is rejected with HTTP 400 and the same error envelope. Both are removed together with the other Workflow endpoints when `DISABLE_WORKFLOW_ENDPOINTS=True`. The saved-definition route forwards `use_cache` and `workflow_version_id` to the ordinary saved-definition lookup.

The Python entry point is `roboflow_workflows.execution_engine.introspection.workload.describe_workflow_workload(definition, init_parameters=None, execution_engine_version=None, model_metadata_provider=None)`. It lives in the standalone `roboflow-workflows` package and needs no inference server.

## Response

The response is a `WorkflowIntrospection` document. Every object in it carries a defaulted `type` discriminator, at every nesting level, so another service can parse it without Python callables; the full JSON Schema is published in the server's `/openapi.json` under `components.schemas.WorkflowIntrospection`, and in Python as `WorkflowIntrospection.model_json_schema()` from `roboflow_workflows.execution_engine.introspection.workload_entities`.

| Field | Meaning |
| --- | --- |
| `schema_version` | `"1"`. Bumped only for an incompatible change of this document. |
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
| `resources` | `Discovery[DependentResource]` — the Roboflow models, Roboflow projects and third-party models the step needs. |
| `restrictions` | `Discovery[RestrictionMetadata]` — portable, conditional restrictions. |
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

Every declaration is wrapped in a `Discovery`:

```json
{"type": "discovery", "items": [...], "complete": true, "unknown_reasons": []}
```

* `complete: true` with an empty `items` means **known absence** — the block declares that it does none of this.
* `complete: false` means the list may be incomplete; `unknown_reasons` then carries at least one stable reason code with step context, for example `step_resources_unknown:$steps.counter` or `custom_python_internal_operations_unknown:$steps.counter`.
* A complete result never carries reasons, and reasons never carry exception traces or secrets.

A block that declares nothing (a third-party plugin that has not been annotated) is reported as unknown, never as empty. Every block registered in this repository — core, enterprise and the server's own plugin — declares all three hooks explicitly; the one deliberate exception is `roboflow_core/inner_workflow@v1`, whose dispatched child may pull anything.

### Restrictions are conditional, not evaluated

A `RestrictionMetadata` carries a `code`, a `severity` and a `when` condition; it is **not** filtered against the answering server's configuration. The condition ANDs across its axes (`runtimes`, `step_execution_modes`, `input_modes`, `configuration_equals`) and ORs within each populated axis; an axis left as `null` is unrestricted. A custom Python step, for example, always reports

```json
{"code": "custom_python_execution_disabled", "severity": "hard",
 "when": {"configuration_equals": {"ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": false,
                                   "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "local"}}}
```

whether or not this server allows custom Python. The consumer decides what the restriction means for its own target configuration. The legacy `get_restrictions()` API, which does filter on the local environment and carries human-readable notes, is unchanged and unrelated.

### Model inventory

`summary.models` is an inventory of model **references**, deduplicated by `(provider, model_id)`, with the referring step ids preserved:

```json
{"provider": "roboflow", "model_id": "my-project/3",
 "used_by_steps": ["$steps.crop_detection", "$steps.detection"],
 "metadata": null, "metadata_status": "disabled"}
```

* Two steps using one model id produce **one** entry with two `used_by_steps`, not two entries.
* Roboflow platform references use the provider `roboflow`; an explicit third-party provider is preserved as declared. Roboflow *projects* are not model entries.
* A model id supplied through a selector (`$inputs.model`) is never invented into an id and never sent to a metadata lookup: it stays a per-step unresolved reference and makes the inventory incomplete with a reason.
* Being listed is not a claim that the reference executes at runtime — a step may only need access to the model.

`metadata_status` is `disabled`, `available` or `unavailable`:

| `USE_INFERENCE_MODELS` | Result |
| --- | --- |
| `False` | every entry is `disabled` with `metadata: null`, and **zero** metadata lookups happen. |
| `True` | the server performs one metadata-only lookup per unique model id (not per step) and maps `model_type`, `model_variant`, `task_type`. A failed or empty lookup yields `unavailable` without dropping the entry, and without making an otherwise complete inventory incomplete. |

A standalone package call with no metadata provider yields `unavailable` — the package makes no claim about any server flag.

#### How the lookup reaches the platform, and what that costs you

The server performs the lookup through the existing registry helper `get_model_metadata_from_inference_models_registry()` (`GET /models/v1/external/stat`). That call is **metadata only**: no weights are downloaded, no model is registered with the model manager, and the model-type resolution paths that would enforce this deployment's model support are deliberately not used. Results are cached under a credential-scoped prefix that is separate from the prefix the execution paths use, so an introspection call can neither populate nor read a cache entry that model loading relies on.

Two consequences of reusing that helper, which is shared with the loading paths and is not modified by this feature:

* **On hosted serverless the request carries the credits-verification header.** When `ENFORCE_CREDITS_VERIFICATION` is on, the helper adds `ENFORCE_CREDITS_VERIFICATION_HEADER` to every call, including this one. A workspace without credits therefore gets a platform error, which surfaces as `metadata_status: "unavailable"` for that entry — the inventory itself is unaffected. On the platform side an introspection lookup is indistinguishable from a model-load lookup.
* **Turning the metadata gate off also turns tensor mode off.** The server defines `ENABLE_TENSOR_DATA_REPRESENTATION` as the environment value **and** `USE_INFERENCE_MODELS`, so a deployment (or a test) that sets `USE_INFERENCE_MODELS=False` to avoid the lookup silently runs in numpy mode as well. This does not change the introspection response — it is identical in both representations — but it is worth knowing before reading a flag matrix.

### Dimensionality summary

`steps_by_dimensionality` counts every compiled step once, by its **input** depth (the reference-lineage depth described above); keys are strings on the wire, values are positive, dimensions with no steps are omitted, and the values sum to the number of steps. `max_dimensionality` is the deepest compiled input or output depth in the graph, including the input-to-output depth of a workflow with no steps at all.

## What this is not

This is compile-time introspection, not an estimator and not a profiler. The document deliberately contains **no**:

* cost score, latency estimate, weight or any other number derived from hardware assumptions;
* lineage identifiers, input bindings or origin paths;
* compute/IO classification of a step;
* execution groups, worker counts or a schedule;
* selected custom-Python backend, selected model backend or any other statement about how *this* server would run the definition.

Answering those is the job of the service that consumes this document.

## Example

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

Response (abridged; `type` discriminators and unchanged fields omitted for readability):

```json
{
  "type": "workflow_introspection",
  "schema_version": "1",
  "execution_engine_version": "1.15.2",
  "nodes": [{"id": "$inputs.image", "kind": "input"},
            {"id": "$steps.detection", "kind": "step"},
            {"id": "$steps.crop", "kind": "step"},
            {"id": "$steps.classification", "kind": "step"},
            {"id": "$outputs.classes", "kind": "output"}],
  "edges": [{"source": "$inputs.image", "target": "$steps.crop", "kind": "data"},
            {"source": "$inputs.image", "target": "$steps.detection", "kind": "data"},
            {"source": "$steps.classification", "target": "$outputs.classes", "kind": "data"},
            {"source": "$steps.crop", "target": "$steps.classification", "kind": "data"},
            {"source": "$steps.detection", "target": "$steps.crop", "kind": "data"}],
  "steps": [
    {"node_id": "$steps.detection", "block_type": "roboflow_core/roboflow_object_detection_model@v3",
     "input_dimensionality": 1, "output_dimensionality": 1, "accepts_batch_input": true,
     "operations": {"items": ["model_inference"], "complete": true, "unknown_reasons": []}},
    {"node_id": "$steps.crop", "block_type": "roboflow_core/dynamic_crop@v1",
     "input_dimensionality": 1, "output_dimensionality": 2, "accepts_batch_input": true,
     "operations": {"items": ["image_crop"], "complete": true, "unknown_reasons": []}},
    {"node_id": "$steps.classification", "block_type": "roboflow_core/roboflow_classification_model@v2",
     "input_dimensionality": 2, "output_dimensionality": 2, "accepts_batch_input": true,
     "operations": {"items": ["model_inference"], "complete": true, "unknown_reasons": []}}
  ],
  "summary": {
    "models": {"items": [
        {"provider": "roboflow", "model_id": "my-other-project/1",
         "used_by_steps": ["$steps.classification"], "metadata": null, "metadata_status": "disabled"},
        {"provider": "roboflow", "model_id": "my-project/3",
         "used_by_steps": ["$steps.detection"], "metadata": null, "metadata_status": "disabled"}],
      "complete": true, "unknown_reasons": []},
    "max_dimensionality": 2,
    "steps_by_dimensionality": {"1": 2, "2": 1}
  }
}
```

The crop raises the dimension, so the classification step reads depth‑2 data: two steps at depth 1, one at depth 2, and the values sum to the three compiled steps.

## Errors

The routes reuse the existing error handling: a malformed definition, an unknown block, an invalid selector or an Execution Engine version outside `>=1.0.0,<2.0.0` returns the error the ordinary compilation path would return. A partially compiled or invented graph is never returned in place of an error.
