# Inner workflows (nested definitions)

The **Inner workflow** block (`roboflow_core/inner_workflow@v1`) lets you embed one workflow definition inside another. At **compile time**, the engine resolves any saved-workflow references, validates **composition** (nesting limits and cycles), validates **parameter bindings**, then **inlines** the child’s steps into the parent. After compilation there is no separate “nested run”: the graph is the same as if you had written those steps at the parent level.

This page describes the block, the compile-time pipeline, limits, and a minimal Python example. For general compilation stages, see [Compilation of Workflow Definition](./workflows_compiler.md).

!!! Note

    This feature is implemented in Execution Engine **v1**. The inner workflow block’s `run()` method is never used at runtime; the step is removed during compilation.

## Inner workflow block

Each inner workflow step is a JSON object in the parent’s `steps` list with `type: "roboflow_core/inner_workflow@v1"`.

### How you supply the child definition

You must provide **either**:

- **`workflow_definition`**: a full nested workflow JSON object (same shape as a root workflow: `version`, `inputs`, `steps`, `outputs`), **or**
- **`workflow_workspace_id`** and **`workflow_id`**, with optional **`workflow_version_id`**, to load a saved workflow spec at compile time.

You must **not** set both an inline `workflow_definition` and the reference fields on the same step.

### `parameter_bindings`

`parameter_bindings` is an object whose **keys** are the **names** of the **child workflow’s** entries in its `inputs` array. Each **value** is a **selector** (or value the engine can coerce) from the **parent** scope, typically:

- `$inputs.<parent_input_name>` for parent workflow inputs, or  
- `$steps.<parent_step_name>.<output_property>` for data produced by earlier parent steps.

**Rules:**

- Every child input that **requires** a value from the parent must appear in `parameter_bindings`, **except** inputs of type `WorkflowParameter` / `InferenceParameter` that declare a **non-null** `default_value` in the child definition. Those may be omitted; the child’s default is applied when the definition is inlined.
- Keys that are not child input names are rejected at compile time.
- Child steps should consume parent data through **`$inputs.<child_input_name>`** in the nested definition; the compiler replaces those references with the bound parent selectors (or injected defaults) during inlining.

### Referencing child outputs from the parent

The nested workflow’s `outputs` array defines **JsonField** entries with a `name` and `selector`. After compilation, the parent treats the inner step like a logical block with outputs named after those JsonField **`name`** values.

From the parent you reference them as:

```text
$steps.<inner_step_name>.<child_output_name>
```

where `<child_output_name>` is the `name` field of a JsonField in the child’s `outputs`, not necessarily the last step’s name.

## Compile-time pipeline (Execution Engine v1)

When `compile_workflow_graph` runs, inner workflows go through the following **before** the main “parse workflow definition” step:

1. **Reference resolution (normalization)**  
   Any step that uses `workflow_workspace_id` / `workflow_id` (and optional `workflow_version_id`) is resolved to an inline `workflow_definition`. This happens recursively inside nested definitions.  
   - Default resolver uses the Roboflow API and **`workflows_core.api_key`** in workflow init parameters (unless the workspace is `"local"` or you supply a custom resolver).  
   - Override with init parameter **`workflows_core.inner_workflow_spec_resolver`**: a callable `(workspace_id, workflow_id, workflow_version_id, init_parameters) -> dict` returning the child workflow JSON.

2. **Composition validation**  
   The engine builds a **composition graph**: one edge per `inner_workflow` step from the parent workflow’s fingerprint to the child definition’s fingerprint. It then checks:  
   - the graph is **acyclic** (no A → B → … → A),  
   - **nesting depth** from the root is within **`WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH`**,  
   - the **total number** of inner-workflow steps (edges) is within **`WORKFLOWS_MAX_INNER_WORKFLOW_COUNT`**.  
   See [Limits and environment variables](#limits-and-environment-variables) below.

3. **Inlining**  
   Each `inner_workflow` step is expanded into ordinary steps: child step names become **`{inner_step_name}__{child_step_name}`** (with collision handling), selectors are rewritten (`$inputs` / `$steps` in the child, and parent references to `$steps.<inner_step_name>…`), then the inner step is removed. The rest of compilation (parse, workflow specification validation, execution graph construction, step initialization) sees only a flat workflow.

4. **Parsing and validation**  
   The flattened JSON is parsed with block manifests, `validate_workflow_specification` runs, and the execution graph is built like any other workflow.

## Example (Python)

The following pattern matches the `examples/workflows/inner_workflows/main.py` example in this repository: resolve a saved workflow by id, bind the parent image into the child’s expected input name, then consume a child output from a downstream parent step.

```python
import json
import os

from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.models.utils import ROBOFLOW_MODEL_TYPES

WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/inner_workflow@v1",
            "name": "inner",
            "workflow_workspace_id": "your-workspace",
            "workflow_id": "your-workflow-id",
            "workflow_version_id": "optional-version-id",
            "parameter_bindings": {
                "image": "$inputs.image",
            },
        },
        {
            "type": "roboflow_core/roboflow_classification_model@v2",
            "name": "classification",
            "images": "$steps.inner.dynamic_crop_output",
            "model_id": "resnet50",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.classification.predictions",
        },
    ],
}

if __name__ == "__main__":
    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    model_manager = ModelManager(model_registry=model_registry)

    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_DEFINITION,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": os.getenv("ROBOFLOW_API_KEY"),
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
    )

    result = execution_engine.run(
        runtime_parameters={
            "image": {
                "type": "file",
                "value": "/path/to/your/image.jpg",
            },
        },
    )

    print(json.dumps(result, indent=2, default=str))
```

Replace workspace, workflow, version, model, output selectors, and image path with values that match your saved workflow and parent graph. The key requirement is that **`parameter_bindings`** keys match the **child** workflow’s `inputs[].name` fields.

## Limits and environment variables

| Variable | Default | Meaning |
|----------|---------|--------|
| `WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH` | `4` | Maximum **depth** of the composition graph from the root workflow: each direct `inner_workflow` child counts as one level along a path. |
| `WORKFLOWS_MAX_INNER_WORKFLOW_COUNT` | `32` | Maximum **number** of `inner_workflow` steps across the whole nested definition (each inner step is one edge in the composition graph). |

**Cycles:** The composition graph must be a **DAG**. You cannot have a cycle of nested references (for example, workflow A embedding B embedding A), even if the per-step execution graph of each workflow is acyclic.

Violations raise compile-time errors (`InnerWorkflowNestingDepthError`, `InnerWorkflowTotalCountError`, `InnerWorkflowCompositionCycleError`, etc.) with messages describing depth, count, or cycle involvement.

## Related reading

- [Workflows definitions](./definitions.md) — JSON shape, inputs, steps, outputs  
- [Compiler](./workflows_compiler.md) — overall compilation stages  
- [Workflow execution](./workflow_execution.md) — runtime behavior after compilation  
