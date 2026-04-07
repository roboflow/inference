# Sub-workflow composition (design)

This document describes a planned **nested workflow** capability (`use_subworkflow`-style steps). It aligns
implementation with the Workflows Execution Engine **v1** compiler and executor; nothing here is a
user-facing product promise until the feature ships.

## Goals

- Reuse a workflow definition inside another workflow with **explicit input/output binding**.
- Keep the **per-workflow step graph** a **DAG** (as today): a sub-workflow is **one step** from the parent’s perspective.
- Enforce **composition** rules at **compile time** where the child definition is known: **no cycles** in the
  meta-graph of “workflow A references workflow B”, and a bounded **nesting depth**.
- Keep the **`use_subworkflow` block** a **declarative manifest** (selectors, ids, optional execution mode).
  **Blocks must not** depend on execution-engine internals (no `ExecutionEngine.init` inside the block).
- Allow **multiple execution backends** later: **local** (in-process), **remote synchronous**, **remote fire-and-forget**,
  selected by engine policy and/or `init_parameters`, not by ad hoc code in the block.

## Concepts

### Per-workflow execution graph (unchanged)

The existing compiler builds a directed acyclic graph of **steps** for a single workflow definition. Cycles there
remain invalid; see [Compilation of Workflow Definition](./workflows_compiler.md).

### Composition graph (new)

Separately, we model **which workflow references which** as another directed graph:

- **Node**: an opaque workflow identity (e.g. canonical hash of a normalized definition, or
  `(workspace_id, workflow_id, version_id)` when loaded from the platform).
- **Edge** `parent → child`: the parent’s definition **directly** contains a sub-workflow reference to `child`.

This graph may **not** contain a **cycle**. A cycle would mean A contains B contains … contains A, which is a
compile-time error when all members are statically resolved.

**Nesting depth** (from a **root** workflow under compilation): length of the longest path of containment edges
starting at that root. A configurable **`max_nesting_depth`** rejects overly deep trees.

### Selector depth (parent JSON)

To keep lineage and tooling simple, parent definitions should only reference **exposed outputs** of the
sub-workflow **step**, e.g. `$steps.my_sub.predictions`, not inner step outputs such as
`$steps.my_sub.inner_model.predictions`. Inner structure stays encapsulated in the child workflow.

## Compilation (execution engine)

A dedicated **compilation pass** (or phase integrated into `compile_workflow_graph`) should:

1. **Discover** sub-workflow steps in the parsed parent definition.
2. **Resolve** child definitions when possible (static id/version, inline spec). Dynamic-only references may
   restrict compile-time checks; policy TBD.
3. **Recursively compile** children using the **same** compiler entry points so rules stay centralized.
4. **Build containment edges** for the current compile tree and run:
   - `validate_subworkflow_composition` (see `inference.core.workflows.execution_engine.v1.subworkflow.composition`)
   - optional caps on total nested workflow count or total nested steps.
5. **Project** child outputs onto the parent step’s **declared** outputs so existing **kind / lineage**
   validation in `graph_constructor` can treat the sub-workflow step like any other block (or via a narrow
   adapter in the compiler, not in the block).

**Compilation cache** keys for the parent must incorporate **resolved child definitions** (or stable
version ids) when static, so cached graphs stay correct.

## Execution (execution engine)

Sub-workflow steps are **not** executed by block code that constructs an engine. Instead:

- The **executor** (or a small **step handler**) recognizes the sub-workflow step type.
- It invokes a **`SubworkflowRunner`** implementation provided through **`init_parameters`** (e.g.
  `workflows_core.subworkflow_runner`), with:
  - **`compiled_child`**: result of nested compilation (type `CompiledWorkflow`).
  - **`runtime_parameters`**: built from the same input assembly rules as other steps (mapped from parent lineage).
  - **`SubworkflowExecutionMode`**: `local` | `remote_sync` | `remote_async` (manifest preference + policy).
  - **`parent_context`**: optional tracing, ids, credentials scope (design only for now).

OSS ships a **stub** `LocalSubworkflowRunner` that raises `NotImplementedError` until wired to the internal
executor path. **Remote** modes remain interfaces only until implemented.

## Block surface (minimal)

The `use_subworkflow` block should expose only:

- Reference to the child (id/version and/or inline handle).
- **Input bindings**: same mechanism as other blocks (manifest fields with selectors to parent inputs/steps).
- Optional **execution mode** hint (validated against allowed set).
- **Output names** matching what the compiler projects from the child (or a fixed convention such as mirroring
  the child workflow’s declared outputs).

No imports from `execution_engine.v1.executor` or `compiler.core` inside the block module.

## Reference implementation stubs

| Location | Role |
| -------- | ---- |
| `inference/core/workflows/execution_engine/v1/subworkflow/composition.py` | Build composition graph, detect cycles, measure depth, `validate_subworkflow_composition`. |
| `inference/core/workflows/execution_engine/v1/subworkflow/runner.py` | `SubworkflowRunner`, `SubworkflowExecutionMode`, `LocalSubworkflowRunner` stub. |
| `inference/core/workflows/execution_engine/v1/subworkflow/errors.py` | `SubworkflowCompositionCycleError`, `SubworkflowNestingDepthError`. |
| `tests/workflows/unit_tests/execution_engine/subworkflow/test_composition.py` | Unit tests for composition validation. |

## See also

- [Compilation of Workflow Definition](./workflows_compiler.md)
- [Workflow definitions syntax](./definitions.md)
