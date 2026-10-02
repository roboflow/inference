# V1 reference cases

This directory runs 45 named workflow cases on the real V1 engine and prints
what happened as JSON. It records rows, errors and every block call. A V2
demo reuses the same definitions and inputs and compares its own results
against this output.

## Run

From the repository root, using the `roboflow-inference-new` environment:

```bash
python -B development/workflows-2.0/02-sequential-parity/reference/run_v1_reference.py > /tmp/v1-reference.json
```

Expect exit code 0 and `"pin_failures": []` in `summary`. Add
`--case <id>` (repeatable) to run a subset. Add `--list` to print the case
catalogue without running it. No models, network or credentials are needed.

## What the runner does

```
run_v1_reference.py  (its own Python process)
  ├─ sys.path: checkout workflows/ and this directory first; checkout inference_models/ and root last
  ├─ WORKFLOWS_PLUGINS=v1_reference_fixtures   # ordinary V1 plugin discovery
  ├─ font downloads off, all socket connections rejected
  └─ for each case, for each session:
       engine = ExecutionEngine.init(definition, max_concurrent_steps=1)
       wrap every compiled block's run() -> record args/result, call original
       for each run in the session: engine.run(inputs) -> rows or error
```

Nothing in V1 is replaced. Core blocks (ContinueIf, SwitchCase,
DimensionCollapse, inner workflow) and the compiler/executor run unchanged.
[`v1_reference_fixtures`](v1_reference_fixtures/__init__.py) holds 15 tiny
model-free plugin blocks. The standalone V1 core catalogue loads without
`onnxruntime`, so no host model catalogue is excluded or stubbed.

Runs that share a `session` number share one compiled engine and its block
instances. A new session compiles a new engine.

## Output

| Field | Meaning |
| --- | --- |
| `cases[].case_id`, `title`, `acceptance`, `origins` | Stable identity, feature, acceptance rows P01–P20, earlier investigation it preserves |
| `cases[].workflow`, `runs[].inputs` | V1 definition and runtime inputs |
| `cases[].comparison` | How V2 should relate to this V1 result (labels below) |
| `sessions[].runs[].rows` / `error` | Workflow output rows, or `{phase, type, message, block_id}` |
| `sessions[].runs[].call_counts` | Live calls per step in this run; `pinned_call_counts` and `pins_ok` check them |
| `invocations[]` | Every block call: `session`, `run`, `step`, `instance`, `arguments`, `result` |
| `sessions[].compiled_steps` | V1 inlined step names, block type, data/execution/control depths |
| `resolver_calls`, `step_error_hook_calls`, `resources`, `resource_bindings` | Saved-workflow resolver calls, caller error hook calls, injected resource objects and which step received which |

Compare invocations per `(session, run, step)`. Calls of one step keep a
stable order. V1 does not define an order between independent sibling steps,
and it changes between processes.

Encoding: a `Batch` is `{"batch": [...], "indices": [[...]]}`. A future is
`{"future": {"done_when_observed": bool, "result": ...}}`; block results never
resolve it. `result_resolved` gives those results after the run, and
`arguments_after_call` appears when a block changed its arguments in place;
both match what V2's observer reports. `inputs_after_run` shows the caller's
input mapping after the run: V1 writes coerced values and defaults back into it
(label `D012-INPUT-PREPARATION`, not a requirement), except for the block's own
mutation in `binding.in_place_mutation`.

## Comparison labels

| `comparison.kind` | Label | Meaning for V2 |
| --- | --- | --- |
| `match` | — | Reproduce values, errors and per-step call counts |
| `v1_defect` | `OBS-V1-01` | V1 ignores an empty gate mask beside a nonempty one. V2 corrects this: zero target calls |
| `v1_defect` | `OBS-V1-02` | V1 gates only the first inlined child step. V2 corrects this: the whole child is gated |
| `v1_defect` | `OBS-V1-03` | Parent gate over deeper existing data gives zero calls in V1. V2 corrects this (decision 007): children of admitted parents run |
| `v2_boundary` | `D012-SELECTOR-ONLY`, `D012-READY-BOUNDARY` | V2 boundary semantics decided differently (decision 012); see `comparison.intended_v2` |
| `v1_quirk` | `V1-QUIRK-EMPTY-EXPANSION` | V1 skips reducers after a genuinely empty expansion. Disclosed difference |

A run may carry its own `comparison`, which overrides the case's: only run 1 of
`outputs.future_results` is `D012-READY-BOUNDARY`. Pins record what V1 does,
including defects. Never use a pin as a V2 expectation when the kind is not
`match`; `../parity.py` holds the V2 expectations.

## Cases

| Case | Feature | Label |
| --- | --- | --- |
| `binding.literal_default_and_selectors` | One parameter bound as literal, default, input selector and step selector | |
| `binding.compound_list_and_dict` | List and dict parameters mixing batch selectors, parameters and literals | |
| `binding.default_versus_explicit_null` | Omitted parameter takes its default; explicit null stays null | |
| `binding.parameter_into_selector_only_field` | Workflow parameter bound to a selector-only batch field is rejected | D012-SELECTOR-ONLY |
| `binding.in_place_mutation` | A block mutates its input in place; the dependent reader sees the change | |
| `validation.step_dependency_cycle` | Steps depending on each other are rejected before any call | |
| `batching.scalar_batch_auto_cast` | Scalar step runs once; a scalar fed to a batch parameter becomes `Batch([x])` | |
| `batching.runtime_input_broadcast` | A scalar or singleton batch input broadcasts to the other batch length | |
| `batching.runtime_input_length_mismatch` | Batch inputs of incompatible or empty outer length are rejected | |
| `batching.mixed_scalar_or_batch` | A scalar-or-batch parameter receives a `Batch` or a plain scalar | |
| `batching.input_free_step` | A literal-only step runs once and feeds every batch element | |
| `control.batch_gate_literal_sink` | Per-index gate drives a literal-only, output-free sink | |
| `control.ungated_literal_sink` | Same outputs without the gate, but the sink runs once | |
| `control.two_gates_intersect` | Two same-level gates on one target admit the intersection | |
| `control.two_gates_single_item_empty_mask` | One of two gates rejects the only item, yet the target runs | OBS-V1-01 |
| `control.two_gates_empty_mask_beside_nonempty` | Gates > 1 and > 100 over `[0, 2, 3, 4]` | OBS-V1-01 |
| `control.parent_and_child_gates_literal_sink` | Parent and child gates jointly select child invocations of a literal sink | |
| `control.shallow_gate_over_deeper_data` | Parent-level gate over already expanded child data | OBS-V1-03 |
| `control.deep_gate_over_shallow_data_rejected` | A child-level gate on parent-level data is a compile error | |
| `control.switch_case_branch_recovery` | SwitchCase routes; an empty-accepting merge recovers the surviving branch | |
| `control.gate_propagates_downstream` | Filtered indices propagate to descendants; ungated steps are unaffected | |
| `control.scalar_gate_and_wildcard_output` | A scalar gate switches its branch; `$steps.x.*` selects all outputs | |
| `control.fan_out_to_two_targets` | One gate governs a sink and a data step | |
| `lineage.expand_scale_reduce` | Expand, per-child step, parent+children join and DimensionCollapse | |
| `lineage.filter_expand_filter_reduce` | Filter parents, ragged expand, filter children, reduce with sparse indices | |
| `lineage.filtered_children_reach_parent_join` | Partially filtered children reach the parent join with original indices | |
| `lineage.all_children_filtered_reducer_runs` | A group whose children are all filtered still reaches the reducers | |
| `lineage.genuine_empty_expansion_reducer_skipped` | Genuinely empty expansions: V1 never calls the reducers | V1-QUIRK-EMPTY-EXPANSION |
| `nested.bound_child_on_expanded_children` | Child bound to expanded data runs per child with its default parameter | |
| `nested.child_step_error_context` | A failing child step reports its inlined identity to the caller hook | |
| `nested.missing_required_binding` | Child input without binding or default is a compile error | |
| `nested.missing_child_output` | Selecting an undeclared child output is a compile error | |
| `nested.repeated_embedded_child` | The same embedded child used twice gets two independent step sets | |
| `nested.repeated_saved_reference` | Two uses of one saved child: resolver called once, both uses execute | |
| `nested.saved_reference_cycle` | A saved child that references itself is a compile error | |
| `nested.gate_on_child_workflow` | Child-level gate on a nested workflow with two independent roots | OBS-V1-02 |
| `nested.scalar_gate_on_two_root_child` | A false scalar gate on a child still lets its second root run | OBS-V1-02 |
| `nested.grouped_reused_child_with_inner_gate` | Reused child with an internal gate keeps the parent's depth-2 grouping | |
| `nested.child_within_child` | Depth-two composition bound to an upstream parent step, leaf default kept | |
| `lifecycle.step_state_across_runs_and_sessions` | One instance per step keeps state across runs; a new engine starts fresh | |
| `lifecycle.plugin_initializer_resource_per_step` | Without an explicit resource, the plugin initializer creates one per step | |
| `lifecycle.explicit_resource_shared_by_steps` | An explicit namespaced resource is passed by identity to every step | |
| `outputs.future_results` | Future-valued outputs are resolved for consumers and, by default, outputs | D012-READY-BOUNDARY (run 1) |
| `dynamic.stateful_inline_python_block` | Inline Python block with init state, compiled from the definition | |
| `dynamic.child_workflow_inline_python_block` | A child workflow's own dynamic block runs inside the parent | |

## Use from a V2 demo

Put `02-sequential-parity/` on `sys.path`. Importing `reference` never
imports V1; `collect_v1_observations` starts the runner as a child process.

```python
from reference import REFERENCE_CASES, collect_v1_observations, get_case

case = get_case("control.two_gates_intersect")
case.workflow, [run.inputs for run in case.runs], case.comparison.kind

v1 = collect_v1_observations(["control.two_gates_intersect"])
v1["cases"][0]["sessions"][0]["runs"][0]["call_counts"]
```

Files: [catalogue.py](catalogue.py) (cases, runs, pins, labels),
[definitions.py](definitions.py) (workflow JSON),
[run_v1_reference.py](run_v1_reference.py) (V1 process),
[v1_observations.py](v1_observations.py) (child-process API).
