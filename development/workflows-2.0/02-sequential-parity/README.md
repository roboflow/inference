# Sequential parity: V1 next to V2

One command runs 45 workflow cases on the real V1 engine and on the V2
engine and compares them. It also runs 20 V2-only examples of features
beyond those cases.

```bash
conda activate roboflow-inference-new
python development/workflows-2.0/02-sequential-parity/run_demo.py \
    --output-dir /tmp/workflows-2.0-parity
```

Expect `verdicts: {'parity': 37, 'difference_confirmed': 8}`, every capability
check `[PASS]`, and exit code 0. No models, network or credentials are needed.

| Option | Effect |
| --- | --- |
| `--list` | Print every parity case (with its label) and capability example |
| `--case <name>` | Run only this case or example; repeatable |
| `--scenario parity` / `capabilities` | Run one group |
| `--output-dir <dir>` | Where JSON artifacts go |

## How a parity case runs

```
reference/ (child process)            this process
V1 ExecutionEngine.init + run         translation.py: V1 JSON -> V2 JSON
  every block call recorded             compile_workflow -> create_session -> run -> rows
  exact V1 call-count pins checked      every call recorded by an ExecutionObserver
                 \                      /
                  comparison.py: rows, calls per step with arguments and indices,
                  error category, resolver/error-hook calls, resources, instances
```

V2 runs ordinary class-owned blocks from [blocks.py](blocks.py): the V1
fixtures plus V2 equivalents of ContinueIf (`fixture/threshold_gate@v1`,
returning `Select`/`Stop`), SwitchCase and DimensionCollapse. Nested cases use
real nested composition and a saved-workflow resolver. Nothing calls V1 from V2.

## Verdicts and labels

| Verdict | Meaning |
| --- | --- |
| `PARITY` | V2 reproduces V1's rows, call counts, call arguments/results, errors and state |
| `DIFFERENCE <label>` | A decided difference. V2 is checked against its own expected values in [parity.py](parity.py); V1 values are shown, never counted as parity |
| `FAIL` | Anything else. The command exits 1 |

| Label | V1 | V2 |
| --- | --- | --- |
| `OBS-V1-01` | An empty gate mask is ignored beside a nonempty one | Gates combine by conjunction: zero calls |
| `OBS-V1-02` | A gate on a nested workflow governs only its first step | The whole nested workflow is gated |
| `OBS-V1-03` | A parent gate over deeper data admits nothing | Children of admitted parents run |
| `V1-QUIRK-EMPTY-EXPANSION` | Reducers skip genuinely empty groups | Reducers receive them |
| `D012-SELECTOR-ONLY` | A parameter bound to a selector-only field fails | Accepted and validated by kind |
| `D012-READY-BOUNDARY` | `resolve_output_futures=False` returns lazy futures | Outputs are always ready |
| `D012-INPUT-PREPARATION` | Input preparation rewrites the caller's input dict | The caller's dict is untouched; the count of affected runs is printed |
| `D021` (capability checks) | Child literals are not decoded; child inputs are not kind-checked | Decoded once per run; checked at the child input |
| Whole-child gate (capability checks) | Forwarded child values bypass a gate on the child | Forwarded values are masked like the child's steps |

The comparison normalizes only: `child__echo` naming of V2 path
`("child", "echo")`, number types (`1 == 1.0`), error classes through a
category map, order between different steps, and the V1 translated core
blocks' arguments. [comparison.py](comparison.py) lists each rule.

## Capability examples

Each reads definitions from [workflows/](workflows/). Change them, or the
inputs in [capabilities.py](capabilities.py), [boundary_examples.py](boundary_examples.py),
[dynamic_examples.py](dynamic_examples.py) or [plugin_examples.py](plugin_examples.py),
and rerun with `--scenario capabilities --case <name>`. Each report lists the
definitions it used, actual and expected values, and error details.

| Example | What to observe |
| --- | --- |
| `selector_crop_mosaic` | Crop regions and tile size come from inputs; one session, two runs |
| `vote_and_csv` | A list of batch selectors gives one call with `list[Batch]`; CSV literal columns stay plain |
| `compound_group_cast` | A scalar beside child groups becomes a one-element group at `(parent, 0)` |
| `per_output_layouts` | One block returns a parent total and per-child values |
| `configured_output_names` | Outputs `2026` and `class-name` named by a parameter; `$steps.x.*` |
| `generated_roots` | Input-free sources; rows equal V1's measured rows |
| `vectorized_calls` | One call over ragged children of several parents, as in V1 |
| `kind_codecs` | `"21.5C"` deserialized at the input, `"70.7F"` serialized at the output |
| `mutation_policy` | In-place mutation: warning by default, error in strict mode |
| `caller_inputs` | The caller's input dict is not rewritten |
| `resource_precedence` | Scoped caller value, unscoped caller value, catalogue provider, default |
| `nested_composition` | Saved-reference diamond fetched once, depth/count limits, duplicate dynamic blocks |
| `inspection_without_execution` | Describing a plan never runs submitted code; a session needs `allow_local_code` |
| `validated_parameters` | Literal and selected values meet the same `Params` rules: shared `Field` bounds, field and model validators; a selected dict keeps its identity; errors name field and index |
| `nested_inputs` | Child literals/defaults decoded once per run, selections passed by identity, wrong payloads rejected at the child input |
| `execution_context` | `self.execution_context` in the constructor and in `run`; nothing stays active after a failure |
| `dynamic_shared_state` | Inline Python `globals` shared by direct and nested steps of a session; a supplied mapping spans sessions |
| `dynamic_representation` | A `tensor_native` block refused by the default policy; `CpuArrayPolicy` feeds it NumPy arrays |
| `plugin_catalogue` | [example_plugin.py](example_plugin.py) loaded as a plugin: `Factory(scope="session")`, per-output conversion, workload through nested inputs |
| `nested_forwarding_gates` | A gate on a nested workflow also masks the inputs, default and literal it forwards, and the consumer declared before the gate; groups `[[1,2],[],[3]]` show a denied versus an admitted empty group. Edit the masks in [gate_examples.py](gate_examples.py) |

## Artifacts

| Path under `--output-dir` | Content |
| --- | --- |
| `parity/<case>/v1.json`, `v2.json` | Each engine's rows, errors and every call with arguments and results |
| `parity/<case>/v2_definition.json` | The V2 definition that ran |
| `parity/<case>/comparison.json` | Every check and the verdict |
| `parity/summary.json` | Verdict counts and V1 pin status |
| `capabilities/<example>.json` | Checks with actual and expected values |

The V1 side alone is described in [reference/README.md](reference/README.md).
