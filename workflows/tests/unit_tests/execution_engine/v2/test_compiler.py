"""Tests of the V2 compiler: structure, selectors, kinds, lineage, gates,
cycles, unsupported features and configuration validation. Every negative
case asserts that no block ``run()`` was invoked."""

import re
from typing import Any, Dict, List, Mapping

import pytest
from roboflow_workflows.execution_engine.v2 import (
    BlockContract,
    CompiledWorkflow,
    ContractError,
    InputSpec,
    OutputSpec,
    Registry,
    WorkflowCompileError,
    compile_workflow,
)

RUNS: List[str] = []


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _Block:
    def __init__(self, config: Mapping[str, Any]):
        if config.get("reject"):
            raise ValueError("configuration rejected on purpose")
        self._config = config

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        RUNS.append(repr(kwargs))
        return {}


def _registry() -> Registry:
    registry = Registry()
    registry.register_kind("number", _is_number)
    registry.register_kind("boolean", lambda v: isinstance(v, bool))
    registry.register_kind("text", lambda v: isinstance(v, str))
    registry.register_block(
        "leaf",
        contract=BlockContract(
            reference="value",
            inputs={"value": InputSpec("number")},
            outputs={"out": OutputSpec("number")},
        ),
        factory=_Block,
    )
    registry.register_block(
        "expand",
        contract=BlockContract(
            reference="value",
            inputs={"value": InputSpec("number")},
            outputs={
                "children": OutputSpec(
                    "number", transform="append", axis="kids", stationary=True
                ),
                "count": OutputSpec("number"),
            },
        ),
        factory=_Block,
    )
    registry.register_block(
        "reduce",
        contract=BlockContract(
            reference="children",
            inputs={
                "parent": InputSpec("number"),
                "children": InputSpec("number", view="batch"),
            },
            outputs={"total": OutputSpec("number", transform="collapse")},
        ),
        factory=_Block,
    )
    registry.register_block(
        "pair",
        contract=BlockContract(
            reference="left",
            inputs={
                "left": InputSpec("number", view="batch"),
                "right": InputSpec("number", view="batch"),
            },
            outputs={"total": OutputSpec("number", transform="collapse")},
        ),
        factory=_Block,
    )
    registry.register_block(
        "gate",
        contract=BlockContract(
            reference="value",
            inputs={"value": InputSpec("number")},
            outputs={"keep": OutputSpec("boolean")},
        ),
        factory=_Block,
    )
    registry.register_block(
        "to_text",
        contract=BlockContract(
            reference="value",
            inputs={"value": InputSpec("number")},
            outputs={"text": OutputSpec("text")},
        ),
        factory=_Block,
    )
    registry.register_block(
        "mutating",
        contract=BlockContract(
            reference="value",
            inputs={"value": InputSpec("number")},
            outputs={"out": OutputSpec("number")},
            mutates_inputs=("value",),
        ),
        factory=_Block,
    )
    return registry


def _samples_input(name: str = "values") -> Dict[str, Any]:
    return {
        "name": name,
        "kind": "number",
        "axes": [{"id": "samples", "kind": "sample"}],
    }


def _definition(steps, outputs=None, inputs=None) -> Dict[str, Any]:
    return {
        "version": "2.0",
        "inputs": inputs if inputs is not None else [_samples_input()],
        "steps": steps,
        "outputs": outputs if outputs is not None else [],
    }


@pytest.fixture(autouse=True)
def _reset_runs():
    RUNS.clear()
    yield
    assert RUNS == [], "compilation must never invoke block.run()"


def test_compile_orders_steps_and_derives_per_output_layouts() -> None:
    definition = _definition(
        steps=[
            {
                "name": "reduce",
                "type": "reduce",
                "inputs": {
                    "parent": "$inputs.values",
                    "children": "$steps.expand.children",
                },
                "when": "$steps.gate.keep",
            },
            {"name": "gate", "type": "gate", "inputs": {"value": "$inputs.values"}},
            {"name": "expand", "type": "expand", "inputs": {"value": "$inputs.values"}},
            {
                "name": "leaf",
                "type": "leaf",
                "inputs": {"value": "$steps.expand.children"},
            },
        ],
        outputs=[
            {"name": "total", "selector": "$steps.reduce.total"},
            {"name": "leaf", "selector": "$steps.leaf.out"},
        ],
    )

    plan = compile_workflow(definition, registry=_registry())

    assert isinstance(plan, CompiledWorkflow)
    order = [step.name for step in plan.steps]
    assert order.index("expand") < order.index("reduce")
    assert order.index("gate") < order.index("reduce")
    assert order.index("expand") < order.index("leaf")
    ports = plan.ports
    assert ports["$inputs.values"].layout.axis_ids == ("samples",)
    assert ports["$steps.expand.children"].layout.axis_ids == ("samples", "expand/kids")
    assert ports["$steps.expand.children"].layout.axes[1].kind == "static_nesting"
    assert ports["$steps.expand.children"].layout.axes[1].stationary is True
    assert ports["$steps.expand.count"].layout.axis_ids == ("samples",)
    assert ports["$steps.reduce.total"].layout.axis_ids == ("samples",)
    assert ports["$steps.leaf.out"].layout.axis_ids == ("samples", "expand/kids")
    reduce_step = next(step for step in plan.steps if step.name == "reduce")
    assert [axis.id for axis in reduce_step.prefix] == ["samples"]
    assert reduce_step.gate == "$steps.gate.keep" and reduce_step.gate_depth == 1
    assert set(reduce_step.dependencies) == {"expand", "gate"}
    described = plan.describe()
    assert described["steps"]["reduce"]["outputs"]["total"]["axes"] == ["samples"]
    assert described["outputs"] == {
        "total": "$steps.reduce.total",
        "leaf": "$steps.leaf.out",
    }


def test_registry_snapshot_shields_plan_from_later_edits() -> None:
    registry = _registry()
    definition = _definition(
        steps=[{"name": "leaf", "type": "leaf", "inputs": {"value": "$inputs.values"}}],
        outputs=[{"name": "out", "selector": "$steps.leaf.out"}],
    )

    plan = compile_workflow(definition, registry=registry)
    registry.register_block(
        "later",
        contract=BlockContract(
            reference="value",
            inputs={"value": InputSpec("number")},
            outputs={"out": OutputSpec("number")},
        ),
        factory=_Block,
    )

    assert "later" not in plan.registry.block_names
    assert plan.registry.is_frozen
    with pytest.raises(ContractError):
        plan.registry.register_kind("x")


def test_shared_input_axis_id_allows_item_pairing_and_distinct_ids_do_not() -> None:
    registry = _registry()
    registry.register_block(
        "combine",
        contract=BlockContract(
            reference="left",
            inputs={"left": InputSpec("number"), "right": InputSpec("number")},
            outputs={"out": OutputSpec("number")},
        ),
        factory=_Block,
    )
    shared = _definition(
        inputs=[_samples_input("a"), _samples_input("b")],
        steps=[
            {
                "name": "c",
                "type": "combine",
                "inputs": {"left": "$inputs.a", "right": "$inputs.b"},
            }
        ],
    )
    distinct = _definition(
        inputs=[
            _samples_input("a"),
            {
                "name": "b",
                "kind": "number",
                "axes": [{"id": "others", "kind": "sample"}],
            },
        ],
        steps=[
            {
                "name": "c",
                "type": "combine",
                "inputs": {"left": "$inputs.a", "right": "$inputs.b"},
            }
        ],
    )

    compile_workflow(shared, registry=registry)
    with pytest.raises(WorkflowCompileError) as info:
        compile_workflow(distinct, registry=registry)

    assert "$steps.c inputs do not share one invocation prefix" in str(info.value)
    assert "('samples',)" in str(info.value) and "('others',)" in str(info.value)


@pytest.mark.parametrize(
    "definition, expected",
    [
        (
            _definition(
                steps=[
                    {"name": "a", "type": "leaf", "inputs": {"value": "$steps.b.out"}},
                    {"name": "b", "type": "leaf", "inputs": {"value": "$steps.a.out"}},
                ]
            ),
            "dependency cycle: $steps.a -> $steps.b -> $steps.a",
        ),
        (
            _definition(
                steps=[
                    {"name": "g", "type": "gate", "inputs": {"value": "$steps.a.out"}},
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.values"},
                        "when": "$steps.g.keep",
                    },
                ]
            ),
            "dependency cycle: $steps.g -> $steps.a -> $steps.g",
        ),
        (
            _definition(
                steps=[
                    {"name": "a", "type": "leaf", "inputs": {"value": "$steps.a.out"}}
                ]
            ),
            "dependency cycle: $steps.a -> $steps.a",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.missing"},
                    }
                ]
            ),
            "references unknown selector '$inputs.missing'",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$steps.nope.out"},
                    }
                ]
            ),
            "references unknown step 'nope'",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.values"},
                    },
                    {"name": "b", "type": "leaf", "inputs": {"value": "$steps.a.nope"}},
                ]
            ),
            "references unknown selector '$steps.a.nope'",
        ),
        (
            _definition(
                steps=[{"name": "a", "type": "leaf", "inputs": {"value": "values"}}]
            ),
            "has invalid selector 'values'",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "a",
                        "type": "unknown",
                        "inputs": {"value": "$inputs.values"},
                    }
                ]
            ),
            "uses unknown block type 'unknown'",
        ),
        (
            _definition(
                inputs=[{"name": "values", "kind": "mystery"}],
                steps=[
                    {"name": "a", "type": "leaf", "inputs": {"value": "$inputs.values"}}
                ],
            ),
            "declares unknown kind 'mystery'",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "t",
                        "type": "to_text",
                        "inputs": {"value": "$inputs.values"},
                    },
                    {"name": "a", "type": "leaf", "inputs": {"value": "$steps.t.text"}},
                ]
            ),
            "expects kind 'number' but '$steps.t.text' produces kind 'text'",
        ),
        (
            _definition(
                inputs=[{"name": "value", "kind": "number"}],
                steps=[
                    {
                        "name": "r",
                        "type": "reduce",
                        "inputs": {
                            "parent": "$inputs.value",
                            "children": "$inputs.value",
                        },
                    }
                ],
            ),
            "declares a batch view but '$inputs.value' is ungrouped",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "e",
                        "type": "expand",
                        "inputs": {"value": "$inputs.values"},
                    },
                    {
                        "name": "r",
                        "type": "reduce",
                        "inputs": {
                            "parent": "$steps.e.children",
                            "children": "$steps.e.children",
                        },
                    },
                ]
            ),
            "inputs do not share one invocation prefix",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "e1",
                        "type": "expand",
                        "inputs": {"value": "$inputs.values"},
                    },
                    {
                        "name": "e2",
                        "type": "expand",
                        "inputs": {"value": "$inputs.values"},
                    },
                    {
                        "name": "p",
                        "type": "pair",
                        "inputs": {
                            "left": "$steps.e1.children",
                            "right": "$steps.e2.children",
                        },
                    },
                ]
            ),
            "batched inputs must share their full lineage",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "e",
                        "type": "expand",
                        "inputs": {"value": "$inputs.values"},
                    },
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.values"},
                        "when": "$steps.e.count",
                    },
                ]
            ),
            "must reference an output of kind 'boolean'",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "e",
                        "type": "expand",
                        "inputs": {"value": "$inputs.values"},
                    },
                    {
                        "name": "g",
                        "type": "gate",
                        "inputs": {"value": "$steps.e.children"},
                    },
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.values"},
                        "when": "$steps.g.keep",
                    },
                ]
            ),
            "not the invocation prefix ('samples',) or an ancestor prefix of it",
        ),
        (
            _definition(
                inputs=[_samples_input(), {"name": "flag", "kind": "boolean"}],
                steps=[
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.values"},
                        "when": "$inputs.flag",
                    }
                ],
            ),
            "must reference a step output",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.values"},
                        "when": "$steps.zzz.keep",
                    }
                ]
            ),
            "references unknown step 'zzz'",
        ),
        (
            _definition(steps=[{"name": "a", "type": "leaf", "inputs": {}}]),
            "Missing ports: ['value']",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {
                            "value": "$inputs.values",
                            "extra": "$inputs.values",
                        },
                    }
                ]
            ),
            "unknown ports: ['extra']",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "a",
                        "type": "mutating",
                        "inputs": {"value": "$inputs.values"},
                    }
                ]
            ),
            "Mutating block implementations are not supported",
        ),
        (
            _definition(
                inputs=[
                    {
                        "name": "frames",
                        "kind": "number",
                        "axes": [{"id": "t", "kind": "time"}],
                    }
                ],
                steps=[
                    {"name": "a", "type": "leaf", "inputs": {"value": "$inputs.frames"}}
                ],
            ),
            "Temporal execution is not supported",
        ),
        (
            _definition(
                inputs=[
                    {
                        "name": "v",
                        "kind": "number",
                        "axes": [
                            {"id": "n", "kind": "dynamic_nesting", "stationary": True}
                        ],
                    }
                ],
                steps=[{"name": "a", "type": "leaf", "inputs": {"value": "$inputs.v"}}],
            ),
            "cannot be stationary",
        ),
        (
            _definition(
                inputs=[
                    {
                        "name": "v",
                        "kind": "number",
                        "axes": [
                            {"id": "n", "kind": "dynamic_nesting"},
                            {"id": "s", "kind": "sample"},
                        ],
                    }
                ],
                steps=[{"name": "a", "type": "leaf", "inputs": {"value": "$inputs.v"}}],
            ),
            "a sample axis must be the first axis",
        ),
        (
            _definition(
                inputs=[
                    _samples_input("a"),
                    {
                        "name": "b",
                        "kind": "number",
                        "axes": [
                            {"id": "samples", "kind": "sample", "stationary": True}
                        ],
                    },
                ],
                steps=[],
            ),
            "Axis 'samples' is declared with different properties",
        ),
        (
            _definition(
                steps=[
                    {"name": "a", "type": "leaf", "inputs": {"value": "$inputs.values"}}
                ],
                outputs=[{"name": "o", "selector": "$steps.a.nope"}],
            ),
            "outputs[0].selector references unknown selector '$steps.a.nope'",
        ),
        (
            _definition(
                steps=[
                    {"name": "a", "type": "leaf", "inputs": {"value": "$inputs.values"}}
                ],
                outputs=[
                    {"name": "o", "selector": "$steps.a.out"},
                    {"name": "o", "selector": "$steps.a.out"},
                ],
            ),
            "Duplicate workflow output name 'o'",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.values"},
                    },
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.values"},
                    },
                ]
            ),
            "Duplicate step name 'a'",
        ),
        (
            _definition(inputs=[_samples_input(), _samples_input()], steps=[]),
            "Duplicate workflow input name 'values'",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "bad name",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.values"},
                    }
                ]
            ),
            "steps[0].name must be an identifier",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.values"},
                        "config": {"reject": True},
                    }
                ]
            ),
            "rejected its configuration {'reject': True}: configuration rejected on purpose",
        ),
        (
            _definition(
                steps=[
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.values"},
                        "extra": 1,
                    }
                ]
            ),
            "Unsupported keys ['extra'] in steps[0]",
        ),
        (
            {**_definition(steps=[]), "parameters": []},
            "Unsupported keys ['parameters'] in definition",
        ),
        (
            {"version": "2.0", "inputs": [], "steps": []},
            "missing required sections: ['outputs']",
        ),
        (
            {**_definition(steps=[]), "version": "1.0"},
            "`version` must be '2.0' for the V2 engine, got '1.0'",
        ),
        (
            {**_definition(steps=[]), "steps": {}},
            "section `steps` must be a list",
        ),
    ],
)
def test_compile_errors_are_useful_and_precede_execution(definition, expected) -> None:
    with pytest.raises(WorkflowCompileError, match=re.escape(expected)):
        compile_workflow(definition, registry=_registry())


def _leaf_step(**overrides: Any) -> Dict[str, Any]:
    step = {"name": "a", "type": "leaf", "inputs": {"value": "$inputs.values"}}
    step.update(overrides)
    return step


def _whole_string_cases() -> List[Any]:
    gate_step = {"name": "g", "type": "gate", "inputs": {"value": "$inputs.values"}}
    return [
        pytest.param(
            _definition(
                inputs=[{**_samples_input(), "name": "values\n"}],
                steps=[],
            ),
            "inputs[0].name must be an identifier",
            "'values\\n'",
            id="input-name-newline",
        ),
        pytest.param(
            _definition(
                steps=[_leaf_step()],
                outputs=[{"name": "out\n", "selector": "$steps.a.out"}],
            ),
            "outputs[0].name must be an identifier",
            "'out\\n'",
            id="output-name-newline",
        ),
        pytest.param(
            _definition(
                inputs=[
                    {
                        "name": "values",
                        "kind": "number",
                        "axes": [{"id": "samples\n", "kind": "sample"}],
                    }
                ],
                steps=[],
            ),
            "$inputs.values.axes[0].id must be an identifier",
            "'samples\\n'",
            id="axis-id-newline",
        ),
        pytest.param(
            _definition(steps=[_leaf_step(name="a\n")]),
            "steps[0].name must be an identifier",
            "'a\\n'",
            id="step-name-newline",
        ),
        pytest.param(
            _definition(steps=[_leaf_step(inputs={"value": "$inputs.values\n"})]),
            "$steps.a.inputs['value'] has invalid selector",
            "'$inputs.values\\n'",
            id="input-selector-newline",
        ),
        pytest.param(
            _definition(
                steps=[_leaf_step()],
                outputs=[{"name": "out", "selector": "$steps.a.out\n"}],
            ),
            "outputs[0].selector has invalid selector",
            "'$steps.a.out\\n'",
            id="step-selector-newline",
        ),
        pytest.param(
            _definition(steps=[gate_step, _leaf_step(when="$steps.g.keep\n")]),
            "$steps.a.when has invalid selector",
            "'$steps.g.keep\\n'",
            id="when-selector-newline",
        ),
        pytest.param(
            _definition(steps=[_leaf_step(inputs={"value": "$inputs.values junk"})]),
            "has invalid selector",
            "'$inputs.values junk'",
            id="input-selector-trailing-junk",
        ),
        pytest.param(
            _definition(
                steps=[_leaf_step()],
                outputs=[{"name": "out", "selector": "$steps.a.out.extra"}],
            ),
            "outputs[0].selector has invalid selector",
            "'$steps.a.out.extra'",
            id="step-selector-trailing-segment",
        ),
        pytest.param(
            _definition(steps=[_leaf_step(name="a;b")]),
            "steps[0].name must be an identifier",
            "'a;b'",
            id="step-name-trailing-junk",
        ),
        pytest.param(
            _definition(
                steps=[
                    {
                        "name": "e",
                        "type": "newline_axis",
                        "inputs": {"value": "$inputs.values"},
                    }
                ]
            ),
            "declares an append transform without a valid `axis` key",
            "'kids\\n'",
            id="append-axis-key-newline",
        ),
    ]


def _registry_with_newline_axis() -> Registry:
    registry = _registry()
    registry.register_block(
        "newline_axis",
        contract=BlockContract(
            reference="value",
            inputs={"value": InputSpec("number")},
            outputs={"kids": OutputSpec("number", transform="append", axis="kids\n")},
        ),
        factory=_Block,
    )
    return registry


@pytest.mark.parametrize("definition, expected, shown", _whole_string_cases())
def test_names_and_selectors_must_match_as_whole_strings(
    definition, expected, shown
) -> None:
    with pytest.raises(WorkflowCompileError) as info:
        compile_workflow(definition, registry=_registry_with_newline_axis())

    assert expected in str(info.value)
    assert shown in str(info.value), "the offending value is shown verbatim"


def test_valid_identifiers_and_selectors_still_compile() -> None:
    definition = _definition(
        inputs=[
            {
                "name": "_Values9",
                "kind": "number",
                "axes": [{"id": "samples_A1", "kind": "sample"}],
            }
        ],
        steps=[
            {"name": "g_1", "type": "gate", "inputs": {"value": "$inputs._Values9"}},
            {
                "name": "Step_2",
                "type": "expand",
                "inputs": {"value": "$inputs._Values9"},
                "when": "$steps.g_1.keep",
            },
        ],
        outputs=[{"name": "out_3", "selector": "$steps.Step_2.children"}],
    )

    plan = compile_workflow(definition, registry=_registry())

    assert plan.ports["$steps.Step_2.children"].layout.axis_ids == (
        "samples_A1",
        "Step_2/kids",
    )


def test_factory_configuration_error_keeps_cause_and_object_without_run_is_rejected() -> (
    None
):
    registry = _registry()
    registry.register_block(
        "runless",
        contract=BlockContract(
            reference="value",
            inputs={"value": InputSpec("number")},
            outputs={"out": OutputSpec("number")},
        ),
        factory=lambda config: object(),
    )

    with pytest.raises(WorkflowCompileError) as info:
        compile_workflow(
            _definition(
                steps=[
                    {
                        "name": "a",
                        "type": "leaf",
                        "inputs": {"value": "$inputs.values"},
                        "config": {"reject": True},
                    }
                ]
            ),
            registry=registry,
        )
    assert isinstance(info.value.__cause__, ValueError)

    with pytest.raises(
        WorkflowCompileError, match=re.escape("without a callable `run` method")
    ):
        compile_workflow(
            _definition(
                steps=[
                    {
                        "name": "a",
                        "type": "runless",
                        "inputs": {"value": "$inputs.values"},
                    }
                ]
            ),
            registry=registry,
        )


def test_compiled_config_is_detached_and_deeply_read_only() -> None:
    definition = _definition(
        steps=[
            {
                "name": "a",
                "type": "leaf",
                "inputs": {"value": "$inputs.values"},
                "config": {"k": [1], "nested": {"items": [{"x": 1}], "tags": {"t"}}},
            }
        ],
        outputs=[{"name": "out", "selector": "$steps.a.out"}],
    )

    plan = compile_workflow(definition, registry=_registry())
    definition["steps"][0]["config"]["k"].append(2)
    definition["steps"][0]["config"]["nested"]["items"][0]["x"] = 99

    step = plan.steps[0]
    assert step.config["k"] == (1,)
    assert step.config["nested"]["items"][0]["x"] == 1
    assert step.config["nested"]["tags"] == frozenset({"t"})
    with pytest.raises(TypeError):
        step.config["k"] = 3
    with pytest.raises(TypeError):
        step.config["nested"]["items"][0]["x"] = 5
    with pytest.raises(AttributeError):
        step.config["k"].append(2)
    materialized = step.materialize_config()
    assert materialized == {"k": [1], "nested": {"items": [{"x": 1}], "tags": {"t"}}}
    assert type(materialized["k"]) is list, "factories get the definition's types"
    materialized["nested"]["items"].append({"x": 2})
    assert step.materialize_config()["nested"]["items"] == [{"x": 1}]
    assert len(step.config["nested"]["items"]) == 1


def test_compile_time_factory_validation_cannot_mutate_the_plan() -> None:
    registry = _registry()

    class _Greedy:
        def __init__(self, config: Dict[str, Any]):
            config["regions"].append("added at validation")
            config["new_key"] = True

        def run(self, value: Any) -> Dict[str, Any]:
            RUNS.append("greedy")
            return {"out": value}

    registry.register_block(
        "greedy",
        contract=BlockContract(
            reference="value",
            inputs={"value": InputSpec("number")},
            outputs={"out": OutputSpec("number")},
        ),
        factory=_Greedy,
    )

    plan = compile_workflow(
        _definition(
            steps=[
                {
                    "name": "g",
                    "type": "greedy",
                    "inputs": {"value": "$inputs.values"},
                    "config": {"regions": []},
                }
            ]
        ),
        registry=registry,
    )

    assert dict(plan.steps[0].config) == {"regions": ()}
    assert plan.steps[0].materialize_config() == {"regions": []}
