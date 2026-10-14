"""Query definitions compile with the workflow, not per invocation.

The engine reruns a step's model validators for every logical invocation with
a selected value. The query blocks compile their definitions in field
validators of literal-only fields instead, so a warm run compiles nothing:

    compile_workflow   one compilation per definition (Params validation)
    first run          one more per block instance (its run cache)
    later runs         none

Selected templates still get their placeholder check per invocation.
"""
import roboflow_workflows.v2.blocks.formatters.expression.v1 as _v2_family_0

import csv
import io
from collections import Counter
from typing import Any, Dict

import pytest
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.control.continue_if import v1 as conditions
from roboflow_workflows.v2.blocks.formatters.csv_formatter import v1 as csv_block
from roboflow_workflows.v2.blocks.formatters._shared import properties as properties
from roboflow_workflows.v2.blocks.formatters.string_template import v1 as text
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)

# The query compilers as each block module calls them.
COMPILERS = [
    (conditions, "compile_condition"),
    (properties, "compile_condition"),
    (properties, "compile_operations"),
    (text, "compile_operations"),
    (csv_block, "compile_operations"),
]
UNDECLARED_RENAME = [{"type": "DetectionsRename", "class_map": "names"}]


def _score_at_least(minimum: int) -> Dict[str, Any]:
    condition = {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {"type": "DynamicOperand", "operand_name": "score"},
                "comparator": {"type": "(Number) >="},
                "right_operand": {"type": "StaticOperand", "value": minimum},
            }
        ],
    }

    return condition


@pytest.fixture
def compilations(monkeypatch) -> Counter:
    counts: Counter = Counter()
    for module, name in COMPILERS:
        original = getattr(module, name)

        def spy(*args, _original=original, _key=f"{module.__name__}.{name}", **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)

    return counts


def _workflow(**step_overrides: Dict[str, Any]) -> Dict[str, Any]:
    steps = [
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "double",
            "data": "$inputs.x",
            "operations": [{"type": "Multiply", "other": 2}],
        },
        {
            "type": "roboflow_core/expression@v1",
            "name": "score",
            "data": {"score": "$steps.double.output", "unused": 1},
            "data_operations": {"score": [{"type": "Multiply", "other": 3}]},
            "switch": {
                "type": "CasesDefinition",
                "cases": [
                    {
                        "type": "CaseDefinition",
                        "condition": _score_at_least(10),
                        "result": {
                            "type": "DynamicCaseResult",
                            "parameter_name": "score",
                            "operations": [{"type": "Divide", "other": 2}],
                        },
                    }
                ],
                "default": {"type": "StaticCaseResult", "value": -1},
            },
        },
        {
            "type": "roboflow_core/continue_if@v1",
            "name": "gate",
            "condition_statement": _score_at_least(0),
            "evaluation_parameters": {"score": "$steps.score.output"},
            "next_steps": ["$steps.text"],
            "stop_delay": "$inputs.delay",
        },
        {
            "type": "roboflow_core/string_template@v1",
            "name": "text",
            "template": "$inputs.template",
            "data": {"value": "$steps.score.output", "unit": "px"},
            "data_operations": {"value": [{"type": "Multiply", "other": 10}]},
        },
        {
            "type": "roboflow_core/csv_formatter@v1",
            "name": "csv",
            "columns_data": {"text": "$steps.text.output", "camera": "gate"},
            "columns_operations": {"text": [{"type": "StringToUpperCase"}]},
        },
    ]
    for step in steps:
        step.update(step_overrides.get(step["name"], {}))
    workflow = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "x"},
            {"type": "WorkflowParameter", "name": "template"},
            {"type": "WorkflowParameter", "name": "delay"},
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": "text", "selector": "$steps.text.output"},
            {"type": "JsonField", "name": "csv", "selector": "$steps.csv.csv_content"},
        ],
    }

    return workflow


def _csv_column(document: str, column: str) -> list:
    column_values = [row[column] for row in csv.DictReader(io.StringIO(document))]

    return column_values


def test_warm_runs_compile_no_query_definition(compilations: Counter) -> None:
    session = compile_workflow(
        _workflow(), catalogue=create_catalogue()
    ).create_session()
    definition = Counter(compilations)
    runs = []
    try:
        # Run 2 switches the gate to a nonzero stop_delay; run 3 repeats it.
        for x, delay in [([1, 2, 3, 4], 0), ([5, 6, 7, 8], 5), ([5, 6, 7, 8], 5)]:
            before = Counter(compilations)
            result = session.run(
                {"x": x, "template": "{value:.0f}{unit}", "delay": delay}
            )
            runs.append((result.rows(), compilations - before))
    finally:
        session.close()

    (first, first_counts), (second, second_counts), (third, third_counts) = runs
    # Gate 1, property 1, expression 3 (operations, condition, result
    # operations), template 1, CSV 1.
    assert sum(definition.values()) == 7
    # The first run fills each block's run cache once, not once per row.
    assert first_counts == definition
    assert second_counts == Counter() and third_counts == Counter()
    assert [row["text"] for row in first] == [None, "60px", "90px", "120px"]
    assert _csv_column(first[-1]["csv"], "text") == ["60PX", "90PX", "120PX"]
    assert [row["text"] for row in second] == ["150px", "180px", "210px", "240px"]
    assert [row["text"] for row in third] == [row["text"] for row in second]


def test_selected_template_is_still_checked_per_invocation() -> None:
    session = compile_workflow(
        _workflow(), catalogue=create_catalogue()
    ).create_session()
    try:
        with pytest.raises(StepExecutionError, match=r"uses \['missing'\]"):
            session.run({"x": [5], "template": "{missing}", "delay": 0})
        with pytest.raises(StepExecutionError, match="format string|format specs"):
            session.run({"x": [5], "template": "{unit:.2f}", "delay": 0})
    finally:
        session.close()


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"gate": {"evaluation_parameters": {"other": "$steps.score.output"}}},
            "operand `score` is not declared",
        ),
        ({"double": {"operations": UNDECLARED_RENAME}}, "`names` is not declared"),
        (
            {"score": {"data_operations": {"missing": []}}},
            "`missing` is not a data variable",
        ),
        (
            {"text": {"data_operations": {"value": UNDECLARED_RENAME}}},
            "`names` is not declared",
        ),
        (
            {"csv": {"columns_operations": {"camera": UNDECLARED_RENAME}}},
            "`names` is not declared",
        ),
        (
            {"csv": {"columns_operations": {"missing": []}}},
            "`missing`, which is not a columns_data column",
        ),
    ],
)
def test_invalid_definitions_still_fail_workflow_compilation(
    overrides: Dict[str, Any], message: str
) -> None:
    with pytest.raises(ParamsValidationError, match=message):
        compile_workflow(_workflow(**overrides), catalogue=create_catalogue())


def test_direct_calls_compile_once_per_definition_and_names(
    compilations: Counter,
) -> None:
    block = _v2_family_0.ExpressionBlock()
    switch = properties.CasesDefinition.model_validate(
        _workflow()["steps"][1]["switch"]
    )
    for _ in range(3):
        output = block.run(data={"score": 4}, data_operations={}, switch=switch)

    # One compilation is a condition plus its result operations.
    assert output == {"output": -1}
    assert sum(compilations.values()) == 2

    # Other variable names are another definition: compiled again.
    block.run(data={"score": 4, "extra": 0}, data_operations={}, switch=switch)
    assert sum(compilations.values()) == 4
