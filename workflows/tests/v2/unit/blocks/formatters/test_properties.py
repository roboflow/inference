"""``v2/property_definition@v1`` and ``v2/expression@v1`` on plain values."""

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.formatters import (
    BLOCKS,
    ExpressionBlock,
    PropertyDefinitionBlock,
)
from roboflow_workflows.v2.blocks.formatters.properties import SWITCH_EXAMPLE
from roboflow_workflows.v2.blocks.query import QueryRuntimeError

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)

TOP_CONFIDENCE = [
    {"type": "ClassificationPropertyExtract", "property_name": "top_class_confidence"}
]
TOP_CLASS = [{"type": "ClassificationPropertyExtract", "property_name": "top_class"}]


def _classification() -> ClassificationPrediction:
    prediction = ClassificationPrediction(
        class_id=torch.tensor([1]),
        confidence=torch.tensor([[0.1, 0.7, 0.2]]),
        images_metadata=[{"class_names": {0: "person", 1: "car", 2: "dog"}}],
    )

    return prediction


def _case(condition_value, result: dict) -> dict:
    case = {
        "type": "CaseDefinition",
        "condition": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {"type": "DynamicOperand", "operand_name": "label"},
                    "comparator": {"type": "=="},
                    "right_operand": {
                        "type": "StaticOperand",
                        "value": condition_value,
                    },
                }
            ],
        },
        "result": result,
    }

    return case


def _switch(*cases: dict, default: dict) -> dict:
    return {"type": "CasesDefinition", "cases": list(cases), "default": default}


def _static(value) -> dict:
    return {"type": "StaticCaseResult", "value": value}


def _run_expression(data: dict, switch: dict, data_operations=None):
    params = ExpressionBlock.Params(
        data={name: f"$inputs.{name}" for name in data},
        data_operations=data_operations or {},
        switch=switch,
    )
    result = ExpressionBlock().run(
        data=data, data_operations=params.data_operations, switch=params.switch
    )

    return result["output"]


def test_the_package_lists_both_formatters() -> None:
    assert PropertyDefinitionBlock in BLOCKS
    assert ExpressionBlock in BLOCKS
    assert PropertyDefinitionBlock.type == "v2/property_definition@v1"
    assert ExpressionBlock.type == "v2/expression@v1"


def test_property_definition_extracts_the_top_class_confidence() -> None:
    params = PropertyDefinitionBlock.Params(
        data="$steps.c.predictions", operations=TOP_CONFIDENCE
    )

    result = PropertyDefinitionBlock().run(
        data=_classification(), operations=params.operations
    )

    assert result["output"] == pytest.approx(0.7)


def test_property_definition_rejects_runtime_names_at_compile_time() -> None:
    operations = [
        {
            "type": "DetectionsFilter",
            "filter_operation": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operations": [
                                {
                                    "type": "ExtractDetectionProperty",
                                    "property_name": "class_name",
                                }
                            ],
                        },
                        "comparator": {"type": "in (Sequence)"},
                        "right_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "classes",
                        },
                    }
                ],
            },
        }
    ]

    with pytest.raises(ValidationError, match="operand `classes` is not declared"):
        PropertyDefinitionBlock.Params(
            data="$steps.m.predictions", operations=operations
        )


def test_property_definition_rejects_an_unknown_operation() -> None:
    with pytest.raises(ValidationError):
        PropertyDefinitionBlock.Params(
            data="$steps.m.predictions", operations=[{"type": "NoSuchOperation"}]
        )


def test_a_multi_label_top_confidence_is_a_list_and_cannot_compare_as_a_number() -> (
    None
):
    prediction = MultiLabelClassificationPrediction(
        class_ids=torch.tensor([0, 2]),
        confidence=torch.tensor([0.9, 0.1, 0.8]),
        image_metadata={"class_names": {0: "person", 1: "car", 2: "dog"}},
    )
    params = PropertyDefinitionBlock.Params(
        data="$steps.c.predictions", operations=TOP_CONFIDENCE
    )
    extracted = PropertyDefinitionBlock().run(
        data=prediction, operations=params.operations
    )
    switch = _switch(default=_static("x"))
    switch["cases"] = [
        {
            "type": "CaseDefinition",
            "condition": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "top",
                        },
                        "comparator": {"type": "(Number) >="},
                        "right_operand": {"type": "StaticOperand", "value": 0.5},
                    }
                ],
            },
            "result": _static("confident"),
        }
    ]

    assert isinstance(extracted["output"], list)
    with pytest.raises(QueryRuntimeError, match="cannot compare"):
        _run_expression({"top": extracted["output"]}, switch)


def test_expression_returns_the_first_matching_case() -> None:
    switch = _switch(
        _case("car", _static("vehicle")),
        _case("car", _static("never reached")),
        default=_static("other"),
    )

    assert _run_expression({"label": "car"}, switch) == "vehicle"
    assert _run_expression({"label": "person"}, switch) == "other"


def test_expression_maps_a_classification_to_a_label_through_data_operations() -> None:
    output = _run_expression(
        {"label": _classification()},
        _switch(_case("car", _static("vehicle")), default=_static("other")),
        data_operations={"label": TOP_CLASS},
    )

    assert output == "vehicle"


def test_a_dynamic_result_applies_its_operations_with_the_variables() -> None:
    switch = _switch(
        default={
            "type": "DynamicCaseResult",
            "parameter_name": "prediction",
            "operations": TOP_CONFIDENCE,
        }
    )
    switch["cases"] = []

    output = _run_expression({"prediction": _classification(), "label": "x"}, switch)

    assert output == pytest.approx(0.7)


def test_the_documented_switch_example_maps_vehicles() -> None:
    assert _run_expression({"top_class": "car"}, SWITCH_EXAMPLE) == "vehicle"
    assert _run_expression({"top_class": "person"}, SWITCH_EXAMPLE) == "other"


@pytest.mark.parametrize(
    "fields, message",
    [
        (
            {"switch": _switch(_case("car", _static("v")), default=_static("o"))},
            "operand `label` is not declared",
        ),
        (
            {
                "switch": _switch(
                    default={"type": "DynamicCaseResult", "parameter_name": "missing"}
                )
            },
            "`missing` is not a data variable",
        ),
        (
            {
                "data_operations": {"missing": TOP_CLASS},
                "switch": _switch(default=_static("o")),
            },
            "`missing` is not a data variable",
        ),
    ],
)
def test_expression_rejects_unknown_names_at_compile_time(fields, message) -> None:
    with pytest.raises(ValidationError, match=message):
        ExpressionBlock.Params(data={"prediction": "$steps.c.predictions"}, **fields)
