"""UQL schema: full literal coverage, defaults, static errors and imports."""

import ast
import json
from pathlib import Path
from typing import Dict

import pytest
from pydantic import BaseModel, ValidationError
from roboflow_workflows.v2.blocks import query
from roboflow_workflows.v2.blocks.query import (
    COMPARATOR_TYPES,
    SUPPORTED_OPERATION_TYPES,
    UNARY_OPERATOR_TYPES,
    QueryDefinitionError,
    StatementGroup,
    compile_condition,
    compile_operations,
)

# Literal ledger from ecosystem-e4-prerequisites/query-coverage.json.
V1_OPERATION_LITERALS = {
    "StringToLowerCase", "StringToUpperCase", "LookupTable", "ToNumber",
    "NumberRound", "SequenceMap", "NumericSequenceAggregate", "SequenceAggregate",
    "ToString", "ToBoolean", "StringSubSequence", "DetectionsPropertyExtract",
    "DetectionsToDictionary", "ClassificationPropertyExtract", "DetectionsSelection",
    "ExtractDetectionProperty", "DetectionsFilter", "SortDetections",
    "DetectionsOffset", "DetectionsShift", "RandomNumber", "ExtractImageProperty",
    "ExtractFrameMetadata", "TimestampToISOFormat", "ConvertImageToJPEG",
    "ConvertDictionaryToJSON", "ConvertImageToBase64", "StringMatches",
    "SequenceLength", "SequenceElementsCount", "SequenceJoin", "SequenceApply",
    "Multiply", "Divide", "DetectionsRename", "PickDetectionsByParentClass",
}  # fmt: skip
V1_COMPARATOR_LITERALS = {
    "(Number) ==", "==", "(Number) !=", "!=", "(Number) >", "(Number) >=",
    "(Number) <", "(Number) <=", "(Number) in range", "(String) startsWith",
    "(String) endsWith", "(String) contains", "in (Sequence)", "all in (Sequence)",
    "any in (Sequence)", "(Detection) in zone",
}  # fmt: skip
V1_UNARY_LITERALS = {
    "Exists", "DoesNotExist", "(Boolean) is True", "(Boolean) is False",
    "(Sequence) is empty", "(Sequence) is not empty",
}  # fmt: skip
QUERY_PACKAGE = Path(query.__file__).parent


def _condition_with_operations(*operations) -> Dict:
    return {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "UnaryStatement",
                "operand": {"type": "DynamicOperand", "operations": list(operations)},
                "operator": {"type": "Exists"},
            }
        ],
    }


def test_every_v1_ledger_literal_is_supported() -> None:
    # then
    assert len(SUPPORTED_OPERATION_TYPES) == 36
    assert set(SUPPORTED_OPERATION_TYPES) == V1_OPERATION_LITERALS
    assert set(COMPARATOR_TYPES) == V1_COMPARATOR_LITERALS
    assert set(UNARY_OPERATOR_TYPES) == V1_UNARY_LITERALS


def test_v1_detections_filter_golden_checks_declared_parameters() -> None:
    # given: FILTERING_OPERATION from tests/workflows/.../test_workflow_with_flow_control.py
    detection_filter = {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "operator": "and",
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
                },
                {
                    "type": "BinaryStatement",
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operations": [
                            {
                                "type": "ExtractDetectionProperty",
                                "property_name": "size",
                            }
                        ],
                    },
                    "comparator": {"type": "(Number) >="},
                    "right_operand": {
                        "type": "DynamicOperand",
                        "operand_name": "image",
                        "operations": [
                            {"type": "ExtractImageProperty", "property_name": "size"},
                            {"type": "Multiply", "other": 0.02},
                        ],
                    },
                },
            ],
        },
    }

    # when
    compile_operations([detection_filter], declared_names={"classes", "image"})
    with pytest.raises(QueryDefinitionError) as error:
        compile_operations([detection_filter], declared_names={"image"})

    # then
    assert error.value.path == "$[0].filter_operation.statements[0].right_operand"
    assert error.value.parameter == "operand_name"
    assert "classes" in str(error.value)


@pytest.mark.parametrize(
    "operations, path, operation, parameter",
    [
        (
            [{"type": "Multiply"}],
            "$.statements[0].operand.operations[0].other",
            "Multiply",
            "other",
        ),
        (
            [{"type": "ToNumber", "cast_to": "double"}],
            "$.statements[0].operand.operations[0].cast_to",
            "ToNumber",
            "cast_to",
        ),
        (
            [{"type": "StringMatches", "regex": "(car"}],
            "$.statements[0].operand.operations[0].regex",
            "StringMatches",
            "regex",
        ),
        (
            [{"type": "NumberRound", "decimal_digits": "two"}],
            "$.statements[0].operand.operations[0].decimal_digits",
            "NumberRound",
            "decimal_digits",
        ),
        (
            [{"type": "SequenceAggregate", "mode": "median"}],
            "$.statements[0].operand.operations[0].mode",
            "SequenceAggregate",
            "mode",
        ),
        (
            [{"type": "NoSuchOperation"}],
            "$.statements[0].operand.operations[0]",
            "NoSuchOperation",
            None,
        ),
        (
            [
                {
                    "type": "SequenceApply",
                    "operations": [
                        {"type": "ToString"},
                        {"type": "ConvertImageToJPEG", "compression_level": 0},
                    ],
                }
            ],
            "$.statements[0].operand.operations[0].operations[1].compression_level",
            "ConvertImageToJPEG",
            "compression_level",
        ),
    ],
)
def test_definition_errors_report_path_operation_and_parameter(
    operations, path, operation, parameter
) -> None:
    # when
    with pytest.raises(QueryDefinitionError) as error:
        compile_condition(_condition_with_operations(*operations))

    # then
    assert error.value.path == path
    assert error.value.operation == operation
    assert error.value.parameter == parameter


@pytest.mark.parametrize(
    "definition, path",
    [
        ({"type": "StatementGroup", "statements": []}, "$.statements"),
        (
            {"type": "StatementGroup", "statements": [{"type": "UnaryStatement"}]},
            "$.statements[0].operand",
        ),
        (
            {
                "type": "StatementGroup",
                "operator": "xor",
                "statements": [
                    {
                        "type": "UnaryStatement",
                        "operand": {"type": "StaticOperand", "value": 1},
                        "operator": {"type": "Exists"},
                    }
                ],
            },
            "$.operator",
        ),
        (
            {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "UnaryStatement",
                        "operand": {"type": "StaticOperand"},
                        "operator": {"type": "Exists"},
                    }
                ],
            },
            "$.statements[0].operand.value",
        ),
        (
            {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "UnaryStatement",
                        "operand": {"type": "StaticOperand", "value": 1},
                        "operator": {"type": "(Number) >"},
                    }
                ],
            },
            "$.statements[0].operator.type",
        ),
    ],
)
def test_structural_definition_errors(definition, path) -> None:
    # when
    with pytest.raises(QueryDefinitionError) as error:
        compile_condition(definition)

    # then
    assert error.value.path == path
    assert isinstance(error.value, ValueError)


def test_current_v1_defaults_are_preserved() -> None:
    # given
    definition = StatementGroup.model_validate(
        _condition_with_operations(
            {"type": "StringSubSequence"},
            {"type": "SequenceJoin"},
            {"type": "RandomNumber"},
            {"type": "NumericSequenceAggregate", "function": "sum"},
        )
    )

    # when
    statement = definition.statements[0]
    sub_sequence, join, random_number, aggregate = statement.operand.operations

    # then
    assert definition.operator == "or"
    assert statement.negate is False
    assert statement.operand.operand_name == "_"
    assert (sub_sequence.start, sub_sequence.end) == (0, -1)
    assert join.separator == ", "
    assert (random_number.min_value, random_number.max_value) == (0.0, 1.0)
    assert aggregate.neutral_value is None


def test_unknown_extra_fields_are_ignored_like_v1() -> None:
    # given
    definition = _condition_with_operations({"type": "ToString", "ui_hint": "x"})
    definition["ui_layout"] = {"collapsed": True}

    # when
    result = compile_condition(definition).evaluate({"_": 1})

    # then
    assert result is True


def test_json_schema_lists_every_literal() -> None:
    # when
    schema_text = json.dumps(StatementGroup.model_json_schema())

    # then
    literals = V1_OPERATION_LITERALS | V1_COMPARATOR_LITERALS | V1_UNARY_LITERALS
    for literal in literals:
        assert json.dumps(literal) in schema_text


def test_block_params_can_embed_the_query_model() -> None:
    # given
    class Params(BaseModel):
        condition_statement: StatementGroup

    # when
    with pytest.raises(ValidationError) as error:
        Params.model_validate(
            {
                "condition_statement": _condition_with_operations(
                    {"type": "ConvertImageToJPEG", "compression_level": 101}
                )
            }
        )

    # then
    assert error.value.errors()[0]["loc"][-1] == "compression_level"


def test_compile_accepts_validated_models() -> None:
    # given
    definition = StatementGroup.model_validate(
        _condition_with_operations({"type": "SequenceLength"})
    )

    # when
    result = compile_condition(definition).evaluate({"_": [1]})

    # then
    assert result is True


def test_query_modules_avoid_v1_engine_registry_and_supervision() -> None:
    # given
    allowed_roots = {
        "base64", "collections", "dataclasses", "datetime", "json", "random", "re",
        "typing", "uuid", "pydantic", "pydantic_core", "typing_extensions", "torch",
        "inference_models",
        # Host boundaries, imported lazily: JPEG encoding and the zone polygon.
        "cv2", "numpy", "shapely",
    }  # fmt: skip
    allowed_modules = {
        "roboflow_workflows.v2.data.frames",
        "roboflow_workflows.v2.data.image",
        "roboflow_workflows.v2.data.masks",
        "roboflow_workflows.v2.data.masks.providers",
        "roboflow_workflows.v2.data.prediction_ops",
        "roboflow_workflows.v2.data.row_identity",
        "roboflow_workflows.v2.data.rows",
        # Defines the V1 response dictionary; imported by DetectionsToDictionary only.
        "roboflow_workflows.core_steps.common.serializers_tensor",
    }
    own_prefix = "roboflow_workflows.v2.blocks.query"

    # when
    imported = set()
    for module_path in QUERY_PACKAGE.glob("*.py"):
        tree = ast.parse(module_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)

    # then
    foreign = {
        name
        for name in imported
        if not name.startswith(own_prefix)
        and name.split(".")[0] not in allowed_roots
        and name not in allowed_modules
    }
    assert foreign == set()
    assert not any("eval(" in path.read_text() for path in QUERY_PACKAGE.glob("*.py"))
