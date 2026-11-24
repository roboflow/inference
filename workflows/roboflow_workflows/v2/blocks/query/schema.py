"""Pydantic schema of the supported UQL (Roboflow query language) JSON.

The models accept the same JSON, field names and defaults as the current V1
query language: 36 operations, 16 comparator spellings, 6 unary operators and
5 structural types. V2 adds two detection properties: ``decoded_text`` and its
alias ``data``, the text a barcode or QR code detector stores per row.
"""

import re
from typing import Any, Dict, List, Literal, Type, Union, get_args

from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from pydantic_core import PydanticCustomError
from roboflow_workflows.v2.blocks.query.errors import QueryDefinitionError
from typing_extensions import Annotated

DEFAULT_OPERAND_NAME = "_"

DETECTION_PROPERTIES = (
    "confidence",
    "class_name",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "size",
    "class_id",
    "center",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "in_out",
    "path_deviation",
    "polygon",
    "time_in_zone",
    "tracker_id",
    "velocity",
    "speed",
    "smoothed_velocity",
    "smoothed_speed",
    "image_dimensions",
    "prediction_type",
    "keypoints_xy",
    "rect",
    "width",
    "height",
    "angle",
    "area_px",
    "area_converted",
    "decoded_text",
    "data",
)

COMPARATOR_TYPES = (
    "==",
    "(Number) ==",
    "!=",
    "(Number) !=",
    "(Number) >",
    "(Number) >=",
    "(Number) <",
    "(Number) <=",
    "(Number) in range",
    "(String) startsWith",
    "(String) endsWith",
    "(String) contains",
    "in (Sequence)",
    "any in (Sequence)",
    "all in (Sequence)",
    "(Detection) in zone",
)
UNARY_OPERATOR_TYPES = (
    "Exists",
    "DoesNotExist",
    "(Boolean) is True",
    "(Boolean) is False",
    "(Sequence) is empty",
    "(Sequence) is not empty",
)


class ParameterlessOperation(BaseModel):
    """Operation that takes no parameters besides its input value."""

    type: Literal[
        "StringToLowerCase",
        "StringToUpperCase",
        "ToString",
        "ToBoolean",
        "SequenceLength",
        "SequenceElementsCount",
        "ConvertDictionaryToJSON",
        "TimestampToISOFormat",
    ] = Field(description="Operation name.")


class LookupOperation(BaseModel):
    """Map the value (LookupTable) or each element (SequenceMap) through a table."""

    type: Literal["LookupTable", "SequenceMap"] = Field(description="Operation name.")
    lookup_table: dict = Field(
        description="Mapping from input values to output values.",
        examples=[{"cat": "animal", "car": "vehicle"}],
    )


class ArithmeticOperation(BaseModel):
    """Multiply or divide the numeric value by ``other``."""

    type: Literal["Multiply", "Divide"] = Field(description="Operation name.")
    other: Union[int, float] = Field(description="Right-hand operand.", examples=[2])


class ToNumber(BaseModel):
    """Convert the value through ``float``; ``int`` then truncates toward zero."""

    type: Literal["ToNumber"] = Field(description="Operation name.")
    cast_to: Literal["int", "float"] = Field(description="Target number type.")


class NumberRound(BaseModel):
    """Round the number with Python ``round``."""

    type: Literal["NumberRound"] = Field(description="Operation name.")
    decimal_digits: int = Field(description="Digits after the decimal point.")


class StringSubSequence(BaseModel):
    """Take ``value[start:end]`` of a string; the default end drops the last character."""

    type: Literal["StringSubSequence"] = Field(description="Operation name.")
    start: int = Field(default=0, description="Slice start index.")
    end: int = Field(default=-1, description="Slice end index (exclusive).")


class StringMatches(BaseModel):
    """Check whether the regex matches anywhere in the string (``re.search``)."""

    type: Literal["StringMatches"] = Field(description="Operation name.")
    regex: str = Field(description="Python regular expression.", examples=["^car"])

    @field_validator("regex")
    @classmethod
    def _check_regex(cls, regex: str) -> str:
        try:
            re.compile(regex)
        except re.error as error:
            raise PydanticCustomError(
                "query_invalid_regex",
                "invalid regular expression: {error}",
                {"error": str(error)},
            ) from error

        return regex


class NumericSequenceAggregate(BaseModel):
    """Aggregate a numeric sequence; an empty sequence returns ``neutral_value``."""

    type: Literal["NumericSequenceAggregate"] = Field(description="Operation name.")
    function: Literal["min", "max", "sum"] = Field(description="Aggregation function.")
    neutral_value: Any = Field(
        default=None, description="Result for an empty sequence."
    )


class SequenceAggregate(BaseModel):
    """Pick one element of a non-empty sequence; count ties keep encounter order."""

    type: Literal["SequenceAggregate"] = Field(description="Operation name.")
    mode: Literal["first", "last", "most_common", "least_common"] = Field(
        description="Element to pick."
    )


class SequenceJoin(BaseModel):
    """Join stringified elements; results above 1,000,000 characters fail."""

    type: Literal["SequenceJoin"] = Field(description="Operation name.")
    separator: str = Field(default=", ", description="Text between elements.")


class RandomNumber(BaseModel):
    """Ignore the value and draw ``min_value + random() * |max_value - min_value|``."""

    type: Literal["RandomNumber"] = Field(description="Operation name.")
    min_value: float = Field(default=0.0, description="Lower end of the range.")
    max_value: float = Field(default=1.0, description="Upper end of the range.")


class SequenceApply(BaseModel):
    """Apply a nested operation chain to every element of a sequence."""

    type: Literal["SequenceApply"] = Field(description="Operation name.")
    operations: List["Operation"] = Field(
        description="Operations applied to each element."
    )


DetectionProperty = Literal[DETECTION_PROPERTIES]


class DetectionsPropertyExtract(BaseModel):
    """Extract one property of every detection, as a list in row order."""

    type: Literal["DetectionsPropertyExtract"] = Field(description="Operation name.")
    property_name: DetectionProperty = Field(description="Property to extract.")


class ExtractDetectionProperty(BaseModel):
    """Extract one property of a single detection (the `_` row of a filter)."""

    type: Literal["ExtractDetectionProperty"] = Field(description="Operation name.")
    property_name: DetectionProperty = Field(description="Property to extract.")


class DetectionsFilter(BaseModel):
    """Keep detections for which ``filter_operation`` holds, with the row bound to `_`."""

    type: Literal["DetectionsFilter"] = Field(description="Operation name.")
    filter_operation: "StatementGroup" = Field(
        description="Condition evaluated once per detection."
    )


class DetectionsSelection(BaseModel):
    """Keep at most one detection, chosen by ``mode``."""

    type: Literal["DetectionsSelection"] = Field(description="Operation name.")
    mode: Literal["left_most", "right_most", "top_confidence", "first", "last"] = Field(
        description="Which detection to keep."
    )


class SortDetections(BaseModel):
    """Reorder detections by a box property."""

    type: Literal["SortDetections"] = Field(description="Operation name.")
    mode: Literal[
        "confidence",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "size",
        "center_x",
        "center_y",
    ] = Field(description="Sort key.")
    ascending: bool = Field(default=True, description="Sort direction.")


class DetectionsOffset(BaseModel):
    """Grow every box by ``offset_x``/``offset_y`` pixels, half on each side."""

    type: Literal["DetectionsOffset"] = Field(description="Operation name.")
    offset_x: int = Field(description="Total width increase in pixels.")
    offset_y: int = Field(description="Total height increase in pixels.")


class DetectionsShift(BaseModel):
    """Move every box by ``shift_x``/``shift_y`` pixels."""

    type: Literal["DetectionsShift"] = Field(description="Operation name.")
    shift_x: int = Field(description="Horizontal shift in pixels.")
    shift_y: int = Field(description="Vertical shift in pixels.")


class DetectionsRename(BaseModel):
    """Rename classes; a string ``class_map``/``strict`` names a runtime parameter."""

    type: Literal["DetectionsRename"] = Field(description="Operation name.")
    class_map: Union[Dict[str, str], str] = Field(
        description="Old-to-new class names, or the name of a runtime parameter "
        "holding that mapping."
    )
    strict: Union[bool, str] = Field(
        default=True,
        description="Require every class to be in class_map, or the name of a "
        "runtime parameter holding that flag.",
    )
    new_classes_id_offset: int = Field(
        default=1024,
        description="First class id given to new class names when strict is false.",
    )


class PickDetectionsByParentClass(BaseModel):
    """Keep parent-class detections and detections centred inside one of them."""

    type: Literal["PickDetectionsByParentClass"] = Field(description="Operation name.")
    parent_class: str = Field(description="Class of parent detections.")


class DetectionsToDictionary(BaseModel):
    """Serialise detections to the inference response dictionary (host boundary)."""

    type: Literal["DetectionsToDictionary"] = Field(description="Operation name.")


class ClassificationPropertyExtract(BaseModel):
    """Extract a property of a single-image classification prediction."""

    type: Literal["ClassificationPropertyExtract"] = Field(
        description="Operation name."
    )
    property_name: Literal[
        "top_class",
        "top_class_confidence",
        "top_class_confidence_single",
        "all_classes",
        "all_confidences",
    ] = Field(
        description=(
            "Single-label: top_class is a name and top_class_confidence is a "
            "number. Multi-label: both return lists of selected classes; use "
            "top_class_confidence_single for the highest selected confidence "
            "as one number (0.0 when none is selected). all_classes and "
            "all_confidences return lists for either kind."
        )
    )


class ExtractImageProperty(BaseModel):
    """Read the image size from its tensor shape; no pixels are copied."""

    type: Literal["ExtractImageProperty"] = Field(description="Operation name.")
    property_name: Literal["size", "height", "width", "aspect_ratio"] = Field(
        description="Property to extract."
    )


class ExtractFrameMetadata(BaseModel):
    """Read video frame metadata of an image, frame or prediction."""

    type: Literal["ExtractFrameMetadata"] = Field(description="Operation name.")
    property_name: Literal["frame_number", "frame_timestamp", "seconds_since_start"] = (
        Field(description="Property to extract.")
    )


class ConvertImageToJPEG(BaseModel):
    """Encode the image as JPEG bytes (host boundary)."""

    type: Literal["ConvertImageToJPEG"] = Field(description="Operation name.")
    compression_level: int = Field(
        default=95, ge=1, le=100, description="JPEG quality."
    )


class ConvertImageToBase64(BaseModel):
    """Encode the image as base64 text of a quality-95 JPEG (host boundary)."""

    type: Literal["ConvertImageToBase64"] = Field(description="Operation name.")


Operation = Annotated[
    Union[
        ParameterlessOperation,
        LookupOperation,
        ArithmeticOperation,
        ToNumber,
        NumberRound,
        StringSubSequence,
        StringMatches,
        NumericSequenceAggregate,
        SequenceAggregate,
        SequenceJoin,
        RandomNumber,
        SequenceApply,
        DetectionsPropertyExtract,
        ExtractDetectionProperty,
        DetectionsFilter,
        DetectionsSelection,
        SortDetections,
        DetectionsOffset,
        DetectionsShift,
        DetectionsRename,
        PickDetectionsByParentClass,
        DetectionsToDictionary,
        ClassificationPropertyExtract,
        ExtractImageProperty,
        ExtractFrameMetadata,
        ConvertImageToJPEG,
        ConvertImageToBase64,
    ],
    Field(discriminator="type"),
]


class StaticOperand(BaseModel):
    """Literal value, optionally transformed by an operation chain."""

    type: Literal["StaticOperand"] = Field(description="Operand kind.")
    value: Any = Field(description="Literal operand value.", examples=[0.5])
    operations: List[Operation] = Field(
        default_factory=list, description="Operations applied to the value."
    )


class DynamicOperand(BaseModel):
    """Runtime value looked up by name, optionally transformed by an operation chain."""

    type: Literal["DynamicOperand"] = Field(description="Operand kind.")
    operand_name: str = Field(
        default=DEFAULT_OPERAND_NAME,
        description="Name of the runtime value; `_` is the current row by convention.",
        examples=["threshold"],
    )
    operations: List[Operation] = Field(
        default_factory=list, description="Operations applied to the value."
    )


Operand = Annotated[Union[StaticOperand, DynamicOperand], Field(discriminator="type")]


class Comparator(BaseModel):
    """Binary comparison between the left and right operands."""

    type: Literal[COMPARATOR_TYPES] = Field(description="Comparator spelling.")


class UnaryOperator(BaseModel):
    """Predicate on one operand. ``(Boolean) is ...`` tests identity, not truthiness."""

    type: Literal[UNARY_OPERATOR_TYPES] = Field(description="Operator spelling.")


class BinaryStatement(BaseModel):
    """Compare two operands; ``negate`` inverts the result."""

    type: Literal["BinaryStatement"] = Field(description="Statement kind.")
    left_operand: Operand = Field(description="Left-hand operand.")
    comparator: Comparator = Field(description="Comparison applied to both operands.")
    right_operand: Operand = Field(description="Right-hand operand.")
    negate: bool = Field(default=False, description="Invert the result.")


class UnaryStatement(BaseModel):
    """Apply a unary predicate to one operand; ``negate`` inverts the result."""

    type: Literal["UnaryStatement"] = Field(description="Statement kind.")
    operand: Operand = Field(description="Tested operand.")
    operator: UnaryOperator = Field(description="Predicate applied to the operand.")
    negate: bool = Field(default=False, description="Invert the result.")


Statement = Annotated[
    Union[BinaryStatement, UnaryStatement, "StatementGroup"],
    Field(discriminator="type"),
]


class StatementGroup(BaseModel):
    """Combine statements with ``and``/``or``; every statement is evaluated."""

    type: Literal["StatementGroup"] = Field(description="Statement kind.")
    statements: List[Statement] = Field(
        min_length=1, description="Statements combined by the operator."
    )
    operator: Literal["and", "or"] = Field(
        default="or", description="How statement results are combined."
    )


SequenceApply.model_rebuild()
DetectionsFilter.model_rebuild()
StatementGroup.model_rebuild()

STRUCTURE_TYPES = (
    "StatementGroup",
    "BinaryStatement",
    "UnaryStatement",
    "StaticOperand",
    "DynamicOperand",
)
_OPERATION_MODELS = get_args(get_args(Operation)[0])
SUPPORTED_OPERATION_TYPES = tuple(
    literal
    for model in _OPERATION_MODELS
    for literal in get_args(model.model_fields["type"].annotation)
)
_TYPE_LITERALS = frozenset(
    SUPPORTED_OPERATION_TYPES
    + COMPARATOR_TYPES
    + UNARY_OPERATOR_TYPES
    + STRUCTURE_TYPES
)
OPERATION_LIST = TypeAdapter(List[Operation])


def validate_definition(
    validator: Union[Type[BaseModel], TypeAdapter],
    definition: Any,
    *,
    path: str = "$",
) -> Any:
    """Validate literal query JSON, reporting the first problem by JSON path.

    Args:
        validator: Model class or type adapter for the expected root node.
        definition: Literal query JSON.
        path: JSON path of ``definition`` inside the caller's document.

    Returns:
        The validated model (or list of models).

    Raises:
        QueryDefinitionError: If the JSON is invalid.
    """
    try:
        if isinstance(validator, TypeAdapter):
            parsed = validator.validate_python(definition)
        else:
            parsed = validator.model_validate(definition)
    except ValidationError as error:
        raise _definition_error(error, path=path) from error

    return parsed


def _definition_error(error: ValidationError, *, path: str) -> QueryDefinitionError:
    issue = error.errors()[0]
    operation = None
    parameter = None
    for item in issue["loc"]:
        if isinstance(item, int):
            path = f"{path}[{item}]"
        elif item in _TYPE_LITERALS:
            # Pydantic inserts the matched discriminator tag into the location.
            operation = item
            parameter = None
        else:
            path = f"{path}.{item}"
            parameter = item
    failing_input = issue.get("input")
    if isinstance(failing_input, dict) and isinstance(failing_input.get("type"), str):
        # The failing node itself (an unknown literal), or the parent of
        # a missing field.
        operation = failing_input["type"]
        if issue["type"] != "missing":
            parameter = None
    remaining = error.error_count() - 1
    detail = issue["msg"]
    if remaining:
        detail = f"{detail} (and {remaining} more validation problem(s))"
    definition_error = QueryDefinitionError(
        detail, path=path, operation=operation, parameter=parameter
    )

    return definition_error
