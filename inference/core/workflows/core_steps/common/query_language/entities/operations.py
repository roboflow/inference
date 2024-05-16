from typing import Any, List, Literal, Union

from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    DetectionsProperty,
    NumberCastingMode,
    SequenceAggregationFunction,
    SequenceAggregationMode,
    SequenceUnwrapMethod,
    StatementsGroupsOperator,
)
from inference.core.workflows.entities.types import STRING_KIND, WILDCARD_KIND, INTEGER_KIND, FLOAT_KIND, \
    FLOAT_ZERO_TO_ONE_KIND, BOOLEAN_KIND, LIST_OF_VALUES_KIND

TYPE_PARAMETER_NAME = "type"
DEFAULT_OPERAND_NAME = "_"


class OperationDefinition(BaseModel):
    type: str


class StringToLowerCase(OperationDefinition):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Executes lowercase operation on input string",
            "compound": False,
            "input_kind": [STRING_KIND],
            "output_kind": [STRING_KIND],
        },
    )
    type: Literal["StringToLowerCase"]


class StringToUpperCase(OperationDefinition):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Executes uppercase operation on input string",
            "compound": False,
            "input_kind": [STRING_KIND],
            "output_kind": [STRING_KIND],
        },
    )
    type: Literal["StringToUpperCase"]


class LookupTable(OperationDefinition):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Changes value according to mapping stated in lookup table",
            "input_kind": [WILDCARD_KIND],
            "output_kind": [WILDCARD_KIND],
        },
    )
    type: Literal["LookupTable"]
    lookup_table: dict


class ToNumber(OperationDefinition):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Changes value into number - float or int depending on configuration",
            "input_kind": [STRING_KIND, BOOLEAN_KIND, INTEGER_KIND, FLOAT_KIND, FLOAT_ZERO_TO_ONE_KIND],
            "output_kind": [INTEGER_KIND, FLOAT_KIND, FLOAT_ZERO_TO_ONE_KIND],
        },
    )
    type: Literal["ToNumber"]
    cast_to: NumberCastingMode


class NumberRound(OperationDefinition):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Rounds the number",
            "input_kind": [INTEGER_KIND, FLOAT_KIND, FLOAT_ZERO_TO_ONE_KIND],
            "output_kind": [INTEGER_KIND, FLOAT_KIND, FLOAT_ZERO_TO_ONE_KIND],
        },
    )
    type: Literal["NumberRound"]
    decimal_digits: int


class SequenceMap(OperationDefinition):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Changes each value of sequence according to mapping stated in lookup table",
            "input_kind": [LIST_OF_VALUES_KIND],
            "output_kind": [LIST_OF_VALUES_KIND],
        },
    )
    type: Literal["SequenceMap"]
    lookup_table: dict


class NumericSequenceAggregate(OperationDefinition):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Aggregates numeric sequence using aggregation function like min or max - adjusted to work on numbers",
            "input_kind": [LIST_OF_VALUES_KIND],
            "output_kind": [INTEGER_KIND, FLOAT_KIND, FLOAT_ZERO_TO_ONE_KIND],
        },
    )
    type: Literal["NumericSequenceAggregate"]
    function: SequenceAggregationFunction


class SequenceAggregate(OperationDefinition):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Aggregates sequence using generic aggregation methods - adjusted to majority data types",
            "input_kind": [LIST_OF_VALUES_KIND],
            "output_kind": [WILDCARD_KIND],
        },
    )
    type: Literal["SequenceAggregate"]
    mode: SequenceAggregationMode


class ToString(OperationDefinition):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Stringifies data",
            "input_kind": [WILDCARD_KIND],
            "output_kind": [STRING_KIND],
        },
    )
    type: Literal["ToString"]


class ToBoolean(OperationDefinition):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Changes input data into boolean",
            "input_kind": [FLOAT_KIND, FLOAT_ZERO_TO_ONE_KIND, INTEGER_KIND],
            "output_kind": [BOOLEAN_KIND],
        },
    )
    type: Literal["ToBoolean"]


class StringSubSequence(OperationDefinition):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Takes sub-string of the input string",
            "input_kind": [STRING_KIND],
            "output_kind": [STRING_KIND],
        },
    )
    type: Literal["StringSubSequence"]
    start: int = Field(default=0)
    end: int = Field(default=-1)


class DetectionsPropertyExtract(OperationDefinition):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Extracts properties from ",
            "input_kind": [STRING_KIND],
            "output_kind": [STRING_KIND],
        },
    )
    type: Literal["DetectionsPropertyExtract"]
    property_name: DetectionsProperty


class ExtractDetectionProperty(OperationDefinition):
    type: Literal["ExtractDetectionProperty"]
    property_name: DetectionsProperty
    unwrap_method: SequenceUnwrapMethod = SequenceUnwrapMethod.FIRST


class DetectionsFilter(OperationDefinition):
    type: Literal["DetectionsFilter"]
    filter_operation: "StatementGroup"


class DetectionsOffset(OperationDefinition):
    type: Literal["DetectionsOffset"]
    offset_x: int
    offset_y: int


class DetectionsShift(OperationDefinition):
    type: Literal["DetectionsShift"]
    shift_x: int
    shift_y: int


class RandomNumber(OperationDefinition):
    type: Literal["RandomNumber"]
    min_value: float = Field(default=0.0)
    max_value: float = Field(default=1.0)


class StringMatches(OperationDefinition):
    type: Literal["StringMatches"]
    regex: str


class SequenceLength(OperationDefinition):
    type: Literal["SequenceLength"]


class SequenceApply(OperationDefinition):
    type: Literal["SequenceApply"]
    operations: List["AllOperationsType"]


AllOperationsType = Annotated[
    Union[
        StringToLowerCase,
        StringToUpperCase,
        LookupTable,
        ToNumber,
        NumberRound,
        SequenceMap,
        SequenceApply,
        NumericSequenceAggregate,
        ToString,
        ToBoolean,
        StringSubSequence,
        DetectionsPropertyExtract,
        SequenceAggregate,
        ExtractDetectionProperty,
        DetectionsFilter,
        DetectionsOffset,
        DetectionsShift,
        RandomNumber,
        StringMatches,
        SequenceLength,
    ],
    Field(discriminator="type"),
]


class OperationsChain(BaseModel):
    operations: List[AllOperationsType]


class BinaryOperator(BaseModel):
    type: str


class Equals(BinaryOperator):
    type: Literal["=="]


class NotEquals(BinaryOperator):
    type: Literal["!="]


class NumerGreater(BinaryOperator):
    type: Literal["(Number) >"]


class NumerGreaterEqual(BinaryOperator):
    type: Literal["(Number) >="]


class NumerLower(BinaryOperator):
    type: Literal["(Number) <"]


class NumerLowerEqual(BinaryOperator):
    type: Literal["(Number) <="]


class StringStartsWith(BinaryOperator):
    type: Literal["(String) startsWith"]


class StringEndsWith(BinaryOperator):
    type: Literal["(String) endsWith"]


class StringContains(BinaryOperator):
    type: Literal["(String) contains"]


class In(BinaryOperator):
    type: Literal["in"]


class UnaryOperator(BaseModel):
    type: str


class Exists(UnaryOperator):
    type: Literal["Exists"]


class DoesNotExist(UnaryOperator):
    type: Literal["DoesNotExist"]


class IsTrue(UnaryOperator):
    type: Literal["(Boolean) is True"]


class IsFalse(UnaryOperator):
    type: Literal["(Boolean) is False"]


class IsEmpty(UnaryOperator):
    type: Literal["(Sequence) is empty"]


class IsNotEmpty(UnaryOperator):
    type: Literal["(Sequence) is not empty"]


class StaticOperand(BaseModel):
    type: Literal["StaticOperand"]
    value: Any
    operations: List["AllOperationsType"] = Field(default_factory=lambda: [])


class DynamicOperand(BaseModel):
    type: Literal["DynamicOperand"]
    operations: List["AllOperationsType"] = Field(default_factory=lambda: [])
    operand_name: str = DEFAULT_OPERAND_NAME


class BinaryStatement(BaseModel):
    type: Literal["BinaryStatement"]
    left_operand: Annotated[
        Union[StaticOperand, DynamicOperand], Field(discriminator="type")
    ]
    comparator: Annotated[
        Union[
            In,
            StringContains,
            StringEndsWith,
            StringStartsWith,
            NumerLowerEqual,
            NumerLower,
            NumerGreaterEqual,
            NumerGreater,
            NotEquals,
            Equals,
        ],
        Field(discriminator="type"),
    ]
    right_operand: Annotated[
        Union[StaticOperand, DynamicOperand], Field(discriminator="type")
    ]
    negate: bool = False


class UnaryStatement(BaseModel):
    type: Literal["UnaryStatement"]
    operand: Annotated[
        Union[StaticOperand, DynamicOperand], Field(discriminator="type")
    ]
    operator: Annotated[
        Union[Exists, DoesNotExist, IsTrue, IsFalse, IsEmpty, IsNotEmpty],
        Field(discriminator="type"),
    ]
    negate: bool = False


class StatementGroup(BaseModel):
    type: Literal["StatementGroup"]
    statements: List[
        Annotated[
            Union[BinaryStatement, UnaryStatement, "StatementGroup"],
            Field(discriminator="type"),
        ]
    ] = Field(min_items=1)
    operator: StatementsGroupsOperator = StatementsGroupsOperator.OR
