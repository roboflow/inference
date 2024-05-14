from typing import List, Literal, Union, Any

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    DetectionsProperty,
    NumberCastingMode,
    SequenceAggregationFunction,
    SequenceAggregationMode,
    SequenceUnwrapMethod, StatementsGroupsOperator,
)

TYPE_PARAMETER_NAME = "type"
DEFAULT_OPERAND_NAME = "_"


class OperationDefinition(BaseModel):
    type: str


class StringToLowerCase(OperationDefinition):
    type: Literal["StringToLowerCase"]


class StringToUpperCase(OperationDefinition):
    type: Literal["StringToUpperCase"]


class LookupTable(OperationDefinition):
    type: Literal["LookupTable"]
    lookup_table: dict


class ToNumber(OperationDefinition):
    type: Literal["ToNumber"]
    cast_to: NumberCastingMode


class NumberRound(OperationDefinition):
    type: Literal["NumberRound"]
    decimal_digits: int


class SequenceMap(OperationDefinition):
    type: Literal["SequenceMap"]
    lookup_table: dict


class NumericSequenceAggregate(OperationDefinition):
    type: Literal["NumericSequenceAggregate"]
    function: SequenceAggregationFunction


class SequenceAggregate(OperationDefinition):
    type: Literal["SequenceAggregate"]
    mode: SequenceAggregationMode


class ToString(OperationDefinition):
    type: Literal["ToString"]


class ToBoolean(OperationDefinition):
    type: Literal["ToBoolean"]


class StringSubSequence(OperationDefinition):
    type: Literal["StringSubSequence"]
    start: int = 0
    end: int = -1


class DetectionsPropertyExtract(OperationDefinition):
    type: Literal["DetectionsPropertyExtract"]
    property_name: DetectionsProperty


class ExtractDetectionProperty(OperationDefinition):
    type: Literal["ExtractDetectionProperty"]
    property_name: DetectionsProperty
    unwrap_method: SequenceUnwrapMethod = SequenceUnwrapMethod.FIRST


class DetectionsFilter(OperationDefinition):
    type: Literal["DetectionsFilter"]
    filter_operation: "AllOperationsType"


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
