from typing import Literal, List, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from inference.core.workflows.core_steps.common.query_language.entities import evaluation
from inference.core.workflows.core_steps.common.query_language.entities.enums import NumberCastingMode, \
    SequenceAggregationFunction, SequenceAggregationMode, DetectionsProperty, SequenceUnwrapMethod


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
    filter_operation: "evaluation.StatementGroup"


class SequenceApply(OperationDefinition):
    type: Literal["SequenceApply"]
    operations: List["AllOperationsType"]


AllOperationsType = Annotated[
    Union[
        StringToLowerCase, StringToUpperCase, LookupTable,
        ToNumber, NumberRound, SequenceMap, SequenceApply,
        NumericSequenceAggregate, ToString, ToBoolean, StringSubSequence,
        DetectionsPropertyExtract, SequenceAggregate, DetectionsFilter,
        ExtractDetectionProperty, DetectionsFilter
    ],
    Field(discriminator="type")
]


class OperationsChain(BaseModel):
    operations: List[AllOperationsType]
