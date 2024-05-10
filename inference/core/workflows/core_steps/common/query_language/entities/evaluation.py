from typing import Literal, Any, List, Union
from typing_extensions import Annotated

from pydantic import BaseModel, Field

from inference.core.workflows.core_steps.common.query_language.entities import operations

DEFAULT_OPERAND_NAME = "_"


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
    operations: List["operations.AllOperationsType"] = Field(default_factory=lambda: [])


class DynamicOperand(BaseModel):
    type: Literal["DynamicOperand"]
    operations: List["operations.AllOperationsType"] = Field(default_factory=lambda: [])
    operand_name: str = DEFAULT_OPERAND_NAME


class BinaryStatement(BaseModel):
    type: Literal["BinaryStatement"]
    left_operand: Annotated[Union[StaticOperand, DynamicOperand], Field(discriminator="type")]
    comparator: Annotated[
        Union[
            In, StringContains, StringEndsWith, StringStartsWith,
            NumerLowerEqual, NumerLower, NumerGreaterEqual,
            NumerGreater, NotEquals, Equals,
        ],
        Field(discriminator="type")
    ]
    right_operand: Annotated[Union[StaticOperand, DynamicOperand], Field(discriminator="type")]
    negate: bool = False


class UnaryStatement(BaseModel):
    type: Literal["UnaryStatement"]
    operand: Annotated[Union[StaticOperand, DynamicOperand], Field(discriminator="type")]
    operator: Annotated[
        Union[Exists, DoesNotExist, IsTrue, IsFalse, IsEmpty, IsNotEmpty],
        Field(discriminator="type")
    ]
    negate: bool = False


class StatementGroup(BaseModel):
    type: Literal["StatementGroup"]
    operator: Literal["and", "or"]
    statements: List[Annotated[Union[BinaryStatement, UnaryStatement, "StatementGroup"], Field(discriminator="type")]]
