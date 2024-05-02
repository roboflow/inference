from enum import Enum


class Operator(Enum):
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LOWER_THAN = "lower_than"
    GREATER_THAN = "greater_than"
    LOWER_THAN_OR_EQUAL = "lower_than_or_equal"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    IN = "in"
    STR_STARTS_WITH = "str_starts_with"
    STR_ENDS_WITH = "str_ends_with"
    STR_CONTAINS = "str_contains"


class BinaryOperator(Enum):
    OR = "or"
    AND = "and"


OPERATORS_FUNCTIONS = {
    Operator.EQUAL: lambda a, b: a == b,
    Operator.NOT_EQUAL: lambda a, b: a != b,
    Operator.LOWER_THAN: lambda a, b: a < b,
    Operator.GREATER_THAN: lambda a, b: a > b,
    Operator.LOWER_THAN_OR_EQUAL: lambda a, b: a <= b,
    Operator.GREATER_THAN_OR_EQUAL: lambda a, b: a >= b,
    Operator.IN: lambda a, b: a in b,
    Operator.STR_STARTS_WITH: lambda a, b: a.startswith(b),
    Operator.STR_ENDS_WITH: lambda a, b: a.endswith(b),
    Operator.STR_CONTAINS: lambda a, b: b in a,
}


BINARY_OPERATORS_FUNCTIONS = {
    BinaryOperator.AND: lambda a, b: a and b,
    BinaryOperator.OR: lambda a, b: a or b,
}
