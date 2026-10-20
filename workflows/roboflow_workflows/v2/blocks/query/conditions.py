"""Recursive UQL condition evaluation on plain Python values.

```
StatementGroup(and|or) ── evaluates EVERY statement, then combines
├── BinaryStatement:  comparator(left_operand, right_operand) [negate]
├── UnaryStatement:   operator(operand) [negate]
└── StatementGroup    (nested)

operand = StaticOperand.value  | runtime values[operand_name]   (default "_")
          then its operation chain
```
"""

from typing import AbstractSet, Any, Callable, Mapping, Optional, Union

from roboflow_workflows.v2.blocks.query.errors import (
    QueryDefinitionError,
    QueryRuntimeError,
    describe_value,
)
from roboflow_workflows.v2.blocks.query.media import point_in_zone
from roboflow_workflows.v2.blocks.query.operations import (
    compile_operation_chain,
    sequence_length,
)
from roboflow_workflows.v2.blocks.query.schema import (
    BinaryStatement,
    DynamicOperand,
    StatementGroup,
    StaticOperand,
    UnaryStatement,
    validate_definition,
)

COMPARATORS = {
    "==": lambda a, b: a == b,
    "(Number) ==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "(Number) !=": lambda a, b: a != b,
    "(Number) >": lambda a, b: a > b,
    "(Number) >=": lambda a, b: a >= b,
    "(Number) <": lambda a, b: a < b,
    "(Number) <=": lambda a, b: a <= b,
    "(Number) in range": lambda a, b: b[0] <= a <= b[1],
    "(String) startsWith": lambda a, b: a.startswith(b),
    "(String) endsWith": lambda a, b: a.endswith(b),
    "(String) contains": lambda a, b: b in a,
    "in (Sequence)": lambda a, b: a in b,
    "any in (Sequence)": lambda a, b: any(item in b for item in a),
    "all in (Sequence)": lambda a, b: all(item in b for item in a),
    "(Detection) in zone": point_in_zone,
}

UNARY_OPERATORS = {
    "Exists": lambda a: a is not None,
    "DoesNotExist": lambda a: a is None,
    # Identity, not truthiness: 1 and "yes" are not `True`.
    "(Boolean) is True": lambda a: a is True,
    "(Boolean) is False": lambda a: a is False,
    "(Sequence) is empty": lambda a: sequence_length(a) == 0,
    "(Sequence) is not empty": lambda a: sequence_length(a) > 0,
}

Values = Mapping[str, Any]
Evaluator = Callable[[Values], Any]


class CompiledCondition:
    """Validated condition, ready to evaluate against runtime named values."""

    def __init__(self, evaluator: Evaluator) -> None:
        self._evaluator = evaluator

    def evaluate(self, values: Values) -> bool:
        """Evaluate the condition.

        Args:
            values: Runtime values by operand name; include ``_`` when the
                caller binds a current row.

        Returns:
            Combined statement result.

        Raises:
            QueryRuntimeError: If an operand is missing or a value cannot be
                processed.
        """
        result = self._evaluator(values)

        return result


def compile_condition(
    definition: Union[StatementGroup, dict],
    *,
    path: str = "$",
    declared_names: Optional[AbstractSet[str]] = None,
) -> CompiledCondition:
    """Validate a literal ``StatementGroup`` once and compile it.

    Args:
        definition: StatementGroup JSON or an already validated model.
        path: JSON path of ``definition``; every error path starts with it.
        declared_names: Operand names the caller will provide at runtime. When
            given, any other ``DynamicOperand.operand_name`` is a definition
            error. When omitted, missing names fail at runtime.

    Returns:
        Compiled condition; call ``evaluate`` per set of runtime values.

    Raises:
        QueryDefinitionError: If the JSON is invalid, uses a pending literal or
            names an undeclared operand.
    """
    if not isinstance(definition, StatementGroup):
        definition = validate_definition(StatementGroup, definition, path=path)
    evaluator = compile_statement(definition, path=path, declared_names=declared_names)
    compiled = CompiledCondition(evaluator)

    return compiled


def compile_statement(statement, *, path: str, declared_names) -> Evaluator:
    """Compile a validated statement or group into ``values -> result``."""
    if isinstance(statement, BinaryStatement):
        return _compile_binary(statement, path=path, declared_names=declared_names)
    if isinstance(statement, UnaryStatement):
        return _compile_unary(statement, path=path, declared_names=declared_names)

    return _compile_group(statement, path=path, declared_names=declared_names)


def _compile_group(group: StatementGroup, *, path: str, declared_names) -> Evaluator:
    evaluators = [
        compile_statement(
            statement, path=f"{path}.statements[{index}]", declared_names=declared_names
        )
        for index, statement in enumerate(group.statements)
    ]
    is_and = group.operator == "and"

    def evaluate_group(values: Values) -> Any:
        # No short circuit: V1 evaluates every statement, so later errors and
        # RandomNumber draws happen even when the result is already known.
        result = evaluators[0](values)
        for evaluator in evaluators[1:]:
            statement_result = evaluator(values)
            result = (
                (result and statement_result)
                if is_and
                else (result or statement_result)
            )

        return result

    return evaluate_group


def _compile_binary(
    statement: BinaryStatement, *, path: str, declared_names
) -> Evaluator:
    left = _compile_operand(
        statement.left_operand,
        path=f"{path}.left_operand",
        declared_names=declared_names,
    )
    right = _compile_operand(
        statement.right_operand,
        path=f"{path}.right_operand",
        declared_names=declared_names,
    )
    comparator_type = statement.comparator.type
    compare = COMPARATORS[comparator_type]

    def evaluate_binary(values: Values) -> Any:
        left_value = left(values)
        right_value = right(values)
        try:
            result = compare(left_value, right_value)
        except Exception as error:
            raise QueryRuntimeError(
                f"cannot compare {describe_value(left_value)} with "
                f"{describe_value(right_value)}: {error}",
                path=f"{path}.comparator",
                operation=comparator_type,
            ) from error

        return not result if statement.negate else result

    return evaluate_binary


def _compile_unary(
    statement: UnaryStatement, *, path: str, declared_names
) -> Evaluator:
    operand = _compile_operand(
        statement.operand, path=f"{path}.operand", declared_names=declared_names
    )
    operator_type = statement.operator.type
    predicate = UNARY_OPERATORS[operator_type]

    def evaluate_unary(values: Values) -> Any:
        value = operand(values)
        try:
            result = predicate(value)
        except Exception as error:
            raise QueryRuntimeError(
                f"cannot test {describe_value(value)}: {error}",
                path=f"{path}.operator",
                operation=operator_type,
            ) from error

        return not result if statement.negate else result

    return evaluate_unary


def _compile_operand(
    operand: Union[StaticOperand, DynamicOperand], *, path: str, declared_names
) -> Evaluator:
    chain = compile_operation_chain(
        operand.operations, path=f"{path}.operations", declared_names=declared_names
    )
    if isinstance(operand, StaticOperand):
        static_value = operand.value
        return lambda values: chain(static_value, values)

    name = operand.operand_name
    if declared_names is not None and name not in declared_names:
        raise QueryDefinitionError(
            f"operand `{name}` is not declared (declared: {sorted(declared_names)})",
            path=path,
            operation="DynamicOperand",
            parameter="operand_name",
        )

    def read_operand(values: Values) -> Any:
        if name not in values:
            raise QueryRuntimeError(
                f"no runtime value named `{name}` (available: {list(values)})",
                path=path,
                operation="DynamicOperand",
                parameter="operand_name",
            )

        return chain(values[name], values)

    return read_operand
