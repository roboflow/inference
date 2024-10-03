from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    StatementsGroupsOperator,
)
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    TYPE_PARAMETER_NAME,
    BinaryStatement,
    DynamicOperand,
    StatementGroup,
    StaticOperand,
    UnaryStatement,
)
from inference.core.workflows.core_steps.common.query_language.entities.types import (
    T,
    V,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    EvaluationEngineError,
    RoboflowQueryLanguageError,
    UndeclaredSymbolError,
)
from inference.core.workflows.core_steps.common.query_language.evaluation_engine.detection.geometry import (
    is_point_in_zone,
)

BINARY_OPERATORS = {
    "==": lambda a, b: a == b,
    "(Number) ==": lambda a, b: a == b,
    "(Number) !=": lambda a, b: a != b,
    "!=": lambda a, b: a != b,
    "(Number) >": lambda a, b: a > b,
    "(Number) >=": lambda a, b: a >= b,
    "(Number) <": lambda a, b: a < b,
    "(Number) <=": lambda a, b: a <= b,
    "(String) startsWith": lambda a, b: a.startswith(b),
    "(String) endsWith": lambda a, b: a.endswith(b),
    "(String) contains": lambda a, b: b in a,
    "in (Sequence)": lambda a, b: a in b,
    "any in (Sequence)": lambda a, b: any(item in b for item in a),
    "all in (Sequence)": lambda a, b: all(item in b for item in a),
    "(Detection) in zone": is_point_in_zone,
}

UNARY_OPERATORS = {
    "Exists": lambda a: a is not None,
    "DoesNotExist": lambda a: a is None,
    "(Boolean) is True": lambda a: a is True,
    "(Boolean) is False": lambda a: a is False,
    "(Sequence) is empty": lambda a: len(a) == 0,
    "(Sequence) is not empty": lambda a: len(a) > 0,
}


def evaluate(values: dict, definition: dict) -> bool:
    parsed_definition = StatementGroup.model_validate(definition)
    eval_function = build_eval_function(parsed_definition)
    return eval_function(values)


def build_eval_function(
    definition: Union[BinaryStatement, UnaryStatement, StatementGroup],
    execution_context: str = "<root>",
) -> Callable[[T], bool]:
    if isinstance(definition, BinaryStatement):
        return build_binary_statement(definition, execution_context=execution_context)
    if isinstance(definition, UnaryStatement):
        return build_unary_statement(definition, execution_context=execution_context)
    statements_functions = []
    for statement_id, statement in enumerate(definition.statements):
        statement_execution_context = f"{execution_context}.statements[{statement_id}]"
        statements_functions.append(
            build_eval_function(
                statement, execution_context=statement_execution_context
            )
        )
    return partial(
        compound_eval,
        statements_functions=statements_functions,
        operator=definition.operator,
        execution_context=execution_context,
    )


def build_binary_statement(
    definition: BinaryStatement,
    execution_context: str,
) -> Callable[[Dict[str, T]], bool]:
    operator = BINARY_OPERATORS[definition.comparator.type]
    operator_parameters_names = [
        t for t in type(definition.comparator).model_fields if t != TYPE_PARAMETER_NAME
    ]
    operator_parameters = {
        a: getattr(definition.comparator, a) for a in operator_parameters_names
    }
    left_operand_builder = create_operand_builder(
        definition=definition.left_operand, execution_context=execution_context
    )
    right_operand_builder = create_operand_builder(
        definition=definition.right_operand, execution_context=execution_context
    )
    return partial(
        binary_eval,
        left_operand_builder=left_operand_builder,
        operator=operator,
        right_operand_builder=right_operand_builder,
        negate=definition.negate,
        operation_type=definition.type,
        execution_context=execution_context,
        operator_parameters=operator_parameters,
    )


def create_operand_builder(
    definition: Union[StaticOperand, DynamicOperand],
    execution_context: str,
) -> Callable[[Dict[str, T]], V]:
    if isinstance(definition, StaticOperand):
        return create_static_operand_builder(
            definition, execution_context=execution_context
        )
    return create_dynamic_operand_builder(
        definition, execution_context=execution_context
    )


def create_static_operand_builder(
    definition: StaticOperand,
    execution_context: str,
) -> Callable[[Dict[str, T]], V]:
    # local import to avoid circular dependency of modules with operations and evaluation
    from inference.core.workflows.core_steps.common.query_language.operations.core import (
        build_operations_chain,
    )

    operations_fun = build_operations_chain(
        operations=definition.operations,
        execution_context=f"{execution_context}.operations",
    )
    return partial(
        static_operand_builder,
        static_value=definition.value,
        operations_function=operations_fun,
    )


def static_operand_builder(
    values: dict,
    static_value: T,
    operations_function: Callable[[T, Dict[str, Any]], V],
) -> V:
    return operations_function(static_value, global_parameters=values)


def create_dynamic_operand_builder(
    definition: DynamicOperand,
    execution_context: str,
) -> Callable[[Dict[str, T]], V]:
    # local import to avoid circular dependency of modules with operations and evaluation
    from inference.core.workflows.core_steps.common.query_language.operations.core import (
        build_operations_chain,
    )

    operations_fun = build_operations_chain(
        operations=definition.operations,
        execution_context=f"{execution_context}.operations",
    )
    return partial(
        dynamic_operand_builder,
        operand_name=definition.operand_name,
        operations_function=operations_fun,
    )


def dynamic_operand_builder(
    values: [Dict[str, T]],
    operand_name: str,
    operations_function: Callable[[T, Dict[str, Any]], V],
) -> V:
    if operand_name not in values:
        raise UndeclaredSymbolError(
            public_message=f"Encountered undefined symbol `{operand_name}`",
            context="unknown",
        )
    return operations_function(values[operand_name], global_parameters=values)


def binary_eval(
    values: Dict[str, T],
    left_operand_builder: Callable[[Dict[str, T]], V],
    operator: Callable[[V, V, Any], bool],
    right_operand_builder: Callable[[Dict[str, T]], V],
    negate: bool,
    operation_type: str,
    execution_context: str,
    operator_parameters: Optional[Dict[str, Any]] = None,
) -> bool:
    if operator_parameters is None:
        operator_parameters = {}
    try:
        left_operand = left_operand_builder(values)
        right_operand = right_operand_builder(values)
        result = operator(left_operand, right_operand, **operator_parameters)
        if negate:
            result = not result
        return result
    except UndeclaredSymbolError as error:
        raise UndeclaredSymbolError(
            public_message=f"Attempted to execute evaluation of type: {operation_type} in context {execution_context}, "
            f"but encountered error: {error.public_message}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        ) from error
    except RoboflowQueryLanguageError as error:
        raise error
    except Exception as error:
        raise EvaluationEngineError(
            public_message=f"Attempted to execute evaluation of type: {operation_type} in context {execution_context}, "
            f"but encountered error: {error}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=error,
        ) from error


def build_unary_statement(
    definition: UnaryStatement, execution_context: str
) -> Callable[[Dict[str, T]], bool]:
    operator = UNARY_OPERATORS[definition.operator.type]
    operator_parameters_names = [
        t for t in type(definition.operator).model_fields if t != TYPE_PARAMETER_NAME
    ]
    operator_parameters = {
        a: getattr(definition.operator, a) for a in operator_parameters_names
    }
    operand_builder = create_operand_builder(
        definition=definition.operand, execution_context=execution_context
    )
    return partial(
        unary_eval,
        operand_builder=operand_builder,
        operator=operator,
        negate=definition.negate,
        operation_type=definition.type,
        execution_context=execution_context,
        operator_parameters=operator_parameters,
    )


def unary_eval(
    values: Dict[str, T],
    operand_builder: Callable[[Dict[str, T]], V],
    operator: Callable[[V, Any], bool],
    negate: bool,
    operation_type: str,
    execution_context: str,
    operator_parameters: Optional[Dict[str, Any]] = None,
) -> bool:
    if operator_parameters is None:
        operator_parameters = {}
    try:
        operand = operand_builder(values)
        result = operator(operand, **operator_parameters)
        if negate:
            result = not result
        return result
    except UndeclaredSymbolError as error:
        raise UndeclaredSymbolError(
            public_message=f"Attempted to execute evaluation of type: {operation_type} in context {execution_context}, "
            f"but encountered error: {error.public_message}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        ) from error
    except RoboflowQueryLanguageError as error:
        raise error
    except Exception as error:
        raise EvaluationEngineError(
            public_message=f"Attempted to execute evaluation of type: {operation_type} in context {execution_context}, "
            f"but encountered error: {error}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=error,
        ) from error


COMPOUND_EVAL_STATEMENTS_COMBINERS = {
    StatementsGroupsOperator.AND: lambda a, b: a and b,
    StatementsGroupsOperator.OR: lambda a, b: a or b,
}


def compound_eval(
    values: Dict[str, T],
    statements_functions: List[Callable[[Dict[str, T]], bool]],
    operator: StatementsGroupsOperator,
    execution_context: str,
) -> bool:
    if not statements_functions:
        raise EvaluationEngineError(
            public_message=f"Attempted to execute evaluation of statements in context of {execution_context}, "
            f"but empty statements list provided.",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    if operator not in COMPOUND_EVAL_STATEMENTS_COMBINERS:
        raise EvaluationEngineError(
            public_message=f"Attempted to execute evaluation of statements in context of {execution_context} "
            f"using operator: {operator} which is not registered.",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    operator_fun = COMPOUND_EVAL_STATEMENTS_COMBINERS[operator]
    result = statements_functions[0](values)
    for fun in statements_functions[1:]:
        fun_result = fun(values)
        result = operator_fun(result, fun_result)
    return result
