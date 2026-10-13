"""Property extraction and switch-case blocks over V1 UQL JSON.

Ports of V1 ``roboflow_core/property_definition@v1`` and
``roboflow_core/expression@v1``. Both compile their UQL while the workflow
compiles; a run only applies the compiled chains::

    property_definition:  data ── operations ──▶ output

    expression:           data[name] ── data_operations[name] ──▶ variables
                          first case whose condition(variables) holds
                            StaticCaseResult  ──▶ output = value
                            DynamicCaseResult ──▶ output = operations(variables[name])
                          no case holds ──▶ the default result

Every name a case uses (operand, filter parameter, ``parameter_name``) and
every ``data_operations`` key must be a ``data`` key; anything else fails
compilation. As in V1, ``data_operations`` chains name no runtime values.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from roboflow_workflows.v2.blocks.query import (
    CompiledCondition,
    CompiledOperations,
    Operation,
    StatementGroup,
    compile_condition,
    compile_operations,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

PROPERTY_OPERATIONS_EXAMPLE = [
    {"type": "ClassificationPropertyExtract", "property_name": "top_class_confidence"}
]


class StaticCaseResult(BaseModel):
    """A fixed result value."""

    type: Literal["StaticCaseResult"] = Field(description="Result kind.")
    value: Any = Field(description="The output value.", examples=["confident"])


class DynamicCaseResult(BaseModel):
    """A ``data`` variable, optionally transformed by operations."""

    type: Literal["DynamicCaseResult"] = Field(description="Result kind.")
    parameter_name: str = Field(
        description="Name of the variable to output.", examples=["prediction"]
    )
    operations: Optional[List[Operation]] = Field(
        default=None,
        description=(
            "Operations applied to the variable; the other variables are their "
            "parameters."
        ),
        examples=[PROPERTY_OPERATIONS_EXAMPLE],
    )


CaseResult = Annotated[
    Union[StaticCaseResult, DynamicCaseResult], Field(discriminator="type")
]


class CaseDefinition(BaseModel):
    """One case: its result applies when its condition holds."""

    type: Literal["CaseDefinition"] = Field(description="Node kind.")
    condition: StatementGroup = Field(description="Condition over the variables.")
    result: CaseResult = Field(description="Output when the condition holds.")


class CasesDefinition(BaseModel):
    """Ordered cases and the default result."""

    type: Literal["CasesDefinition"] = Field(description="Node kind.")
    cases: List[CaseDefinition] = Field(
        description="Cases tried in order; the first that holds wins."
    )
    default: CaseResult = Field(description="Output when no case holds.")


SWITCH_EXAMPLE = {
    "type": "CasesDefinition",
    "cases": [
        {
            "type": "CaseDefinition",
            "condition": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "BinaryStatement",
                        "left_operand": {
                            "type": "DynamicOperand",
                            "operand_name": "top_class",
                        },
                        "comparator": {"type": "in (Sequence)"},
                        "right_operand": {
                            "type": "StaticOperand",
                            "value": ["car", "truck"],
                        },
                    }
                ],
            },
            "result": {"type": "StaticCaseResult", "value": "vehicle"},
        }
    ],
    "default": {"type": "StaticCaseResult", "value": "other"},
}


class PropertyDefinitionBlock(Block):
    """Extract a value with a UQL operation chain.

    Example: ``[{"type": "ClassificationPropertyExtract", "property_name":
    "top_class_confidence"}]`` turns a single-label classification whose top
    class scored 0.92 into ``0.92``.
    """

    type = "v2/property_definition@v1"
    metadata = {
        "name": "Property Definition",
        "port_of": "roboflow_core/property_definition@v1",
        "section": "advanced",
    }
    outputs = {
        "output": Output(
            source="data", description="Result of the last operation, any kind."
        ),
    }

    class Params(BlockParams):
        data: Ref() = Field(
            description="Value the operations start from, of any kind.",
            examples=["$steps.classifier.predictions"],
        )
        operations: List[Operation] = Field(
            description="UQL operations applied in order; they name no runtime values.",
            examples=[PROPERTY_OPERATIONS_EXAMPLE],
        )

        # Not a model validator: the engine reruns those per invocation.
        @field_validator("operations")
        @classmethod
        def _operations_compile(cls, value: List[Operation]) -> List[Operation]:
            compile_property_operations(value)

            return value

    def __init__(self) -> None:
        self._compiled: Optional[Tuple[list, CompiledOperations]] = None

    def run(self, *, data: Any, operations: List[Any]) -> Dict[str, Any]:
        """Apply the operations to ``data``.

        Args:
            data: Resolved input value.
            operations: Validated UQL operations.

        Returns:
            ``output``: the chain's result.

        Raises:
            QueryRuntimeError: When an operation cannot process its input.
        """
        cached = self._compiled
        if cached is None or cached[0] != operations:
            cached = (list(operations), compile_property_operations(operations))
            self._compiled = cached
        result = cached[1].apply(data, parameters={})

        return {"output": result}


def compile_property_operations(operations: List[Any]) -> CompiledOperations:
    """Compile a property chain that names no runtime values.

    Args:
        operations: UQL operations, as JSON objects or validated models.

    Returns:
        The compiled chain.

    Raises:
        QueryDefinitionError: On an invalid chain or a nested filter naming a
            runtime value.
    """
    chain = compile_operations(
        operations, path="$.operations", declared_names=frozenset()
    )

    return chain


class ExpressionBlock(Block):
    """Return the result of the first switch case whose condition holds.

    Example: ``data={"top_class": "$steps.label.output"}`` with
    ``SWITCH_EXAMPLE`` outputs ``"vehicle"`` for ``car`` and ``"other"`` for
    ``person``.
    """

    type = "v2/expression@v1"
    metadata = {
        "name": "Expression",
        "port_of": "roboflow_core/expression@v1",
        "section": "advanced",
    }
    outputs = {
        "output": Output(description="The selected case's result, any kind."),
    }

    class Params(BlockParams):
        data: Dict[str, Any | Ref()] = Field(
            description=(
                "Named variables for conditions and results, e.g. "
                "{'prediction': '$steps.classifier.predictions'}."
            ),
            examples=[{"prediction": "$steps.classifier.predictions"}],
        )
        data_operations: Dict[str, List[Operation]] = Field(
            default_factory=dict,
            description=(
                "Operations that transform a variable before the cases run, by "
                "variable name. Every key must be a data key; the operations "
                "name no runtime values."
            ),
            examples=[{"prediction": PROPERTY_OPERATIONS_EXAMPLE}],
        )
        switch: CasesDefinition = Field(
            description="Ordered cases with conditions and results, plus a default.",
            examples=[SWITCH_EXAMPLE],
        )

        # Not a model validator: the engine reruns those per invocation.
        # This literal-only field is validated once per definition, and
        # selectors replace data values, never keys.
        @field_validator("switch")
        @classmethod
        def _switch_compiles(
            cls, value: CasesDefinition, info: ValidationInfo
        ) -> CasesDefinition:
            if "data" in info.data and "data_operations" in info.data:
                compile_expression(
                    info.data["data_operations"],
                    value,
                    variable_names=tuple(info.data["data"]),
                )

            return value

    def __init__(self) -> None:
        self._compiled: Optional[Tuple[Any, "CompiledExpression"]] = None

    def run(
        self,
        *,
        data: Dict[str, Any],
        data_operations: Dict[str, List[Any]],
        switch: CasesDefinition,
    ) -> Dict[str, Any]:
        """Transform the variables and return the first matching case's result.

        Args:
            data: Resolved variables by name.
            data_operations: Validated operations by variable name.
            switch: Validated cases and default.

        Returns:
            ``output``: the selected result.

        Raises:
            QueryRuntimeError: When an operation or condition cannot process
                its values.
        """
        key = (tuple(data), data_operations, switch)
        cached = self._compiled
        if cached is None or cached[0] != key:
            expression = compile_expression(
                data_operations, switch, variable_names=tuple(data)
            )
            cached = (key, expression)
            self._compiled = cached
        result = cached[1].evaluate(data)

        return {"output": result}


class CompiledExpression:
    """Compiled ``data_operations`` and switch of one expression step."""

    def __init__(
        self,
        *,
        variable_operations: Dict[str, CompiledOperations],
        cases: List[Tuple[CompiledCondition, "_CompiledResult"]],
        default: "_CompiledResult",
    ) -> None:
        self._variable_operations = variable_operations
        self._cases = cases
        self._default = default

    def evaluate(self, data: Dict[str, Any]) -> Any:
        """Return the result of the first case that holds, else the default.

        Args:
            data: Variables by name.

        Returns:
            The selected result value.

        Raises:
            QueryRuntimeError: When an operation or condition fails.
        """
        variables = dict(data)
        for name, operations in self._variable_operations.items():
            # As in V1, these chains see no runtime names.
            variables[name] = operations.apply(variables[name], parameters={})
        for condition, result in self._cases:
            if condition.evaluate(variables):
                return result.build(variables)

        default_value = self._default.build(variables)

        return default_value


def compile_expression(
    data_operations: Dict[str, List[Any]],
    switch: CasesDefinition,
    *,
    variable_names: Tuple[str, ...],
) -> CompiledExpression:
    """Validate and compile an expression's operations and cases.

    Args:
        data_operations: Operations by variable name.
        switch: Validated cases and default.
        variable_names: Keys of ``data``.

    Returns:
        The compiled expression.

    Raises:
        QueryDefinitionError: On an invalid definition or a name that is not
            a ``data`` key.
        ValueError: When a ``data_operations`` key or a ``parameter_name`` is
            not a ``data`` key.
    """
    declared = frozenset(variable_names)
    variable_operations = {}
    for name, operations in data_operations.items():
        _require_variable(name, declared=declared, path=f"$.data_operations.{name}")
        variable_operations[name] = compile_operations(
            operations, path=f"$.data_operations.{name}", declared_names=frozenset()
        )
    cases = [
        (
            compile_condition(
                case.condition,
                path=f"$.switch.cases[{index}].condition",
                declared_names=declared,
            ),
            _CompiledResult(
                case.result,
                declared=declared,
                path=f"$.switch.cases[{index}].result",
            ),
        )
        for index, case in enumerate(switch.cases)
    ]
    default = _CompiledResult(
        switch.default, declared=declared, path="$.switch.default"
    )
    expression = CompiledExpression(
        variable_operations=variable_operations, cases=cases, default=default
    )

    return expression


class _CompiledResult:
    def __init__(self, result: Any, *, declared: frozenset, path: str) -> None:
        self._static = isinstance(result, StaticCaseResult)
        self._value = result.value if self._static else None
        self._name = None if self._static else result.parameter_name
        self._operations = None
        if self._static:
            return

        _require_variable(self._name, declared=declared, path=f"{path}.parameter_name")
        if result.operations:
            self._operations = compile_operations(
                result.operations, path=f"{path}.operations", declared_names=declared
            )

    def build(self, variables: Dict[str, Any]) -> Any:
        if self._static:
            return self._value

        value = variables[self._name]
        if self._operations is not None:
            value = self._operations.apply(value, parameters=variables)

        return value


def _require_variable(name: str, *, declared: frozenset, path: str) -> None:
    if name not in declared:
        raise ValueError(
            f"{path}: `{name}` is not a data variable (data: {sorted(declared)})"
        )
