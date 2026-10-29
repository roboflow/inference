"""Return the result of the first switch case whose condition holds.

Canonical type: ``roboflow_core/expression@v1``.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import Field, ValidationInfo, field_validator
from roboflow_workflows.v2.blocks.formatters._shared.properties import (
    PROPERTY_OPERATIONS_EXAMPLE,
    CasesDefinition,
    CompiledExpression,
    compile_expression,
)
from roboflow_workflows.v2.blocks.query import Operation
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


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


class ExpressionBlock(Block):
    """Return the result of the first switch case whose condition holds.

    Example: ``data={"top_class": "$steps.label.output"}`` with
    ``SWITCH_EXAMPLE`` outputs ``"vehicle"`` for ``car`` and ``"other"`` for
    ``person``.
    """

    type = "roboflow_core/expression@v1"
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
        self._compiled: Optional[Tuple[Any, CompiledExpression]] = None

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
