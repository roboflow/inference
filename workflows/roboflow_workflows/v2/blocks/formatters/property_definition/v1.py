"""Extract a value with a UQL operation chain.

Canonical type: ``roboflow_core/property_definition@v1``.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import Field, field_validator
from roboflow_workflows.v2.blocks.formatters._shared.properties import (
    PROPERTY_OPERATIONS_EXAMPLE,
    compile_property_operations,
)
from roboflow_workflows.v2.blocks.query import CompiledOperations, Operation
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class PropertyDefinitionBlock(Block):
    """Extract a value with a UQL operation chain.

    Example: ``[{"type": "ClassificationPropertyExtract", "property_name":
    "top_class_confidence"}]`` turns a single-label classification whose top
    class scored 0.92 into ``0.92``.
    """

    type = "roboflow_core/property_definition@v1"
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
