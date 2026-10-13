"""Run a UQL operation chain on predictions: filter and transformation blocks.

Ports of V1 ``roboflow_core/detections_filter@v1`` and
``roboflow_core/detections_transformation@v1``. As in V1, both run the same
chain and differ only in name and description::

    predictions ──▶ step 1 ──▶ step 2 ──▶ ... ──▶ predictions (same carrier)

Allowed top-level steps keep a detections carrier: ``DetectionsFilter``,
``DetectionsSelection``, ``SortDetections``, ``DetectionsOffset``,
``DetectionsShift``, ``DetectionsRename`` and ``PickDetectionsByParentClass``.
Compilation rejects other steps, an invalid chain, an operand name missing
from ``operations_parameters`` and the reserved name ``_``.

Rows keep their masks, keypoints, row metadata and row record. Host work is
the query helper's: a filter condition runs in Python per row and copies each
box column it reads to the host once per call.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, model_validator
from roboflow_workflows.v2.blocks.query import (
    DETECTIONS_PRESERVING_OPERATIONS,
    CompiledOperations,
    Operation,
    compile_operations,
)
from roboflow_workflows.v2.blocks.query.schema import DEFAULT_OPERAND_NAME
from roboflow_workflows.v2.blocks.transforms.carriers import (
    PREDICTION_KINDS,
    carrier_name,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

OPERATIONS_EXAMPLE = [
    {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operations": [
                            {
                                "type": "ExtractDetectionProperty",
                                "property_name": "class_name",
                            }
                        ],
                    },
                    "comparator": {"type": "in (Sequence)"},
                    "right_operand": {
                        "type": "DynamicOperand",
                        "operand_name": "classes",
                    },
                }
            ],
        },
    }
]


class OperationsChainParams(BlockParams):
    """Parameters shared by the filter and transformation blocks."""

    predictions: Ref(*PREDICTION_KINDS) = Field(
        description=(
            "Detections, instance segmentation (dense, RLE or view), keypoint "
            "predictions or prediction rows."
        ),
        examples=["$steps.object_detection_model.predictions"],
    )
    operations: List[Operation] = Field(
        description=(
            "UQL steps applied in order. Each step must keep a detections "
            "carrier: DetectionsFilter, DetectionsSelection, SortDetections, "
            "DetectionsOffset, DetectionsShift, DetectionsRename or "
            "PickDetectionsByParentClass."
        ),
        examples=[OPERATIONS_EXAMPLE],
    )
    operations_parameters: Dict[str, Any | Ref()] = Field(
        default_factory=dict,
        description=(
            "Runtime values the steps name, e.g. {'classes': '$inputs.classes'}. "
            "The name '_' is reserved for the detection a filter evaluates."
        ),
        examples=[{"classes": "$inputs.classes"}],
    )

    @model_validator(mode="after")
    def _chain_compiles(self) -> "OperationsChainParams":
        compile_chain(
            self.operations, parameter_names=tuple(self.operations_parameters)
        )

        return self


class _OperationsChainBlock(Block):
    """Common run of the filter and transformation blocks."""

    outputs = {
        "predictions": Output(
            *PREDICTION_KINDS,
            source="predictions",
            description="The predictions after every step; same carrier type.",
        ),
    }
    Params = OperationsChainParams

    def __init__(self) -> None:
        self._compiled: Optional[Tuple[list, Tuple[str, ...], CompiledOperations]] = (
            None
        )

    def run(
        self,
        *,
        predictions: Any,
        operations: List[Any],
        operations_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply the operation chain to one prediction.

        Args:
            predictions: A carrier listed in ``transforms.carriers``.
            operations: Validated UQL steps.
            operations_parameters: Resolved runtime values by name.

        Returns:
            ``predictions``: the chain's result, same carrier type.

        Raises:
            QueryRuntimeError: When a step cannot process its input.
            ValueError: When the chain returns another carrier type.
        """
        chain = self._chain(operations, tuple(operations_parameters))
        result = chain.apply(predictions, parameters=operations_parameters)
        if carrier_name(result) != carrier_name(predictions):
            raise ValueError(
                f"{self.type}: the operations turned {carrier_name(predictions)} "
                f"into {carrier_name(result)}; the output must keep the input type"
            )

        return {"predictions": result}

    def _chain(
        self, operations: List[Any], parameter_names: Tuple[str, ...]
    ) -> CompiledOperations:
        # Compiled once per step; recompiled only if a graph update changes it.
        cached = self._compiled
        if cached is None or cached[0] != operations or cached[1] != parameter_names:
            chain = compile_chain(operations, parameter_names=parameter_names)
            cached = (list(operations), parameter_names, chain)
            self._compiled = cached

        return cached[2]


def compile_chain(
    operations: List[Any], *, parameter_names: Tuple[str, ...]
) -> CompiledOperations:
    """Validate and compile a detections-preserving operation chain.

    Args:
        operations: UQL steps, as JSON objects or validated models.
        parameter_names: Names declared in ``operations_parameters``.

    Returns:
        The compiled chain.

    Raises:
        QueryDefinitionError: On an invalid chain, an undeclared operand or
            a step that does not return detections.
        ValueError: When ``parameter_names`` contains ``_``.
    """
    if DEFAULT_OPERAND_NAME in parameter_names:
        raise ValueError(
            f"operations_parameters cannot use the name '{DEFAULT_OPERAND_NAME}': "
            "it is reserved for the detection a DetectionsFilter evaluates"
        )

    chain = compile_operations(
        operations, path="$.operations", declared_names=set(parameter_names)
    )
    for index, operation in enumerate(operations):
        operation_type = (
            operation["type"] if isinstance(operation, dict) else operation.type
        )
        if operation_type not in DETECTIONS_PRESERVING_OPERATIONS:
            raise ValueError(
                f"$.operations[{index}]: {operation_type} does not return "
                "detections; use only "
                f"{', '.join(sorted(DETECTIONS_PRESERVING_OPERATIONS))}"
            )

    return chain


class DetectionsFilterBlock(_OperationsChainBlock):
    """Keep the detections that satisfy a UQL condition.

    Example: with ``operations_parameters={"classes": ["car"]}`` and the
    ``OPERATIONS_EXAMPLE`` condition, rows ``car, person, car`` become
    ``car, car``, each with its own mask, keypoints and row identity.
    """

    type = "v2/detections_filter@v1"
    metadata = {
        "name": "Detections Filter",
        "port_of": "roboflow_core/detections_filter@v1",
        "section": "flow_control",
    }


class DetectionsTransformationBlock(_OperationsChainBlock):
    """Filter, sort, select, rename, shift or pad detections with UQL steps.

    Example: ``[{"type": "SortDetections", "mode": "confidence",
    "ascending": false}]`` orders rows by falling confidence; masks, keypoints
    and row identity follow their rows.
    """

    type = "v2/detections_transformation@v1"
    metadata = {
        "name": "Detections Transformation",
        "port_of": "roboflow_core/detections_transformation@v1",
        "section": "transformation",
    }
