"""Filter, sort, select, rename, shift or pad detections with UQL steps.

Canonical type: ``roboflow_core/detections_transformation@v1``.
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import Field, model_validator
from roboflow_workflows.v2.blocks.query import CompiledOperations, Operation
from roboflow_workflows.v2.blocks.transforms._shared.operations_chain import (
    OPERATIONS_EXAMPLE,
    compile_chain,
)
from roboflow_workflows.v2.blocks.transforms.carriers import (
    PREDICTION_KINDS,
    carrier_name,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class DetectionsTransformationBlock(Block):
    """Filter, sort, select, rename, shift or pad detections with UQL steps.

    Example: ``[{"type": "SortDetections", "mode": "confidence",
    "ascending": false}]`` orders rows by falling confidence; masks, keypoints
    and row identity follow their rows.
    """

    type = "roboflow_core/detections_transformation@v1"
    metadata = {
        "name": "Detections Transformation",
        "port_of": "roboflow_core/detections_transformation@v1",
        "section": "transformation",
    }
    outputs = {
        "predictions": Output(
            *PREDICTION_KINDS,
            source="predictions",
            description="The predictions after every step; same carrier type.",
        ),
    }

    class Params(BlockParams):
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
        def _chain_compiles(self) -> "DetectionsTransformationBlock.Params":
            compile_chain(
                self.operations, parameter_names=tuple(self.operations_parameters)
            )

            return self

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
