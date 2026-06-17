"""Tensor-native sibling of `roboflow_core/florence_2@v2`.

SCRATCH — first pass for review. v2 is a thin variant of v1 that only renames the
model selector field (`model_id` instead of `model_version`) and bumps the type
literal; all inference logic lives in v1. So the tensor sibling simply subclasses
the tensor-native `Florence2BlockV1` (from `v1_tensor`) and reuses the verbatim,
mode-agnostic `V2BlockManifest`.
"""

from typing import List, Optional, Type, Union

from inference.core.workflows.core_steps.models.foundation.florence2.v1 import (
    GroundingSelectionMode,
    TaskType,
)
from inference.core.workflows.core_steps.models.foundation.florence2.v1_tensor import (
    Florence2BlockV1,
    TensorNativeGrounding,
)
from inference.core.workflows.core_steps.models.foundation.florence2.v2 import (
    V2BlockManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest


class Florence2BlockV2(Florence2BlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return V2BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        task_type: TaskType,
        prompt: Optional[str],
        classes: Optional[List[str]],
        grounding_detection: Optional[
            Union[Batch[TensorNativeGrounding], List[int], List[float]]
        ],
        grounding_selection_mode: GroundingSelectionMode,
    ) -> BlockResult:
        return super().run(
            images=images,
            model_version=model_id,
            task_type=task_type,
            prompt=prompt,
            classes=classes,
            grounding_detection=grounding_detection,
            grounding_selection_mode=grounding_selection_mode,
        )
