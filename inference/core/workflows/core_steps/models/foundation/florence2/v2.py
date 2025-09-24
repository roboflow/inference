from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.models.foundation.florence2.v1 import (
    LONG_DESCRIPTION,
    BaseManifest,
    Florence2BlockV1,
    GroundingSelectionMode,
    TaskType,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    ROBOFLOW_MODEL_ID_KIND,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest


class V2BlockManifest(BaseManifest):
    type: Literal["roboflow_core/florence_2@v2"]
    model_id: Union[WorkflowParameterSelector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = (
        Field(
            default="florence-2-base",
            description="Model to be used",
            examples=["florence-2-base"],
            json_schema_extra={"always_visible": True},
        )
    )
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Florence-2 Model",
            "version": "v2",
            "short_description": "Run Florence-2 on an image",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["Florence", "Florence-2", "Microsoft"],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 5.5,
            },
        },
        protected_namespaces=(),
    )


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
            Union[Batch[sv.Detections], List[int], List[float]]
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
