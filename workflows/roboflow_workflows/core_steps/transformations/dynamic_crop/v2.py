from typing import List, Literal, Type

from pydantic import ConfigDict
from roboflow_workflows.core_steps.transformations.dynamic_crop.v1 import (
    BlockManifest as V1BlockManifest,
)
from roboflow_workflows.core_steps.transformations.dynamic_crop.v1 import (
    DynamicCropBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    DependentResource,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = (
    V1BlockManifest.model_config["json_schema_extra"]["long_description"]
    + "\n\nEach crop also returns its source image and the selected prediction "
    "in that image's coordinates. Use this pair to keep the original alongside a crop."
)


class BlockManifest(V1BlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            **V1BlockManifest.model_config["json_schema_extra"],
            "version": "v2",
            "long_description": LONG_DESCRIPTION,
        }
    )
    type: Literal["roboflow_core/dynamic_crop@v2"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        crop, prediction = super().describe_outputs()
        return [
            crop,
            prediction,
            OutputDefinition(name="source_image", kind=crop.kind),
            OutputDefinition(name="source_prediction", kind=prediction.kind),
        ]

    def discover_work_operations(self) -> List[WorkOperation]:
        return [WorkOperation.IMAGE_CROP]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        return []


class DynamicCropBlockV2(DynamicCropBlockV1):
    # switches on the `source_image` / `source_prediction` keys in `crop_image`
    include_source = True

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest
