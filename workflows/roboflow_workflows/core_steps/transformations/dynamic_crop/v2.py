from typing import List, Literal, Type

from pydantic import ConfigDict
from roboflow_workflows.environment import ENABLE_TENSOR_DATA_REPRESENTATION
from roboflow_workflows.execution_engine.entities.base import OutputDefinition

if ENABLE_TENSOR_DATA_REPRESENTATION:
    from roboflow_workflows.core_steps.transformations.dynamic_crop.v1_tensor import (
        BlockManifest as CropManifestV1,
    )
    from roboflow_workflows.core_steps.transformations.dynamic_crop.v1_tensor import (
        DynamicCropBlockV1,
    )
else:
    from roboflow_workflows.core_steps.transformations.dynamic_crop.v1 import (
        BlockManifest as CropManifestV1,
    )
    from roboflow_workflows.core_steps.transformations.dynamic_crop.v1 import (
        DynamicCropBlockV1,
    )


class BlockManifest(CropManifestV1):
    type: Literal["roboflow_core/dynamic_crop@v2"]
    model_config = ConfigDict(
        json_schema_extra={
            **CropManifestV1.model_config["json_schema_extra"],
            "version": "v2",
            "long_description": CropManifestV1.model_config["json_schema_extra"][
                "long_description"
            ]
            + "\n\nEach crop also returns its source image and the selected prediction "
            "in that image's coordinates. Use this pair to keep the original alongside a crop.",
        }
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        crop, prediction = super().describe_outputs()
        return [
            crop,
            prediction,
            OutputDefinition(name="source_image", kind=crop.kind),
            OutputDefinition(name="source_prediction", kind=prediction.kind),
        ]


class DynamicCropBlockV2(DynamicCropBlockV1):
    include_source = True

    @classmethod
    def get_manifest(cls) -> Type[BlockManifest]:
        return BlockManifest
