from typing import List, Literal, Optional, Type, Union

import cv2
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = "Convert an RGB image to grayscale."
LONG_DESCRIPTION: str = """
Block to convert an RGB image to grayscale. The output image will have only one channel.
"""


class ConvertGrayscaleManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/convert_grayscale@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Image Convert Grayscale",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-palette",
                "blockPriority": 7,
                "opencv": True,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    IMAGE_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ConvertGrayscaleBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ConvertGrayscaleManifest]:
        return ConvertGrayscaleManifest

    def run(
        self,
        image: WorkflowImageData,
        *args,
        **kwargs,
    ) -> BlockResult:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image.numpy_image, cv2.COLOR_BGR2GRAY)
        output = WorkflowImageData.copy_and_replace(
            origin_image_data=image, numpy_image=gray
        )
        return {OUTPUT_IMAGE_KEY: output}
