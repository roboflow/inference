from typing import List, Literal, Optional, Type, Union

import numpy as np
from pydantic import AliasChoices, ConfigDict, Field
from skimage import exposure

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = "Apply contrast equalization to an image."
LONG_DESCRIPTION: str = """
Apply contrast equalization to an image
These are the same options provided for model preprocessing
"""


class ContrastEqualizationManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/contrast_equalization@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Contrast Equalization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-image",
                "blockPriority": 5,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    equalization_type: Union[
        Selector(kind=[STRING_KIND]),
        Literal[
            "Contrast Stretching", "Histogram Equalization", "Adaptive Equalization"
        ],
    ] = Field(
        default="Histogram Equalization",
        description="Type of contrast equalization to use.",
        examples=["Equalization", "$inputs.type"],
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


class ContrastEqualizationBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[ContrastEqualizationManifest]:
        return ContrastEqualizationManifest

    def run(
        self,
        image: WorkflowImageData,
        equalization_type: str,
    ) -> BlockResult:
        # Apply contrast equalization to the image
        updated_image = update_image(image.numpy_image, equalization_type)
        output = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=updated_image,
        )
        return {OUTPUT_IMAGE_KEY: output}


def update_image(img: np.ndarray, how: str):

    if how == "Contrast Stretching":
        # grab 2nd and 98 percentile
        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        # rescale
        img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
        return img_rescale

    elif how == "Histogram Equalization":
        img = img.astype(np.float32) / 255
        img_eq = exposure.equalize_hist(img) * 255
        return img_eq.astype(np.uint8)

    elif how == "Adaptive Equalization":
        img = img.astype(np.float32) / 255
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03) * 255
        return img_adapteq.astype(np.uint8)

    raise ValueError(f"contrast equalization type `{how}` not implemented!")
