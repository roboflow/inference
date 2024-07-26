from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Type, Union
import numpy as np
from sklearn.cluster import MiniBatchKMeans


import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.utils import str_to_color
from inference.core.workflows.entities.base import OutputDefinition, WorkflowImageData
from inference.core.workflows.entities.types import (
    BATCH_OF_STRING_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

TYPE: str = "DominantColor"
SHORT_DESCRIPTION: str = "Get the dominant color of an image."
LONG_DESCRIPTION: str = "Get the dominant color of an image as a list of RGB values."

class DominantColorManifest(WorkflowBlockManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
        }
    )

    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="rgb_color", kind=[BATCH_OF_STRING_KIND]),
        ]


class DominantColorBlock(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[DominantColorManifest]:
        return DominantColorManifest

    async def run(
        self,
        image: WorkflowImageData,
        *args,
        **kwargs
    ) -> BlockResult:

        np_image = image.numpy_image
        pixels = np_image.reshape(-1, 3).astype(np.float32)

        # Use MiniBatchKMeans for faster processing
        kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=1000)
        kmeans.fit(pixels)

        # Get the colors and their counts
        colors = kmeans.cluster_centers_
        counts = np.bincount(kmeans.labels_)

        # Find the most dominant color
        dominant_color = colors[np.argmax(counts)]
        rgb_color = np.round(dominant_color[::-1]).astype(int).tolist()

        return {"rgb_color": rgb_color}
