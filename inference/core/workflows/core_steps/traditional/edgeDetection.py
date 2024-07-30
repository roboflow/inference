## Required Libraries:
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

#### Traditional CV (Opencv) Import 
import cv2
####

from inference.core.workflows.core_steps.visualizations.utils import str_to_color
# TODO: Is this kosher?
from inference.core.workflows.core_steps.visualizations.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.entities.base import OutputDefinition, WorkflowImageData, Batch
from inference.core.workflows.entities.types import (
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_IMAGES_KIND,
    STRING_KIND,
    INTEGER_KIND,
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

TYPE: str = "ImageEdgeDetection"
SHORT_DESCRIPTION: str = "Apply a blur to an image."
LONG_DESCRIPTION: str = "Apply a blur to an image."

class EdgeDetectionManifest(WorkflowBlockManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "traditional",
        }
    )

    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    edge_type: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]), Literal["sobel", "laplacian", "canny"]
    ] = Field(
        description="Type of Edge Detection to perform.", examples=["canny", "$inputs.edge_type"]
    )

    kernel_size: Union[WorkflowParameterSelector(kind=[INTEGER_KIND]), int] = Field(
        description="(sobel) Kernel size for the edge detection filter.",
        examples=[5, "$inputs.kernel_size"],
    )

    threshold1: Union[WorkflowParameterSelector(kind=[INTEGER_KIND]), int] = Field(
        description="(canny) First threshold for the Canny edge detector.",
        examples=[100, "$inputs.threshold1"],
    )

    threshold2: Union[WorkflowParameterSelector(kind=[INTEGER_KIND]), int] = Field(
        description="(canny) Second threshold for the Canny edge detector.",
        examples=[200, "$inputs.threshold2"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    BATCH_OF_IMAGES_KIND,
                ],
            ),
        ]


class EdgeDetectionBlock(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[EdgeDetectionManifest]:
        return EdgeDetectionManifest
    
    # TODO: Fix image type
    def apply_edge_detection(self, image, edge_type: str, ksize: int = 5, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
        """
        Applies the specified edge detection to the image.

        Args:
            image: Input image.
            edge_type (str): Type of edge detection ('sobel', 'laplacian', 'canny').
            threshold1 (int, optional): First threshold for the Canny edge detector. Defaults to 100.
            threshold2 (int, optional): Second threshold for the Canny edge detector. Defaults to 200.

        Returns:
            np.ndarray: Image with edges detected.
        """

        if edge_type == 'sobel':
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
            edges = cv2.magnitude(sobelx, sobely)
        elif edge_type == 'laplacian':
            edges = cv2.Laplacian(image, cv2.CV_64F)
        elif edge_type == 'canny':
            edges = cv2.Canny(image, threshold1, threshold2, ksize)
        else:
            raise ValueError(f"Unknown edge type: {edge_type}")

        return edges

    # TODO: Check this is all good and robust.
    async def run(self, image: WorkflowImageData, edge_type: str, kernel_size: int, threshold1: int, threshold2: int, *args, **kwargs) -> BlockResult:
        # Apply blur to the image
        # TODO: Edit so we don't have to have valid values in UI if not relevant to edge detection type.
        edges_image = self.apply_edge_detection(image.numpy_image, edge_type, kernel_size, threshold1, threshold2)

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=edges_image,
        )

        return {OUTPUT_IMAGE_KEY: output}