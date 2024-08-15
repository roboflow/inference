from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_IMAGES_KIND,
    INTEGER_KIND,
    NUMPY_ARRAY_KIND,
    STRING_KIND,
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

TYPE: str = "TemplateMatching"
SHORT_DESCRIPTION: str = "Apply Template Matching to an image."
LONG_DESCRIPTION: str = "Apply Template Matching to an image."


class TemplateMatchingManifest(WorkflowBlockManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Template Matching",
            "version": "v1",
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

    template: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Template Image",
        description="The template image for this step.",
        examples=["$inputs.template", "$steps.cropping.template"],
        validation_alias=AliasChoices("template", "templates"),
    )

    threshold: float = Field(
        title="Matching Threshold",
        description="The threshold value for template matching.",
        default=0.8,
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    BATCH_OF_IMAGES_KIND,
                ],
            ),
            OutputDefinition(
                name="num_matches",
                kind=[
                    INTEGER_KIND,
                ],
            ),
        ]


class TemplateMatchingBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[TemplateMatchingManifest]:
        return TemplateMatchingManifest

    def apply_template_matching(
        self, image: np.ndarray, template: np.ndarray, threshold: float = 0.8
    ) -> (np.ndarray, int):
        """
        Applies Template Matching to the image.
        Args:
            image: Input image.
            template: Template image.
            threshold: Matching threshold.
        Returns:
            np.ndarray: Template image with rectangles drawn around matched regions.
            int: Number of matched regions.
        """
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        w, h = template_gray.shape[::-1]

        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        num_matches = 0

        for pt in zip(*loc[::-1]):
            top_left = pt
            bottom_right = (pt[0] + w, pt[1] + h)
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
            num_matches += 1

        return image, num_matches

    def run(
        self,
        image: Union[WorkflowImageData, str],
        template: Union[WorkflowImageData, str],
        threshold: float = 0.8,
        *args,
        **kwargs,
    ) -> BlockResult:
        # Ensure inputs are WorkflowImageData objects

        # Apply Template Matching to the image
        template_with_matches, num_matches = self.apply_template_matching(
            image.numpy_image, template.numpy_image, threshold
        )

        output_image = WorkflowImageData(
            parent_metadata=template.parent_metadata,
            workflow_root_ancestor_metadata=template.workflow_root_ancestor_metadata,
            numpy_image=template_with_matches,
        )

        return {
            OUTPUT_IMAGE_KEY: output_image,  # Template image with matches
            "num_matches": num_matches,
        }
