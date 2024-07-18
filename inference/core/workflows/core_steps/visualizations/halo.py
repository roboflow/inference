from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.base import (
    VisualizationBlock,
    VisualizationManifest,
)
from inference.core.workflows.entities.base import WorkflowImageData
from inference.core.workflows.entities.types import (
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    FloatZeroToOne,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

OUTPUT_IMAGE_KEY: str = "image"

TYPE: str = "HaloVisualization"
SHORT_DESCRIPTION = "Paints a halo around detected objects in an image."
LONG_DESCRIPTION = """
The `HaloVisualization` block uses a detected polygon
from an instance segmentation to draw a halo using
`sv.HaloAnnotator`.
"""


class HaloManifest(VisualizationManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
        }
    )

    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions",
        examples=["$steps.instance_segmentation_model.predictions"],
    )

    opacity: Union[FloatZeroToOne, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Transparency of the halo overlay.",
        default=0.8,
        examples=[0.8, "$inputs.opacity"],
    )

    kernel_size: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Size of the average pooling kernel used for creating the halo.",
        default=40,
        examples=[40, "$inputs.kernel_size"],
    )


class HaloVisualizationBlock(VisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return HaloManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        opacity: float,
        kernel_size: int,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_palette,
                    palette_size,
                    color_axis,
                    opacity,
                    kernel_size,
                ],
            )
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            self.annotatorCache[key] = sv.HaloAnnotator(
                color=palette,
                color_lookup=getattr(sv.ColorLookup, color_axis),
                opacity=opacity,
            )

        return self.annotatorCache[key]

    async def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        color_palette: Optional[str],
        palette_size: Optional[int],
        custom_colors: Optional[List[str]],
        color_axis: Optional[str],
        opacity: Optional[float],
        kernel_size: Optional[int],
    ) -> BlockResult:
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            opacity,
            kernel_size,
        )

        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions,
        )

        output = WorkflowImageData(
            parent_metadata=image.parent_metadata,
            workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
            numpy_image=annotated_image,
        )

        return {OUTPUT_IMAGE_KEY: output}
