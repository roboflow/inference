from typing import Any, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field, field_validator
from supervision.annotators.base import BaseAnnotator

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.core_steps.visualizations.common.base_colorable import (
    ColorableVisualizationBlock,
    ColorableVisualizationManifest,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

SHORT_DESCRIPTION = "Draw traces based on detections tracking results."
LONG_DESCRIPTION = """
The `TraceVisualization` block draws tracker results on an image using Supervision's `sv.TraceAnnotator`.
"""


class TraceManifest(ColorableVisualizationManifest):
    type: Literal["roboflow_core/trace_visualization@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Trace Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-scribble",
                "blockPriority": 17,
                "supervision": True,
                "warnings": [
                    {
                        "property": "copy_image",
                        "value": False,
                        "message": "This setting will mutate its input image. If the input is used by other blocks, it may cause unexpected behavior.",
                    }
                ],
            },
        }
    )

    position: Union[
        Literal[
            "CENTER",
            "CENTER_LEFT",
            "CENTER_RIGHT",
            "TOP_CENTER",
            "TOP_LEFT",
            "TOP_RIGHT",
            "BOTTOM_LEFT",
            "BOTTOM_CENTER",
            "BOTTOM_RIGHT",
            "CENTER_OF_MASS",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="CENTER",
        description="The anchor position for placing the label.",
        examples=["CENTER", "$inputs.text_position"],
    )
    trace_length: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=30,
        description="Maximum number of historical tracked objects positions to display.",
        examples=[30, "$inputs.trace_length"],
    )
    thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the track visualization line.",
        default=1,
        examples=[1, "$inputs.track_thickness"],
    )

    @field_validator("trace_length", "thickness")
    @classmethod
    def ensure_max_entries_per_file_is_correct(cls, value: Any) -> Any:
        if isinstance(value, int) and value < 1:
            raise ValueError("`trace_length` and `thickness` cannot be lower than 1.")
        return value

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class TraceVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TraceManifest

    def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        color_palette: Optional[str],
        palette_size: Optional[int],
        custom_colors: Optional[List[str]],
        color_axis: Optional[str],
        position: str,
        trace_length: int,
        thickness: int,
    ) -> BlockResult:
        if predictions.tracker_id is None:
            raise ValueError(
                "Expected tracked predictions in `roboflow_core/trace_visualization@v1` block."
            )
        annotator = self.getAnnotator(
            color_palette=color_palette,
            palette_size=palette_size,
            custom_colors=custom_colors,
            color_axis=color_axis,
            position=position,
            trace_length=trace_length,
            thickness=thickness,
        )
        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        position: str,
        trace_length: int,
        thickness: int,
    ) -> BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_palette,
                    palette_size,
                    color_axis,
                    position,
                    trace_length,
                    thickness,
                ],
            )
        )
        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)
            self.annotatorCache[key] = sv.TraceAnnotator(
                color=palette,
                position=getattr(sv.Position, position),
                trace_length=trace_length,
                thickness=thickness,
                color_lookup=getattr(sv.ColorLookup, color_axis),
            )
        return self.annotatorCache[key]
