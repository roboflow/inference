from typing import Dict, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    PredictionsVisualizationBlock,
    PredictionsVisualizationManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    VIDEO_METADATA_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/heatmap_visualization@v1"
SHORT_DESCRIPTION = "Draw a heatmap based on detections in an image."
LONG_DESCRIPTION = """
Draw heatmaps on an image based on provided detections. Heat accumulates over time and is drawn as a semi-transparent overlay of blurred circles.

## How This Block Works

This block takes an image and detection predictions and draws a heatmap. The block:

1. Takes an image and predictions as input.
2. Accumulates heat based on the position of detections.
3. Draws a semi-transparent overlay of blurred circles representing the heat.

## Common Use Cases

- **Density Analysis**: Visualize the density of objects in a scene.
- **Traffic Monitoring**: Identify high-traffic areas.
- **Retail Analytics**: Analyze foot traffic in stores.
"""


class HeatmapManifest(PredictionsVisualizationManifest):
    type: Literal[f"{TYPE}", "HeatmapVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Heatmap Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator", "heatmap"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "fas fa-fire",
                "blockPriority": 4,
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

    metadata: Selector(kind=[VIDEO_METADATA_KIND]) = Field(
        description="Video metadata containing video_identifier to maintain separate state for different videos.",
        default=None,
        examples=["$inputs.video_metadata"],
    )

    position: Union[
        Literal[
            "CENTER",
            "CENTER_LEFT",
            "CENTER_RIGHT",
            "TOP_CENTER",
            "TOP_LEFT",
            "TOP_RIGHT",
            "BOTTOM_CENTER",
            "BOTTOM_LEFT",
            "BOTTOM_RIGHT",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="BOTTOM_CENTER",
        description="The position of the heatmap relative to the detection.",
        examples=["BOTTOM_CENTER", "$inputs.position"],
    )

    opacity: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        description="Opacity of the overlay mask, between 0 and 1.",
        default=0.2,
        examples=[0.2, "$inputs.opacity"],
    )

    radius: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Radius of the heat circle.",
        default=40,
        examples=[40, "$inputs.radius"],
    )

    kernel_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Kernel size for blurring the heatmap.",
        default=25,
        examples=[25, "$inputs.kernel_size"],
    )

    top_hue: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Hue at the top of the heatmap. Defaults to 0 (red).",
        default=0,
        examples=[0, "$inputs.top_hue"],
    )

    low_hue: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Hue at the bottom of the heatmap. Defaults to 125 (blue).",
        default=125,
        examples=[125, "$inputs.low_hue"],
    )

    ignore_stationary: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description="If True, only moving objects (based on tracker ID) will contribute to the heatmap.",
        default=True,
        examples=[True, "$inputs.ignore_stationary"],
    )

    motion_threshold: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Minimum movement in pixels required to consider an object as moving.",
        default=25,
        examples=[25, "$inputs.motion_threshold"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class HeatmapVisualizationBlockV1(PredictionsVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}
        # Dictionary to store track history: {video_id: {tracker_id: (x, y)}}
        self._track_history: Dict[str, Dict[int, tuple]] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return HeatmapManifest

    def getAnnotator(
        self,
        position: str,
        opacity: float,
        radius: int,
        kernel_size: int,
        top_hue: int,
        low_hue: int,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    position,
                    opacity,
                    radius,
                    kernel_size,
                    top_hue,
                    low_hue,
                ],
            )
        )

        if key not in self.annotatorCache:
            position_enum = getattr(sv.Position, position)
            self.annotatorCache[key] = sv.HeatMapAnnotator(
                position=position_enum,
                opacity=opacity,
                radius=radius,
                kernel_size=kernel_size,
                top_hue=top_hue,
                low_hue=low_hue,
            )

        return self.annotatorCache[key]

    def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        position: Optional[str],
        opacity: Optional[float],
        radius: Optional[int],
        kernel_size: Optional[int],
        top_hue: Optional[int],
        low_hue: Optional[int],
        metadata: Optional[VideoMetadata] = None,
        ignore_stationary: bool = True,
        motion_threshold: int = 25,
    ) -> BlockResult:
        detections_to_plot = predictions

        if ignore_stationary and predictions.tracker_id is not None:
            video_id = metadata.video_identifier if metadata else "default_video"
            if video_id not in self._track_history:
                self._track_history[video_id] = {}

            current_history = self._track_history[video_id]
            moving_indices = []

            # Calculate centers for current detections
            # Use the specified position anchor for tracking consistency
            anchor_position = getattr(sv.Position, position) if position else sv.Position.BOTTOM_CENTER
            anchors = predictions.get_anchors_coordinates(anchor=anchor_position)

            for i, (tracker_id, point) in enumerate(zip(predictions.tracker_id, anchors)):
                tracker_id = int(tracker_id)
                x, y = point
                
                if tracker_id in current_history:
                    # Check for movement
                    prev_x, prev_y = current_history[tracker_id]
                    dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                    
                    if dist >= motion_threshold:
                        moving_indices.append(i)
                        # Update history
                        current_history[tracker_id] = (x, y)
                else:
                    # New track, initialize history
                    current_history[tracker_id] = (x, y)
            
            # Filter detections
            if len(moving_indices) > 0:
                detections_to_plot = predictions[np.array(moving_indices)]
            else:
                detections_to_plot = sv.Detections.empty()
        
        annotator = self.getAnnotator(
            position,
            opacity,
            radius,
            kernel_size,
            top_hue,
            low_hue,
        )
        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=detections_to_plot,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
