from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.core_steps.visualizations.common.base_colorable import (
    ColorableVisualizationBlock,
    ColorableVisualizationManifest,
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.execution_engine.constants import (
    AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
    AREA_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/label_visualization@v1"
SHORT_DESCRIPTION = (
    "Draw labels on an image at specific coordinates based on provided detections."
)
LONG_DESCRIPTION = """
Draw text labels on detected objects with customizable content, position, styling, and background colors to display information like class names, confidence scores, tracking IDs, or other detection metadata.

## How This Block Works

This block takes an image and detection predictions and draws text labels on each detected object. The block:

1. Takes an image and predictions as input
2. Extracts label text for each detection based on the selected text option (class name, confidence, tracker ID, dimensions, area, time in zone, or index)
3. Determines label position based on the selected anchor point (center, corners, edges, or center of mass)
4. Applies background color styling based on the selected color palette, with colors assigned by class, index, or track ID
5. Renders text labels with customizable text color, scale, thickness, padding, and border radius using Supervision's LabelAnnotator
6. Returns an annotated image with text labels overlaid on the original image

The block supports various text content options including class names, confidence scores, combination of class and confidence, tracker IDs (for tracked objects), time in zone (for zone analysis), object dimensions (center coordinates and width/height), area, or detection index. Labels are rendered with colored backgrounds that match the object's assigned color from the palette, and text styling (color, size, thickness) can be customized for optimal visibility. The labels can be positioned at any anchor point relative to each detection, allowing flexible placement for different visualization needs.

## Common Use Cases

- **Information Display on Detections**: Add informative text labels showing class names, confidence scores, or other metadata directly on detected objects for quick identification and validation
- **Model Performance Visualization**: Display confidence scores or class predictions on detected objects to visualize model certainty, identify low-confidence detections, and validate model performance
- **Object Tracking Visualization**: Show tracker IDs on tracked objects to visualize object tracking across frames, monitor persistent object identities, or debug tracking algorithms
- **Zone Analysis and Monitoring**: Display "Time In Zone" labels on objects to visualize how long objects have been in specific zones for occupancy monitoring, dwell time analysis, or compliance tracking
- **Spatial Information Display**: Show object dimensions (center coordinates, width, height) or area measurements directly on detections for spatial analysis, measurement workflows, or quality control
- **Professional Presentation and Reporting**: Create clean, informative visualizations with labeled detections for reports, dashboards, or presentations that combine visual results with textual information

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Bounding Box Visualization, Polygon Visualization, Dot Visualization) to combine text labels with geometric annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with labels for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with labels to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with labels as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with labels for live monitoring, tracking visualization, or post-processing analysis
"""


class LabelManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}", "LabelVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Label Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-tag",
                "blockPriority": 2,
                "popular": True,
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

    text: Union[
        Literal[
            "Class",
            "Confidence",
            "Class and Confidence",
            "Index",
            "Dimensions",
            "Area",
            "Area (mask)",
            "Area (converted)",
            "Tracker Id",
            "Time In Zone",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="Class",
        description="Content to display in text labels. Options: 'Class' (class name), 'Confidence' (confidence score), 'Class and Confidence' (both), 'Tracker Id' (tracking ID for tracked objects), 'Time In Zone' (time spent in zone), 'Dimensions' (center coordinates and width x height), 'Area' (bounding box area in pixels), 'Area (mask)' (mask area in pixels from Mask Area Measurement block), 'Area (converted)' (mask area in converted units from Mask Area Measurement block), or 'Index' (detection index).",
        examples=["LABEL", "$inputs.text"],
        json_schema_extra={
            "always_visible": True,
        },
    )

    text_position: Union[
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
        default="TOP_LEFT",
        description="Anchor position for placing labels relative to each detection's bounding box. Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object).",
        examples=["CENTER", "$inputs.text_position"],
    )

    text_color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the label text. Can be a color name (e.g., 'WHITE', 'BLACK') or color code in HEX format (e.g., '#FFFFFF') or RGB format (e.g., 'rgb(255, 255, 255)').",
        default="WHITE",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.text_color"],
    )

    text_scale: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        description="Scale factor for text size. Higher values create larger text. Default is 1.0.",
        default=1.0,
        examples=[1.0, "$inputs.text_scale"],
    )

    text_thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of text characters in pixels. Higher values create bolder, thicker text for better visibility.",
        default=1,
        examples=[1, "$inputs.text_thickness"],
    )

    text_padding: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Padding around the text in pixels. Controls the spacing between the text and the label background border.",
        default=10,
        examples=[10, "$inputs.text_padding"],
    )

    border_radius: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Border radius of the label background in pixels. Set to 0 for square corners. Higher values create more rounded corners for a softer appearance.",
        default=0,
        examples=[0, "$inputs.border_radius"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LabelVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LabelManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        text_position: str,
        text_color: str,
        text_scale: float,
        text_thickness: int,
        text_padding: int,
        border_radius: int,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_palette,
                    palette_size,
                    color_axis,
                    text_position,
                    text_color,
                    text_scale,
                    text_thickness,
                    text_padding,
                    border_radius,
                ],
            )
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            text_color = str_to_color(text_color)

            self.annotatorCache[key] = sv.LabelAnnotator(
                color=palette,
                color_lookup=getattr(sv.ColorLookup, color_axis),
                text_position=getattr(sv.Position, text_position),
                text_color=text_color,
                text_scale=text_scale,
                text_thickness=text_thickness,
                text_padding=text_padding,
                border_radius=border_radius,
            )

        return self.annotatorCache[key]

    def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        color_palette: Optional[str],
        palette_size: Optional[int],
        custom_colors: Optional[List[str]],
        color_axis: Optional[str],
        text: Optional[str],
        text_position: Optional[str],
        text_color: Optional[str],
        text_scale: Optional[float],
        text_thickness: Optional[int],
        text_padding: Optional[int],
        border_radius: Optional[int],
    ) -> BlockResult:
        if len(predictions) == 0:
            return {
                OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                    origin_image_data=image,
                    numpy_image=(
                        image.numpy_image.copy() if copy_image else image.numpy_image
                    ),
                )
            }
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            text_position,
            text_color,
            text_scale,
            text_thickness,
            text_padding,
            border_radius,
        )
        if text == "Class":
            labels = predictions["class_name"]
        elif text == "Tracker Id":
            if predictions.tracker_id is not None:
                labels = [
                    str(t) if t else "No Tracker ID" for t in predictions.tracker_id
                ]
            else:
                labels = ["No Tracker ID"] * len(predictions)
        elif text == "Time In Zone":
            if "time_in_zone" in predictions.data:
                labels = [
                    f"In zone: {round(t, 2)}s" if t else "In zone: N/A"
                    for t in predictions.data["time_in_zone"]
                ]
            else:
                labels = [f"In zone: N/A"] * len(predictions)
        elif text == "Confidence":
            labels = [f"{confidence:.2f}" for confidence in predictions.confidence]
        elif text == "Class and Confidence":
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(
                    predictions["class_name"], predictions.confidence
                )
            ]
        elif text == "Index":
            labels = [str(i) for i in range(len(predictions))]
        elif text == "Dimensions":
            # rounded ints: center x, center y wxh from predictions[i].xyxy
            labels = []
            for i in range(len(predictions)):
                x1, y1, x2, y2 = predictions.xyxy[i]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                labels.append(f"{int(cx)}, {int(cy)} {int(w)}x{int(h)}")
        elif text == "Area":
            labels = [str(int(area)) for area in predictions.area]
        elif text == "Area (mask)":
            if AREA_KEY_IN_SV_DETECTIONS in predictions.data:
                labels = [
                    str(round(float(a), 2)) if a is not None else "Area (mask): N/A"
                    for a in predictions.data[AREA_KEY_IN_SV_DETECTIONS]
                ]
            else:
                labels = ["Area (mask): N/A"] * len(predictions)
        elif text == "Area (converted)":
            if AREA_CONVERTED_KEY_IN_SV_DETECTIONS in predictions.data:
                labels = [
                    str(round(float(a), 2)) if a is not None else "Area (conv): N/A"
                    for a in predictions.data[AREA_CONVERTED_KEY_IN_SV_DETECTIONS]
                ]
            else:
                labels = ["Area (conv): N/A"] * len(predictions)
        else:
            try:
                labels = [str(d) if d else "" for d in predictions[text]]
            except Exception:
                raise ValueError(f"Invalid text type: {text}")
        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions,
            labels=labels,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
