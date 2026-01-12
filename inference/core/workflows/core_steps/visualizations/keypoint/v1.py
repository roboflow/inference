from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    VisualizationBlock,
    VisualizationManifest,
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/keypoint_visualization@v1"
SHORT_DESCRIPTION = "Draw keypoints on detected objects in an image."
LONG_DESCRIPTION = """
Visualize keypoints (landmark points) detected on objects by drawing point markers, connecting edges, or labeled vertices, providing pose estimation visualization for anatomical points, structural landmarks, or object key features.

## How This Block Works

This block takes an image and keypoint detection predictions and visualizes the detected keypoints using one of three visualization modes. The block:

1. Takes an image and keypoint detection predictions as input (predictions must include keypoint coordinates, confidence scores, and class names)
2. Extracts keypoint data (coordinates, confidence values, and class names) from the predictions
3. Converts the detection data into a KeyPoints format suitable for visualization
4. Applies one of three visualization modes based on the annotator_type setting:
   - **Edge mode**: Draws connecting lines (edges) between keypoints using specified edge pairs to show keypoint relationships (e.g., skeleton connections in pose estimation)
   - **Vertex mode**: Draws circular markers at each keypoint location without connections, showing individual keypoint positions
   - **Vertex label mode**: Draws circular markers with text labels identifying each keypoint class name, providing labeled keypoint visualization
5. Applies color styling, sizing, and optional text labeling based on the selected parameters
6. Returns an annotated image with keypoints visualized according to the selected mode

The block supports three visualization styles to suit different use cases. Edge mode connects related keypoints with lines (useful for pose estimation skeletons or structural relationships), vertex mode shows individual keypoint locations as circular markers, and vertex label mode adds text labels to identify each keypoint type. This visualization is essential for pose estimation workflows, anatomical point detection, or any application where specific landmark points on objects need to be identified and visualized.

## Common Use Cases

- **Human Pose Estimation**: Visualize human body keypoints (joints, body parts) for pose estimation, activity recognition, or motion analysis applications where anatomical points need to be displayed with skeleton connections or labeled markers
- **Animal Pose Estimation**: Display animal keypoints for behavior analysis, veterinary applications, or wildlife monitoring where anatomical landmarks need to be visualized for pose analysis or movement tracking
- **Structural Landmark Detection**: Visualize keypoints on objects, structures, or machinery for structural analysis, quality control, or measurement workflows where specific landmark points need to be identified and displayed
- **Facial Landmark Detection**: Display facial keypoints (eye corners, nose tip, mouth corners, etc.) for facial recognition, expression analysis, or face alignment applications where facial features need to be visualized
- **Sports and Movement Analysis**: Visualize keypoints for sports analysis, biomechanics, or movement studies where body positions, joint angles, or movement patterns need to be analyzed and displayed
- **Quality Control and Inspection**: Display keypoints for manufacturing, quality assurance, or inspection workflows where specific points on products or components need to be identified, measured, or validated

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Keypoint Detection Model blocks** to receive keypoint predictions that are visualized with point markers, edges, or labeled vertices
- **Other visualization blocks** (e.g., Bounding Box Visualization, Label Visualization, Polygon Visualization) to combine keypoint visualization with additional annotations for comprehensive pose or structure visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with keypoint visualizations for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with keypoints to external systems, APIs, or web applications for display in dashboards, pose analysis tools, or monitoring interfaces
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with keypoints as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with keypoint visualizations for live pose estimation, movement analysis, or post-processing workflows
"""


class KeypointManifest(VisualizationManifest):
    type: Literal[f"{TYPE}", "KeypointVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Keypoint Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-braille",
                "blockPriority": 20,
            },
        }
    )

    predictions: Selector(
        kind=[
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Keypoint detection predictions containing keypoint coordinates, confidence scores, and class names. Predictions must include keypoints_xy (keypoint coordinates), keypoints_confidence (confidence values), and keypoints_class_name (keypoint class/type names). Requires outputs from a keypoint detection model block.",
        examples=["$steps.keypoint_detection_model.predictions"],
    )

    annotator_type: Literal["edge", "vertex", "vertex_label"] = Field(
        description="Type of keypoint visualization mode. Options: 'edge' (draws connecting lines between keypoints using edge pairs, useful for skeleton/pose visualization), 'vertex' (draws circular markers at keypoint locations without connections), 'vertex_label' (draws circular markers with text labels identifying each keypoint class name).",
        default="edge",
        json_schema_extra={"always_visible": True},
    )

    color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the keypoint markers, edges, or labels. Can be specified as a color name (e.g., 'green', 'red', 'blue'), hex color code (e.g., '#A351FB', '#FF0000'), or RGB format. Used for keypoint circles (vertex/vertex_label modes) or edge lines (edge mode).",
        default="#A351FB",
        examples=["#A351FB", "green", "$inputs.color"],
    )

    text_color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the text labels displayed on keypoints (vertex_label mode only). Can be specified as a color name (e.g., 'black', 'white'), hex color code, or RGB format. Only applies when annotator_type is 'vertex_label'.",
        default="black",
        examples=["black", "$inputs.text_color"],
        json_schema_extra={
            "relevant_for": {
                "annotator_type": {
                    "values": ["vertex_label"],
                },
            },
        },
    )
    text_scale: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        description="Scale factor for keypoint label text size (vertex_label mode only). Controls the size of text labels displayed on keypoints. Values greater than 1.0 make text larger, values less than 1.0 make text smaller. Only applies when annotator_type is 'vertex_label'. Typical values range from 0.3 to 1.0.",
        default=0.5,
        examples=[0.5, "$inputs.text_scale"],
        json_schema_extra={
            "relevant_for": {
                "annotator_type": {
                    "values": ["vertex_label"],
                },
            },
        },
    )

    text_thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the keypoint label text characters in pixels (vertex_label mode only). Controls how bold the text labels appear. Higher values create thicker, bolder text. Only applies when annotator_type is 'vertex_label'. Typical values range from 1 to 3.",
        default=1,
        examples=[1, "$inputs.text_thickness"],
        json_schema_extra={
            "relevant_for": {
                "annotator_type": {
                    "values": ["vertex_label"],
                },
            },
        },
    )

    text_padding: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Padding around keypoint label text in pixels (vertex_label mode only). Controls the spacing between the text label and its background border. Higher values create more space around text. Only applies when annotator_type is 'vertex_label'. Typical values range from 5 to 20 pixels.",
        default=10,
        examples=[10, "$inputs.text_padding"],
        json_schema_extra={
            "relevant_for": {
                "annotator_type": {
                    "values": ["vertex_label"],
                },
            },
        },
    )

    thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the edge lines connecting keypoints in pixels (edge mode only). Controls how thick the connecting lines between keypoints appear. Higher values create thicker, more visible edges. Only applies when annotator_type is 'edge'. Typical values range from 1 to 5 pixels.",
        default=2,
        examples=[2, "$inputs.thickness"],
        json_schema_extra={
            "relevant_for": {
                "annotator_type": {
                    "values": ["edge"],
                },
            },
        },
    )

    radius: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Radius of the circular keypoint markers in pixels (vertex and vertex_label modes only). Controls the size of circular markers drawn at keypoint locations. Higher values create larger, more visible markers. Only applies when annotator_type is 'vertex' or 'vertex_label'. Typical values range from 5 to 20 pixels.",
        default=10,
        examples=[10, "$inputs.radius"],
        json_schema_extra={
            "relevant_for": {
                "annotator_type": {
                    "values": ["vertex", "vertex_label"],
                },
            },
        },
    )
    edges: Union[list, Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Edge connections between keypoints (edge mode only). List of pairs of keypoint indices (e.g., [(0, 1), (1, 2), ...]) defining which keypoints should be connected with lines. For pose estimation, this typically represents skeleton connections (e.g., connecting joints). Only applies when annotator_type is 'edge'. Required for edge visualization.",
        default=None,
        examples=["$inputs.edges"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.2.0,<2.0.0"


class KeypointVisualizationBlockV1(VisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return KeypointManifest

    def getAnnotator(
        self,
        color: str,
        text_color: str,
        text_scale: float,
        text_thickness: int,
        text_padding: int,
        thickness: int,
        radius: int,
        annotator_type: str,
        edges: List[Tuple[int, int]],
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color,
                    text_color,
                    text_scale,
                    text_thickness,
                    text_padding,
                    thickness,
                    radius,
                    annotator_type,
                ],
            )
        )

        if key not in self.annotatorCache:
            color = str_to_color(color)
            text_color = str_to_color(text_color)

            if annotator_type == "edge":
                self.annotatorCache[key] = sv.EdgeAnnotator(
                    color=color,
                    thickness=thickness,
                    edges=edges,
                )
            elif annotator_type == "vertex":
                self.annotatorCache[key] = sv.VertexAnnotator(
                    color=color,
                    radius=radius,
                )
            elif annotator_type == "vertex_label":
                self.annotatorCache[key] = sv.VertexLabelAnnotator(
                    color=color,
                    text_color=text_color,
                    text_scale=text_scale,
                    text_thickness=text_thickness,
                    text_padding=text_padding,
                    border_radius=radius,
                )

        return self.annotatorCache[key]

    # Function to convert detections to keypoints
    def convert_detections_to_keypoints(self, detections):
        if len(detections) == 0:
            return sv.KeyPoints.empty()
        keypoints_xy = detections.data["keypoints_xy"]
        keypoints_confidence = detections.data["keypoints_confidence"]
        keypoints_class_name = detections.data["keypoints_class_name"]
        class_id = detections.class_id

        keypoints = sv.KeyPoints(
            xy=np.array(keypoints_xy, dtype=np.float32),
            confidence=np.array(keypoints_confidence, dtype=np.float32),
            class_id=np.array(class_id, dtype=int),
            data={"class_name": np.array(keypoints_class_name, dtype=object)},
        )
        return keypoints

    def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        annotator_type: Optional[str],
        color: Optional[str],
        text_color: Optional[str],
        text_scale: Optional[float],
        text_thickness: Optional[int],
        text_padding: Optional[int],
        thickness: Optional[int],
        radius: Optional[int],
        edges: Optional[List[Tuple[int, int]]] = None,
    ) -> BlockResult:
        annotator: sv.EdgeAnnotator = self.getAnnotator(
            color,
            text_color,
            text_scale,
            text_thickness,
            text_padding,
            thickness,
            radius,
            annotator_type,
            edges,
        )

        keypoints = self.convert_detections_to_keypoints(predictions)

        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            key_points=keypoints,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
