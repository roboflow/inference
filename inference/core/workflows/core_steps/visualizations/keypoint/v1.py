from typing import Literal, Optional, Type, Union

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
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/keypoint_visualization@v1"
SHORT_DESCRIPTION = "Draw keypoints on detected objects in an image."
LONG_DESCRIPTION = """
The `KeypointVisualization` block uses a detections from an
keypoint detection model to draw keypoints on objects using
`sv.VertexAnnotator`.
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
        description="Predictions",
        examples=["$steps.keypoint_detection_model.predictions"],
    )

    annotator_type: Literal["edge", "vertex", "vertex_label"] = Field(
        description="Type of annotator to be used for keypoint visualization.",
        default="edge",
        json_schema_extra={"always_visible": True},
    )

    color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the keypoint.",
        default="#A351FB",
        examples=["#A351FB", "green", "$inputs.color"],
    )

    text_color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Text color of the keypoint.",
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
        description="Scale of the text.",
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
        description="Thickness of the text characters.",
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
        description="Padding around the text in pixels.",
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
        description="Thickness of the outline in pixels.",
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
        description="Radius of the keypoint in pixels.",
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
    ) -> BlockResult:
        annotator = self.getAnnotator(
            color,
            text_color,
            text_scale,
            text_thickness,
            text_padding,
            thickness,
            radius,
            annotator_type,
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
