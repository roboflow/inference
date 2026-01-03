from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
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
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    CLASSIFICATION_PREDICTION_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

SHORT_DESCRIPTION = "Visualize both single-label and multi-label classification predictions with customizable display options."

LONG_DESCRIPTION = """
Visualize classification predictions as text labels positioned on images, automatically handling both single-label and multi-label classification formats with customizable styling and positioning.

## How This Block Works

This block takes an image and classification predictions (for entire image classification, not object detection) and displays text labels showing the predicted class names and confidence scores. The block:

1. Takes an image and classification predictions as input
2. Automatically detects whether predictions are single-label (one class per image) or multi-label (multiple classes per image)
3. For single-label predictions: selects the highest confidence prediction to display
4. For multi-label predictions: formats and sorts all predicted classes by confidence score (highest first)
5. Extracts label text based on the selected text option (class name, confidence score, or both)
6. Positions labels on the image at the specified location (top, center, or bottom edges, with left/center/right alignment)
7. Applies background color styling based on the selected color palette, with colors assigned by class
8. Renders text labels with customizable text color, scale, thickness, padding, and border radius
9. Returns an annotated image with classification labels overlaid on the original image

Unlike the regular Label Visualization block (which labels detected objects with bounding boxes), this block is designed for image-level classification where the entire image is classified into one or more categories. Labels are positioned at the edges or center of the image itself, not relative to object locations. For multi-label predictions, multiple labels are stacked vertically at the chosen position, making it easy to see all predicted classes and their confidence scores.

## Common Use Cases

- **Image Classification Results Display**: Visualize the predicted class and confidence score for classified images in applications like content moderation, product categorization, or medical image analysis
- **Multi-Class Probability Visualization**: Display multiple predicted classes with their confidence scores for multi-label classification tasks, such as tagging images with multiple attributes, detecting multiple defects, or identifying multiple objects in scene classification
- **Model Performance Validation**: Show classification predictions directly on images to validate model performance, verify correct classifications, and identify misclassifications during model development or testing
- **User Interface Integration**: Create clean, professional displays of classification results for applications, dashboards, or mobile apps where users need to see what an image was classified as
- **Documentation and Reporting**: Generate annotated images showing classification results for reports, documentation, or training data review to demonstrate model predictions
- **Quality Control Workflows**: Display classification results on production images for quality control, content filtering, or automated categorization workflows where visual confirmation of predictions is needed

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with classification labels for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with classification labels to external systems, APIs, or web applications for display in dashboards or classification monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with classification labels as visual evidence in alerts or reports when specific classes are detected
- **Video output blocks** to create annotated video streams or recordings with classification labels for live monitoring, real-time classification display, or post-processing analysis
- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on classification results or confidence scores displayed in the labels
"""


class ClassificationLabelManifest(ColorableVisualizationManifest):
    type: Literal["roboflow_core/classification_label_visualization@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Classification Label Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-tags",
                "blockPriority": 2.5,
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
        Literal["Class", "Confidence", "Class and Confidence"],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="Class",
        description="Content to display in text labels. Options: 'Class' (class name only), 'Confidence' (confidence score only, formatted as decimal), or 'Class and Confidence' (both class name and confidence score).",
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
        ],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="TOP_LEFT",
        description="Position for placing labels on the image. Options include: TOP (TOP_LEFT, TOP_CENTER, TOP_RIGHT), CENTER (CENTER_LEFT, CENTER, CENTER_RIGHT), or BOTTOM (BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT). For multi-label predictions, labels are stacked vertically at the chosen position.",
        examples=["CENTER", "$inputs.text_position"],
    )

    predictions: StepOutputSelector(kind=[CLASSIFICATION_PREDICTION_KIND]) = Field(  # type: ignore
        description="Classification predictions from a single-label or multi-label classification model. The block automatically detects the prediction format and handles both types accordingly.",
        examples=["$steps.classification_model.predictions"],
    )

    text_color: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the label text. Can be a color name (e.g., 'WHITE', 'BLACK') or color code in HEX format (e.g., '#FFFFFF') or RGB format (e.g., 'rgb(255, 255, 255)').",
        default="WHITE",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.text_color"],
    )

    text_scale: Union[float, WorkflowParameterSelector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        description="Scale factor for text size. Higher values create larger text. Default is 1.0.",
        default=1.0,
        examples=[1.0, "$inputs.text_scale"],
    )

    text_thickness: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of text characters in pixels. Higher values create bolder, thicker text for better visibility.",
        default=1,
        examples=[1, "$inputs.text_thickness"],
    )

    text_padding: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Padding around the text in pixels. Controls the spacing between the text and the label background border, and the spacing between multiple labels in multi-label predictions.",
        default=10,
        examples=[10, "$inputs.text_padding"],
    )

    border_radius: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Border radius of the label background in pixels. Set to 0 for square corners. Higher values create more rounded corners for a softer appearance.",
        default=0,
        examples=[0, "$inputs.border_radius"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ClassificationLabelVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ClassificationLabelManifest

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
        predictions: dict,
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
        try:
            # Detect task type from predictions
            task_type = detect_prediction_type(predictions)

            # Get sorted predictions based on detected task type
            if task_type == "single-label":
                sorted_predictions = sorted(
                    predictions["predictions"],
                    key=lambda x: x["confidence"],
                    reverse=True,
                )[:1]
            else:  # multi-label
                formatted_predictions = format_multi_label_predictions(predictions)
                sorted_predictions = sorted(
                    formatted_predictions, key=lambda x: x["confidence"], reverse=True
                )

            # Early return if no predictions
            if not sorted_predictions:
                return {
                    OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                        origin_image_data=image,
                        numpy_image=(
                            image.numpy_image.copy()
                            if copy_image
                            else image.numpy_image
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

            # Calculate spacing based on all relevant parameters
            base_text_height = 20  # OpenCV's base text height in pixels
            scaled_text_height = base_text_height * text_scale
            padding_space = (
                text_padding * 2
            )  # multiply by 2 since padding is applied top and bottom
            thickness_space = text_thickness * 2  # account for text thickness
            total_spacing = (
                scaled_text_height + padding_space + thickness_space + 5
            )  # added small buffer
            initial_offset = total_spacing
            w = predictions["image"]["width"]
            h = predictions["image"]["height"]

            # Both single and multi-label use the same visualization creation
            xyxy, labels, predictions_to_use = create_label_visualization(
                sorted_predictions=sorted_predictions,
                text_position=text_position,
                text=text,
                w=w,
                h=h,
                initial_offset=initial_offset,
                total_spacing=total_spacing,
                text_scale=text_scale,
                text_padding=text_padding,
            )

            if not predictions_to_use:
                # If no predictions, return the original image
                return {
                    OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                        origin_image_data=image,
                        numpy_image=(
                            image.numpy_image.copy()
                            if copy_image
                            else image.numpy_image
                        ),
                    )
                }

            pseudo_detections = sv.Detections(
                xyxy=xyxy,
                class_id=np.array([p["class_id"] for p in predictions_to_use]),
                confidence=np.array([p["confidence"] for p in predictions_to_use]),
                tracker_id=np.array([0 for _ in predictions_to_use]),
            )

            annotated_image = annotator.annotate(
                scene=image.numpy_image.copy() if copy_image else image.numpy_image,
                detections=pseudo_detections,
                labels=labels,
            )

            return {
                OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                    origin_image_data=image, numpy_image=annotated_image
                )
            }

        except ValueError as e:
            raise ValueError(
                f"Invalid prediction format: {str(e)}. Please check if the task_type matches your model's output format."
            ) from e


def handle_bottom_position(
    sorted_predictions: List[dict],
    text: str,
    w: int,
    h: int,
    initial_offset: float,
    total_spacing: float,
) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Handle visualization layout for bottom positions."""
    reversed_predictions = sorted_predictions[::-1]
    xyxy = np.array(
        [
            [0, 0, w, h - (initial_offset + i * total_spacing)]
            for i in range(len(reversed_predictions))
        ]
    )
    labels = format_labels(reversed_predictions, text)
    return xyxy, labels, reversed_predictions


def handle_center_position(
    sorted_predictions: List[dict],
    text: str,
    text_position: str,
    w: int,
    h: int,
    total_spacing: float,
    text_scale: float,
    text_padding: int,
) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Handle visualization layout for center positions."""
    labels = format_labels(sorted_predictions, text)
    n_predictions = len(sorted_predictions)
    total_height = total_spacing * n_predictions
    start_y = max(0, min((h - total_height) / 2, h - total_height))

    max_label_length = max(len(label) for label in labels)
    char_width = 15
    label_width = (max_label_length * char_width * text_scale) + (text_padding * 2)
    extra_padding = 20 + max(0, 10 - text_padding) * 3

    if text_position == "CENTER_LEFT":
        x_start = label_width + extra_padding
        xyxy = np.array(
            [
                [
                    x_start,
                    start_y + i * total_spacing,
                    w,
                    start_y + (i + 1) * total_spacing,
                ]
                for i in range(n_predictions)
            ]
        )
    elif text_position == "CENTER_RIGHT":
        x_end = w - (label_width + extra_padding)
        xyxy = np.array(
            [
                [
                    0,
                    start_y + i * total_spacing,
                    x_end,
                    start_y + (i + 1) * total_spacing,
                ]
                for i in range(n_predictions)
            ]
        )
    else:  # CENTER
        xyxy = np.array(
            [
                [0, start_y + i * total_spacing, w, start_y + (i + 1) * total_spacing]
                for i in range(n_predictions)
            ]
        )

    return xyxy, labels, sorted_predictions


def handle_top_position(
    sorted_predictions: List[dict],
    text: str,
    w: int,
    h: int,
    initial_offset: float,
    total_spacing: float,
) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Handle visualization layout for top positions."""
    xyxy = np.array(
        [
            [0, initial_offset + i * total_spacing, w, h]
            for i in range(len(sorted_predictions))
        ]
    )
    labels = format_labels(sorted_predictions, text)
    return xyxy, labels, sorted_predictions


def create_label_visualization(
    sorted_predictions: List[dict],
    text_position: str,
    text: str,
    w: int,
    h: int,
    initial_offset: float,
    total_spacing: float,
    text_scale: float,
    text_padding: int,
) -> Tuple[np.ndarray, List[str], List[dict]]:
    """Create visualization layout for classification labels."""
    if text_position in ["BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT"]:
        return handle_bottom_position(
            sorted_predictions, text, w, h, initial_offset, total_spacing
        )
    elif text_position in ["CENTER", "CENTER_LEFT", "CENTER_RIGHT"]:
        return handle_center_position(
            sorted_predictions,
            text,
            text_position,
            w,
            h,
            total_spacing,
            text_scale,
            text_padding,
        )
    else:  # Top positions
        return handle_top_position(
            sorted_predictions, text, w, h, initial_offset, total_spacing
        )


def detect_prediction_type(predictions: dict) -> str:
    """
    Detect whether predictions are single-label or multi-label based on structure.

    Args:
        predictions (dict): The predictions dictionary

    Returns:
        str: 'single-label' or 'multi-label'
    """
    if isinstance(predictions.get("predictions"), list):
        return "single-label"
    return "multi-label"


def validate_prediction_format(predictions: dict, task_type: str) -> None:
    """
    Validate that the predictions format matches the specified task type.

    Args:
        predictions (dict): The predictions dictionary
        task_type (str): The specified task type ('single-label' or 'multi-label')

    Raises:
        ValueError: If prediction format doesn't match task type
    """
    actual_type = detect_prediction_type(predictions)

    if actual_type != task_type:
        if actual_type == "single-label":
            raise ValueError(
                "Received single-label predictions but task_type is set to 'multi-label'. Please correct the task_type setting."
            )
        else:
            raise ValueError(
                "Received multi-label predictions but task_type is set to 'single-label'. Please correct the task_type setting."
            )


def format_multi_label_predictions(predictions: dict) -> List[dict]:
    """
    Transform multi-label predictions from predicted_classes into standard format.

    Args:
        predictions (dict): The predictions dictionary

    Returns:
        List[dict]: Formatted predictions list
    """
    formatted_predictions = []
    for class_name in predictions["predicted_classes"]:
        pred_info = predictions["predictions"][class_name]
        formatted_predictions.append(
            {
                "class": class_name,
                "class_id": pred_info["class_id"],
                "confidence": pred_info["confidence"],
            }
        )
    return formatted_predictions


def format_labels(predictions, text="Class and Confidence"):
    """
    Format labels based on specified text option.

    Args:
        predictions (list): List of prediction dictionaries containing 'class' and 'confidence'
        text (str): One of "class", "confidence", or "class and confidence"

    Returns:
        list: Formatted label strings
    """
    if text == "Class":
        labels = [f"{p['class']}" for p in predictions]
    elif text == "Confidence":
        labels = [f"{p['confidence']:.2f}" for p in predictions]
    elif text == "Class and Confidence":
        labels = [f"{p['class']} {p['confidence']:.2f}" for p in predictions]
    else:
        raise ValueError(
            "text must be one of: 'class', 'confidence', or 'class and confidence'"
        )

    return labels
