from typing import List, Literal, Optional, Type, Union

import supervision as sv
import numpy as np
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
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    WorkflowParameterSelector,
    StepOutputSelector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/classification_label_visualization@v1"
SHORT_DESCRIPTION = (
    "Visualizes classification predictions as stacked labels with confidence scores."
)

LONG_DESCRIPTION = """
The `ClassificationLabelVisualization` block displays classification predictions as stacked labels
with confidence scores. Labels can be positioned at the top, bottom, or center of the image,
with customizable alignment (left, center, right). Each label shows the class name and 
confidence score, ordered by confidence. The visualization uses Supervision's `sv.LabelAnnotator` 
and `sv.BoxAnnotator` for rendering, with support for customizable text properties including 
padding, scale, and color.
"""

class ClassificationLabelManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}", "ClassificationLabelVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Classification Label Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
        }
    )

    text: Union[
        Literal[
            "Class",
            "Confidence",
            "Class and Confidence"
        ],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="Class",
        description="The type of text to display.",
        examples=["LABEL", "$inputs.text"],
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
        description="The anchor position for placing the label.",
        examples=["CENTER", "$inputs.text_position"],
    )

    predictions: StepOutputSelector(kind=[CLASSIFICATION_PREDICTION_KIND]) = Field(  # type: ignore
        description="Classification predictions.",
        examples=["$steps.classification_model.predictions"],
    )

    task_type: Literal["single-label", "multi-label"] = Field(
        description="The type of task to visualize.",
    )
    single_labels_show: Literal["top", "all"] = Field(
        title="Single Labels To Show",
        description="Whether to show all single labels or just the top one.",
        default="top",
        examples=["top", "$inputs.single_labels_show"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": ["single-label"],
                    "required": True,
                },
            },
        },
    )
    single_labels_num_show: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Number of Single Labels to Show",
        description="The number of single labels to show.",
        default=5,
        examples=[5, "$inputs.single_labels_num_show"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": ["single-label"],
                    "required": True,
                },
                "single_labels_show": {
                    "values": ["all"],
                    "required": True,
                },
            },
        },
    )


    text_color: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the text.",
        default="WHITE",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.text_color"],
    )

    text_scale: Union[float, WorkflowParameterSelector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        description="Scale of the text.",
        default=1.0,
        examples=[1.0, "$inputs.text_scale"],
    )

    text_thickness: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the text characters.",
        default=1,
        examples=[1, "$inputs.text_thickness"],
    )

    text_padding: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Padding around the text in pixels.",
        default=10,
        examples=[10, "$inputs.text_padding"],
    )

    border_radius: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Radius of the label in pixels.",
        default=0,
        examples=[0, "$inputs.border_radius"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.2.0,<2.0.0"


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

    def _handle_bottom_position(
        self,
        sorted_predictions: List[dict],
        text: str,
        w: int,
        h: int,
        initial_offset: float,
        total_spacing: float,
    ) -> tuple[np.ndarray, List[str], List[dict]]:
        """Handle visualization layout for bottom positions."""
        reversed_predictions = sorted_predictions[::-1]
        xyxy = np.array([
            [0, 0, w, h - (initial_offset + i*total_spacing)] 
            for i in range(len(reversed_predictions))
        ])
        labels = format_labels(reversed_predictions, text)
        return xyxy, labels, reversed_predictions

    def _handle_center_position(
        self,
        sorted_predictions: List[dict],
        text: str,
        text_position: str,
        w: int,
        h: int,
        total_spacing: float,
        text_scale: float,
        text_padding: int,
    ) -> tuple[np.ndarray, List[str], List[dict]]:
        """Handle visualization layout for center positions."""
        labels = format_labels(sorted_predictions, text)
        n_predictions = len(sorted_predictions)
        total_height = total_spacing * n_predictions
        start_y = max(0, min((h - total_height) / 2, h - total_height))
        
        max_label_length = max(len(label) for label in labels)
        char_width = 15
        label_width = (max_label_length * char_width * text_scale) + (text_padding * 2)
        extra_padding = 20 + max(0, 10 - text_padding) * 3
        
        if text_position == 'CENTER_LEFT':
            x_start = label_width + extra_padding
            xyxy = np.array([
                [x_start, start_y + i*total_spacing, w, start_y + (i+1)*total_spacing] 
                for i in range(n_predictions)
            ])
        elif text_position == 'CENTER_RIGHT':
            x_end = w - (label_width + extra_padding)
            xyxy = np.array([
                [0, start_y + i*total_spacing, x_end, start_y + (i+1)*total_spacing] 
                for i in range(n_predictions)
            ])
        else:  # CENTER
            xyxy = np.array([
                [0, start_y + i*total_spacing, w, start_y + (i+1)*total_spacing] 
                for i in range(n_predictions)
            ])
        
        return xyxy, labels, sorted_predictions

    def _handle_top_position(
        self,
        sorted_predictions: List[dict],
        text: str,
        w: int,
        h: int,
        initial_offset: float,
        total_spacing: float,
    ) -> tuple[np.ndarray, List[str], List[dict]]:
        """Handle visualization layout for top positions."""
        xyxy = np.array([
            [0, initial_offset + i*total_spacing, w, h] 
            for i in range(len(sorted_predictions))
        ])
        labels = format_labels(sorted_predictions, text)
        return xyxy, labels, sorted_predictions

    def _create_label_visualization(
        self,
        sorted_predictions: List[dict],
        text_position: str,
        text: str,
        w: int,
        h: int,
        initial_offset: float,
        total_spacing: float,
        text_scale: float,
        text_padding: int,
    ) -> tuple[np.ndarray, List[str], List[dict]]:
        """Create visualization layout for classification labels.
        
        Args:
            sorted_predictions: List of prediction dictionaries sorted by confidence (descending).
                Each prediction should have the structure:
                {
                    'class': str,           # The class name
                    'class_id': int,        # The numeric class ID
                    'confidence': float,    # Confidence score between 0 and 1
                }
            text_position: Position of the text labels
            text: Type of text to display
            w: Image width
            h: Image height
            initial_offset: Initial vertical offset
            total_spacing: Total spacing between labels
            text_scale: Scale of the text
            text_padding: Padding around text
            
        Returns:
            tuple containing:
            - xyxy: numpy array of bounding box coordinates
            - labels: list of formatted label strings
            - predictions_to_use: list of predictions in correct order
        """
        if text_position in ['BOTTOM_LEFT', 'BOTTOM_CENTER', 'BOTTOM_RIGHT']:
            return self._handle_bottom_position(
                sorted_predictions, text, w, h, initial_offset, total_spacing
            )
        elif text_position in ['CENTER', 'CENTER_LEFT', 'CENTER_RIGHT']:
            return self._handle_center_position(
                sorted_predictions, text, text_position, w, h, 
                total_spacing, text_scale, text_padding
            )
        else:  # Top positions
            return self._handle_top_position(
                sorted_predictions, text, w, h, initial_offset, total_spacing
            )

    def _format_multi_label_predictions(
        self, 
        predictions: dict
    ) -> List[dict]:
        """Transform multi-label predictions from predicted_classes into standard format.
        
        Args:
            predictions: Prediction dictionary containing 'predicted_classes' and 'predictions' fields.
                Example:
                {
                    'predicted_classes': ['class1', 'class2'],
                    'predictions': {
                        'class1': {'confidence': 0.9, 'class_id': 0},
                        'class2': {'confidence': 0.8, 'class_id': 1},
                        ...
                    }
                }
            
        Returns:
            List of prediction dictionaries in format:
            [
                {'class': str, 'class_id': int, 'confidence': float},
                ...
            ]
        """
        formatted_predictions = []
        for class_name in predictions['predicted_classes']:
            pred_info = predictions['predictions'][class_name]
            formatted_predictions.append({
                'class': class_name,
                'class_id': pred_info['class_id'],
                'confidence': pred_info['confidence']
            })
        return formatted_predictions

    def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Classifications,
        task_type: str,
        copy_image: bool,
        single_labels_show: Optional[str],
        single_labels_num_show: Optional[int],
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
        padding_space = text_padding * 2  # multiply by 2 since padding is applied top and bottom
        thickness_space = text_thickness * 2  # account for text thickness
        total_spacing = scaled_text_height + padding_space + thickness_space + 5  # added small buffer
        initial_offset = total_spacing
        w = predictions['image']['width']
        h = predictions['image']['height']

        

        if task_type == "single-label":
            if single_labels_show == "top":
                n = 1
            else:
                n = single_labels_num_show
            
            sorted_predictions = sorted(predictions['predictions'], 
                                     key=lambda x: x['confidence'], 
                                     reverse=True)[:n]
        else:  # multi-label
            # Only use the predicted classes
            formatted_predictions = self._format_multi_label_predictions(predictions)
            sorted_predictions = sorted(formatted_predictions,
                                     key=lambda x: x['confidence'],
                                     reverse=True)
        
        # Both single and multi-label use the same visualization creation
        xyxy, labels, predictions_to_use = self._create_label_visualization(
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
                    numpy_image=image.numpy_image.copy() if copy_image else image.numpy_image
                )
            }

        pseudo_detections = sv.Detections(
            xyxy=xyxy,
            class_id=np.array([p['class_id'] for p in predictions_to_use]),
            confidence=np.array([p['confidence'] for p in predictions_to_use]),
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
        labels = [
            f"{p['class']}" for p in predictions
        ]
    elif text == "Confidence":
        labels = [
            f"{p['confidence']:.2f}" for p in predictions
        ]
    elif text == "Class and Confidence":
        labels = [
            f"{p['class']} {p['confidence']:.2f}" for p in predictions
        ]
    else:
        raise ValueError("text must be one of: 'class', 'confidence', or 'class and confidence'")
    
    return labels