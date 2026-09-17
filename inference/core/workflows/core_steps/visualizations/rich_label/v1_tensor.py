from typing import List, Optional, Type, Union

from pydantic import Field

from inference.core.workflows.core_steps.common.tensor_native import (
    TensorNativeDetections,
    TensorNativePrediction,
    split_key_point_prediction,
)
from inference.core.workflows.core_steps.visualizations.common.base_tensor import (
    OUTPUT_IMAGE_KEY,
    empty_predictions_passthrough,
    to_supervision_for_annotation,
)
from inference.core.workflows.core_steps.visualizations.common.label_text import (
    TEXT_SIZE_MODE_MANUAL,
    build_detection_labels,
    compute_adaptive_rich_font_size,
)
from inference.core.workflows.core_steps.visualizations.rich_label.v1 import TYPE
from inference.core.workflows.core_steps.visualizations.rich_label.v1 import (
    RichLabelManifest as _NumpyRichLabelManifest,
)
from inference.core.workflows.core_steps.visualizations.rich_label.v1 import (
    RichLabelVisualizationBlockV1 as _NumpyRichLabelVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import Selector
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest


class RichLabelManifest(_NumpyRichLabelManifest):
    """The numpy manifest reused verbatim (same ``type`` literal, fields,
    font-family validator and I/O contract) - only the ``predictions`` selector
    is re-declared with the tensor-native kinds, mirroring
    ``base_tensor.PredictionsVisualizationManifest``."""

    predictions: Selector(
        kind=[
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Model predictions to visualize.",
        examples=["$steps.object_detection_model.predictions"],
    )


class RichLabelVisualizationBlockV1(_NumpyRichLabelVisualizationBlockV1):
    """Tensor-native sibling of Rich Label Visualization v1.

    All drawing internals are reused from the numpy block: ``getAnnotator``
    (approved-font resolution via ``fonts.resolve_font_path`` +
    ``sv.RichLabelAnnotator`` construction/caching) is inherited, label text
    comes from the shared ``label_text.build_detection_labels`` and adaptive
    sizing from ``label_text.compute_adaptive_rich_font_size``. This class only
    adds the tensor-side glue: device-resident empty passthrough and native->sv
    materialisation before the inherently-CPU Pillow text rendering.
    """

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return RichLabelManifest

    def run(
        self,
        image: WorkflowImageData,
        predictions: Union[TensorNativePrediction, TensorNativeDetections],
        copy_image: bool,
        color_palette: Optional[str],
        palette_size: Optional[int],
        custom_colors: Optional[List[str]],
        color_axis: Optional[str],
        text: Optional[str],
        text_position: Optional[str],
        text_color: Optional[str],
        font_family: Optional[str],
        text_size_mode: Optional[str],
        font_size: Optional[int],
        text_padding: Optional[int],
        border_radius: Optional[int],
        max_line_length: Optional[int],
    ) -> BlockResult:
        detections = (
            split_key_point_prediction(predictions)[1]
            if isinstance(predictions, tuple)
            else predictions
        )
        passthrough = empty_predictions_passthrough(
            image=image, detections=detections, copy_image=copy_image
        )
        if passthrough is not None:
            return passthrough
        # `.mask` is read for exactly two configurations, and only for
        # instance-segmentation input (see label/v1_tensor.py for details):
        #   * text == "Area": `sv.Detections.area` reports MASK area when a
        #     mask is present and BOX area when it is None - flag-off shows
        #     mask area on IS input, so the mask must be materialised to match;
        #   * text_position == "CENTER_OF_MASS": the annotator anchors on the
        #     mask centroid; `get_anchors_coordinates` RAISES without a mask.
        # Every other label reads xyxy / confidence / per-box metadata, so the
        # device->host dense-mask copy is skipped for them.
        needs_masks = text == "Area" or text_position == "CENTER_OF_MASS"
        sv_detections = to_supervision_for_annotation(
            predictions, materialise_masks=needs_masks
        )
        # Non-empty frame: rich-label rendering is inherently CPU work (Pillow
        # TrueType rasterisation), so the numpy image is materialised here.
        height, width = image.numpy_image.shape[:2]
        effective_font_size = compute_adaptive_rich_font_size(
            height,
            width,
            manual_font_size=font_size,
            text_size_mode=text_size_mode or TEXT_SIZE_MODE_MANUAL,
        )
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            text_position,
            text_color,
            font_family,
            effective_font_size,
            text_padding,
            border_radius,
            max_line_length,
        )
        labels = build_detection_labels(sv_detections, text)
        scene = image.numpy_image
        if copy_image:
            scene = scene.copy()
        else:
            image.declare_numpy_image_mutated()
        annotated_image = annotator.annotate(
            scene=scene,
            detections=sv_detections,
            labels=labels,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
