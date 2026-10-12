"""Opt-in V2 tensor image, temporal and native prediction catalogue.

Callers select these blocks and the built-in operators explicitly::

    from roboflow_workflows.execution_engine.v2.blocks import create_catalogue

    catalogue = create_catalogue()

``Catalogue.from_modules(["roboflow_workflows.execution_engine.v2.blocks"])``
reads the same catalogue through ``WORKFLOWS_V2_CATALOGUE``. The generic V2
engine never imports this package, and the legacy V1 loader does not discover
it. Every contract detail lives on the block classes themselves.
"""

from roboflow_workflows.execution_engine.v2.blocks.boundaries import (
    MEDIA_WILDCARD_KIND,
    NUMPY_ARRAY_KIND,
)
from roboflow_workflows.execution_engine.v2.blocks.control import ContinueIfBlock
from roboflow_workflows.execution_engine.v2.blocks.detection import ObjectDetectionBlock
from roboflow_workflows.execution_engine.v2.blocks.image import (
    CropBlock,
    HasBrightnessBlock,
    InvertBlock,
    MosaicBlock,
    ResizeBlock,
    StaticCropBlock,
)
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import (
    CROP_SUMMARY_KIND,
    IMAGE_KIND,
)
from roboflow_workflows.execution_engine.v2.blocks.prediction_ops import (
    select_predictions,
)
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    BAR_CODE_DETECTION_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    DETECTION_KIND,
    EMBEDDING_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    NATIVE_KINDS,
    OBJECT_DETECTION_PREDICTION_KIND,
    QR_CODE_DETECTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
    TENSOR_KIND,
)
from roboflow_workflows.execution_engine.v2.blocks.recording_codecs import (
    NATIVE_RECORDING_CODECS,
)
from roboflow_workflows.execution_engine.v2.blocks.segmentation_views import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    MaskGridGeometry,
    SegmentationView,
)
from roboflow_workflows.execution_engine.v2.blocks.state import (
    StateCompareAndSetBlock,
    StateGetBlock,
    StateIncrementBlock,
    StateSetBlock,
)
from roboflow_workflows.execution_engine.v2.blocks.state_machine import (
    StateMachineSetBlock,
)
from roboflow_workflows.execution_engine.v2.blocks.temporal import (
    BestFrameBlock,
    TopKBrightestBlock,
)
from roboflow_workflows.execution_engine.v2.blocks.tracking import (
    ByteTrackerBlock,
    TrackerCapacityError,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization import (
    BLOCKS as VISUALIZATION_BLOCKS,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization import (
    DETECTION_DRAWING_KIND,
    BoxVisualizationBlock,
    DetectionDrawing,
    DetectionDrawingBlock,
    DetectionVisualizationBlock,
    DrawingBackendUnavailable,
    InPlaceLabelVisualizationBlock,
    LabelVisualizationBlock,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.operators.alignment import Align
from roboflow_workflows.execution_engine.v2.operators.window import Window


def create_catalogue() -> Catalogue:
    """Collect the opt-in V2 vision, temporal and managed-state blocks.

    Models remain caller-owned resources. The generic V2 engine does not import
    this catalogue; hosts select it explicitly. Triton is loaded only when its
    drawing backend is requested.

    Returns:
        Catalogue with image transforms, caller-loaded batched object detection,
        per-source tracking, detection painters, conditional and temporal blocks,
        managed-state blocks, alignment/window operators and native media kinds.
        Recording codecs preserve tensors, images and native predictions as raw
        binary data. Prepared drawings and live segmentation views have no codec;
        a segmentation view's ``full_res()`` is its explicit dense boundary.
    """
    catalogue = Catalogue(
        [
            CropBlock,
            StaticCropBlock,
            ResizeBlock,
            InvertBlock,
            MosaicBlock,
            HasBrightnessBlock,
            ContinueIfBlock,
            BestFrameBlock,
            TopKBrightestBlock,
            StateGetBlock,
            StateSetBlock,
            StateIncrementBlock,
            StateCompareAndSetBlock,
            StateMachineSetBlock,
            ObjectDetectionBlock,
            ByteTrackerBlock,
            *VISUALIZATION_BLOCKS,
        ],
        operators=[Align, Window],
        kinds=[
            IMAGE_KIND,
            CROP_SUMMARY_KIND,
            DETECTION_DRAWING_KIND,
            *NATIVE_KINDS,
            INSTANCE_SEGMENTATION_VIEW_KIND,
            MEDIA_WILDCARD_KIND,
            NUMPY_ARRAY_KIND,
        ],
        codecs=NATIVE_RECORDING_CODECS,
    )

    return catalogue


WORKFLOWS_V2_CATALOGUE = create_catalogue

__all__ = [
    "Align",
    "BAR_CODE_DETECTION_KIND",
    "BestFrameBlock",
    "BoxVisualizationBlock",
    "ByteTrackerBlock",
    "CLASSIFICATION_PREDICTION_KIND",
    "CROP_SUMMARY_KIND",
    "ContinueIfBlock",
    "CropBlock",
    "DETECTION_DRAWING_KIND",
    "DETECTION_KIND",
    "DetectionDrawing",
    "DetectionDrawingBlock",
    "DetectionVisualizationBlock",
    "DrawingBackendUnavailable",
    "EMBEDDING_KIND",
    "HasBrightnessBlock",
    "IMAGE_KIND",
    "INSTANCE_SEGMENTATION_PREDICTION_KIND",
    "INSTANCE_SEGMENTATION_VIEW_KIND",
    "ImageData",
    "InPlaceLabelVisualizationBlock",
    "InvertBlock",
    "KEYPOINT_DETECTION_PREDICTION_KIND",
    "LabelVisualizationBlock",
    "MEDIA_WILDCARD_KIND",
    "MaskGridGeometry",
    "MosaicBlock",
    "NATIVE_KINDS",
    "NATIVE_RECORDING_CODECS",
    "NUMPY_ARRAY_KIND",
    "OBJECT_DETECTION_PREDICTION_KIND",
    "ObjectDetectionBlock",
    "QR_CODE_DETECTION_KIND",
    "RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND",
    "ResizeBlock",
    "SEMANTIC_SEGMENTATION_PREDICTION_KIND",
    "SegmentationView",
    "StateCompareAndSetBlock",
    "StateGetBlock",
    "StateIncrementBlock",
    "StateMachineSetBlock",
    "StateSetBlock",
    "StaticCropBlock",
    "TENSOR_KIND",
    "TopKBrightestBlock",
    "TrackerCapacityError",
    "WORKFLOWS_V2_CATALOGUE",
    "Window",
    "create_catalogue",
    "select_predictions",
]
