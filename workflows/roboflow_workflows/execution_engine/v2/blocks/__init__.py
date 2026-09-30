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
from roboflow_workflows.execution_engine.v2.blocks.temporal import (
    BestFrameBlock,
    TopKBrightestBlock,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.operators.alignment import Align
from roboflow_workflows.execution_engine.v2.operators.window import Window


def create_catalogue() -> Catalogue:
    """Collect the image and temporal blocks, the operators and native media kinds.

    Returns:
        Immutable catalogue with ``v2/crop``, ``v2/static_crop``,
        ``v2/invert``, ``v2/mosaic``, ``v2/resize``, ``v2/has_brightness``,
        ``v2/continue_if``, ``v2/best_frame`` and ``v2/top_k_brightest``; the
        ``v2/align@v1`` and ``v2/window@v1`` operators; the ``image`` and
        ``crop_summary`` kinds, all eleven native prediction/tensor kinds, and
        the built-in kinds these blocks use. Wildcard media outputs and the
        legacy ``numpy_array`` tensor label use explicit boundary policies.
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
        ],
        operators=[Align, Window],
        kinds=[
            IMAGE_KIND,
            CROP_SUMMARY_KIND,
            *NATIVE_KINDS,
            MEDIA_WILDCARD_KIND,
            NUMPY_ARRAY_KIND,
        ],
    )

    return catalogue


WORKFLOWS_V2_CATALOGUE = create_catalogue

__all__ = [
    "BAR_CODE_DETECTION_KIND",
    "CLASSIFICATION_PREDICTION_KIND",
    "DETECTION_KIND",
    "EMBEDDING_KIND",
    "INSTANCE_SEGMENTATION_PREDICTION_KIND",
    "KEYPOINT_DETECTION_PREDICTION_KIND",
    "NATIVE_KINDS",
    "MEDIA_WILDCARD_KIND",
    "NUMPY_ARRAY_KIND",
    "OBJECT_DETECTION_PREDICTION_KIND",
    "QR_CODE_DETECTION_KIND",
    "RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND",
    "SEMANTIC_SEGMENTATION_PREDICTION_KIND",
    "TENSOR_KIND",
    "CROP_SUMMARY_KIND",
    "Align",
    "BestFrameBlock",
    "ContinueIfBlock",
    "CropBlock",
    "HasBrightnessBlock",
    "IMAGE_KIND",
    "ImageData",
    "InvertBlock",
    "ResizeBlock",
    "select_predictions",
    "MosaicBlock",
    "StaticCropBlock",
    "TopKBrightestBlock",
    "Window",
    "WORKFLOWS_V2_CATALOGUE",
    "create_catalogue",
]
