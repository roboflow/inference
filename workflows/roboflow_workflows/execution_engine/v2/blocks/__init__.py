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
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.operators.alignment import Align
from roboflow_workflows.execution_engine.v2.operators.window import Window


def create_catalogue() -> Catalogue:
    """Collect the image and temporal blocks, the operators and native media kinds.

    Returns:
        Immutable catalogue with ``v2/crop``, ``v2/static_crop``,
        ``v2/invert``, ``v2/mosaic``, ``v2/resize``, ``v2/has_brightness``,
        ``v2/continue_if``, ``v2/best_frame``, ``v2/top_k_brightest`` and the
        managed state blocks ``v2/state_get``, ``v2/state_set``,
        ``v2/state_increment`` and ``v2/state_compare_and_set``; the
        ``v2/align@v1`` and ``v2/window@v1`` operators; the ``image`` and
        ``crop_summary`` kinds, all eleven native prediction/tensor kinds, the
        ``instance_segmentation_view`` kind, and the built-in kinds these
        blocks use. Wildcard media outputs and the
        legacy ``numpy_array`` tensor label use explicit boundary policies.
        Recording codecs store tensors, ``ImageData`` and the native
        prediction carriers as raw binary data. A live ``SegmentationView``
        has no codec: ``full_res()`` is its explicit dense boundary for
        recording and the wire.
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
        ],
        operators=[Align, Window],
        kinds=[
            IMAGE_KIND,
            CROP_SUMMARY_KIND,
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
    "BAR_CODE_DETECTION_KIND",
    "CLASSIFICATION_PREDICTION_KIND",
    "DETECTION_KIND",
    "EMBEDDING_KIND",
    "INSTANCE_SEGMENTATION_PREDICTION_KIND",
    "INSTANCE_SEGMENTATION_VIEW_KIND",
    "KEYPOINT_DETECTION_PREDICTION_KIND",
    "NATIVE_KINDS",
    "NATIVE_RECORDING_CODECS",
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
    "MaskGridGeometry",
    "ResizeBlock",
    "SegmentationView",
    "select_predictions",
    "MosaicBlock",
    "StateCompareAndSetBlock",
    "StateGetBlock",
    "StateIncrementBlock",
    "StateMachineSetBlock",
    "StateSetBlock",
    "StaticCropBlock",
    "TopKBrightestBlock",
    "Window",
    "WORKFLOWS_V2_CATALOGUE",
    "create_catalogue",
]
