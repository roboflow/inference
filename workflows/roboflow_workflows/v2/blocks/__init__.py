"""Opt-in V2 image, model, prediction, formatter and painter catalogue.

Callers select these blocks and the built-in operators explicitly::

    from roboflow_workflows.v2.blocks import create_catalogue

    catalogue = create_catalogue()

``Catalogue.from_modules(["roboflow_workflows.v2.blocks"])``
reads the same catalogue through ``WORKFLOWS_V2_CATALOGUE``. The generic V2
engine never imports this package, and the legacy V1 loader does not discover
it. Every contract detail lives on the block classes themselves.

Where an older V2 prototype and a later V1 port do the same job, both stay
registered under different types::

    v2/continue_if                     ready boolean (ContinueIfBlock)
    v2/continue_if@v1                  V1 UQL condition (ContinueIfQueryBlock)
    v2/label_visualization@v1          fast prototype (LabelVisualizationBlock)
    v2/label_visualization@v2          V1 port (AnnotationLabelVisualizationBlock)
    v2/box_visualization@v1            fast prototype (BoxVisualizationBlock)
    v2/bounding_box_visualization@v1   V1 port (BoundingBoxVisualizationBlock)
"""

from roboflow_workflows.v2.blocks.combine import BLOCKS as COMBINE_BLOCKS
from roboflow_workflows.v2.blocks.combine import (
    DetectionsCombineBlock,
    DetectionsMergeBlock,
)
from roboflow_workflows.v2.blocks.control import (
    ContinueIfBlock,
    ContinueIfQueryBlock,
    DeltaFilterBlock,
    HoldCapacityError,
    ObservationOrderError,
    RateLimiterBlock,
    SwitchCaseBlock,
)
from roboflow_workflows.v2.blocks.formatters import BLOCKS as FORMATTER_BLOCKS
from roboflow_workflows.v2.blocks.formatters import (
    TIMESTAMP_KIND,
    CsvFormatterBlock,
    CurrentTimeBlock,
    ExpressionBlock,
    FirstNonEmptyOrDefaultBlock,
    JsonParserBlock,
    PropertyDefinitionBlock,
    StringTemplateBlock,
)
from roboflow_workflows.v2.blocks.fusion import BLOCKS as FUSION_BLOCKS
from roboflow_workflows.v2.blocks.fusion import (
    DetectionsConsensusBlock,
    DetectionsDifferenceBlock,
)
from roboflow_workflows.v2.blocks.geometry import BLOCKS as GEOMETRY_BLOCKS
from roboflow_workflows.v2.blocks.geometry import (
    BoundingRectBlock,
    DetectionOffsetBlock,
    DetectionsClassesReplacementBlock,
    DetectionsStitchBlock,
    DimensionCollapseBlock,
)
from roboflow_workflows.v2.blocks.image import (
    AbsoluteStaticCropBlock,
    CropBlock,
    DynamicCropBlock,
    HasBrightnessBlock,
    ImageCopyBlock,
    ImagePreprocessingBlock,
    ImageSlicerBlock,
    InvertBlock,
    MosaicBlock,
    RelativeStaticCropBlock,
    ResizeBlock,
    StaticCropBlock,
)
from roboflow_workflows.v2.blocks.models import BLOCKS as MODEL_BLOCKS
from roboflow_workflows.v2.blocks.models import (
    ClassificationBlock,
    InstanceSegmentationBlock,
    KeypointDetectionBlock,
    MultiLabelClassificationBlock,
    ObjectDetectionBlock,
    SemanticSegmentationBlock,
)
from roboflow_workflows.v2.blocks.sampling import BLOCKS as SAMPLING_BLOCKS
from roboflow_workflows.v2.blocks.sampling import (
    IdentifyChangesBlock,
    IdentifyOutliersBlock,
    SamplingCapacityError,
)
from roboflow_workflows.v2.blocks.state.machines import StateMachineSetBlock
from roboflow_workflows.v2.blocks.state.object_cache import (
    CacheGetBlock,
    CacheSetBlock,
    ObjectCache,
)
from roboflow_workflows.v2.blocks.state.values import (
    StateCompareAndSetBlock,
    StateGetBlock,
    StateIncrementBlock,
    StateSetBlock,
)
from roboflow_workflows.v2.blocks.temporal.blocks import (
    BestFrameBlock,
    TopKBrightestBlock,
)
from roboflow_workflows.v2.blocks.temporal.frame_delay import (
    FrameDelayBlock,
    FrameHistoryCapacityError,
)
from roboflow_workflows.v2.blocks.tracking.byte_track import (
    ByteTrackerBlock,
    TrackerCapacityError,
)
from roboflow_workflows.v2.blocks.transforms import BLOCKS as TRANSFORM_BLOCKS
from roboflow_workflows.v2.blocks.transforms import (
    DetectionsFilterBlock,
    DetectionsTransformationBlock,
    PerClassConfidenceFilterBlock,
)
from roboflow_workflows.v2.blocks.visualization import BLOCKS as VISUALIZATION_BLOCKS
from roboflow_workflows.v2.blocks.visualization import (
    DETECTION_DRAWING_KIND,
    BoxVisualizationBlock,
    DetectionDrawing,
    DetectionDrawingBlock,
    DetectionVisualizationBlock,
    InPlaceLabelVisualizationBlock,
    LabelVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BLOCKS as ANNOTATION_BLOCKS,
)
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BlurVisualizationBlock,
    BoundingBoxVisualizationBlock,
    ClassificationLabelVisualizationBlock,
    CropVisualizationBlock,
    KeypointVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations import (
    LabelVisualizationBlock as AnnotationLabelVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations import (
    MaskVisualizationBlock,
    PixelateVisualizationBlock,
    PolygonVisualizationBlock,
    SemanticSegmentationVisualizationBlock,
)
from roboflow_workflows.v2.data.boundaries import MEDIA_WILDCARD_KIND, NUMPY_ARRAY_KIND
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import CROP_SUMMARY_KIND, IMAGE_KIND
from roboflow_workflows.v2.data.masks.grid import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    MaskGridGeometry,
    SegmentationView,
)
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.predictions import (
    BAR_CODE_DETECTION_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    DETECTION_KIND,
    EMBEDDING_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    NATIVE_KINDS,
    NATIVE_SEMANTIC_SEGMENTATION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    QR_CODE_DETECTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
    TENSOR_KIND,
)
from roboflow_workflows.v2.data.rows import PREDICTION_ROWS_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.operators.alignment import Align
from roboflow_workflows.v2.engine.operators.window import Window
from roboflow_workflows.v2.engine.resources import Factory
from roboflow_workflows.v2.recording_codecs import NATIVE_RECORDING_CODECS


def create_catalogue() -> Catalogue:
    """Collect the opt-in V2 vision, temporal and managed-state blocks.

    Models remain caller-owned resources: model blocks read the session's
    ``models`` resource. Building the catalogue loads no model and creates
    no resource. Cache blocks receive one lazily created ObjectCache per session;
    callers can supply an object_cache resource to share it explicitly.
    The generic V2 engine does not import this catalogue; hosts
    select it explicitly.

    Returns:
        Catalogue with image blocks and crop ports, caller-loaded batched model
        blocks, per-source tracking, region and parent-mapping, prediction
        transforms, merge and combine, formatters, boolean and query gates,
        prototype detection painters and V1 visualization ports, temporal
        and managed-state blocks, alignment/window operators and native
        media kinds, including native semantic grids and opt-in
        ``PredictionRows``. Kinds that only a block's outputs name, such as
        ``TIMESTAMP_KIND``, register with that block.
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
            AbsoluteStaticCropBlock,
            RelativeStaticCropBlock,
            DynamicCropBlock,
            ImagePreprocessingBlock,
            ImageSlicerBlock,
            ImageCopyBlock,
            DeltaFilterBlock,
            RateLimiterBlock,
            SwitchCaseBlock,
            ContinueIfBlock,
            ContinueIfQueryBlock,
            BestFrameBlock,
            TopKBrightestBlock,
            FrameDelayBlock,
            CacheGetBlock,
            CacheSetBlock,
            StateGetBlock,
            StateSetBlock,
            StateIncrementBlock,
            StateCompareAndSetBlock,
            StateMachineSetBlock,
            *MODEL_BLOCKS,
            ByteTrackerBlock,
            *GEOMETRY_BLOCKS,
            *TRANSFORM_BLOCKS,
            *COMBINE_BLOCKS,
            *FUSION_BLOCKS,
            *SAMPLING_BLOCKS,
            *FORMATTER_BLOCKS,
            *VISUALIZATION_BLOCKS,
            *ANNOTATION_BLOCKS,
        ],
        operators=[Align, Window],
        kinds=[
            IMAGE_KIND,
            CROP_SUMMARY_KIND,
            DETECTION_DRAWING_KIND,
            *NATIVE_KINDS,
            PREDICTION_ROWS_KIND,
            INSTANCE_SEGMENTATION_VIEW_KIND,
            MEDIA_WILDCARD_KIND,
            NUMPY_ARRAY_KIND,
        ],
        codecs=NATIVE_RECORDING_CODECS,
        providers={"object_cache": Factory(ObjectCache)},
    )

    return catalogue


WORKFLOWS_V2_CATALOGUE = create_catalogue

__all__ = [
    "AbsoluteStaticCropBlock",
    "Align",
    "AnnotationLabelVisualizationBlock",
    "BAR_CODE_DETECTION_KIND",
    "BestFrameBlock",
    "BlurVisualizationBlock",
    "BoundingBoxVisualizationBlock",
    "BoundingRectBlock",
    "BoxVisualizationBlock",
    "ByteTrackerBlock",
    "CacheGetBlock",
    "CacheSetBlock",
    "CLASSIFICATION_PREDICTION_KIND",
    "CROP_SUMMARY_KIND",
    "ClassificationBlock",
    "ClassificationLabelVisualizationBlock",
    "ContinueIfBlock",
    "ContinueIfQueryBlock",
    "CropBlock",
    "CropVisualizationBlock",
    "CsvFormatterBlock",
    "CurrentTimeBlock",
    "DETECTION_DRAWING_KIND",
    "DETECTION_KIND",
    "DeltaFilterBlock",
    "DetectionDrawing",
    "DetectionDrawingBlock",
    "DetectionOffsetBlock",
    "DetectionVisualizationBlock",
    "DetectionsClassesReplacementBlock",
    "DetectionsCombineBlock",
    "DetectionsConsensusBlock",
    "DetectionsDifferenceBlock",
    "DetectionsFilterBlock",
    "DetectionsMergeBlock",
    "DetectionsStitchBlock",
    "DetectionsTransformationBlock",
    "DimensionCollapseBlock",
    "DynamicCropBlock",
    "EMBEDDING_KIND",
    "ExpressionBlock",
    "FirstNonEmptyOrDefaultBlock",
    "FrameDelayBlock",
    "FrameHistoryCapacityError",
    "HasBrightnessBlock",
    "HoldCapacityError",
    "IMAGE_KIND",
    "INSTANCE_SEGMENTATION_PREDICTION_KIND",
    "INSTANCE_SEGMENTATION_VIEW_KIND",
    "ImageData",
    "ImageCopyBlock",
    "IdentifyChangesBlock",
    "IdentifyOutliersBlock",
    "ImagePreprocessingBlock",
    "ImageSlicerBlock",
    "InPlaceLabelVisualizationBlock",
    "InstanceSegmentationBlock",
    "InvertBlock",
    "JsonParserBlock",
    "KEYPOINT_DETECTION_PREDICTION_KIND",
    "KeypointDetectionBlock",
    "KeypointVisualizationBlock",
    "LabelVisualizationBlock",
    "MEDIA_WILDCARD_KIND",
    "MaskGridGeometry",
    "MaskVisualizationBlock",
    "MosaicBlock",
    "MultiLabelClassificationBlock",
    "NATIVE_KINDS",
    "NATIVE_RECORDING_CODECS",
    "NATIVE_SEMANTIC_SEGMENTATION_KIND",
    "NUMPY_ARRAY_KIND",
    "OBJECT_DETECTION_PREDICTION_KIND",
    "ObjectDetectionBlock",
    "ObjectCache",
    "ObservationOrderError",
    "PREDICTION_ROWS_KIND",
    "PerClassConfidenceFilterBlock",
    "PixelateVisualizationBlock",
    "PolygonVisualizationBlock",
    "PropertyDefinitionBlock",
    "QR_CODE_DETECTION_KIND",
    "RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND",
    "RateLimiterBlock",
    "RelativeStaticCropBlock",
    "ResizeBlock",
    "SEMANTIC_SEGMENTATION_PREDICTION_KIND",
    "SegmentationView",
    "SamplingCapacityError",
    "SemanticSegmentationBlock",
    "SemanticSegmentationVisualizationBlock",
    "StateCompareAndSetBlock",
    "StateGetBlock",
    "StateIncrementBlock",
    "StateMachineSetBlock",
    "StateSetBlock",
    "StaticCropBlock",
    "StringTemplateBlock",
    "SwitchCaseBlock",
    "TENSOR_KIND",
    "TIMESTAMP_KIND",
    "TopKBrightestBlock",
    "TrackerCapacityError",
    "WORKFLOWS_V2_CATALOGUE",
    "Window",
    "create_catalogue",
    "select_predictions",
]
