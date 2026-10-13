"""Model blocks and the host-owned model cache they read models from."""

from roboflow_workflows.v2.blocks.models.action_recognition import (
    ActionRecognitionBlock,
)
from roboflow_workflows.v2.blocks.models.cache import (
    ModelCache,
    ModelLoadError,
    ModelLoadRecursionError,
    ModelRequest,
    ModelRequestConflict,
    auto_model_loader,
)
from roboflow_workflows.v2.blocks.models.classification import (
    ClassificationBlock,
    MultiLabelClassificationBlock,
)
from roboflow_workflows.v2.blocks.models.detection import (
    ObjectDetectionBlock,
)
from roboflow_workflows.v2.blocks.models.instance_segmentation import (
    InstanceSegmentationBlock,
)
from roboflow_workflows.v2.blocks.models.keypoints import KeypointDetectionBlock
from roboflow_workflows.v2.blocks.models.sam_interactive import (
    SegmentAnything3InteractiveBlock,
)
from roboflow_workflows.v2.blocks.models.sam_video import (
    SegmentAnything2VideoBlock,
    SegmentAnything3VideoBlock,
)
from roboflow_workflows.v2.blocks.models.segment_anything import (
    SegmentAnything2Block,
    SegmentAnything3Block,
)
from roboflow_workflows.v2.blocks.models.semantic_segmentation import (
    SemanticSegmentationBlock,
)

from roboflow_workflows.v2.blocks.models.cosmos import Cosmos3EdgeBlock
from roboflow_workflows.v2.blocks.models.florence import Florence2Block
from roboflow_workflows.v2.blocks.models.remote_ocr import GoogleVisionOCRBlock
from roboflow_workflows.v2.blocks.models.moondream import Moondream2Block
from roboflow_workflows.v2.blocks.models.qwen_vlm import QwenVLMBlock

BLOCKS = (
    Cosmos3EdgeBlock,
    Florence2Block,
    GoogleVisionOCRBlock,
    Moondream2Block,
    QwenVLMBlock,

    ObjectDetectionBlock,
    ClassificationBlock,
    MultiLabelClassificationBlock,
    KeypointDetectionBlock,
    InstanceSegmentationBlock,
    SemanticSegmentationBlock,
    ActionRecognitionBlock,
    SegmentAnything2Block,
    SegmentAnything2VideoBlock,
    SegmentAnything3Block,
    SegmentAnything3InteractiveBlock,
    SegmentAnything3VideoBlock,
)

__all__ = [
    "Cosmos3EdgeBlock",
    "Florence2Block",
    "GoogleVisionOCRBlock",
    "Moondream2Block",
    "QwenVLMBlock",
    "BLOCKS",
    "ActionRecognitionBlock",
    "ClassificationBlock",
    "InstanceSegmentationBlock",
    "KeypointDetectionBlock",
    "ModelCache",
    "ModelLoadError",
    "ModelLoadRecursionError",
    "ModelRequest",
    "ModelRequestConflict",
    "MultiLabelClassificationBlock",
    "ObjectDetectionBlock",
    "SemanticSegmentationBlock",
    "auto_model_loader",
    "SegmentAnything2Block",
    "SegmentAnything2VideoBlock",
    "SegmentAnything3Block",
    "SegmentAnything3InteractiveBlock",
    "SegmentAnything3VideoBlock",
]
