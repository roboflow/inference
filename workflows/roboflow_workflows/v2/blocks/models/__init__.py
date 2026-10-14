"""Model blocks and the host-owned model cache they read models from."""

from roboflow_workflows.v2.blocks.models.roboflow_action_recognition_model.v1 import ActionRecognitionBlock
from roboflow_workflows.v2.blocks.models.cache import (
    ModelCache,
    ModelLoadError,
    ModelLoadRecursionError,
    ModelRequest,
    ModelRequestConflict,
    auto_model_loader,
)
from roboflow_workflows.v2.blocks.models.roboflow_classification_model.v3 import ClassificationBlock
from roboflow_workflows.v2.blocks.models.roboflow_multi_label_classification_model.v3 import MultiLabelClassificationBlock
from roboflow_workflows.v2.blocks.models.roboflow_object_detection_model.v3 import ObjectDetectionBlock
from roboflow_workflows.v2.blocks.models.roboflow_instance_segmentation_model.v4 import InstanceSegmentationBlock
from roboflow_workflows.v2.blocks.models.roboflow_keypoint_detection_model.v3 import KeypointDetectionBlock
from roboflow_workflows.v2.blocks.models.sam3_interactive.v1 import SegmentAnything3InteractiveBlock
from roboflow_workflows.v2.blocks.models.segment_anything_2_video.v1 import SegmentAnything2VideoBlock
from roboflow_workflows.v2.blocks.models.sam3_video.v1 import SegmentAnything3VideoBlock
from roboflow_workflows.v2.blocks.models.segment_anything.v1 import SegmentAnything2Block
from roboflow_workflows.v2.blocks.models.sam3.v3 import SegmentAnything3Block
from roboflow_workflows.v2.blocks.models.roboflow_semantic_segmentation_model.v2 import SemanticSegmentationBlock

from roboflow_workflows.v2.blocks.models.cosmos3_edge.v1 import Cosmos3EdgeBlock
from roboflow_workflows.v2.blocks.models.florence_2.v2 import Florence2Block
from roboflow_workflows.v2.blocks.models.google_vision_ocr.v1 import GoogleVisionOCRBlock
from roboflow_workflows.v2.blocks.models.moondream2.v1 import Moondream2Block
from roboflow_workflows.v2.blocks.models.qwen_vlm.v4 import QwenVLMBlock

from roboflow_workflows.v2.blocks.models.qwen3_5vl.v2 import Qwen35VLBlock

from roboflow_workflows.v2.blocks.models.clip_comparison.v2 import ClipComparisonBlock
from roboflow_workflows.v2.blocks.models.depth_estimation.v1 import DepthEstimationBlock
from roboflow_workflows.v2.blocks.models.clip.v1 import ClipEmbeddingBlock
from roboflow_workflows.v2.blocks.models.perception_encoder.v1 import PerceptionEncoderEmbeddingBlock
from roboflow_workflows.v2.blocks.models.glm_ocr.v1 import GlmOcrBlock
from roboflow_workflows.v2.blocks.models.ocr_model.v1 import OCRModelBlock
from roboflow_workflows.v2.blocks.models.easy_ocr.v1 import EasyOCRBlock
from roboflow_workflows.v2.blocks.models.pp_ocr.v1 import PPOCRBlock
from roboflow_workflows.v2.blocks.models.smolvlm2.v1 import SmolVLM2Block

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
    Qwen35VLBlock,
    ClipComparisonBlock,
    ClipEmbeddingBlock,
    PerceptionEncoderEmbeddingBlock,
    DepthEstimationBlock,
    GlmOcrBlock,
    OCRModelBlock,
    EasyOCRBlock,
    PPOCRBlock,
    SmolVLM2Block,
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
    "Qwen35VLBlock",
    "ClipComparisonBlock",
    "ClipEmbeddingBlock",
    "PerceptionEncoderEmbeddingBlock",
    "DepthEstimationBlock",
    "GlmOcrBlock",
    "OCRModelBlock",
    "EasyOCRBlock",
    "PPOCRBlock",
    "SmolVLM2Block",
]
