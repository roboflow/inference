"""Model blocks and the host-owned model cache they read models from."""

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
from roboflow_workflows.v2.blocks.models.semantic_segmentation import (
    SemanticSegmentationBlock,
)

BLOCKS = (
    ObjectDetectionBlock,
    ClassificationBlock,
    MultiLabelClassificationBlock,
    KeypointDetectionBlock,
    InstanceSegmentationBlock,
    SemanticSegmentationBlock,
)

__all__ = [
    "BLOCKS",
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
]
