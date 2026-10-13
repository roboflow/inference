"""Roboflow platform blocks behind explicit, offline-by-default ports."""

from roboflow_workflows.v2.blocks.roboflow_platform.asset_library import (
    AssetLibraryAttributesBlock,
)
from roboflow_workflows.v2.blocks.roboflow_platform.custom_metadata import (
    RoboflowCustomMetadataBlock,
)
from roboflow_workflows.v2.blocks.roboflow_platform.dataset_upload import (
    DatasetUploadBlock,
)
from roboflow_workflows.v2.blocks.roboflow_platform.model_monitoring import (
    ModelMonitoringInferenceAggregatorBlock,
)
from roboflow_workflows.v2.blocks.roboflow_platform.vision_events import (
    RoboflowVisionEventsBlock,
    VisionEventBundleBlock,
)
from roboflow_workflows.v2.blocks.roboflow_platform.visual_search import (
    VisualSearchBlock,
    VisualSearchClassifierBlock,
)

PLATFORM_BLOCKS = (
    AssetLibraryAttributesBlock,
    DatasetUploadBlock,
    ModelMonitoringInferenceAggregatorBlock,
    RoboflowCustomMetadataBlock,
    VisualSearchBlock,
    VisualSearchClassifierBlock,
    RoboflowVisionEventsBlock,
    VisionEventBundleBlock,
)

BLOCKS = PLATFORM_BLOCKS

__all__ = [
    "BLOCKS",
    "PLATFORM_BLOCKS",
    "AssetLibraryAttributesBlock",
    "DatasetUploadBlock",
    "ModelMonitoringInferenceAggregatorBlock",
    "RoboflowCustomMetadataBlock",
    "RoboflowVisionEventsBlock",
    "VisionEventBundleBlock",
    "VisualSearchBlock",
    "VisualSearchClassifierBlock",
]
