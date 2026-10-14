"""Roboflow platform blocks behind explicit, offline-by-default ports."""

from roboflow_workflows.v2.blocks.roboflow_platform.asset_library_attributes.v1 import AssetLibraryAttributesBlock
from roboflow_workflows.v2.blocks.roboflow_platform.roboflow_custom_metadata.v1 import RoboflowCustomMetadataBlock
from roboflow_workflows.v2.blocks.roboflow_platform.roboflow_dataset_upload.v2 import DatasetUploadBlock
from roboflow_workflows.v2.blocks.roboflow_platform.model_monitoring_inference_aggregator.v1 import ModelMonitoringInferenceAggregatorBlock
from roboflow_workflows.v2.blocks.roboflow_platform.roboflow_vision_events.v1 import RoboflowVisionEventsBlock
from roboflow_workflows.v2.blocks.roboflow_platform.vision_event_bundle.v1 import VisionEventBundleBlock
from roboflow_workflows.v2.blocks.roboflow_platform.visual_search.v1 import VisualSearchBlock
from roboflow_workflows.v2.blocks.roboflow_platform.visual_search_classifier.v1 import VisualSearchClassifierBlock

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
