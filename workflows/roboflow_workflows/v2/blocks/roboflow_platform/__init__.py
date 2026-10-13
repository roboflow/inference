"""Roboflow platform blocks behind explicit, offline-by-default ports."""

from roboflow_workflows.v2.blocks.roboflow_platform.asset_library import (
    AssetLibraryAttributesBlock,
)
from roboflow_workflows.v2.blocks.roboflow_platform.dataset_upload import (
    DatasetUploadBlock,
)
from roboflow_workflows.v2.blocks.roboflow_platform.model_monitoring import (
    ModelMonitoringInferenceAggregatorBlock,
)

PLATFORM_BLOCKS = (
    AssetLibraryAttributesBlock,
    DatasetUploadBlock,
    ModelMonitoringInferenceAggregatorBlock,
)

__all__ = [
    "PLATFORM_BLOCKS",
    "AssetLibraryAttributesBlock",
    "DatasetUploadBlock",
    "ModelMonitoringInferenceAggregatorBlock",
]
