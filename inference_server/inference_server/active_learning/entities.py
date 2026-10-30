"""Entities of active learning: task types, errors and project configuration."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from roboflow_workflows.core_steps.sinks.roboflow.dataset_upload.active_learning import (
    ImageDimensions,
    StrategyLimit,
    StrategyLimitType,
)

__all__ = [
    "ActiveLearningConfiguration",
    "ActiveLearningConfigurationDecodingError",
    "ActiveLearningConfigurationError",
    "ActiveLearningError",
    "BatchReCreationInterval",
    "CLASSIFICATION_TASK",
    "INSTANCE_SEGMENTATION_TASK",
    "ImageDimensions",
    "KEYPOINTS_DETECTION_TASK",
    "MULTI_LABEL_CLASSIFICATION_TASK",
    "OBJECT_DETECTION_TASK",
    "PredictionFormatNotSupported",
    "RoboflowProjectMetadata",
    "SamplingMethod",
    "StrategyLimit",
    "StrategyLimitType",
]

CLASSIFICATION_TASK = "classification"
MULTI_LABEL_CLASSIFICATION_TASK = "multi-label-classification"
OBJECT_DETECTION_TASK = "object-detection"
INSTANCE_SEGMENTATION_TASK = "instance-segmentation"
KEYPOINTS_DETECTION_TASK = "keypoint-detection"

DatasetID = str
WorkspaceID = str
LocalImageIdentifier = str
PredictionType = str
Prediction = dict
SerialisedPrediction = str
PredictionFileType = str


class ActiveLearningError(Exception):
    """Base class of active learning errors."""


class PredictionFormatNotSupported(ActiveLearningError):
    """Raised when a prediction has no annotation format the platform accepts."""


class ActiveLearningConfigurationDecodingError(ActiveLearningError):
    """Raised when a project configuration document cannot be decoded."""


class ActiveLearningConfigurationError(ActiveLearningError):
    """Raised when a project configuration is inconsistent."""


@dataclass(frozen=True)
class SamplingMethod:
    """Named sampling function deciding whether a datapoint is registered."""

    name: str
    sample: Callable[[np.ndarray, Prediction, PredictionType], bool]


class BatchReCreationInterval(Enum):
    """How often a new labeling batch name is started."""

    NEVER = "never"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass(frozen=True)
class ActiveLearningConfiguration:
    """Active learning configuration of one target project."""

    max_image_size: Optional[ImageDimensions]
    jpeg_compression_level: int
    persist_predictions: bool
    sampling_methods: List[SamplingMethod]
    batches_name_prefix: str
    batch_recreation_interval: BatchReCreationInterval
    max_batch_images: Optional[int]
    workspace_id: WorkspaceID
    dataset_id: DatasetID
    model_id: str
    strategies_limits: Dict[str, List[StrategyLimit]]
    tags: List[str]
    strategies_tags: Dict[str, List[str]]

    @classmethod
    def init(
        cls,
        roboflow_api_configuration: Dict[str, Any],
        sampling_methods: List[SamplingMethod],
        workspace_id: WorkspaceID,
        dataset_id: DatasetID,
        model_id: str,
    ) -> "ActiveLearningConfiguration":
        """Build the configuration from the platform document.

        Args:
            roboflow_api_configuration: Configuration document of the project.
            sampling_methods: Sampling methods initialised from the document.
            workspace_id: Workspace datapoints are registered in.
            dataset_id: Project datapoints are registered in.
            model_id: Model whose predictions are registered.

        Returns:
            The decoded configuration.

        Raises:
            ActiveLearningConfigurationDecodingError: If a required key is missing
                or a value is not allowed.
        """
        try:
            max_image_size = roboflow_api_configuration.get("max_image_size")
            if max_image_size is not None:
                max_image_size = ImageDimensions(
                    height=roboflow_api_configuration["max_image_size"][0],
                    width=roboflow_api_configuration["max_image_size"][1],
                )
            strategies_limits = {
                strategy["name"]: [
                    StrategyLimit.from_dict(specification=specification)
                    for specification in strategy.get("limits", [])
                ]
                for strategy in roboflow_api_configuration["sampling_strategies"]
            }
            strategies_tags = {
                strategy["name"]: strategy.get("tags", [])
                for strategy in roboflow_api_configuration["sampling_strategies"]
            }
            configuration = cls(
                max_image_size=max_image_size,
                jpeg_compression_level=roboflow_api_configuration.get(
                    "jpeg_compression_level", 95
                ),
                persist_predictions=roboflow_api_configuration["persist_predictions"],
                sampling_methods=sampling_methods,
                batches_name_prefix=roboflow_api_configuration["batching_strategy"][
                    "batches_name_prefix"
                ],
                batch_recreation_interval=BatchReCreationInterval(
                    roboflow_api_configuration["batching_strategy"][
                        "recreation_interval"
                    ]
                ),
                max_batch_images=roboflow_api_configuration["batching_strategy"].get(
                    "max_batch_images"
                ),
                workspace_id=workspace_id,
                dataset_id=dataset_id,
                model_id=model_id,
                strategies_limits=strategies_limits,
                tags=roboflow_api_configuration.get("tags", []),
                strategies_tags=strategies_tags,
            )
        except (KeyError, ValueError) as error:
            raise ActiveLearningConfigurationDecodingError(
                f"Failed to initialise Active Learning configuration. Cause: {str(error)}"
            ) from error

        return configuration


@dataclass(frozen=True)
class RoboflowProjectMetadata:
    """Project facts the configuration is built from, as kept in the cache."""

    dataset_id: DatasetID
    workspace_id: WorkspaceID
    dataset_type: str
    active_learning_configuration: dict
