import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Optional, Tuple, Type, Union

import supervision as sv
from fastapi import BackgroundTasks
from pydantic import AliasChoices, ConfigDict, Field
from typing_extensions import Annotated

from inference.core.cache.base import BaseCache
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v1 import (
    register_datapoint_at_roboflow,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

FloatZeroToHundred = Annotated[float, Field(ge=0.0, le=100.0)]

SHORT_DESCRIPTION = "Save images and predictions to your Roboflow Dataset."

LONG_DESCRIPTION = """
Block let users save their images and predictions into Roboflow Dataset. Persisting data from
production environments helps iteratively building more robust models. 

Block provides configuration options to decide how data should be stored and what are the limits 
to be applied. We advice using this block in combination with rate limiter blocks to effectively 
collect data that the model struggle with.
"""

WORKSPACE_NAME_CACHE_EXPIRE = 900  # 15 min
TIMESTAMP_FORMAT = "%Y_%m_%d"
DUPLICATED_STATUS = "Duplicated image"
BatchCreationFrequency = Literal["never", "daily", "weekly", "monthly"]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Roboflow Dataset Upload",
            "version": "v2",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-upload",
                "blockPriority": 0,
                "popular": True,
                "requires_rf_key": True,
            },
        }
    )
    type: Literal["roboflow_core/roboflow_dataset_upload@v2"]
    images: Selector(kind=[IMAGE_KIND]) = Field(
        title="Image",
        description="The image to upload.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    target_project: Union[Selector(kind=[ROBOFLOW_PROJECT_KIND]), str] = Field(
        description="Roboflow project where data will be saved.",
        examples=["my_dataset", "$inputs.target_al_dataset"],
    )
    predictions: Optional[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
                CLASSIFICATION_PREDICTION_KIND,
            ]
        )
    ] = Field(
        default=None,
        description="Model predictions to be uploaded.",
        examples=["$steps.object_detection_model.predictions"],
        json_schema_extra={"always_visible": True},
    )
    data_percentage: Union[FloatZeroToHundred, Selector(kind=[FLOAT_KIND])] = Field(
        default=100,
        description="Percent of data that will be saved (0.0 to 100.0).",
        examples=[True, False, "$inputs.persist_predictions"],
    )
    minutely_usage_limit: int = Field(
        default=10,
        description="Maximum number of image uploads allowed per minute.",
        examples=[10, 60],
    )
    hourly_usage_limit: int = Field(
        default=100,
        description="Maximum number of image uploads allowed per hour.",
        examples=[10, 60],
    )
    daily_usage_limit: int = Field(
        default=1000,
        description="Maximum number of image uploads allowed per day.",
        examples=[10, 60],
    )
    usage_quota_name: str = Field(
        description="A unique identifier for tracking usage quotas (minutely, hourly, daily limits).",
        examples=["quota-for-data-sampling-1"],
        json_schema_extra={"hidden": True},
    )
    max_image_size: Tuple[int, int] = Field(
        default=(1920, 1080),
        description="Maximum size of the image to be saved. Bigger images will be "
        "downsized preserving aspect ratio.",
        examples=[(1920, 1080), (512, 512)],
    )
    compression_level: int = Field(
        default=95,
        gt=0,
        le=100,
        description="Compression level for the registered image.",
        examples=[95, 75],
    )
    registration_tags: List[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default_factory=list,
        description="Tags to be attached to the registered image.",
        examples=[["location-florida", "factory-name", "$inputs.dynamic_tag"]],
    )
    persist_predictions: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to specify if model predictions should be saved along with the image.",
        examples=[True, False, "$inputs.persist_predictions"],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Boolean flag to disable block execution.",
        examples=[True, "$inputs.disable_active_learning"],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to run the block asynchronously (True) for faster workflows or  "
        "synchronously (False) for debugging and error handling.",
    )
    labeling_batch_prefix: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="workflows_data_collector",
        description="Target batch name for the registered image.",
        examples=["my_labeling_batch_name"],
    )
    labeling_batches_recreation_frequency: BatchCreationFrequency = Field(
        default="never",
        description="Frequency in which new labeling batches are created for uploaded images.",
        examples=["never", "daily"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images", "predictions"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowDatasetUploadBlockV2(WorkflowBlock):

    def __init__(
        self,
        cache: BaseCache,
        api_key: Optional[str],
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        self._cache = cache
        self._api_key = api_key
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["cache", "api_key", "background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        predictions: Optional[Batch[Union[sv.Detections, dict]]],
        target_project: str,
        usage_quota_name: str,
        data_percentage: float,
        minutely_usage_limit: int,
        persist_predictions: bool,
        hourly_usage_limit: int,
        daily_usage_limit: int,
        max_image_size: Tuple[int, int],
        compression_level: int,
        registration_tags: List[str],
        disable_sink: bool,
        fire_and_forget: bool,
        labeling_batch_prefix: str,
        labeling_batches_recreation_frequency: BatchCreationFrequency,
    ) -> BlockResult:
        if self._api_key is None:
            raise ValueError(
                "RoboflowDataCollector block cannot run without Roboflow API key. "
                "If you do not know how to get API key - visit "
                "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to "
                "retrieve one."
            )
        if disable_sink:
            return [
                {
                    "error_status": False,
                    "message": "Sink was disabled by parameter `disable_sink`",
                }
                for _ in range(len(images))
            ]
        result = []
        predictions = [None] * len(images) if predictions is None else predictions
        for image, prediction in zip(images, predictions):
            error_status, message = maybe_register_datapoint_at_roboflow(
                image=image,
                prediction=prediction,
                target_project=target_project,
                usage_quota_name=usage_quota_name,
                data_percentage=data_percentage,
                persist_predictions=persist_predictions,
                minutely_usage_limit=minutely_usage_limit,
                hourly_usage_limit=hourly_usage_limit,
                daily_usage_limit=daily_usage_limit,
                max_image_size=max_image_size,
                compression_level=compression_level,
                registration_tags=registration_tags,
                fire_and_forget=fire_and_forget,
                labeling_batch_prefix=labeling_batch_prefix,
                new_labeling_batch_frequency=labeling_batches_recreation_frequency,
                cache=self._cache,
                background_tasks=self._background_tasks,
                thread_pool_executor=self._thread_pool_executor,
                api_key=self._api_key,
            )
            result.append({"error_status": error_status, "message": message})
        return result


def maybe_register_datapoint_at_roboflow(
    image: WorkflowImageData,
    prediction: Optional[Union[sv.Detections, dict]],
    target_project: str,
    usage_quota_name: str,
    data_percentage: float,
    persist_predictions: bool,
    minutely_usage_limit: int,
    hourly_usage_limit: int,
    daily_usage_limit: int,
    max_image_size: Tuple[int, int],
    compression_level: int,
    registration_tags: List[str],
    fire_and_forget: bool,
    labeling_batch_prefix: str,
    new_labeling_batch_frequency: BatchCreationFrequency,
    cache: BaseCache,
    background_tasks: Optional[BackgroundTasks],
    thread_pool_executor: Optional[ThreadPoolExecutor],
    api_key: str,
) -> Tuple[bool, str]:
    normalised_probability = data_percentage / 100
    if random.random() < normalised_probability:
        return register_datapoint_at_roboflow(
            image=image,
            prediction=prediction,
            target_project=target_project,
            usage_quota_name=usage_quota_name,
            persist_predictions=persist_predictions,
            minutely_usage_limit=minutely_usage_limit,
            hourly_usage_limit=hourly_usage_limit,
            daily_usage_limit=daily_usage_limit,
            max_image_size=max_image_size,
            compression_level=compression_level,
            registration_tags=registration_tags,
            fire_and_forget=fire_and_forget,
            labeling_batch_prefix=labeling_batch_prefix,
            new_labeling_batch_frequency=new_labeling_batch_frequency,
            cache=cache,
            background_tasks=background_tasks,
            thread_pool_executor=thread_pool_executor,
            api_key=api_key,
        )
    return False, "Registration skipped due to sampling settings"
