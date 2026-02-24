"""
*****************************************************************
*                           WARNING!                            *
*****************************************************************
This module contains the utility functions used by
RoboflowDatasetUploadBlockV2.

We do not recommend making multiple blocks dependent on the same code,
but the change between v1 and v2 was basically the default value of
some parameter - hence we decided not to replicate the code.

If you need to modify this module beware that you may introduce
change to RoboflowDatasetUploadBlockV2! If that happens,
probably that's the time to disentangle those blocks and copy the
code.
"""

import hashlib
import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import partial
from typing import List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import supervision as sv
from fastapi import BackgroundTasks
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.active_learning.cache_operations import (
    return_strategy_credit,
    use_credit_of_matching_strategy,
)
from inference.core.active_learning.core import prepare_image_to_registration
from inference.core.active_learning.entities import (
    ImageDimensions,
    StrategyLimit,
    StrategyLimitType,
)
from inference.core.cache.base import BaseCache
from inference.core.roboflow_api import (
    annotate_image_at_roboflow,
    get_roboflow_workspace,
    register_image_at_roboflow,
)
from inference.core.workflows.core_steps.common.serializers import (
    serialise_sv_detections,
)
from inference.core.workflows.core_steps.common.utils import scale_sv_detections
from inference.core.workflows.execution_engine.constants import INFERENCE_ID_KEY
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Save images and predictions to your Roboflow Dataset."

LONG_DESCRIPTION = """
Upload images and model predictions to a Roboflow dataset for active learning, model improvement, and data collection, with configurable usage quotas, batch organization, image compression, and optional annotation persistence.

## How This Block Works

This block uploads workflow images and predictions to your Roboflow dataset for storage, labeling, and model training. The block:

1. Takes images and optional model predictions (object detection, instance segmentation, keypoint detection, or classification) as input
2. Validates the Roboflow API key is available (required for uploading)
3. Checks usage quotas (minutely, hourly, daily limits) to ensure uploads stay within configured rate limits for active learning strategies
4. Prepares images by resizing if they exceed maximum size (maintaining aspect ratio) and compressing to specified quality level
5. Generates labeling batch names based on the prefix and batch creation frequency (never, daily, weekly, or monthly), organizing uploaded data into batches
6. Optionally persists model predictions as annotations if `persist_predictions` is enabled, allowing predictions to serve as pre-labels for review and correction
7. Attaches registration tags to images for organization and filtering in the Roboflow platform
8. Registers the image (and annotations if enabled) to the specified Roboflow project via the Roboflow API
9. Executes synchronously or asynchronously based on `fire_and_forget` setting, allowing non-blocking uploads for faster workflow execution
10. Returns error status and messages indicating upload success or failure

The block supports active learning workflows by implementing usage quotas that prevent excessive data collection, helping focus on collecting valuable training data within rate limits. Images are organized into labeling batches that can be automatically recreated on a schedule (daily, weekly, monthly), making it easier to manage and review collected data over time. The block can operate in fire-and-forget mode for asynchronous execution, allowing workflows to continue processing without waiting for uploads to complete, or synchronously for debugging and error handling.

## Requirements

**API Key Required**: This block requires a valid Roboflow API key to upload data. The API key must be configured in your environment or workflow configuration. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to retrieve an API key.

## Common Use Cases

- **Active Learning Data Collection**: Collect images and predictions from production environments where models struggle or are uncertain (e.g., low-confidence detections, edge cases), enabling iterative model improvement by gathering challenging examples for retraining
- **Production Data Logging**: Continuously upload production inference data to Roboflow datasets for monitoring, analysis, and future model training, creating a growing dataset from real-world deployments
- **Pre-Labeled Data Collection**: Upload images with model predictions as pre-labels (when `persist_predictions` is enabled), accelerating annotation workflows by providing initial labels that can be reviewed and corrected rather than starting from scratch
- **Stratified Data Sampling**: Use rate limiting and quotas to selectively collect data based on specific criteria (e.g., combine with Rate Limiter or Continue If blocks), ensuring diverse and balanced dataset collection without overwhelming storage or annotation resources
- **Batch-Based Labeling Workflows**: Organize uploaded data into batches with automatic recreation schedules (daily, weekly, monthly), making it easier to manage labeling tasks, track progress, and organize data collection efforts over time
- **Tagged Data Organization**: Attach metadata tags to uploaded images (e.g., location, camera ID, time period, model version), enabling filtering and organization of collected data in Roboflow for better dataset management and analysis

## Connecting to Other Blocks

This block receives data from workflow steps and uploads it to Roboflow:

- **After detection or analysis blocks** (e.g., Object Detection Model, Instance Segmentation Model, Classification Model, Keypoint Detection Model) to upload images along with their predictions, enabling active learning by collecting inference data with model outputs for annotation and retraining
- **After filtering or analytics blocks** (e.g., Detections Filter, Continue If, Overlap Filter) to selectively upload only specific types of data (e.g., low-confidence detections, overlapping objects, specific classes), focusing data collection on valuable edge cases or interesting scenarios
- **After rate limiter blocks** (e.g., Rate Limiter) to throttle upload frequency and stay within usage quotas, ensuring controlled data collection that respects rate limits and prevents excessive storage usage
- **Image inputs or preprocessing blocks** to upload raw images or processed images (e.g., crops, transformed images) without predictions, enabling collection of image data for future labeling or analysis
- **Conditional workflows** using flow control blocks (e.g., Continue If) to upload data only when certain conditions are met (e.g., upload only when detection count exceeds threshold, upload only errors or failures), enabling selective data collection based on workflow state
- **Batch processing workflows** where multiple images or predictions are generated, allowing bulk upload of workflow outputs to Roboflow datasets for organized data collection and management
"""

WORKSPACE_NAME_CACHE_EXPIRE = 900  # 15 min
TIMESTAMP_FORMAT = "%Y_%m_%d"
DUPLICATED_STATUS = "Duplicated image"
BatchCreationFrequency = Literal["never", "daily", "weekly", "monthly"]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Roboflow Dataset Upload",
            "version": "v1",
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
    type: Literal["roboflow_core/roboflow_dataset_upload@v1", "RoboflowDatasetUpload"]
    images: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Image(s) to upload to the Roboflow dataset. Can be a single image or batch of images from workflow inputs or processing steps. Images are resized if they exceed max_image_size and compressed before uploading. Supports batch processing.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
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
        description="Optional model predictions to upload alongside images. Predictions are saved as annotations (pre-labels) in the Roboflow dataset when persist_predictions is enabled, allowing predictions to serve as starting points for annotation review and correction. Supports object detection, instance segmentation, keypoint detection, and classification predictions. If None, only images are uploaded.",
        examples=["$steps.object_detection_model.predictions"],
    )
    target_project: Union[Selector(kind=[ROBOFLOW_PROJECT_KIND]), str] = Field(
        description="Roboflow project identifier where uploaded images and annotations will be saved. Must be a valid project in your Roboflow workspace. The project name can be specified directly or referenced from workflow inputs.",
        examples=["my_project", "$inputs.target_project"],
    )
    minutely_usage_limit: int = Field(
        default=10,
        description="Maximum number of image uploads allowed per minute for this quota. Part of the usage quota system that enforces rate limits for active learning data collection. Uploads exceeding this limit are skipped to prevent excessive data collection. Works together with hourly_usage_limit and daily_usage_limit to provide multi-level rate limiting.",
        examples=[10, 60],
    )
    hourly_usage_limit: int = Field(
        default=100,
        description="Maximum number of image uploads allowed per hour for this quota. Part of the usage quota system that enforces rate limits for active learning data collection. Uploads exceeding this limit are skipped to prevent excessive data collection. Works together with minutely_usage_limit and daily_usage_limit to provide multi-level rate limiting.",
        examples=[10, 60],
    )
    daily_usage_limit: int = Field(
        default=1000,
        description="Maximum number of image uploads allowed per day for this quota. Part of the usage quota system that enforces rate limits for active learning data collection. Uploads exceeding this limit are skipped to prevent excessive data collection. Works together with minutely_usage_limit and hourly_usage_limit to provide multi-level rate limiting.",
        examples=[10, 60],
    )
    usage_quota_name: str = Field(
        description="Unique identifier for tracking usage quotas (minutely, hourly, daily limits). Used internally to manage rate limiting across multiple upload operations. Each unique quota name maintains separate counters, allowing different upload strategies or data collection workflows to have independent rate limits.",
        examples=["quota-for-data-sampling-1"],
        json_schema_extra={"hidden": True},
    )
    max_image_size: Tuple[int, int] = Field(
        default=(512, 512),
        description="Maximum dimensions (width, height) for uploaded images. Images exceeding these dimensions are automatically resized while preserving aspect ratio before uploading. Smaller sizes reduce storage and bandwidth but may lose image quality. Use larger sizes (e.g., (1920, 1080)) for high-resolution data collection, or smaller sizes (e.g., (512, 512)) for efficient storage and faster uploads.",
        examples=[(512, 512), (1920, 1080)],
    )
    compression_level: int = Field(
        default=75,
        gt=0,
        le=100,
        description="JPEG compression quality level for uploaded images, ranging from 1 (highest compression, smallest file size, lower quality) to 100 (no compression, largest file size, highest quality). Higher values preserve more image quality but increase storage and bandwidth usage. Typical values range from 70-90 for balanced quality and size. Default of 75 provides good quality with reasonable file sizes.",
        examples=[75],
    )
    registration_tags: Union[
        List[Union[Selector(kind=[STRING_KIND]), str]],
        Selector(kind=[LIST_OF_VALUES_KIND]),
    ] = Field(
        default_factory=list,
        description="List of tags to attach to uploaded images for organization and filtering in Roboflow. Tags can be static strings (e.g., 'location-florida', 'camera-1') or dynamic values from workflow inputs. Tags help organize collected data, filter images in Roboflow, and add metadata for dataset management. Can be an empty list if no tags are needed.",
        examples=[
            ["location-florida", "factory-name", "$inputs.dynamic_tag"],
            "$inputs.tags",
        ],
    )
    persist_predictions: bool = Field(
        default=True,
        description="If True, model predictions are saved as annotations (pre-labels) in the Roboflow dataset alongside images. This enables predictions to serve as starting points for annotation, allowing reviewers to correct or approve labels rather than creating them from scratch. If False, only images are uploaded without annotations. Enabling this accelerates annotation workflows by providing initial labels.",
        examples=[True, False],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="If True, the block execution is disabled and no uploads occur. This allows temporarily disabling data collection without removing the block from workflows, useful for testing, debugging, or conditional data collection. When disabled, returns a message indicating the sink was disabled. Default is False (uploads enabled).",
        examples=[True, "$inputs.disable_active_learning"],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="If True, uploads execute asynchronously (fire-and-forget mode), allowing the workflow to continue immediately without waiting for upload completion. This improves workflow performance but prevents error handling. If False, uploads execute synchronously, blocking workflow execution until completion and allowing proper error handling and status reporting. Use async mode (True) for production workflows where speed is prioritized, and sync mode (False) for debugging or when error handling is critical.",
        examples=[True],
    )
    labeling_batch_prefix: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="workflows_data_collector",
        description="Prefix used to generate labeling batch names for organizing uploaded images in Roboflow. Combined with the batch recreation frequency and timestamps to create batch names like 'workflows_data_collector_2024_01_15'. Batches help organize collected data for labeling, making it easier to manage and review uploaded images in groups. Can be customized to match your organization scheme.",
        examples=["my_labeling_batch_name"],
    )
    labeling_batches_recreation_frequency: BatchCreationFrequency = Field(
        default="never",
        description="Frequency at which new labeling batches are automatically created for uploaded images. Options: 'never' (all images go to the same batch), 'daily' (new batch each day), 'weekly' (new batch each week), 'monthly' (new batch each month). Batch timestamps are appended to the labeling_batch_prefix to create unique batch names. Automatically organizing uploads into time-based batches simplifies dataset management and makes it easier to track and review collected data over time.",
        examples=["never", "daily"],
    )
    image_name: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Optional custom name for the uploaded image. If provided, this name will be used instead of an auto-generated UUID. This is useful when you want to preserve the original filename or use a meaningful identifier (e.g., serial number, timestamp) for the image in the Roboflow dataset. The name should not include file extension. If not provided, a UUID will be generated automatically.",
        examples=["serial_12345", "camera1_frame_001", "$inputs.filename"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images", "predictions", "image_name"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowDatasetUploadBlockV1(WorkflowBlock):

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
        image_name: Optional[Batch[Optional[str]]] = None,
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
        image_names = [None] * len(images) if image_name is None else image_name
        for image, prediction, img_name in zip(images, predictions, image_names):
            error_status, message = register_datapoint_at_roboflow(
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
                new_labeling_batch_frequency=labeling_batches_recreation_frequency,
                cache=self._cache,
                background_tasks=self._background_tasks,
                thread_pool_executor=self._thread_pool_executor,
                api_key=self._api_key,
                image_name=img_name,
            )
            result.append({"error_status": error_status, "message": message})
        return result


def register_datapoint_at_roboflow(
    image: WorkflowImageData,
    prediction: Optional[Union[sv.Detections, dict]],
    target_project: str,
    usage_quota_name: str,
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
    image_name: Optional[str] = None,
) -> Tuple[bool, str]:
    registration_task = partial(
        execute_registration,
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
        labeling_batch_prefix=labeling_batch_prefix,
        new_labeling_batch_frequency=new_labeling_batch_frequency,
        cache=cache,
        api_key=api_key,
        image_name=image_name,
    )
    if fire_and_forget and background_tasks:
        background_tasks.add_task(registration_task)
        return False, "Element registration happens in the background task"
    if fire_and_forget and thread_pool_executor:
        thread_pool_executor.submit(registration_task)
        return False, "Element registration happens in the background task"
    return registration_task()


def execute_registration(
    image: WorkflowImageData,
    prediction: Optional[Union[sv.Detections, dict]],
    target_project: str,
    persist_predictions: bool,
    usage_quota_name: str,
    minutely_usage_limit: int,
    hourly_usage_limit: int,
    daily_usage_limit: int,
    max_image_size: Tuple[int, int],
    compression_level: int,
    registration_tags: List[str],
    labeling_batch_prefix: str,
    new_labeling_batch_frequency: BatchCreationFrequency,
    cache: BaseCache,
    api_key: str,
    image_name: Optional[str] = None,
) -> Tuple[bool, str]:
    matching_strategies_limits = OrderedDict(
        {
            usage_quota_name: [
                StrategyLimit(
                    limit_type=StrategyLimitType.MINUTELY, value=minutely_usage_limit
                ),
                StrategyLimit(
                    limit_type=StrategyLimitType.HOURLY, value=hourly_usage_limit
                ),
                StrategyLimit(
                    limit_type=StrategyLimitType.DAILY, value=daily_usage_limit
                ),
            ]
        }
    )
    workspace_name = get_workspace_name(api_key=api_key, cache=cache)
    strategy_with_spare_credit = use_credit_of_matching_strategy(
        cache=cache,
        workspace=workspace_name,
        project=target_project,
        matching_strategies_limits=matching_strategies_limits,
    )
    if strategy_with_spare_credit is None:
        return False, "Registration skipped due to usage quota exceeded"
    credit_to_be_returned = False
    try:
        local_image_id = image_name if image_name else str(uuid4())
        encoded_image, scaling_factor = prepare_image_to_registration(
            image=image.numpy_image,
            desired_size=ImageDimensions(
                width=max_image_size[0], height=max_image_size[1]
            ),
            jpeg_compression_level=compression_level,
        )
        batch_name = generate_batch_name(
            labeling_batch_prefix=labeling_batch_prefix,
            new_labeling_batch_frequency=new_labeling_batch_frequency,
        )
        if isinstance(prediction, sv.Detections):
            prediction = scale_sv_detections(
                detections=prediction, scale=scaling_factor
            )
        status = register_datapoint(
            target_project=target_project,
            encoded_image=encoded_image,
            local_image_id=local_image_id,
            prediction=prediction if persist_predictions else None,
            api_key=api_key,
            batch_name=batch_name,
            tags=registration_tags,
        )
        if status == DUPLICATED_STATUS:
            credit_to_be_returned = True
        return False, status
    except Exception as error:
        credit_to_be_returned = True
        logging.exception("Failed to register datapoint on the Roboflow platform")
        return (
            True,
            f"Error while registration. Error type: {type(error)}. Details: {error}",
        )
    finally:
        if credit_to_be_returned:
            return_strategy_credit(
                cache=cache,
                workspace=workspace_name,
                project=target_project,
                strategy_name=strategy_with_spare_credit,
            )


def get_workspace_name(
    api_key: str,
    cache: BaseCache,
) -> str:
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cached_workspace_name = cache.get(cache_key)
    if cached_workspace_name:
        return cached_workspace_name
    workspace_name_from_api = get_roboflow_workspace(api_key=api_key)
    cache.set(
        key=cache_key, value=workspace_name_from_api, expire=WORKSPACE_NAME_CACHE_EXPIRE
    )
    return workspace_name_from_api


def generate_batch_name(
    labeling_batch_prefix: str,
    new_labeling_batch_frequency: BatchCreationFrequency,
) -> str:
    if new_labeling_batch_frequency == "never":
        return labeling_batch_prefix
    timestamp_generator = RECREATION_INTERVAL2TIMESTAMP_GENERATOR[
        new_labeling_batch_frequency
    ]
    timestamp = timestamp_generator()
    return f"{labeling_batch_prefix}_{timestamp}"


def generate_today_timestamp() -> str:
    return datetime.today().strftime(TIMESTAMP_FORMAT)


def generate_start_timestamp_for_this_week() -> str:
    today = datetime.today()
    return (today - timedelta(days=today.weekday())).strftime(TIMESTAMP_FORMAT)


def generate_start_timestamp_for_this_month() -> str:
    return datetime.today().replace(day=1).strftime(TIMESTAMP_FORMAT)


RECREATION_INTERVAL2TIMESTAMP_GENERATOR = {
    "daily": generate_today_timestamp,
    "weekly": generate_start_timestamp_for_this_week,
    "monthly": generate_start_timestamp_for_this_month,
}


def register_datapoint(
    target_project: str,
    encoded_image: bytes,
    local_image_id: str,
    prediction: Optional[Union[sv.Detections, dict]],
    api_key: str,
    batch_name: str,
    tags: List[str],
) -> str:
    inference_id = None
    if isinstance(prediction, dict):
        inference_id = prediction.get(INFERENCE_ID_KEY)
    if isinstance(prediction, sv.Detections) and len(prediction) > 0:
        # TODO: Lack of inference ID for empty prediction -
        #  dependent on https://github.com/roboflow/inference/issues/567
        inference_id_array = prediction.data.get(INFERENCE_ID_KEY)
        if inference_id_array is not None:
            inference_id_list = inference_id_array.tolist()
            inference_id = inference_id_list[0]
    roboflow_image_id = safe_register_image_at_roboflow(
        target_project=target_project,
        encoded_image=encoded_image,
        local_image_id=local_image_id,
        api_key=api_key,
        batch_name=batch_name,
        tags=tags,
        inference_id=inference_id,
    )
    if roboflow_image_id is None:
        return DUPLICATED_STATUS
    if is_prediction_registration_forbidden(prediction=prediction):
        return "Successfully registered image"
    encoded_prediction, prediction_format = encode_prediction(prediction=prediction)
    _ = annotate_image_at_roboflow(
        api_key=api_key,
        dataset_id=target_project,
        local_image_id=local_image_id,
        roboflow_image_id=roboflow_image_id,
        annotation_content=encoded_prediction,
        annotation_file_type=prediction_format,
        is_prediction=True,
    )
    return "Successfully registered image and annotation"


def safe_register_image_at_roboflow(
    target_project: str,
    encoded_image: bytes,
    local_image_id: str,
    api_key: str,
    batch_name: str,
    tags: List[str],
    inference_id: Optional[str],
) -> Optional[str]:
    registration_response = register_image_at_roboflow(
        api_key=api_key,
        dataset_id=target_project,
        local_image_id=local_image_id,
        image_bytes=encoded_image,
        batch_name=batch_name,
        tags=tags,
        inference_id=inference_id,
    )
    image_duplicated = registration_response.get("duplicate", False)
    if image_duplicated:
        logging.warning(f"Image duplication detected: {registration_response}.")
        return None
    return registration_response["id"]


def is_prediction_registration_forbidden(
    prediction: Optional[Union[sv.Detections, dict]],
) -> bool:
    if prediction is None:
        return True
    if isinstance(prediction, sv.Detections) and len(prediction) == 0:
        return True
    if isinstance(prediction, dict) and all(
        k not in prediction for k in ["top", "predicted_classes"]
    ):
        return True
    return False


def encode_prediction(
    prediction: Union[sv.Detections, dict],
) -> Tuple[str, str]:
    if isinstance(prediction, dict):
        if "predicted_classes" in prediction:
            return ",".join(prediction["predicted_classes"]), "txt"
        return prediction["top"], "txt"
    detections_in_inference_format = serialise_sv_detections(detections=prediction)
    return json.dumps(detections_in_inference_format), "json"
