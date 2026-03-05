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
    LIST_OF_VALUES_KIND,
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
Upload images and model predictions to a Roboflow dataset for active learning, model improvement, and data collection, with configurable usage quotas, probabilistic sampling, batch organization, image compression, and optional annotation persistence.

## How This Block Works

This block uploads workflow images and predictions to your Roboflow dataset for storage, labeling, and model training. The block:

1. Takes images and optional model predictions (object detection, instance segmentation, keypoint detection, or classification) as input
2. Validates the Roboflow API key is available (required for uploading)
3. Applies probabilistic sampling based on `data_percentage` setting, randomly selecting a percentage of inputs to upload (e.g., 50% uploads half the data, 100% uploads everything)
4. Checks usage quotas (minutely, hourly, daily limits) to ensure uploads stay within configured rate limits for active learning strategies
5. Prepares images by resizing if they exceed maximum size (maintaining aspect ratio) and compressing to specified quality level
6. Generates labeling batch names based on the prefix and batch creation frequency (never, daily, weekly, or monthly), organizing uploaded data into batches
7. Optionally persists model predictions as annotations if `persist_predictions` is enabled, allowing predictions to serve as pre-labels for review and correction
8. Attaches registration tags to images for organization and filtering in the Roboflow platform
9. Registers the image (and annotations if enabled) to the specified Roboflow project via the Roboflow API
10. Executes synchronously or asynchronously based on `fire_and_forget` setting, allowing non-blocking uploads for faster workflow execution
11. Returns error status and messages indicating upload success, failure, or sampling skip

The block supports active learning workflows by implementing usage quotas that prevent excessive data collection, helping focus on collecting valuable training data within rate limits. The probabilistic sampling feature (new in v2) allows you to randomly sample a percentage of data for upload, enabling cost-effective data collection strategies where you want to collect representative samples rather than all data. Images are organized into labeling batches that can be automatically recreated on a schedule (daily, weekly, monthly), making it easier to manage and review collected data over time. The block can operate in fire-and-forget mode for asynchronous execution, allowing workflows to continue processing without waiting for uploads to complete, or synchronously for debugging and error handling.

## Version Differences (v2 vs v1)

**New Features in v2:**

- **Probabilistic Data Sampling**: Added `data_percentage` parameter (0-100%) that enables random sampling of data for upload. This allows you to upload only a percentage of workflow inputs (e.g., 25% samples one in four images), reducing storage and annotation costs while still collecting representative data. When sampling skips an upload, the block returns a message indicating the skip.

- **Improved Default Settings**: 
  - `max_image_size` default increased from (512, 512) to (1920, 1080) for higher resolution data collection
  - `compression_level` default increased from 75 to 95 for better image quality preservation

**Behavior Changes:**

- By default, `data_percentage` is set to 100, so v2 behaves identically to v1 unless sampling is explicitly configured
- The block now uses probabilistic sampling before quota checking and image preparation, allowing efficient filtering before resource-intensive operations

## Requirements

**API Key Required**: This block requires a valid Roboflow API key to upload data. The API key must be configured in your environment or workflow configuration. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to retrieve an API key.

## Common Use Cases

- **Active Learning Data Collection**: Collect images and predictions from production environments where models struggle or are uncertain (e.g., low-confidence detections, edge cases), enabling iterative model improvement by gathering challenging examples for retraining
- **Probabilistic Data Sampling**: Use `data_percentage` to randomly sample a subset of data for upload (e.g., upload 20% of all detections, 50% of low-confidence cases), enabling cost-effective data collection strategies that reduce storage and annotation overhead while maintaining dataset diversity
- **Production Data Logging**: Continuously upload production inference data to Roboflow datasets for monitoring, analysis, and future model training, creating a growing dataset from real-world deployments
- **Pre-Labeled Data Collection**: Upload images with model predictions as pre-labels (when `persist_predictions` is enabled), accelerating annotation workflows by providing initial labels that can be reviewed and corrected rather than starting from scratch
- **Stratified Data Sampling**: Combine probabilistic sampling with rate limiting and quotas to selectively collect data based on specific criteria (e.g., sample 30% of detections that pass filters), ensuring diverse and balanced dataset collection without overwhelming storage or annotation resources
- **Batch-Based Labeling Workflows**: Organize uploaded data into batches with automatic recreation schedules (daily, weekly, monthly), making it easier to manage labeling tasks, track progress, and organize data collection efforts over time

## Connecting to Other Blocks

This block receives data from workflow steps and uploads it to Roboflow:

- **After detection or analysis blocks** (e.g., Object Detection Model, Instance Segmentation Model, Classification Model, Keypoint Detection Model) to upload images along with their predictions, enabling active learning by collecting inference data with model outputs for annotation and retraining
- **After filtering or analytics blocks** (e.g., Detections Filter, Continue If, Overlap Filter) to selectively upload only specific types of data (e.g., low-confidence detections, overlapping objects, specific classes), focusing data collection on valuable edge cases or interesting scenarios
- **After rate limiter blocks** (e.g., Rate Limiter) to throttle upload frequency and stay within usage quotas, ensuring controlled data collection that respects rate limits and prevents excessive storage usage
- **Image inputs or preprocessing blocks** to upload raw images or processed images (e.g., crops, transformed images) without predictions, enabling collection of image data for future labeling or analysis
- **Conditional workflows** using flow control blocks (e.g., Continue If) to upload data only when certain conditions are met (e.g., upload only when detection count exceeds threshold, upload only errors or failures), enabling selective data collection based on workflow state
- **Batch processing workflows** where multiple images or predictions are generated, allowing bulk upload of workflow outputs to Roboflow datasets with probabilistic sampling for organized and cost-effective data collection
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
        description="Image(s) to upload to the Roboflow dataset. Can be a single image or batch of images from workflow inputs or processing steps. Images are randomly sampled based on data_percentage, resized if they exceed max_image_size, and compressed before uploading. Supports batch processing.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    target_project: Union[Selector(kind=[ROBOFLOW_PROJECT_KIND]), str] = Field(
        description="Roboflow project identifier where uploaded images and annotations will be saved. Must be a valid project in your Roboflow workspace. The project name can be specified directly or referenced from workflow inputs.",
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
        description="Optional model predictions to upload alongside images. Predictions are saved as annotations (pre-labels) in the Roboflow dataset when persist_predictions is enabled, allowing predictions to serve as starting points for annotation review and correction. Supports object detection, instance segmentation, keypoint detection, and classification predictions. If None, only images are uploaded.",
        examples=["$steps.object_detection_model.predictions"],
        json_schema_extra={"always_visible": True},
    )
    data_percentage: Union[FloatZeroToHundred, Selector(kind=[FLOAT_KIND])] = Field(
        default=100,
        description="Percentage of input data (0.0 to 100.0) to randomly sample for upload. This enables probabilistic data collection where only a subset of inputs are uploaded, reducing storage and annotation costs. For example, 25.0 uploads approximately 25% of images (one in four on average), 50.0 uploads half, and 100.0 uploads everything (no sampling). Random sampling occurs before quota checking and image processing, making it efficient for large-scale data collection workflows.",
        examples=[100, 25, 50, "$inputs.sampling_rate"],
    )
    minutely_usage_limit: int = Field(
        default=10,
        description="Maximum number of image uploads allowed per minute for this quota. Part of the usage quota system that enforces rate limits for active learning data collection. Uploads exceeding this limit are skipped to prevent excessive data collection. Works together with hourly_usage_limit and daily_usage_limit to provide multi-level rate limiting. Note: This quota is checked after probabilistic sampling via data_percentage.",
        examples=[10, 60],
    )
    hourly_usage_limit: int = Field(
        default=100,
        description="Maximum number of image uploads allowed per hour for this quota. Part of the usage quota system that enforces rate limits for active learning data collection. Uploads exceeding this limit are skipped to prevent excessive data collection. Works together with minutely_usage_limit and daily_usage_limit to provide multi-level rate limiting. Note: This quota is checked after probabilistic sampling via data_percentage.",
        examples=[10, 60],
    )
    daily_usage_limit: int = Field(
        default=1000,
        description="Maximum number of image uploads allowed per day for this quota. Part of the usage quota system that enforces rate limits for active learning data collection. Uploads exceeding this limit are skipped to prevent excessive data collection. Works together with minutely_usage_limit and hourly_usage_limit to provide multi-level rate limiting. Note: This quota is checked after probabilistic sampling via data_percentage.",
        examples=[10, 60],
    )
    usage_quota_name: str = Field(
        description="Unique identifier for tracking usage quotas (minutely, hourly, daily limits). Used internally to manage rate limiting across multiple upload operations. Each unique quota name maintains separate counters, allowing different upload strategies or data collection workflows to have independent rate limits.",
        examples=["quota-for-data-sampling-1"],
        json_schema_extra={"hidden": True},
    )
    max_image_size: Tuple[int, int] = Field(
        default=(1920, 1080),
        description="Maximum dimensions (width, height) for uploaded images. Images exceeding these dimensions are automatically resized while preserving aspect ratio before uploading. Default is (1920, 1080) for higher resolution data collection. Use smaller sizes (e.g., (512, 512)) for efficient storage and faster uploads, or keep the default for preserving image quality.",
        examples=[(1920, 1080), (512, 512)],
    )
    compression_level: int = Field(
        default=95,
        gt=0,
        le=100,
        description="JPEG compression quality level for uploaded images, ranging from 1 (highest compression, smallest file size, lower quality) to 100 (no compression, largest file size, highest quality). Default is 95 for better image quality preservation. Higher values preserve more image quality but increase storage and bandwidth usage. Typical values range from 70-95 for balanced quality and size.",
        examples=[95, 75],
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
    persist_predictions: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="If True, model predictions are saved as annotations (pre-labels) in the Roboflow dataset alongside images. This enables predictions to serve as starting points for annotation, allowing reviewers to correct or approve labels rather than creating them from scratch. If False, only images are uploaded without annotations. Enabling this accelerates annotation workflows by providing initial labels.",
        examples=[True, False, "$inputs.persist_predictions"],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="If True, the block execution is disabled and no uploads occur. This allows temporarily disabling data collection without removing the block from workflows, useful for testing, debugging, or conditional data collection. When disabled, returns a message indicating the sink was disabled. Default is False (uploads enabled).",
        examples=[True, "$inputs.disable_active_learning"],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="If True, uploads execute asynchronously (fire-and-forget mode), allowing the workflow to continue immediately without waiting for upload completion. This improves workflow performance but prevents error handling. If False, uploads execute synchronously, blocking workflow execution until completion and allowing proper error handling and status reporting. Use async mode (True) for production workflows where speed is prioritized, and sync mode (False) for debugging or when error handling is critical.",
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
        description="Optional custom name for the uploaded image. This is useful when you want to preserve the original filename or use a meaningful identifier (e.g., serial number, timestamp) for the image in the Roboflow dataset. The name should not include file extension. If not provided, a UUID will be generated automatically.",
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
                image_name=img_name,
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
    image_name: Optional[str] = None,
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
            image_name=image_name,
        )
    return False, "Registration skipped due to sampling settings"
