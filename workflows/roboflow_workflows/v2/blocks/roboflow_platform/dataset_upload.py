"""Dataset collection with host-owned shared quota and bounded sink effects.

The host owns cache and BackgroundEffects lifetime. Reservation, encoding,
registration and conditional refund happen together inside the effect, including
when it runs on a worker thread. A denied or unscheduled effect spends no credit.
"""

import copy
import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union

from pydantic import AliasChoices, Field, StrictBool
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.asset_library import _per_row
from roboflow_workflows.v2.blocks.roboflow_platform.ports import api_key_for_dispatch
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BackgroundEffects,
    deliver,
    disabled_sink_message,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

BATCH_ALWAYS = "always"
BATCH_IF_VARYING = "if_varying"


@dataclass(frozen=True)
class DatasetUploadRequest:
    """Detached inputs for one upload, before quota admission or encoding.

    Args:
        image: Native RGB image snapshot.
        prediction: Native prediction snapshot, or None.
        target_project: Destination project slug.
        usage_quota_name: V1 shared strategy identifier.
        minutely_usage_limit: Minute limit; zero denies.
        hourly_usage_limit: Hour limit; zero denies.
        daily_usage_limit: Day limit; zero denies.
        max_image_size: Maximum JPEG width and height.
        compression_level: JPEG quality.
        registration_tags: Image tags.
        labeling_batch_prefix: Batch name prefix.
        labeling_batches_recreation_frequency: Date suffix schedule.
        image_name: Optional filename without extension.
        metadata: Optional image attributes.
    """

    image: ImageData
    prediction: Any
    target_project: str
    usage_quota_name: str
    minutely_usage_limit: int
    hourly_usage_limit: int
    daily_usage_limit: int
    max_image_size: Tuple[int, int]
    compression_level: int
    registration_tags: List[str]
    labeling_batch_prefix: str
    labeling_batches_recreation_frequency: str
    image_name: Optional[str]
    metadata: Optional[Dict[str, Any]]


class DatasetUploadPort(Protocol):
    """One host operation owns shared reservation, upload and conditional refund."""

    def upload(
        self, request: DatasetUploadRequest, *, api_key: Secret
    ) -> Tuple[bool, str]:
        """Upload one image using the caller-owned quota cache.

        Args:
            request: Detached image, predictions and collection policy.
            api_key: Unrevealed credential.

        Returns:
            V1 error flag and message; denial is a successful skip.
        """


_OFFLINE_MESSAGE = (
    "The host provided no 'dataset_upload_port' resource. Pass "
    "resources={'dataset_upload_port': "
    "RoboflowAssetsAdapter(cache=shared_cache)} when creating the session. "
    "Import RoboflowAssetsAdapter from "
    "inference.roboflow_workflows_plugin.v2.platform_assets_adapter; "
    "shared_cache must be a caller-owned BaseCache with quota locks."
)


class _OfflineDatasetUploadPort:
    def upload(
        self, request: DatasetUploadRequest, *, api_key: Secret
    ) -> Tuple[bool, str]:
        raise PlatformUnavailableError(_OFFLINE_MESSAGE)


class DatasetUploadBlock(Block):
    """Collect images with optional native predictions, preserving V1 row order.

    Args:
        dataset_upload_port: Host adapter with explicit shared cache ownership.
        api_key: Host credential, string or Secret.
        disable_sinks: Execution policy disabling collection.
        sink_effects: Optional caller-owned BackgroundEffects, drained by the host.
    """

    type = "v2/roboflow_dataset_upload@v2"
    aliases = ("RoboflowDatasetUpload",)
    metadata = {
        "name": "Roboflow Dataset Upload",
        "port_of": "roboflow_core/roboflow_dataset_upload@v2",
    }
    outputs = {
        "error_status": Output(
            BOOLEAN_KIND, description="True on upload failure or background rejection."
        ),
        "message": Output(
            STRING_KIND,
            description="Registration, skip, or background admission outcome.",
        ),
    }

    class Params(BlockParams):
        images: Ref(IMAGE_KIND, batch=BATCH_ALWAYS) = Field(
            description="Images to collect in input order.",
            validation_alias=AliasChoices("images", "image"),
        )
        target_project: str | Ref(STRING_KIND) = Field(
            description="Destination project slug."
        )
        usage_quota_name: str = Field(
            description="Shared V1 quota strategy identifier."
        )
        predictions: Optional[
            Ref(
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
                CLASSIFICATION_PREDICTION_KIND,
                batch=BATCH_IF_VARYING,
                allow_none=True,
            )
        ] = Field(default=None, description="Optional native predictions per image.")
        data_percentage: float | Ref(FLOAT_KIND) = Field(
            default=100.0,
            ge=0,
            le=100,
            description="Random sampling percentage before quota or encoding.",
        )
        minutely_usage_limit: int = Field(
            default=10, description="Uploads per minute; zero disables admission."
        )
        hourly_usage_limit: int = Field(
            default=100, description="Uploads per hour; zero disables admission."
        )
        daily_usage_limit: int = Field(
            default=1000, description="Uploads per day; zero disables admission."
        )
        max_image_size: Tuple[int, int] = Field(
            default=(1920, 1080), description="Maximum JPEG width and height."
        )
        compression_level: int = Field(
            default=95, gt=0, le=100, description="JPEG quality from 1 to 100."
        )
        registration_tags: List[str | Ref(STRING_KIND, batch=BATCH_IF_VARYING)] | Ref(
            LIST_OF_VALUES_KIND,
            batch=BATCH_IF_VARYING,
        ) = Field(default_factory=list, description="Shared or per-row image tags.")
        persist_predictions: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True, description="Save predictions as annotations."
        )
        disable_sink: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False, description="Skip collection when true."
        )
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description="Use caller-owned sink_effects; run inline when absent.",
        )
        labeling_batch_prefix: str | Ref(STRING_KIND) = Field(
            default="workflows_data_collector",
            description="Labeling batch name prefix.",
        )
        labeling_batches_recreation_frequency: Literal[
            "never", "daily", "weekly", "monthly"
        ] = Field(
            default="never", description="Local-date suffix frequency, matching V1."
        )
        image_name: Optional[str] | Ref(
            STRING_KIND,
            batch=BATCH_IF_VARYING,
            allow_none=True,
        ) = Field(
            default=None,
            description="Custom filename without extension; null generates a UUID.",
        )
        metadata: Optional[
            Dict[str, Union[str, int, float, bool, Ref(batch=BATCH_IF_VARYING)]]
        ] | Ref(DICTIONARY_KIND, batch=BATCH_IF_VARYING, allow_none=True) = Field(
            default=None,
            description="Shared or per-image attributes included in multipart upload.",
        )

    def __init__(
        self,
        *,
        dataset_upload_port: DatasetUploadPort = _OfflineDatasetUploadPort(),
        api_key: Optional[Union[str, Secret]] = None,
        disable_sinks: bool = False,
        sink_effects: Optional[BackgroundEffects] = None
    ) -> None:
        self._port = dataset_upload_port
        self._api_key = api_key
        self._disable_sinks = disable_sinks
        self._effects = sink_effects

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Declare the literal target project.

        Args:
            params: Validated step parameters.

        Returns:
            A Roboflow project dependency, or none for a dynamic selector.
        """
        if params.target_project.startswith("$"):
            return []

        return [
            DependentResource(
                resource_type="roboflow_platform_project",
                identifier=params.target_project,
            )
        ]

    def run(
        self,
        *,
        images: Batch[ImageData],
        predictions: Any,
        target_project: str,
        usage_quota_name: str,
        data_percentage: float,
        minutely_usage_limit: int,
        hourly_usage_limit: int,
        daily_usage_limit: int,
        max_image_size: Tuple[int, int],
        compression_level: int,
        registration_tags: Any,
        persist_predictions: bool,
        disable_sink: bool,
        fire_and_forget: bool,
        labeling_batch_prefix: str,
        labeling_batches_recreation_frequency: str,
        image_name: Any,
        metadata: Any
    ) -> List[Dict[str, Any]]:
        """Sample rows and deliver detached registration callbacks.

        Args:
            images: Engine image batch.
            predictions: Native predictions, Batch or None.
            target_project: Destination project.
            usage_quota_name: Shared strategy identifier.
            data_percentage: Sampling probability in percent.
            minutely_usage_limit: Minute limit.
            hourly_usage_limit: Hour limit.
            daily_usage_limit: Day limit.
            max_image_size: Maximum JPEG dimensions.
            compression_level: JPEG quality.
            registration_tags: Shared or per-row tags.
            persist_predictions: Include annotations when true.
            disable_sink: Per-step disabling policy.
            fire_and_forget: Request bounded background delivery.
            labeling_batch_prefix: Batch prefix.
            labeling_batches_recreation_frequency: Batch date suffix policy.
            image_name: Shared or per-row filename.
            metadata: Shared or per-row attributes.

        Returns:
            One error_status/message dict per image, in input order.

        Raises:
            ValueError: Missing credentials or misaligned per-row values.
        """
        if self._disable_sinks:
            return [
                {
                    "error_status": False,
                    "message": disabled_sink_message(by_policy=True),
                }
                for _ in images
            ]

        key = api_key_for_dispatch(self._api_key, block_name="RoboflowDataCollector")
        if disable_sink:
            return [
                {
                    "error_status": False,
                    "message": disabled_sink_message(by_policy=False),
                }
                for _ in images
            ]

        count = len(images)
        prediction_rows = _per_row(predictions, count)
        names = _per_row(image_name, count)
        tags = _per_row(registration_tags, count)
        attributes = _per_row(metadata, count)
        results = []
        for index, image in enumerate(images):
            if random.random() >= data_percentage / 100:
                results.append(
                    {
                        "error_status": False,
                        "message": "Registration skipped due to sampling settings",
                    }
                )
                continue

            request = DatasetUploadRequest(
                image=image.copy(),
                prediction=(
                    copy.deepcopy(prediction_rows[index])
                    if persist_predictions
                    else None
                ),
                target_project=target_project,
                usage_quota_name=usage_quota_name,
                minutely_usage_limit=minutely_usage_limit,
                hourly_usage_limit=hourly_usage_limit,
                daily_usage_limit=daily_usage_limit,
                max_image_size=max_image_size,
                compression_level=compression_level,
                registration_tags=copy.deepcopy(tags[index]),
                labeling_batch_prefix=labeling_batch_prefix,
                labeling_batches_recreation_frequency=labeling_batches_recreation_frequency,
                image_name=names[index],
                metadata=copy.deepcopy(attributes[index]),
            )
            outcome = deliver(
                partial(self._port.upload, request, api_key=key),
                fire_and_forget=fire_and_forget,
                sink_effects=self._effects,
                label=self.type,
            )
            results.append({"error_status": outcome[0], "message": outcome[1]})

        return results
