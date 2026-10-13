"""Attach custom metadata to Model Monitoring results.

Port of V1 ``roboflow_core/roboflow_custom_metadata@v1`` (latest tensor
sibling). One call tags one image's inference result::

    disable_sinks ─▶ skipped, error_status=False
    api_key       ─▶ None: ValueError; replayed secret: SecretUnavailableError
    inference id  ─▶ none found: error_status=True, V1's message
    port.get_workspace          a failure fails the step, as in V1
    port.add_custom_metadata ─▶ ok:     (False, "Custom metadata upload was successful")
                                failed: (True,  "Error while custom metadata registration. ...")

Kept from V1: parameters, outputs, messages, the order of checks, and which
failures fail the step. V1 resolved the workspace outside its error handling,
so a failed lookup still fails the step here.

Differences from V1:

* ``inference_id`` is a new optional parameter. V2 model blocks emit the id as
  their own ``inference_id`` output and do not write it into prediction
  metadata, where V1 looked. When given, it is the only id sent; otherwise the
  id comes from ``image_metadata['inference_id']`` as in V1.
* ``fire_and_forget`` always waits for the request, which is what V1 did when
  the host had no background scheduler or thread pool. Background delivery
  needs the host's sink effects runtime, which this module does not use yet.
* Workspace lookups are cached by the host's port, not by a 15-minute V1
  ``WorkflowsCache`` entry keyed by an MD5 of the API key.
* Without a host port the step fails with ``PlatformUnavailableError``.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, StrictBool
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.ports import (
    OFFLINE_CUSTOM_METADATA_PORT,
    CustomMetadataPort,
    PlatformRequestError,
    api_key_for_dispatch,
)
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, STRING_KIND
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

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)

LOGGER = logging.getLogger(__name__)

BLOCK_NAME = "RoboflowCustomMetadata"
INFERENCE_ID_KEY = "inference_id"
PREDICTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    CLASSIFICATION_PREDICTION_KIND,
)

DISABLED_MESSAGE = "Sink was disabled by workflow execution policy"
SUCCESS_MESSAGE = "Custom metadata upload was successful"
NO_INFERENCE_ID_MESSAGE = (
    "Custom metadata upload failed because no inference_ids were received. "
    "This is known bug (https://github.com/roboflow/inference/issues/567). "
    "Please provide a report for the problem under mentioned issue."
)

__all__ = ["RoboflowCustomMetadataBlock"]


class RoboflowCustomMetadataBlock(Block):
    """Tag one inference result in Roboflow Model Monitoring.

    Resources:
        custom_metadata_port: Host adapter for the two platform calls; the
            default refuses with ``PlatformUnavailableError``.
        api_key: Roboflow API key, a ``str`` or a ``Secret``. Only the port
            reveals it.
        disable_sinks: Host policy that skips every call, as V1's.
    """

    type = "v2/roboflow_custom_metadata@v1"
    aliases = ("RoboflowCustomMetadata",)
    metadata = {
        "name": "Roboflow Custom Metadata",
        "port_of": "roboflow_core/roboflow_custom_metadata@v1",
        "section": "advanced",
        "requires_rf_key": True,
    }
    outputs = {
        "error_status": Output(
            BOOLEAN_KIND,
            description=(
                "True when no inference id was found or the platform rejected "
                "the request; False on success or when sinks are disabled."
            ),
        ),
        "message": Output(STRING_KIND, description="Human-readable outcome."),
    }

    class Params(BlockParams):
        predictions: Ref(*PREDICTION_KINDS) = Field(
            description=(
                "Predictions of the tagged image. Its id is read from "
                "image_metadata['inference_id'] unless inference_id is given."
            ),
            examples=["$steps.object_detection.predictions"],
        )
        field_name: str = Field(
            description="Metadata field to set in Model Monitoring.",
            examples=["location", "camera_id"],
        )
        field_value: str | Ref(STRING_KIND) = Field(
            description="Value of the metadata field.",
            examples=["toronto", "$steps.expression.output"],
        )
        inference_id: Optional[str] | Ref(STRING_KIND) = Field(
            default=None,
            description=(
                "Id of the tagged result, e.g. a V2 model block's inference_id "
                "output. When set, the id in the predictions is not used."
            ),
            examples=["$steps.object_detection.inference_id"],
        )
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "Accepted for V1 compatibility. The request always completes "
                "before the step returns, as in V1 without a background runner."
            ),
            examples=[True, False],
        )

    def __init__(
        self,
        *,
        custom_metadata_port: CustomMetadataPort = OFFLINE_CUSTOM_METADATA_PORT,
        api_key: Optional[Union[str, Secret]] = None,
        disable_sinks: bool = False,
    ) -> None:
        self._port = custom_metadata_port
        self._api_key = api_key
        self._disable_sinks = disable_sinks

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report nothing: tagging stays inside the API key's own workspace."""
        return []

    def run(
        self,
        *,
        predictions: Any,
        field_name: str,
        field_value: str,
        inference_id: Optional[str],
        fire_and_forget: bool,
    ) -> Dict[str, Any]:
        """Send one custom metadata pair for the image's inference result.

        Args:
            predictions: Native predictions of one image.
            field_name: Metadata field name.
            field_value: Metadata field value.
            inference_id: Explicit result id, or ``None`` to read it from
                ``predictions``.
            fire_and_forget: Ignored; see the parameter description.

        Returns:
            ``{"error_status": bool, "message": str}``.

        Raises:
            ValueError: When the host gave no API key.
            SecretUnavailableError: When the API key is a replayed secret.
            PlatformUnavailableError: When the host gave no port.
            PlatformRequestError: When the workspace lookup fails.
        """
        if self._disable_sinks:
            return {"error_status": False, "message": DISABLED_MESSAGE}

        api_key = api_key_for_dispatch(self._api_key, block_name=BLOCK_NAME)
        inference_ids = (
            [inference_id]
            if inference_id is not None
            else _metadata_inference_ids(predictions)
        )
        if not inference_ids:
            return {"error_status": True, "message": NO_INFERENCE_ID_MESSAGE}

        workspace_id = self._port.get_workspace(api_key=api_key)
        try:
            self._port.add_custom_metadata(
                api_key=api_key,
                workspace_id=workspace_id,
                inference_ids=inference_ids,
                field_name=field_name,
                field_value=field_value,
            )
        except PlatformUnavailableError:
            raise
        except Exception as error:
            LOGGER.warning(
                "Could not add custom metadata for inference IDs: %s. Reason: %s",
                inference_ids,
                error,
            )
            message = (
                "Error while custom metadata registration. "
                f"Error type: {_error_type(error)}. Details: {error}"
            )
            return {"error_status": True, "message": message}

        return {"error_status": False, "message": SUCCESS_MESSAGE}


def _metadata_inference_ids(predictions: Any) -> List[str]:
    # V1 tensor rule: one per-image id in the prediction's image metadata.
    if isinstance(predictions, ClassificationPrediction):
        images_metadata = predictions.images_metadata or [{}]
        image_metadata = images_metadata[0]
    elif isinstance(predictions, tuple):
        key_points, detections = predictions
        carrier = detections if detections is not None else key_points
        image_metadata = carrier.image_metadata
    elif isinstance(predictions, MultiLabelClassificationPrediction):
        image_metadata = predictions.image_metadata
    else:
        image_metadata = predictions.image_metadata

    inference_id = (image_metadata or {}).get(INFERENCE_ID_KEY)
    if inference_id is None:
        return []

    return [inference_id]


def _error_type(error: Exception) -> str:
    # V1 printed type(error); a port error names the class it replaced.
    if isinstance(error, PlatformRequestError):
        return f"<class '{error.error_type}'>"

    return str(type(error))
