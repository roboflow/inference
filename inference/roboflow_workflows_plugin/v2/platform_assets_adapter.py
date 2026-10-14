"""Inference host adapter for asset attributes and complete dataset uploads.

Cache lifetime belongs to the caller. SharedUploadQuota and V1 uploaders must
receive that same cache to compete for the same workspace/project/strategy.
The boundary enters inside upload(), hence on the actual effects worker thread.
"""

import hashlib
from typing import Any, Dict, Sequence, Tuple
from uuid import uuid4

from roboflow_workflows.v2.blocks.roboflow_platform.roboflow_dataset_upload.v2 import DatasetUploadRequest
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError
from roboflow_workflows.v2.data.secrets import Secret, reveal_secrets

from inference.core import roboflow_api
from inference.core.cache.base import BaseCache
from inference.roboflow_workflows_plugin.v2.credential_boundary import (
    HostOperation,
    call_host,
    register_diagnostic_filters,
)
from inference.roboflow_workflows_plugin.v2.platform_upload_quota import (
    SharedUploadQuota,
)


def _operation(name: str) -> HostOperation:
    operation = getattr(HostOperation, name, None)
    if operation is None:
        raise RuntimeError(
            "Platform assets require the host credential-boundary operation extension"
        )

    return operation


class RoboflowAssetsAdapter:
    """Host-managed asset writes and upload accounting, with no cache fallback.

    Args:
        cache: Explicit shared host cache with working quota locks. The caller
            retains ownership and closes it after sessions and effects drain.
    """

    def __init__(self, *, cache: BaseCache) -> None:
        self._quota = SharedUploadQuota(cache)
        self._cache = cache
        register_diagnostic_filters()

    def get_workspace(self, *, api_key: Secret) -> str:
        """Resolve and cache the workspace using V1's 15-minute shared key.

        Args:
            api_key: Unrevealed host credential.

        Returns:
            Workspace slug.

        Raises:
            PlatformRequestError: Fixed, unchained diagnostic on host failure.
        """
        workspace = call_host(
            HostOperation.WORKSPACE, self._workspace, platform=True, api_key=api_key
        )

        return workspace

    def _workspace(self, *, api_key: Secret) -> str:
        plaintext = reveal_secrets(api_key).value
        digest = hashlib.md5(
            plaintext.encode("utf-8"), usedforsecurity=False
        ).hexdigest()
        cache_key = f"workflows:api_key_to_workspace:{digest}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        workspace = roboflow_api.get_roboflow_workspace(api_key=plaintext)
        self._cache.set(key=cache_key, value=workspace, expire=900)

        return workspace

    def update_attributes(
        self, *, api_key: Secret, workspace_id: str, updates: Sequence[Dict[str, Any]]
    ) -> None:
        """Submit a single update or V1 batch request; require batch taskId.

        Args:
            api_key: Unrevealed host credential.
            workspace_id: Workspace slug.
            updates: Merged imageId/metadata/addTags payloads.

        Raises:
            PlatformRequestError: Fixed, unchained diagnostic on request failure.
            RuntimeError: The host boundary lacks the required operation label.
        """
        call_host(
            _operation("ASSET_ATTRIBUTES"),
            self._update,
            platform=True,
            api_key=api_key,
            workspace_id=workspace_id,
            updates=updates,
        )

    def _update(
        self, *, api_key: Secret, workspace_id: str, updates: Sequence[Dict[str, Any]]
    ) -> None:
        plaintext = reveal_secrets(api_key).value
        if not updates:
            return

        if len(updates) == 1:
            update = updates[0]
            roboflow_api.update_image_metadata_at_roboflow(
                api_key=plaintext,
                workspace_id=workspace_id,
                image_id=update["imageId"],
                metadata=update.get("metadata"),
                add_tags=update.get("addTags"),
            )
        else:
            response = roboflow_api.batch_update_image_metadata_at_roboflow(
                api_key=plaintext, workspace_id=workspace_id, updates=list(updates)
            )
            if not response.get("taskId"):
                raise ValueError(
                    "Malformed image metadata batch response: missing taskId"
                )

    def upload(
        self, request: DatasetUploadRequest, *, api_key: Secret
    ) -> Tuple[bool, str]:
        """Reserve, encode and register one image; refund once on duplicate/error.

        Args:
            request: Detached per-image native input and collection policy.
            api_key: Unrevealed host credential.

        Returns:
            V1 error flag and fixed success/skip or scrubbed error message.

        Raises:
            PlatformRequestError: Workspace or reservation failure, as in V1.
            RuntimeError: The host boundary lacks the required operation label.

        Note:
            Refunds retain V1 non-idempotence and current-window semantics. No
            refund is attempted for a denied reservation. Successful image plus
            failed annotation refunds the whole credit, as in V1.
        """
        operation = _operation("DATASET_UPLOAD")
        outcome = call_host(
            operation,
            self._upload,
            platform=True,
            request=request,
            api_key=api_key,
            boundary_operation=operation,
        )

        return outcome

    def _upload(
        self,
        *,
        request: DatasetUploadRequest,
        api_key: Secret,
        boundary_operation: HostOperation,
    ) -> Tuple[bool, str]:
        workspace = self._workspace(api_key=api_key)
        project = request.target_project
        strategy = self._quota.reserve(
            workspace=workspace,
            project=project,
            usage_quota_name=request.usage_quota_name,
            minutely_usage_limit=request.minutely_usage_limit,
            hourly_usage_limit=request.hourly_usage_limit,
            daily_usage_limit=request.daily_usage_limit,
        )
        if strategy is None:
            return False, "Registration skipped due to usage quota exceeded"

        refund = True
        try:
            try:
                status = call_host(
                    boundary_operation,
                    self._register,
                    platform=True,
                    request=request,
                    api_key=api_key,
                )
            except PlatformRequestError as error:
                return (
                    True,
                    f"Error while registration. Error type: <class '{error.error_type}'>. Details: {error}",
                )

            refund = status == "Duplicated image"
            return False, status
        finally:
            if refund:
                self._quota.return_credit(
                    workspace=workspace, project=project, strategy_name=strategy
                )

    def _register(self, *, request: DatasetUploadRequest, api_key: Secret) -> str:
        # Reuse pinned V1 native conversion and exact JPEG canvas scaling. These
        # helpers do not send requests or log; registration below bypasses V1's
        # root-logger duplicate/error helpers, which are outside the audited set.
        from inference.core.active_learning.core import (
            prepare_image_to_registration_with_metadata,
        )
        from inference.core.active_learning.entities import ImageDimensions
        from inference.roboflow_workflows_plugin.sinks.dataset_upload import (
            v1_tensor as native,
        )

        plaintext = reveal_secrets(api_key).value
        pixels = request.image.tensor_image.detach().cpu().permute(1, 2, 0).numpy()
        if pixels.shape[2] == 3:
            pixels = pixels[:, :, ::-1].copy()
        prepared = prepare_image_to_registration_with_metadata(
            image=pixels,
            desired_size=ImageDimensions(
                width=request.max_image_size[0], height=request.max_image_size[1]
            ),
            jpeg_compression_level=request.compression_level,
        )
        prediction = request.prediction
        if native._is_tensor_native_detection_prediction(prediction):
            prediction = native.scale_tensor_native_prediction(
                prediction=prediction,
                scale=(prepared.scale_x, prepared.scale_y),
                target_size_wh=prepared.final_size_wh,
                update_scaling_metadata=False,
            )

        inference_id = None
        if native._is_tensor_native_classification_prediction(prediction):
            inference_id = native._read_classification_inference_id(
                prediction=prediction
            )
        elif native._is_tensor_native_detection_prediction(prediction):
            _, detections = native.split_key_point_prediction(prediction)
            if len(detections):
                inference_id = (detections.image_metadata or {}).get("inference_id")
        name = request.image_name or str(uuid4())
        batch = native.generate_batch_name(
            labeling_batch_prefix=request.labeling_batch_prefix,
            new_labeling_batch_frequency=request.labeling_batches_recreation_frequency,
        )
        response = roboflow_api.register_image_at_roboflow(
            api_key=plaintext,
            dataset_id=request.target_project,
            local_image_id=name,
            image_bytes=prepared.encoded_image,
            batch_name=batch,
            tags=request.registration_tags,
            inference_id=inference_id,
            metadata=request.metadata,
        )
        if response.get("duplicate", False):
            return "Duplicated image"

        image_id = response["id"]
        if native.is_prediction_registration_forbidden(prediction=prediction):
            return "Successfully registered image"

        content, file_type = native.encode_prediction(prediction=prediction)
        roboflow_api.annotate_image_at_roboflow(
            api_key=plaintext,
            dataset_id=request.target_project,
            local_image_id=name,
            roboflow_image_id=image_id,
            annotation_content=content,
            annotation_file_type=file_type,
            is_prediction=True,
        )

        return "Successfully registered image and annotation"
