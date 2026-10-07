"""Tensor-input sibling of the action recognition workflow block.

The manifest and frame predictions use the tensor-native classification kind.
"""

from typing import List, Optional

import numpy as np
import torch
from roboflow_workflows.core_steps.common.deserializers_tensor import (
    deserialize_native_classification_prediction_kind,
)
from roboflow_workflows.core_steps.common.workload_presets import (
    STATEFUL_VIDEO_ACTUAL_RESTRICTION,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    ActionRecognitionModelBlockV1 as _NumpyActionRecognitionModelBlockV1,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    BlockManifest as _NumpyBlockManifest,
)
from roboflow_workflows.core_steps.models.workload_presets import (
    REMOTE_STEP_EXECUTION_NOT_SUPPORTED,
    REQUIRES_GPU_FOR_LOCAL_EXECUTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
)
from roboflow_workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    DependentResource,
    ModelExecutionLocation,
    ModelRequiredAction,
    actual_restrictions_of,
    roboflow_platform_model,
)


class BlockManifest(_NumpyBlockManifest):
    """Declare tensor-native current-frame classification predictions."""

    @classmethod
    def describe_outputs(cls) -> list[OutputDefinition]:
        """Describe the timeline, error status and tensor classification output.

        Returns:
            Output definitions using the native kind for frame predictions.
        """
        outputs = super().describe_outputs()
        for output in outputs:
            if output.name == "frame_predictions":
                output.kind = [TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND]
        return outputs

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        """Declare the action recognition model of the block's local run path.

        The block supports LOCAL step execution only (its run path rejects any
        other mode) and owns its model loading: it asks the model provider for
        the model through `load_action_recognition_model()`, not through the
        generic `add_model()` registration. The declared dependency describes
        that supported execution path, so it is LOCAL and kept away from the
        generic preloader. The configured id is returned verbatim, selector
        included. Both representations declare the same model resources.

        Returns:
            The configured action recognition model.
        """
        return [
            roboflow_platform_model(
                self.model_id,
                required_action=ModelRequiredAction.EXECUTION,
                execution_location=ModelExecutionLocation.LOCAL,
                preloadable=False,
            )
        ]

    def discover_work_operations(self) -> List[WorkOperation]:
        """Declare model inference over buffered video frames.

        Returns:
            Model inference and temporal buffering operations.
        """
        return [
            WorkOperation.MODEL_INFERENCE,
            WorkOperation.TEMPORAL_BUFFERING,
        ]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        """Declare the REMOTE, GPU, cross-frame state-loss and still-image caveats.

        Args:
            ignore_environment_restrictions: If True, return every declaration
                with its condition intact (the portable view). If False,
                evaluate configuration predicates against this host and drop
                entries that definitively do not apply here.

        Returns:
            The step's restrictions. In the host view the discovery is
            incomplete when a configuration predicate cannot be evaluated.
        """
        return actual_restrictions_of(
            declared=[
                STATEFUL_VIDEO_ACTUAL_RESTRICTION,
                REQUIRES_GPU_FOR_LOCAL_EXECUTION,
                STILL_IMAGE_INPUT_SOFT_RESTRICTION,
                REMOTE_STEP_EXECUTION_NOT_SUPPORTED,
            ],
            node_id=f"$steps.{getattr(self, 'name', '')}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )


class ActionRecognitionModelBlockV1(_NumpyActionRecognitionModelBlockV1):
    @classmethod
    def get_manifest(cls) -> type[BlockManifest]:
        """Return the manifest for tensor-native action recognition.

        Returns:
            The tensor-native block manifest.
        """
        return BlockManifest

    def _build_frame_predictions(self, image, bookkeeping):
        predictions = super()._build_frame_predictions(
            image=image, bookkeeping=bookkeeping
        )
        native_prediction = deserialize_native_classification_prediction_kind(
            parameter="frame_predictions", value=predictions
        )
        # Sparse model ids leave zero-filled gaps in the native confidence
        # vector. Omit those gaps from the serialized classification response.
        native_prediction.image_metadata["classification_confidence_threshold"] = 1.0
        return native_prediction

    def _extract_frame(self, image: WorkflowImageData):
        if image.is_tensor_materialised():
            frame = image.tensor_image
            if frame.dim() != 3 or frame.shape[0] != 3:
                raise ValueError(
                    "Action Recognition Model expects a CHW RGB frame tensor."
                )
            return frame
        return np.ascontiguousarray(image.numpy_image[:, :, ::-1])

    @staticmethod
    def _cap_frame_side(frame, max_side):
        """Shrink a CHW frame tensor on the device it already sits on.

        A numpy frame takes the parent's cv2 path. A tensor stays put: moving
        it to the host to resize would undo the single batched transfer the
        buffer is crossed with.

        ``area`` is the torch counterpart of the ``INTER_AREA`` the model uses,
        so the frames match closely rather than exactly. The two agree on what
        they average, not on every rounded byte.
        """
        if not max_side or max_side <= 0:
            return frame
        if isinstance(frame, np.ndarray):
            return _NumpyActionRecognitionModelBlockV1._cap_frame_side(
                frame=frame, max_side=max_side
            )
        height, width = frame.shape[1], frame.shape[2]
        scale = max_side / max(height, width)
        if scale >= 1.0:
            return frame
        resized = torch.nn.functional.interpolate(
            frame.unsqueeze(0).to(torch.float32),
            size=(round(height * scale), round(width * scale)),
            mode="area",
        ).squeeze(0)
        if frame.dtype == torch.uint8:
            return resized.round_().clamp_(0, 255).to(torch.uint8)
        return resized.to(frame.dtype)
