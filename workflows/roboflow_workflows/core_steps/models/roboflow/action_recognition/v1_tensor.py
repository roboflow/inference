"""Tensor-input sibling of the action recognition workflow block.

The manifest and frame predictions use the tensor-native classification kind.
"""

import numpy as np
import torch
from roboflow_workflows.core_steps.common.deserializers_tensor import (
    deserialize_native_classification_prediction_kind,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    ActionRecognitionModelBlockV1 as _NumpyActionRecognitionModelBlockV1,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    BlockManifest as _NumpyBlockManifest,
)
from roboflow_workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND,
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
