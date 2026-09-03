"""Tensor-input sibling of the action recognition workflow block.

``BlockManifest`` is re-exported so a caller loading either module finds the
same manifest.
"""

import numpy as np
import torch

from inference.core.workflows.core_steps.models.roboflow.action_recognition.v1 import (
    ActionRecognitionModelBlockV1 as _NumpyActionRecognitionModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.action_recognition.v1 import (  # noqa: F401
    BlockManifest,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


class ActionRecognitionModelBlockV1(_NumpyActionRecognitionModelBlockV1):
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
