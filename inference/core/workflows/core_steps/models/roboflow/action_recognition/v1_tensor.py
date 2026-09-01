"""Tensor-input sibling of the action recognition workflow block.

``BlockManifest`` is re-exported so a caller loading either module finds the
same manifest.
"""

import numpy as np

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
