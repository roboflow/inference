"""Tensor-input sibling of the v2 Action Recognition Model."""

from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1_tensor import (
    _TensorFrameInput,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v2 import (
    ActionRecognitionModelBlockV2 as _NumpyActionRecognitionModelBlockV2,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v2 import (  # noqa: F401
    BlockManifest,
)


class ActionRecognitionModelBlockV2(
    _TensorFrameInput, _NumpyActionRecognitionModelBlockV2
):
    pass
