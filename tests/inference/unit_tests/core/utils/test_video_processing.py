from unittest.mock import MagicMock

import pytest

from inference.core.entities.requests.action_recognition import (
    ActionRecognitionInferenceRequest,
    InferenceRequestVideo,
)
from inference.core.managers.base import ModelManager
from inference.core.utils.video_processing import (
    VideoProcessingCancelledError,
    VideoProcessingControl,
    VideoProcessingTimeoutError,
)


def test_deadline_and_disconnection_stop_processing(monkeypatch):
    monkeypatch.setattr("inference.core.utils.video_processing.monotonic", lambda: 10)
    control = VideoProcessingControl(timeout_seconds=5)
    control.check()
    monkeypatch.setattr("inference.core.utils.video_processing.monotonic", lambda: 15)
    with pytest.raises(VideoProcessingTimeoutError):
        control.check()

    disconnected = VideoProcessingControl(
        timeout_seconds=5, is_disconnected=lambda: True
    )
    with pytest.raises(VideoProcessingCancelledError):
        disconnected.check()


def test_manager_forwards_control_only_to_action_recognition():
    manager = ModelManager.__new__(ModelManager)
    model = MagicMock()
    manager._get_model_reference = MagicMock(return_value=model)
    control = VideoProcessingControl(timeout_seconds=5)
    request = ActionRecognitionInferenceRequest(
        model_id="workspace/model",
        video=InferenceRequestVideo(type="url", value="https://example.com/video"),
    )
    manager.model_infer_sync("workspace/model", request, processing_control=control)
    model.infer_from_request.assert_called_once_with(
        request, processing_control=control
    )
    model.reset_mock()
    other_request = object()
    manager.model_infer_sync("other", other_request, processing_control=control)
    model.infer_from_request.assert_called_once_with(other_request)
