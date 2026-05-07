"""Verify VideoSource._start dispatches rosbridge:// URLs to the new producer
without affecting the existing rtsp:// / file / device routes."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_roslibpy(monkeypatch):
    fake = types.ModuleType("roslibpy")

    class _Ros:
        def __init__(self, host, port, is_secure=False):
            self.is_connected = False

        def run(self, timeout=None):
            self.is_connected = True

        def close(self):
            self.is_connected = False

    class _Topic:
        def __init__(self, *a, **kw):
            self._cb = None

        def subscribe(self, cb):
            self._cb = cb

        def unsubscribe(self):
            pass

    fake.Ros = _Ros
    fake.Topic = _Topic
    fake.Message = lambda d: d
    monkeypatch.setitem(sys.modules, "roslibpy", fake)
    from inference.core.workflows.core_steps.common.rosbridge import connection
    connection._REGISTRY = None
    yield fake
    connection._REGISTRY = None


def test_helper_recognises_rosbridge_scheme():
    from inference.core.interfaces.camera.video_source import (
        _is_rosbridge_reference,
    )
    assert _is_rosbridge_reference("rosbridge://h/t") is True
    assert _is_rosbridge_reference("ROSBRIDGE://h/t") is True
    assert _is_rosbridge_reference("rtsp://h/t") is False
    assert _is_rosbridge_reference("/path/file.mp4") is False
    assert _is_rosbridge_reference(0) is False


def test_dispatch_constructs_rosbridge_producer(mock_roslibpy):
    """``VideoSource._start`` should construct ``RosbridgeImageFrameProducer``
    via the ``from_url`` classmethod when given a ``rosbridge://`` reference,
    and not fall through to the cv2 path."""
    from inference.core.interfaces.camera import (
        rosbridge_image_producer as producer_mod,
    )

    fake_producer = MagicMock(name="RosbridgeImageFrameProducer")
    fake_producer.isOpened.return_value = True
    fake_producer.discover_source_properties.return_value = MagicMock(
        is_file=False
    )

    with patch.object(
        producer_mod.RosbridgeImageFrameProducer,
        "from_url",
        return_value=fake_producer,
    ) as from_url:
        from inference.core.interfaces.camera.video_source import VideoSource

        source = VideoSource.init(
            video_reference="rosbridge://robot.local/cam/raw"
            "?type=sensor_msgs/CompressedImage",
        )
        # Patch the source's video consumer to abort the consume loop quickly.
        source._video_consumer = MagicMock()
        source._video_consumer.consume_frame.return_value = False
        source._video_consumer.buffer_filling_strategy = None
        source.start()
        try:
            from_url.assert_called_once_with(
                "rosbridge://robot.local/cam/raw?type=sensor_msgs/CompressedImage"
            )
            assert source._video is fake_producer
        finally:
            # Stream consumer thread will exit on first consume_frame=False.
            if source._stream_consumption_thread is not None:
                source._stream_consumption_thread.join(timeout=2.0)




def test_rtsp_path_unchanged():
    """Sanity: rtsp URLs do not trigger the rosbridge branch."""
    from inference.core.interfaces.camera.video_source import (
        _is_rosbridge_reference,
        _is_test_pattern_reference,
    )
    assert _is_rosbridge_reference("rtsp://camera.local/stream") is False
    assert _is_test_pattern_reference("rtsp://camera.local/stream") is False
