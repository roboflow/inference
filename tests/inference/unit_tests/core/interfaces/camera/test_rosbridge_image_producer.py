import base64
import sys
import threading
import types

import cv2
import numpy as np
import pytest


@pytest.fixture
def mock_roslibpy(monkeypatch):
    """Install a minimal in-process replacement for roslibpy that drives
    subscription callbacks synchronously when ``deliver`` is called."""
    fake = types.ModuleType("roslibpy")

    class _FakeRos:
        def __init__(self, host, port, is_secure=False):
            self.host = host
            self.port = port
            self.is_connected = False
            self.subscribers = []

        def run(self, timeout=None):
            self.is_connected = True

        def close(self):
            self.is_connected = False

    class _FakeTopic:
        def __init__(self, ros, name, message_type, **kwargs):
            self.ros = ros
            self.name = name
            self.message_type = message_type
            self.kwargs = kwargs
            self._cb = None
            self.published = []
            self.advertised = False
            self.unsubscribed = False

        def subscribe(self, cb):
            self._cb = cb
            self.ros.subscribers.append(self)

        def unsubscribe(self):
            self.unsubscribed = True

        def advertise(self):
            self.advertised = True

        def publish(self, msg):
            self.published.append(msg)

        def deliver(self, msg):
            assert self._cb is not None, "subscribe() must be called first"
            self._cb(msg)

    fake.Ros = _FakeRos
    fake.Topic = _FakeTopic

    class _FakeMessage:
        def __init__(self, data):
            self.data = data

    fake.Message = _FakeMessage
    monkeypatch.setitem(sys.modules, "roslibpy", fake)

    # Reset the global rosbridge registry so each test gets a fresh state.
    from inference.core.workflows.core_steps.common.rosbridge import connection
    connection._REGISTRY = None
    yield fake
    connection._REGISTRY = None


def _compressed_image_msg(bgr: np.ndarray) -> dict:
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return {"data": base64.b64encode(buf.tobytes()).decode("ascii")}


def test_from_url_parses_all_fields(mock_roslibpy):
    from inference.core.interfaces.camera.rosbridge_image_producer import (
        RosbridgeImageFrameProducer,
    )
    url = (
        "rosbridge://robot.local:9091/cam/image_raw"
        "?type=sensor_msgs/Image&compression=cbor-raw&ssl=true&queue_size=3"
    )
    producer = RosbridgeImageFrameProducer.from_url(url)
    try:
        assert producer._host == "robot.local"
        assert producer._port == 9091
        assert producer._ssl is True
        assert producer._topic == "/cam/image_raw"
        assert producer._message_type == "sensor_msgs/Image"
        assert producer._compression == "cbor-raw"
        assert producer._queue_size == 3
    finally:
        producer.release()


def test_from_url_uses_defaults_when_query_missing(mock_roslibpy):
    from inference.core.interfaces.camera.rosbridge_image_producer import (
        RosbridgeImageFrameProducer,
    )
    producer = RosbridgeImageFrameProducer.from_url(
        "rosbridge://robot.local/cam/image_raw"
    )
    try:
        assert producer._port == 9090
        assert producer._ssl is False
        assert producer._message_type == "sensor_msgs/CompressedImage"
        assert producer._compression == "none"
    finally:
        producer.release()


def test_from_url_rejects_other_scheme(mock_roslibpy):
    from inference.core.interfaces.camera.rosbridge_image_producer import (
        RosbridgeImageFrameProducer,
    )
    with pytest.raises(ValueError):
        RosbridgeImageFrameProducer.from_url("rtsp://x/y")


def test_grab_retrieve_release_contract(mock_roslibpy):
    from inference.core.interfaces.camera.rosbridge_image_producer import (
        RosbridgeImageFrameProducer,
    )
    producer = RosbridgeImageFrameProducer(
        host="h", topic="/x", message_type="sensor_msgs/CompressedImage",
        first_frame_timeout=2.0,
    )
    img = np.full((16, 16, 3), 50, dtype=np.uint8)
    msg = _compressed_image_msg(img)
    fake_topic = mock_roslibpy.Ros(host="h", port=9090).subscribers
    # The producer registered against a different fake Ros via the registry —
    # find its subscription via the topic handle held on the producer.
    producer._topic_handle.deliver(msg)
    assert producer.isOpened() is True
    assert producer.grab() is True
    ok, frame = producer.retrieve()
    assert ok is True
    assert frame.shape == img.shape
    producer.release()
    assert producer.isOpened() is False


def test_release_decrements_registry_refcount(mock_roslibpy):
    from inference.core.interfaces.camera.rosbridge_image_producer import (
        RosbridgeImageFrameProducer,
    )
    from inference.core.workflows.core_steps.common.rosbridge.connection import (
        get_registry,
    )
    p1 = RosbridgeImageFrameProducer(host="h", topic="/a")
    p2 = RosbridgeImageFrameProducer(host="h", topic="/b")
    reg = get_registry()
    key = ("h", 9090, False)
    assert key in reg._entries
    assert reg._entries[key].refcount == 2
    p1.release()
    assert reg._entries[key].refcount == 1
    p2.release()
    assert key not in reg._entries


def test_cbor_raw_path_consumes_bytes_directly(mock_roslibpy):
    """Subscribed with compression=cbor-raw, `data` arrives as bytes."""
    from inference.core.interfaces.camera.rosbridge_image_producer import (
        RosbridgeImageFrameProducer,
    )
    producer = RosbridgeImageFrameProducer(
        host="h", topic="/x", compression="cbor-raw", first_frame_timeout=2.0,
    )
    img = np.full((10, 12, 3), 80, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    producer._topic_handle.deliver({"data": buf.tobytes()})
    assert producer.grab() is True
    ok, frame = producer.retrieve()
    assert ok is True
    assert frame.shape == img.shape
    producer.release()


def test_grab_returns_false_on_timeout(mock_roslibpy):
    from inference.core.interfaces.camera.rosbridge_image_producer import (
        RosbridgeImageFrameProducer,
    )
    producer = RosbridgeImageFrameProducer(
        host="h", topic="/x", first_frame_timeout=0.1
    )
    assert producer.grab() is False
    producer.release()
