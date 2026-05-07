import sys
import time
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.sinks.rosbridge_publish.v1 import (
    BlockManifest,
    RosbridgePublishSinkBlockV1,
)


_VALID_MANIFEST = {
    "type": "roboflow_core/rosbridge_publish_sink@v1",
    "name": "ros_pub",
    "host": "robot.local",
    "topic": "/inference/detections",
    "message_type": "vision_msgs/Detection2DArray",
    "payload": "$steps.model.predictions",
}


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
        instances = []

        def __init__(self, ros, name, message_type, latch=False, **kwargs):
            self.ros = ros
            self.name = name
            self.message_type = message_type
            self.latch = latch
            self.advertised = False
            self.unadvertised = False
            self.published = []
            _Topic.instances.append(self)

        def advertise(self):
            self.advertised = True

        def unadvertise(self):
            self.unadvertised = True

        def publish(self, msg):
            self.published.append(msg)

    class _Message:
        def __init__(self, data):
            self.data = data

    fake.Ros = _Ros
    fake.Topic = _Topic
    fake.Message = _Message
    monkeypatch.setitem(sys.modules, "roslibpy", fake)

    from inference.core.workflows.core_steps.common.rosbridge import connection
    connection._REGISTRY = None
    yield fake
    connection._REGISTRY = None
    _Topic.instances.clear()


def test_manifest_validation_succeeds_for_built_in_serializer():
    m = BlockManifest.model_validate(_VALID_MANIFEST)
    assert m.message_type == "vision_msgs/Detection2DArray"
    assert m.cooldown_seconds == 0
    assert m.fire_and_forget is True


def test_manifest_rejects_unknown_message_type():
    bad = dict(_VALID_MANIFEST, message_type="bogus")
    with pytest.raises(Exception):
        BlockManifest.model_validate(bad)


def test_run_disabled_short_circuits():
    block = RosbridgePublishSinkBlockV1()
    out = block.run(
        host="h", port=9090, ssl=False, topic="/x",
        message_type="vision_msgs/Detection2DArray",
        payload=None, json_payload={}, json_payload_operations={},
        frame_id="inference", ros_version=2, latch=False,
        cooldown_seconds=0, fire_and_forget=False, disable_sink=True,
    )
    assert out["error_status"] is False
    assert out["throttling_status"] is False


def test_run_returns_error_when_payload_missing_for_builtin():
    block = RosbridgePublishSinkBlockV1()
    out = block.run(
        host="h", port=9090, ssl=False, topic="/x",
        message_type="vision_msgs/Detection2DArray",
        payload=None, json_payload={}, json_payload_operations={},
        frame_id="inference", ros_version=2, latch=False,
        cooldown_seconds=0, fire_and_forget=False, disable_sink=False,
    )
    assert out["error_status"] is True
    assert "payload" in out["message"]


def test_run_publishes_detection2d_array(mock_roslibpy):
    block = RosbridgePublishSinkBlockV1()
    det = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=float),
        class_id=np.array([1]),
        confidence=np.array([0.8]),
        data={"class_name": np.array(["x"], dtype=object)},
    )
    out = block.run(
        host="robot", port=9090, ssl=False, topic="/inference/detections",
        message_type="vision_msgs/Detection2DArray",
        payload=det, json_payload={}, json_payload_operations={},
        frame_id="cam0", ros_version=2, latch=False,
        cooldown_seconds=0, fire_and_forget=False, disable_sink=False,
    )
    assert out["error_status"] is False
    topics = mock_roslibpy.Topic.instances
    assert len(topics) == 1
    assert topics[0].name == "/inference/detections"
    assert topics[0].message_type == "vision_msgs/msg/Detection2DArray"
    assert topics[0].advertised is True
    assert len(topics[0].published) == 1


def test_run_publishes_instance_seg_companion_topics(mock_roslibpy):
    block = RosbridgePublishSinkBlockV1()
    h, w = 6, 6
    m = np.zeros((h, w), dtype=bool); m[1:3, 1:3] = True
    det = sv.Detections(
        xyxy=np.array([[1, 1, 3, 3]], dtype=float),
        class_id=np.array([1]),
        confidence=np.array([0.9]),
        mask=np.stack([m]),
        data={
            "class_name": np.array(["red"], dtype=object),
            "image_dimensions": np.array([[h, w]]),
        },
    )
    out = block.run(
        host="robot", port=9090, ssl=False, topic="/seg",
        message_type="instance_segmentation",
        payload=det, json_payload={}, json_payload_operations={},
        frame_id="cam0", ros_version=2, latch=False,
        cooldown_seconds=0, fire_and_forget=False, disable_sink=False,
    )
    assert out["error_status"] is False
    names = sorted(t.name for t in mock_roslibpy.Topic.instances)
    assert names == [
        "/seg/classes",
        "/seg/detections",
        "/seg/instances",
        "/seg/label_info",
    ]
    label_info = next(t for t in mock_roslibpy.Topic.instances if t.name == "/seg/label_info")
    assert label_info.latch is True


def test_run_cooldown_throttles(mock_roslibpy):
    block = RosbridgePublishSinkBlockV1()
    args = dict(
        host="r", port=9090, ssl=False, topic="/t",
        message_type="std_msgs/String", payload="hi",
        json_payload={}, json_payload_operations={},
        frame_id="inference", ros_version=2, latch=False,
        cooldown_seconds=5, fire_and_forget=False, disable_sink=False,
    )
    first = block.run(**args)
    assert first["error_status"] is False
    second = block.run(**args)
    assert second["throttling_status"] is True


def test_topic_advertised_only_once_across_runs(mock_roslibpy):
    block = RosbridgePublishSinkBlockV1()
    args = dict(
        host="r", port=9090, ssl=False, topic="/t",
        message_type="std_msgs/String", payload="hi",
        json_payload={}, json_payload_operations={},
        frame_id="inference", ros_version=2, latch=False,
        cooldown_seconds=0, fire_and_forget=False, disable_sink=False,
    )
    block.run(**args)
    block.run(**args)
    topics = [t for t in mock_roslibpy.Topic.instances if t.name == "/t"]
    assert len(topics) == 1
    assert len(topics[0].published) == 2


def test_custom_message_type_wraps_dict(mock_roslibpy):
    block = RosbridgePublishSinkBlockV1()
    out = block.run(
        host="r", port=9090, ssl=False, topic="/t",
        message_type="custom",
        payload=None,
        json_payload={"foo": "bar", "n": 42},
        json_payload_operations={},
        frame_id="inference", ros_version=2, latch=False,
        cooldown_seconds=0, fire_and_forget=False, disable_sink=False,
    )
    assert out["error_status"] is False
    topics = [t for t in mock_roslibpy.Topic.instances if t.name == "/t"]
    assert topics[0].message_type == "std_msgs/msg/String"
    published_data = topics[0].published[0].data["data"]
    import json as _json
    assert _json.loads(published_data) == {"foo": "bar", "n": 42}
