import base64

import cv2
import numpy as np
import pytest

from inference.core.workflows.core_steps.common.rosbridge import encoding


def test_now_stamp_ros2_shape():
    s = encoding.now_stamp(ros_version=2)
    assert set(s.keys()) == {"sec", "nanosec"}
    assert isinstance(s["sec"], int) and isinstance(s["nanosec"], int)


def test_now_stamp_ros1_shape():
    s = encoding.now_stamp(ros_version=1)
    assert set(s.keys()) == {"secs", "nsecs"}


def test_decode_compressed_image_base64_roundtrip():
    img = np.full((32, 48, 3), 100, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    msg = {"data": base64.b64encode(buf.tobytes()).decode("ascii")}
    out = encoding.decode_image_message(msg, "sensor_msgs/CompressedImage")
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_decode_compressed_image_cbor_raw_path():
    """CBOR-raw subscriptions deliver `data` as raw bytes, not a base64 str."""
    img = np.full((24, 32, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    # Same payload as base64 path but as bytes — should decode identically.
    msg = {"data": buf.tobytes()}
    out = encoding.decode_image_message(msg, "sensor_msgs/CompressedImage")
    assert out.shape == img.shape


def test_decode_raw_image_bgr8():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[..., 0] = 50  # B
    img[..., 1] = 100  # G
    img[..., 2] = 150  # R
    msg = {
        "encoding": "bgr8",
        "height": 10,
        "width": 20,
        "step": 60,
        "data": base64.b64encode(img.tobytes()).decode("ascii"),
    }
    out = encoding.decode_image_message(msg, "sensor_msgs/Image")
    assert out.shape == img.shape
    np.testing.assert_array_equal(out, img)


def test_decode_raw_image_rgb8_swaps_to_bgr():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 20
    rgb[..., 2] = 30
    msg = {
        "encoding": "rgb8",
        "height": 4,
        "width": 6,
        "step": 18,
        "data": base64.b64encode(rgb.tobytes()).decode("ascii"),
    }
    out = encoding.decode_image_message(msg, "sensor_msgs/Image")
    # BGR-swapped: B=30, G=20, R=10
    assert out[0, 0, 0] == 30
    assert out[0, 0, 1] == 20
    assert out[0, 0, 2] == 10


def test_decode_raw_image_mono8():
    arr = (np.arange(8 * 12) % 256).astype(np.uint8).reshape(8, 12)
    msg = {
        "encoding": "mono8",
        "height": 8,
        "width": 12,
        "step": 12,
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }
    out = encoding.decode_image_message(msg, "sensor_msgs/Image")
    np.testing.assert_array_equal(out, arr)


def test_encode_compressed_image_payload_shape():
    img = np.full((16, 16, 3), 42, dtype=np.uint8)
    payload = encoding.encode_compressed_image(img, frame_id="cam0")
    assert payload["format"] == "jpeg"
    assert payload["header"]["frame_id"] == "cam0"
    raw = base64.b64decode(payload["data"])
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == img.shape


def test_encode_label_image_mono8():
    img = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    payload = encoding.encode_label_image(img)
    assert payload["encoding"] == "mono8"
    assert payload["height"] == 2
    assert payload["width"] == 3
    assert payload["step"] == 3
    raw = base64.b64decode(payload["data"])
    np.testing.assert_array_equal(
        np.frombuffer(raw, dtype=np.uint8).reshape(2, 3), img
    )


def test_encode_label_image_mono16():
    img = np.array([[0, 1000, 2000]], dtype=np.uint16)
    payload = encoding.encode_label_image(img)
    assert payload["encoding"] == "mono16"
    assert payload["step"] == 6  # 3 cols * 2 bytes
