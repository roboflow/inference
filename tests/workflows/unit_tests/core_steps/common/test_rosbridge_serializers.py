import json

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.rosbridge import serializers
from inference.core.workflows.core_steps.common.rosbridge.serializers import (
    SerializerContext,
    serialize,
)


def _det(xyxy, class_id, confidence, class_name=None, mask=None, image_dims=None,
         data_extra=None):
    data = {}
    if class_name is not None:
        data["class_name"] = np.array(class_name, dtype=object)
    if image_dims is not None:
        data["image_dimensions"] = np.array([image_dims])
    if data_extra:
        data.update(data_extra)
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=float),
        class_id=np.array(class_id),
        confidence=np.array(confidence),
        mask=mask,
        data=data,
    )


def test_detection2d_array_basic_shape():
    det = _det(
        xyxy=[[10, 20, 30, 40]],
        class_id=[5],
        confidence=[0.9],
        class_name=["hat"],
    )
    [out] = serialize("vision_msgs/Detection2DArray", det, SerializerContext())
    assert out.message_type == "vision_msgs/Detection2DArray"
    assert out.topic_suffix == ""
    box = out.payload["detections"][0]["bbox"]
    assert box["center"]["position"]["x"] == pytest.approx(20.0)
    assert box["center"]["position"]["y"] == pytest.approx(30.0)
    assert box["size_x"] == pytest.approx(20.0)
    assert box["size_y"] == pytest.approx(20.0)
    hyp = out.payload["detections"][0]["results"][0]["hypothesis"]
    assert hyp["class_id"] == "hat"
    assert hyp["score"] == pytest.approx(0.9)


def test_detection2d_array_uses_class_id_when_class_name_missing():
    det = _det(
        xyxy=[[0, 0, 10, 10]],
        class_id=[3],
        confidence=[0.5],
    )
    [out] = serialize("vision_msgs/Detection2DArray", det, SerializerContext())
    assert out.payload["detections"][0]["results"][0]["hypothesis"]["class_id"] == "3"


def test_classification_dict_top_k():
    value = {
        "predictions": [
            {"class_name": "cat", "confidence": 0.7},
            {"class_name": "dog", "confidence": 0.2},
        ],
    }
    [out] = serialize("vision_msgs/Classification", value, SerializerContext())
    assert out.message_type == "vision_msgs/Classification"
    assert len(out.payload["results"]) == 2
    assert out.payload["results"][0]["hypothesis"]["class_id"] == "cat"


def test_classification_dict_multi_label():
    value = {
        "predictions": {
            "wet": {"confidence": 0.8, "class_id": 0},
            "indoor": {"confidence": 0.6, "class_id": 1},
        }
    }
    [out] = serialize("vision_msgs/Classification", value, SerializerContext())
    classes = {r["hypothesis"]["class_id"] for r in out.payload["results"]}
    assert classes == {"wet", "indoor"}


def test_keypoints_emits_string_and_marker_array():
    det = sv.Detections(
        xyxy=np.array([[0, 0, 100, 100]], dtype=float),
        class_id=np.array([0]),
        confidence=np.array([0.95]),
        data={
            "class_name": np.array(["person"], dtype=object),
            "keypoints_xy": np.array(
                [np.array([[10.0, 20.0], [30.0, 40.0]])], dtype=object
            ),
            "keypoints_confidence": np.array(
                [np.array([0.9, 0.8])], dtype=object
            ),
            "keypoints_class_name": np.array(
                [np.array(["nose", "ear"], dtype=object)], dtype=object
            ),
        },
    )
    msgs = serialize("keypoints", det, SerializerContext())
    assert len(msgs) == 2
    json_msg, marker_msg = msgs
    assert json_msg.message_type == "std_msgs/String"
    decoded = json.loads(json_msg.payload["data"])
    assert decoded["detections"][0]["class_name"] == "person"
    assert len(decoded["detections"][0]["keypoints"]) == 2
    assert decoded["detections"][0]["keypoints"][0]["name"] == "nose"
    assert marker_msg.message_type == "visualization_msgs/MarkerArray"
    assert marker_msg.topic_suffix == "/markers"
    assert len(marker_msg.payload["markers"][0]["points"]) == 2


def test_semantic_seg_emits_label_image_and_label_info():
    h, w = 8, 12
    # Inference-native shape: COCO RLE dicts {"size": [H, W], "counts": "..."}
    rle_a = {
        "size": [h, w],
        "counts": sv.mask_to_rle(
            np.zeros((h, w), dtype=bool), compressed=True
        ),
    }
    mask_b = np.zeros((h, w), dtype=bool)
    mask_b[2:6, 4:9] = True
    rle_b = {
        "size": [h, w],
        "counts": sv.mask_to_rle(mask_b, compressed=True),
    }
    det = sv.Detections(
        xyxy=np.array([[0, 0, w, h], [4, 2, 9, 6]], dtype=float),
        class_id=np.array([0, 1]),
        confidence=np.array([0.5, 0.7]),
        data={
            "class_name": np.array(["bg", "obj"], dtype=object),
            "rle_mask": np.array([rle_a, rle_b], dtype=object),
            "image_dimensions": np.array([[h, w], [h, w]]),
        },
    )
    msgs = serialize("semantic_segmentation", det, SerializerContext())
    assert len(msgs) == 2
    label_img_msg, label_info_msg = msgs
    assert label_img_msg.message_type == "sensor_msgs/Image"
    assert label_img_msg.payload["height"] == h
    assert label_img_msg.payload["width"] == w
    assert label_info_msg.message_type == "vision_msgs/LabelInfo"
    assert label_info_msg.latch is True
    class_map = label_info_msg.payload["class_map"]
    assert {c["class_id"] for c in class_map} == {0, 1}


def test_instance_seg_emits_four_topics_with_id_join():
    h, w = 6, 6
    mask_a = np.zeros((h, w), dtype=bool); mask_a[1:3, 1:3] = True
    mask_b = np.zeros((h, w), dtype=bool); mask_b[3:5, 3:5] = True
    det = sv.Detections(
        xyxy=np.array([[1, 1, 3, 3], [3, 3, 5, 5]], dtype=float),
        class_id=np.array([1, 2]),
        confidence=np.array([0.9, 0.8]),
        mask=np.stack([mask_a, mask_b]),
        data={
            "class_name": np.array(["red", "blue"], dtype=object),
            "image_dimensions": np.array([[h, w], [h, w]]),
        },
    )
    msgs = serialize("instance_segmentation", det, SerializerContext())
    suffixes = [m.topic_suffix for m in msgs]
    assert suffixes == ["/instances", "/classes", "/detections", "/label_info"]
    instance_msg = msgs[0]
    assert instance_msg.payload["encoding"] == "mono16"
    detections_msg = msgs[2]
    ids = [d["id"] for d in detections_msg.payload["detections"]]
    assert ids == ["1", "2"]
    label_info = msgs[3]
    assert label_info.latch is True


def test_compressed_image_serializer():
    img = np.full((16, 24, 3), 99, dtype=np.uint8)
    [out] = serialize("sensor_msgs/CompressedImage", img, SerializerContext())
    assert out.payload["format"] == "jpeg"


def test_scalar_serializers():
    assert serialize("std_msgs/String", "hi", SerializerContext())[0].payload == {
        "data": "hi"
    }
    assert serialize("std_msgs/Int32", 7, SerializerContext())[0].payload == {
        "data": 7
    }
    assert serialize("std_msgs/Float64", 1.5, SerializerContext())[0].payload == {
        "data": 1.5
    }
    assert serialize("std_msgs/Bool", True, SerializerContext())[0].payload == {
        "data": True
    }


def test_custom_dict_wrapped_in_string():
    [out] = serialize(
        "custom",
        {"x": np.array([1, 2, 3]), "y": "hi"},
        SerializerContext(),
    )
    assert out.message_type == "std_msgs/String"
    parsed = json.loads(out.payload["data"])
    assert parsed == {"x": [1, 2, 3], "y": "hi"}


def test_short_form_routing_accepts_ros2_pkg_msg_path():
    det = _det(
        xyxy=[[0, 0, 1, 1]],
        class_id=[0],
        confidence=[0.5],
        class_name=["x"],
    )
    [out] = serialize(
        "vision_msgs/msg/Detection2DArray", det, SerializerContext()
    )
    assert out.message_type == "vision_msgs/Detection2DArray"


def test_unsupported_message_type_raises():
    with pytest.raises(ValueError):
        serialize("bogus_msgs/Nope", {}, SerializerContext())
