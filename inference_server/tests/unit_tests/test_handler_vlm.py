from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from inference_server.framework.entities import CommonRequestParams
from inference_server.framework.registry import DYNAMIC_MODELS_HANDLERS
from inference_server.handlers.vlm.handler import handle_vlm
from inference_server.handlers.vlm.input_parser import parse_vlm_input
from inference_server.handlers.vlm.introspection import (
    get_vlm_detections_prompt_interface,
    get_vlm_keypoints_image_only_interface,
)
from inference_server.handlers.vlm.output_serializer import serialize_vlm_detections


def _common() -> CommonRequestParams:
    return CommonRequestParams(model_id="acme/1", api_key="")


def test_segment_phrase_and_ground_phrase_bind_prompt_input_and_detections_output():
    for action in ("segment_phrase", "ground_phrase"):
        desc = DYNAMIC_MODELS_HANDLERS[("vlm", action)]
        assert desc.input_parser is parse_vlm_input, action
        assert desc.handler is handle_vlm, action
        assert desc.output_serializer is serialize_vlm_detections, action
        assert desc.interface_provider is get_vlm_detections_prompt_interface, action


def test_mask_bearing_prediction_serializes_as_instance_segmentation():
    pred = SimpleNamespace(
        xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=torch.ones(1, 2, 2, dtype=torch.uint8),
    )
    resp = serialize_vlm_detections([pred], _common())
    body = json.loads(resp.body)
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]
    assert p["type"] == "roboflow-instance-segmentation-compact-v1"
    assert p["xyxy"] == [[1.0, 2.0, 3.0, 4.0]]
    assert p["mask"] == [[[1, 1], [1, 1]]]


def test_mask_free_prediction_still_serializes_as_object_detection():
    pred = SimpleNamespace(
        xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
    )
    resp = serialize_vlm_detections([pred], _common())
    body = json.loads(resp.body)
    p = body["predictions"][0]
    assert p["type"] == "roboflow-object-detection-compact-v1"
    assert p["xyxy"] == [[1.0, 2.0, 3.0, 4.0]]


def test_multi_image_point_prediction_serializes_keypoints_per_image():
    def _points(x: float) -> SimpleNamespace:
        return SimpleNamespace(
            xy=torch.tensor([[x, x]]),
            class_id=torch.tensor([0]),
            confidence=torch.tensor([0.9]),
        )

    resp = serialize_vlm_detections([[_points(1.0)], [_points(2.0)]], _common())
    body = json.loads(resp.body)
    assert len(body["predictions"]) == 2
    for i, p in enumerate(body["predictions"]):
        assert p["type"] == "roboflow-keypoints-compact-v1"
        assert p["xy"] == [[float(i + 1), float(i + 1)]]


def test_point_interface_advertises_classes_and_keypoints_output():
    interface = get_vlm_keypoints_image_only_interface()
    assert interface.task == "vlm"
    assert interface.params["images"] == {"type": "image", "required": True}
    assert interface.params["classes"] == {"type": "list[str]", "required": True}
    assert "max_new_tokens" in interface.params
    assert interface.output_schema == {"type": "roboflow-keypoints-compact-v1"}


def test_point_registration_uses_keypoints_interface():
    desc = DYNAMIC_MODELS_HANDLERS[("vlm", "point")]
    assert desc.interface_provider is get_vlm_keypoints_image_only_interface
    assert desc.output_serializer is serialize_vlm_detections
