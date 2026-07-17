"""Tests for the Modal wire serializer's tensor-pivot defense-in-depth
(Step 7 of the tensor_compatibility plan): native ``inference_models`` objects
must never be silently stringified on the wire — flag-on they raise, flag-off
the pre-existing generic-object contract stays byte-identical."""

import json
from unittest import mock

import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
    representation_boundary,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    serialize_for_modal_remote_execution,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.representation_boundary import (
    RepresentationBoundaryError,
)
from inference_models.models.base.object_detection import Detections

_flag_on = mock.patch.object(
    representation_boundary, "_TENSOR_REPRESENTATION_ACTIVE", True
)
_flag_off = mock.patch.object(
    representation_boundary, "_TENSOR_REPRESENTATION_ACTIVE", False
)


def _native_detections() -> Detections:
    return Detections(
        xyxy=torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
    )


class _ArbitraryObject:
    def __init__(self):
        self.field = "value"

    def __str__(self):
        return "arbitrary-object"


def test_modal_serializer_raises_on_unconverted_native_detections_when_flag_on() -> (
    None
):
    # when
    with _flag_on, pytest.raises(RepresentationBoundaryError) as error:
        _ = serialize_for_modal_remote_execution(
            inputs={"predictions": _native_detections()}
        )

    # then
    assert "Detections" in str(error.value)
    assert "reached the Modal wire serializer unconverted" in str(error.value)


def test_modal_serializer_raises_on_bare_tensor_when_flag_on() -> None:
    # when
    with _flag_on, pytest.raises(RepresentationBoundaryError) as error:
        _ = serialize_for_modal_remote_execution(
            inputs={"value": torch.tensor([1.0, 2.0])}
        )

    # then
    assert "Tensor" in str(error.value)


def test_modal_serializer_raises_on_native_nested_in_containers_when_flag_on() -> None:
    # given - natives hide inside dicts/lists; the JSON encoder is the choke
    # point, so nesting must not smuggle them past the guard
    inputs = {"payload": {"items": [_native_detections()]}}

    # when
    with _flag_on, pytest.raises(RepresentationBoundaryError):
        _ = serialize_for_modal_remote_execution(inputs=inputs)


def test_modal_serializer_keeps_generic_fallback_for_arbitrary_objects_when_flag_on() -> (
    None
):
    # when
    with _flag_on:
        payload = json.loads(
            serialize_for_modal_remote_execution(inputs={"value": _ArbitraryObject()})
        )

    # then - pre-existing generic contract untouched for non-native objects
    assert payload["value"] == {
        "_type": "object",
        "class": "_ArbitraryObject",
        "value": "arbitrary-object",
    }


def test_modal_serializer_flag_off_behavior_is_byte_identical_to_legacy() -> None:
    # given - flag-off natives cannot exist in real flows, but the guard must be
    # provably inert: everything with __dict__ stringifies exactly as before
    inputs = {
        "native": _native_detections(),
        "tensor": torch.tensor([1.0]),
        "object": _ArbitraryObject(),
    }

    # when
    with _flag_off:
        payload = json.loads(serialize_for_modal_remote_execution(inputs=inputs))

    # then
    assert payload["native"]["_type"] == "object"
    assert payload["native"]["class"] == "Detections"
    assert payload["tensor"]["_type"] == "object"
    assert payload["tensor"]["class"] == "Tensor"
    assert payload["object"] == {
        "_type": "object",
        "class": "_ArbitraryObject",
        "value": "arbitrary-object",
    }


def test_modal_serializer_sv_detections_ride_dedicated_arm_when_flag_on() -> None:
    # given - the Option-A input leg: boundary-converted sv rides the dedicated
    # wire arm, never the generic fallback
    detections = sv.Detections(
        xyxy=np.array([[10.0, 10.0, 60.0, 60.0]], dtype=np.float32),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float32),
        data={
            "class_name": np.array(["widget"]),
            "detection_id": np.array(["det-1"]),
            "image_dimensions": np.array([[480, 640]]),
        },
    )

    # when
    with _flag_on:
        payload = json.loads(
            serialize_for_modal_remote_execution(inputs={"predictions": detections})
        )

    # then
    assert payload["predictions"]["_type"] == "sv_detections"
    assert payload["predictions"]["predictions"][0]["class"] == "widget"
