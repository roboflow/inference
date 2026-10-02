"""Explicit wildcard and legacy-labelled media boundaries over real Kind hooks."""

import json
from collections import UserDict
from functools import partial

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from pydantic import Field
from roboflow_workflows.execution_engine.v2.blocks.boundaries import (
    MEDIA_WILDCARD_KIND,
    NUMPY_ARRAY_KIND,
)
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    BAR_CODE_DETECTION_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    DETECTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    NATIVE_KINDS,
    OBJECT_DETECTION_PREDICTION_KIND,
    QR_CODE_DETECTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
    TENSOR_KIND,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    WorkflowExecutionError,
)
from roboflow_workflows.execution_engine.v2.kinds import WILDCARD_KIND

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks


def _metadata():
    return {
        "parent_id": "crop",
        "root_parent_id": "input",
        "image_dimensions": [4, 6],
        "parent_dimensions": [20, 30],
        "root_parent_dimensions": [20, 30],
        "parent_coordinates": [5, 7],
        "root_parent_coordinates": [5, 7],
        "scaling_relative_to_parent": [2.0, 0.5],
        "scaling_relative_to_root_parent": [2.0, 0.5],
        "class_names": {0: "object"},
        "source_note": "keep arbitrary metadata",
    }


def _detections(*, prediction_type="object-detection", device="cpu", empty=False):
    metadata = {**_metadata(), "prediction_type": prediction_type}
    detections = Detections(
        xyxy=torch.tensor(
            [] if empty else [[0.0, 0.0, 4.0, 2.0]], device=device
        ).reshape(-1, 4),
        class_id=torch.tensor([] if empty else [0], dtype=torch.int64, device=device),
        confidence=torch.tensor([] if empty else [0.75], device=device),
        image_metadata=metadata,
        bboxes_metadata=[] if empty else [{"detection_id": "box", "data": "code text"}],
    )

    return detections


def _instances(*, rle=False, semantic=False):
    boxes = _detections()
    dense = np.ones((4, 6), dtype=np.uint8)
    dense[1:3, 2:4] = 0
    if rle:
        mask = InstancesRLEMasks.from_coco_rle_masks(
            image_size=(4, 6), masks=[mask_utils.encode(np.asfortranarray(dense))]
        )
    else:
        mask = torch.from_numpy(dense).bool().unsqueeze(0)

    metadata = {
        **boxes.image_metadata,
        "prediction_type": (
            "semantic-segmentation" if semantic else "instance-segmentation"
        ),
    }
    instances = InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        mask=mask,
        image_metadata=metadata,
        bboxes_metadata=boxes.bboxes_metadata,
    )

    return instances


def _keypoints(*, with_boxes=True):
    boxes = _detections()
    keypoints = KeyPoints(
        xy=torch.tensor([[[1.0, 1.0], [3.0, 2.0]]]),
        class_id=boxes.class_id,
        confidence=torch.tensor([[0.75, 0.5]]),
        image_metadata=boxes.image_metadata,
        key_points_metadata=[{"keypoint_names": ["first", "second"]}],
        covariance=torch.eye(2).repeat(1, 2, 1, 1),
        detection_confidence=boxes.confidence,
    )

    return keypoints, boxes if with_boxes else None


def _classification(*, multilabel=False):
    metadata = {**_metadata(), "classification_style": "model"}
    if multilabel:
        prediction = MultiLabelClassificationPrediction(
            class_ids=torch.tensor([0]),
            confidence=torch.tensor([0.75]),
            image_metadata=metadata,
        )
    else:
        prediction = ClassificationPrediction(
            class_id=torch.tensor([0]),
            confidence=torch.tensor([[0.75]]),
            images_metadata=[metadata],
        )

    return prediction


def _image():
    image = ImageData.from_tensor(
        torch.arange(3 * 10 * 14, dtype=torch.uint8).reshape(3, 10, 14),
        image_id="image",
    ).crop((2, 2, 10, 8), image_id="image-crop")
    resized = image.resize((3, 4), image_id="image-resized")

    return resized


def _detection():
    row = next(iter(_detections()))

    return row


NATIVE_CASES = [
    pytest.param(IMAGE_KIND, _image, id="image"),
    pytest.param(
        TENSOR_KIND, lambda: torch.ones(2, 3, dtype=torch.float16), id="tensor"
    ),
    pytest.param(
        TENSOR_KIND, lambda: torch.empty(0, 8, dtype=torch.float16), id="empty-tensor"
    ),
    pytest.param(CLASSIFICATION_PREDICTION_KIND, _classification, id="classification"),
    pytest.param(
        CLASSIFICATION_PREDICTION_KIND,
        partial(_classification, multilabel=True),
        id="multilabel",
    ),
    pytest.param(DETECTION_KIND, _detection, id="detection-tuple"),
    pytest.param(OBJECT_DETECTION_PREDICTION_KIND, _detections, id="detections"),
    pytest.param(
        OBJECT_DETECTION_PREDICTION_KIND,
        partial(_detections, empty=True),
        id="empty-detections",
    ),
    pytest.param(
        QR_CODE_DETECTION_KIND,
        partial(_detections, prediction_type="qrcode-detection"),
        id="qr",
    ),
    pytest.param(
        BAR_CODE_DETECTION_KIND,
        partial(_detections, prediction_type="barcode-detection"),
        id="barcode",
    ),
    pytest.param(INSTANCE_SEGMENTATION_PREDICTION_KIND, _instances, id="dense-mask"),
    pytest.param(
        RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        partial(_instances, rle=True),
        id="rle-mask",
    ),
    pytest.param(
        SEMANTIC_SEGMENTATION_PREDICTION_KIND,
        partial(_instances, rle=True, semantic=True),
        id="semantic-mask",
    ),
    pytest.param(KEYPOINT_DETECTION_PREDICTION_KIND, _keypoints, id="keypoints"),
    pytest.param(
        KEYPOINT_DETECTION_PREDICTION_KIND,
        partial(_keypoints, with_boxes=False),
        id="keypoints-without-boxes",
    ),
]


@pytest.mark.parametrize("kind,factory", NATIVE_CASES)
def test_wildcard_delegates_native_family_encoding_to_its_typed_kind(kind, factory):
    payload = factory()

    assert MEDIA_WILDCARD_KIND.to_payload(payload) is payload
    assert (
        MEDIA_WILDCARD_KIND.to_output(payload, options={"coordinates_system": "own"})
        is payload
    )
    wire = MEDIA_WILDCARD_KIND.to_serialized(payload)

    assert wire == {
        "__workflows_v2_media__": 1,
        "kind": kind.name,
        "value": kind.to_serialized(payload),
    }
    decoded = MEDIA_WILDCARD_KIND.to_payload(
        json.loads(json.dumps(wire, allow_nan=False))
    )
    kind.check(decoded)
    assert type(decoded) is type(payload)
    assert kind.to_serialized(decoded) == wire["value"]


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_native_nested_ingress_and_own_output_keep_identity_and_device(device):
    tensor = torch.empty(0, 8, device=device, dtype=torch.float16)
    boxes = _detections(device=device)
    value = UserDict({"items": [tensor, boxes], "config": ("text", 7, False, None)})

    assert MEDIA_WILDCARD_KIND.to_payload(value) is value
    assert (
        MEDIA_WILDCARD_KIND.to_output(value, options={"coordinates_system": "own"})
        is value
    )
    assert value["items"][0] is tensor
    assert tensor.device.type == boxes.xyxy.device.type == device


def test_nested_mixed_wire_restores_native_values_without_decoding_lookalike_config():
    config = {
        "image_settings": {"type": "base64", "value": "configuration, not pixels"},
        "prediction_settings": {"image": {"height": 4}, "predictions": []},
        "tensor_settings": {"dtype": "float16", "shape": [0, 8], "data": []},
    }
    original = {
        "image": _image(),
        "items": [torch.empty(0, 8, dtype=torch.float16), _detections()],
        "special": (_detection(), _keypoints(with_boxes=False)),
        "config": config,
    }

    assert MEDIA_WILDCARD_KIND.to_payload(config) is config
    wire = MEDIA_WILDCARD_KIND.to_serialized(original)
    decoded = MEDIA_WILDCARD_KIND.to_payload(json.loads(json.dumps(wire)))

    assert decoded["config"] == config
    assert decoded["items"][0].shape == (0, 8)
    assert decoded["items"][0].dtype == torch.float16
    assert decoded["items"][1].image_metadata == original["items"][1].image_metadata
    assert isinstance(decoded["special"], list)
    assert isinstance(decoded["special"][0], tuple)
    assert isinstance(decoded["special"][1][0], KeyPoints)
    assert decoded["special"][1][1] is None
    assert decoded["image"].image_id == original["image"].image_id
    assert decoded["image"].root == original["image"].root
    assert torch.equal(decoded["image"].tensor_image, original["image"].tensor_image)


@pytest.mark.parametrize("empty", [False, True])
def test_wildcard_detections_keep_metadata_including_empty_predictions(empty):
    original = _detections(empty=empty)

    decoded = MEDIA_WILDCARD_KIND.to_payload(
        json.loads(json.dumps(MEDIA_WILDCARD_KIND.to_serialized(original)))
    )

    assert decoded.image_metadata == original.image_metadata
    assert decoded.bboxes_metadata == original.bboxes_metadata
    assert torch.equal(decoded.xyxy, original.xyxy)
    assert decoded.xyxy.shape == original.xyxy.shape


@pytest.mark.parametrize("rle", [False, True])
def test_wildcard_mask_roundtrip_preserves_holes(rle):
    original = _instances(rle=rle)

    decoded = MEDIA_WILDCARD_KIND.to_payload(
        json.loads(json.dumps(MEDIA_WILDCARD_KIND.to_serialized(original)))
    )

    if rle:
        assert decoded.mask.image_size == original.mask.image_size
        assert decoded.mask.masks == original.mask.masks
    else:
        assert torch.equal(decoded.mask, original.mask)


def test_special_detection_tuple_accepts_native_python_scalars():
    row = _detection()
    row = (*row[:2], int(row[2]), float(row[3]), *row[4:])

    wire = MEDIA_WILDCARD_KIND.to_serialized(row)

    assert wire["kind"] == "detection"
    assert MEDIA_WILDCARD_KIND.to_payload(row) is row


@pytest.mark.parametrize(
    "envelope,message",
    [
        ({"__workflows_v2_media__": 1}, "exactly"),
        ({"__workflows_v2_media__": 2, "kind": "tensor", "value": []}, "version"),
        ({"__workflows_v2_media__": True, "kind": "tensor", "value": []}, "version"),
        ({"__workflows_v2_media__": 1, "kind": "unknown", "value": []}, "kind"),
        ({"__workflows_v2_media__": 1, "kind": [], "value": []}, "kind"),
        (
            {"__workflows_v2_media__": 1, "kind": "image", "value": "not an image"},
            "image",
        ),
    ],
)
def test_malformed_explicit_media_envelopes_fail_usefully(envelope, message):
    with pytest.raises((ContractError, TypeError, ValueError), match=message):
        MEDIA_WILDCARD_KIND.to_payload(envelope)


@pytest.mark.parametrize("value", [0, 2.5, True, None, "plain"])
def test_ordinary_scalars_stay_ordinary(value):
    assert MEDIA_WILDCARD_KIND.to_payload(value) is value
    assert MEDIA_WILDCARD_KIND.to_serialized(value) is value
    assert (
        MEDIA_WILDCARD_KIND.to_output(value, options={"coordinates_system": "parent"})
        is value
    )


def test_unknown_objects_are_preserved_in_process_and_rejected_at_serialization():
    class UnknownPayload:
        def __repr__(self):
            raise AssertionError("The boundary must not stringify unknown objects")

    unknown = UnknownPayload()
    value = {"payload": [unknown]}

    assert MEDIA_WILDCARD_KIND.to_payload(value) is value
    assert (
        MEDIA_WILDCARD_KIND.to_output(value, options={"coordinates_system": "own"})
        is value
    )
    with pytest.raises(ContractError, match="Cannot serialize UnknownPayload"):
        MEDIA_WILDCARD_KIND.to_serialized(value)


@pytest.mark.parametrize(
    "value,message", [({1: "not a JSON key"}, "string keys"), (float("nan"), "finite")]
)
def test_non_json_values_fail_explicitly(value, message):
    with pytest.raises(ContractError, match=message):
        MEDIA_WILDCARD_KIND.to_serialized(value)


@pytest.mark.parametrize("coordinates_system", ["parent", "root"])
def test_wildcard_coordinate_output_recurses_through_containers(coordinates_system):
    original = _detections()
    untouched = torch.ones(2)
    value = {"predictions": [(original,)], "tensor": untouched, "image": _image()}

    converted = MEDIA_WILDCARD_KIND.to_output(
        value, options={"coordinates_system": coordinates_system}
    )

    boxes = converted["predictions"][0][0]
    assert boxes is not original
    assert torch.equal(boxes.xyxy, torch.tensor([[5.0, 7.0, 7.0, 11.0]]))
    assert boxes.image_metadata["parent_id"] == "input"
    assert torch.equal(original.xyxy, torch.tensor([[0.0, 0.0, 4.0, 2.0]]))
    assert converted["tensor"] is untouched
    assert converted["image"] is value["image"]
    assert (
        MEDIA_WILDCARD_KIND.to_output(
            converted, options={"coordinates_system": coordinates_system}
        )
        is converted
    )


def test_wildcard_special_tuples_use_typed_coordinate_hooks_before_recursing():
    detection = _detection()
    keypoints = _keypoints(with_boxes=False)

    row, skeleton = MEDIA_WILDCARD_KIND.to_output(
        [detection, keypoints], options={"coordinates_system": "parent"}
    )

    assert torch.equal(row[0], torch.tensor([5.0, 7.0, 7.0, 11.0]))
    assert torch.equal(skeleton[0].xy, torch.tensor([[[5.5, 9.0], [6.5, 11.0]]]))
    assert skeleton[1] is None


def test_wildcard_composite_prediction_rejects_root_restoration():
    composite = ImageData.composite(torch.zeros(3, 4, 6, dtype=torch.uint8), sources=())
    boxes = _detections()
    boxes.image_metadata = composite.prediction_metadata()

    assert (
        MEDIA_WILDCARD_KIND.to_output(boxes, options={"coordinates_system": "own"})
        is boxes
    )
    with pytest.raises(ValueError, match="composite|Composite"):
        MEDIA_WILDCARD_KIND.to_output(
            {"boxes": [boxes]}, options={"coordinates_system": "parent"}
        )


def test_wildcard_prediction_rejects_unknown_coordinate_system():
    with pytest.raises(ValueError, match="Unknown"):
        MEDIA_WILDCARD_KIND.to_output(
            _detections(), options={"coordinates_system": "screen"}
        )


@pytest.mark.parametrize(
    "array",
    [
        np.arange(6, dtype=np.float16).reshape(2, 3),
        np.empty((0, 8), dtype=np.float16),
        np.array(3, dtype=np.int64),
        np.arange(6, dtype=np.float32)[::-1],
        np.arange(6, dtype=">f4"),
    ],
)
def test_numpy_compatibility_ingress_normalizes_once_and_uses_faithful_tensor_wire(
    array,
):
    tensor = NUMPY_ARRAY_KIND.to_payload(array)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == array.shape
    assert np.array_equal(tensor.numpy(), array)
    assert NUMPY_ARRAY_KIND.to_payload(tensor) is tensor
    wire = NUMPY_ARRAY_KIND.to_serialized(tensor)
    assert wire == TENSOR_KIND.to_serialized(tensor)
    decoded = NUMPY_ARRAY_KIND.to_payload(json.loads(json.dumps(wire)))
    assert decoded.dtype == tensor.dtype
    assert decoded.shape == tensor.shape
    assert torch.equal(decoded, tensor)


def test_numpy_compatibility_keeps_native_storage_and_device():
    tensor = torch.empty(0, 8, dtype=torch.float16, device="meta")

    assert NUMPY_ARRAY_KIND.to_payload(tensor) is tensor
    assert (
        NUMPY_ARRAY_KIND.to_output(tensor, options={"coordinates_system": "parent"})
        is tensor
    )
    NUMPY_ARRAY_KIND.check(tensor)


@pytest.mark.parametrize(
    "array", [np.array(["text"]), np.array([object()]), np.array([1j])]
)
def test_numpy_compatibility_rejects_non_real_arrays(array):
    with pytest.raises(ContractError, match="real numeric"):
        NUMPY_ARRAY_KIND.to_payload(array)


class _WildcardEcho(Block):
    type = "test/boundary-wildcard-echo"
    outputs = {"value": Output(WILDCARD_KIND)}

    class Params(BlockParams):
        value: Ref(WILDCARD_KIND) = Field(description="Payload to pass through.")

    def run(self, *, value):
        return {"value": value}


class _DepthEcho(Block):
    type = "test/boundary-depth-echo"
    outputs = {"value": Output(NUMPY_ARRAY_KIND)}

    class Params(BlockParams):
        value: Ref(NUMPY_ARRAY_KIND) = Field(
            description="Depth tensor to pass through."
        )

    def run(self, *, value):
        return {"value": value}


def _compiled_echo(block=_WildcardEcho, *, kind="*", coordinates_system="own"):
    catalogue = Catalogue(
        [block],
        kinds=[MEDIA_WILDCARD_KIND, NUMPY_ARRAY_KIND, IMAGE_KIND, *NATIVE_KINDS],
    )
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "values", "kind": [kind]}],
        "steps": [{"type": block.type, "name": "echo", "value": "$inputs.values"}],
        "outputs": [
            {
                "type": "JsonField",
                "name": "value",
                "selector": "$steps.echo.value",
                "coordinates_system": coordinates_system,
            }
        ],
    }
    compiled = compile_workflow(definition, catalogue=catalogue)

    return compiled


def test_compiled_wildcard_uses_media_policy_and_preserves_native_in_process_values():
    tensor = torch.empty(0, 8, dtype=torch.float16)
    original = {"tensor": tensor, "boxes": _detections(), "image": _image()}
    plan = _compiled_echo()

    result = plan.create_session().run({"values": [original]})

    assert result.rows()[0]["value"] is original
    wire = result.rows(serialize=True)[0]["value"]
    decoded = (
        plan.create_session()
        .run({"values": [json.loads(json.dumps(wire))]})
        .rows()[0]["value"]
    )
    assert decoded["tensor"].dtype == torch.float16
    assert decoded["tensor"].shape == (0, 8)
    assert decoded["boxes"].image_metadata == original["boxes"].image_metadata
    assert decoded["image"].image_id == original["image"].image_id


def test_compiled_numpy_label_normalizes_to_tensor_before_the_block():
    plan = _compiled_echo(_DepthEcho, kind="numpy_array")
    value = np.empty((0, 8), dtype=np.float16)

    result = plan.create_session().run({"values": [value]})
    tensor = result.rows()[0]["value"]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (0, 8)
    assert tensor.dtype == torch.float16
    wire = result.rows(serialize=True)[0]["value"]
    decoded = NUMPY_ARRAY_KIND.to_payload(json.loads(json.dumps(wire)))
    assert decoded.shape == tensor.shape
    assert decoded.dtype == tensor.dtype


def test_compiled_wildcard_root_conversion_and_unknown_serialization_failures():
    original = _detections()
    plan = _compiled_echo(coordinates_system="parent")

    rows = plan.create_session().run({"values": [{"boxes": original}]}).rows()

    assert torch.equal(
        rows[0]["value"]["boxes"].xyxy, torch.tensor([[5.0, 7.0, 7.0, 11.0]])
    )
    unknown = object()
    result = plan.create_session().run({"values": [unknown]})
    assert result.rows()[0]["value"] is unknown
    with pytest.raises(WorkflowExecutionError, match="Cannot serialize object"):
        result.rows(serialize=True)
