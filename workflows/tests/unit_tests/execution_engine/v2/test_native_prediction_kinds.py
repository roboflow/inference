"""Tests of the native prediction kinds in ``v2/blocks/predictions.py``.

Each native payload family is built the way tensor-native producers build it.
The tests check the boundary promises of every kind:

* native ingress returns the same object, with storage and device untouched;
* validation reads only tensor metadata (``meta`` tensors have no data);
* serialization writes the faithful native envelope, and decoding it gives
  CPU payloads that encode to the same wire value;
* the formerly lossy cases (empty tensors, empty detections with provenance,
  anisotropic scales, keypoints without boxes, disconnected RLE, custom
  metadata) keep every field;
* legacy V1 JSON is imported only where declared, and malformed payloads fail
  with errors naming the problem.

The last tests run the real compiler and executor with a pass-through block
and ``own`` / ``root`` outputs of a non-unit, anisotropic frame.
"""

import json
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from pydantic import Field
from roboflow_workflows.execution_engine.entities import tensor_native_types
from roboflow_workflows.execution_engine.v2.blocks.coordinates import (
    convert_prediction_output,
)
from roboflow_workflows.execution_engine.v2.blocks.payload_codec import (
    NATIVE_ENVELOPE_KEY,
    encode_native,
)
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    BAR_CODE_DETECTION_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    DETECTION_KIND,
    EMBEDDING_KIND,
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
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.kinds import BUILTIN_KINDS, Kind

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

HEIGHT, WIDTH = 20, 30
CLASS_NAMES = {0: "car", 1: "truck"}
# Frame of a crop of a resized crop, as ImageData.prediction_metadata() writes
# it: root 400x300, root offset (40, 82), V1 scale [0.5, 0.25] (local pixels
# per root pixel), so local (2, 4) maps to root (2 / 0.5 + 40, 4 / 0.25 + 82).
CROP_METADATA = {
    "class_names": CLASS_NAMES,
    "image_dimensions": [HEIGHT, WIDTH],
    "parent_id": "crop-2",
    "parent_frame_id": "resize-1",
    "parent_coordinates": [5.0, 3.0],
    "parent_dimensions": [60, 80],
    "scaling_relative_to_parent": 1.0,
    "root_parent_id": "image-0",
    "root_parent_coordinates": [40.0, 82.0],
    "root_parent_dimensions": [300, 400],
    "scaling_relative_to_root_parent": [0.5, 0.25],
}
BOXES = [[2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 3.0, 3.0]]
ROOT_BOXES = [[44.0, 98.0, 52.0, 114.0], [40.0, 82.0, 46.0, 94.0]]


def _masks() -> np.ndarray:
    masks = np.zeros((2, HEIGHT, WIDTH), dtype=bool)
    masks[0, 2:8, 1:5] = True
    masks[1, 0:3, 0:3] = True

    return masks


def _rle_masks(masks: np.ndarray) -> InstancesRLEMasks:
    encoded = [
        mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"]
        for mask in masks
    ]

    return InstancesRLEMasks(image_size=(HEIGHT, WIDTH), masks=encoded)


def _box_fields(device: str = "cpu") -> Dict[str, Any]:
    fields = {
        "xyxy": torch.tensor(BOXES, device=device),
        "class_id": torch.tensor([0, 1], device=device),
        "confidence": torch.tensor([0.9, 0.5], device=device),
        "image_metadata": dict(CROP_METADATA),
        "bboxes_metadata": [
            {"detection_id": "a", "tracker_id": 3},
            {"detection_id": "b", "class": "plate text"},
        ],
    }

    return fields


def _detections(device: str = "cpu") -> Detections:
    return Detections(**_box_fields(device=device))


def _code_detections() -> Detections:
    detections = _detections()
    detections.bboxes_metadata = [
        {"detection_id": "a", "data": "https://roboflow.com"},
        {"detection_id": "b", "data": "second code"},
    ]

    return detections


def _dense_instances(device: str = "cpu") -> InstanceDetections:
    mask = torch.as_tensor(_masks(), device=device)

    return InstanceDetections(mask=mask, **_box_fields(device=device))


def _rle_instances() -> InstanceDetections:
    return InstanceDetections(mask=_rle_masks(_masks()), **_box_fields())


def _semantic_segmentation() -> InstanceDetections:
    # Class 0 covers two disconnected regions.
    masks = np.zeros((2, HEIGHT, WIDTH), dtype=bool)
    masks[0, 0:4, 0:4] = True
    masks[0, 15:20, 25:30] = True
    masks[1, 8:12, 10:20] = True
    prediction = InstanceDetections(mask=_rle_masks(masks), **_box_fields())
    prediction.image_metadata["prediction_type"] = "semantic-segmentation"

    return prediction


def _key_points(device: str = "cpu") -> KeyPoints:
    return KeyPoints(
        xy=torch.tensor([[[2.0, 4.0], [6.0, 8.0]]] * 2, device=device),
        class_id=torch.tensor([0, 1], device=device),
        confidence=torch.tensor([[0.8, 0.0]] * 2, device=device),
        image_metadata=dict(CROP_METADATA),
        key_points_metadata=[{"skeleton": "person"}, {"skeleton": "person"}],
        covariance=torch.tensor([[[[4.0, 1.0], [1.0, 2.0]]] * 2] * 2, device=device),
        detection_confidence=torch.tensor([0.7, 0.6], device=device),
    )


def _key_point_prediction(device: str = "cpu") -> Tuple[KeyPoints, Detections]:
    return _key_points(device=device), _detections(device=device)


def _key_points_without_boxes() -> Tuple[KeyPoints, None]:
    return _key_points(), None


def _classification() -> ClassificationPrediction:
    return ClassificationPrediction(
        class_id=torch.tensor([1]),
        confidence=torch.tensor([[0.25, 0.75]]),
        images_metadata=[
            {
                "class_names": CLASS_NAMES,
                "image_dimensions": [HEIGHT, WIDTH],
                "parent_id": "image-0",
                "time": 0.25,
            }
        ],
    )


def _multi_label_classification() -> MultiLabelClassificationPrediction:
    return MultiLabelClassificationPrediction(
        class_ids=torch.tensor([0]),
        confidence=torch.tensor([0.75, 0.25]),
        image_metadata={"class_names": CLASS_NAMES, "parent_id": "image-0"},
    )


def _detection_tuple() -> tuple:
    return next(iter(_rle_instances()))


def _empty_detections() -> Detections:
    return Detections(
        xyxy=torch.zeros(0, 4, dtype=torch.float16),
        class_id=torch.zeros(0, dtype=torch.int32),
        confidence=torch.zeros(0, dtype=torch.float16),
        image_metadata=dict(CROP_METADATA),
        bboxes_metadata=[],
    )


NATIVE_PAYLOADS: List[Tuple[Kind, Callable[[], Any]]] = [
    (EMBEDDING_KIND, lambda: torch.tensor([0.25, -0.5, 1.0], dtype=torch.float64)),
    (TENSOR_KIND, lambda: torch.zeros(0, 8, dtype=torch.float16)),
    (TENSOR_KIND, lambda: torch.tensor(7, dtype=torch.int16)),
    (CLASSIFICATION_PREDICTION_KIND, _classification),
    (CLASSIFICATION_PREDICTION_KIND, _multi_label_classification),
    (DETECTION_KIND, _detection_tuple),
    (OBJECT_DETECTION_PREDICTION_KIND, _detections),
    (OBJECT_DETECTION_PREDICTION_KIND, _empty_detections),
    (QR_CODE_DETECTION_KIND, _code_detections),
    (BAR_CODE_DETECTION_KIND, _code_detections),
    (INSTANCE_SEGMENTATION_PREDICTION_KIND, _dense_instances),
    (INSTANCE_SEGMENTATION_PREDICTION_KIND, _rle_instances),
    (RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND, _rle_instances),
    (SEMANTIC_SEGMENTATION_PREDICTION_KIND, _semantic_segmentation),
    (KEYPOINT_DETECTION_PREDICTION_KIND, _key_point_prediction),
    (KEYPOINT_DETECTION_PREDICTION_KIND, _key_points_without_boxes),
]
PAYLOAD_IDS = [
    f"{kind.name}-{factory.__name__.strip('_<>')}-{index}"
    for index, (kind, factory) in enumerate(NATIVE_PAYLOADS)
]
GEOMETRY_KINDS = (
    DETECTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    QR_CODE_DETECTION_KIND,
    BAR_CODE_DETECTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)


def _tensors_of(payload: Any) -> List[torch.Tensor]:
    if isinstance(payload, torch.Tensor):
        return [payload]
    if isinstance(payload, (tuple, list)):
        return [tensor for item in payload for tensor in _tensors_of(item)]
    if hasattr(payload, "__dataclass_fields__"):
        values = [getattr(payload, name) for name in payload.__dataclass_fields__]
        return [tensor for value in values for tensor in _tensors_of(value)]

    return []


def _wire_round_trip(kind: Kind, payload: Any) -> Any:
    wire = json.loads(json.dumps(kind.to_serialized(payload), allow_nan=False))

    return kind.to_payload(wire)


def test_native_kinds_cover_every_non_image_v1_tensor_native_kind() -> None:
    # given
    v1_names = {
        value.name
        for name, value in vars(tensor_native_types).items()
        if name.endswith("_KIND")
    }

    # when
    v2_names = [kind.name for kind in NATIVE_KINDS]

    # then
    assert sorted(v2_names) == sorted(v1_names)
    assert not set(v2_names) & {kind.name for kind in BUILTIN_KINDS}
    assert all(
        kind.validate and kind.deserialize and kind.serialize for kind in NATIVE_KINDS
    )
    assert set(Catalogue(kinds=NATIVE_KINDS).kinds) >= set(v2_names)


def test_geometry_and_classification_kinds_use_the_coordinate_hook() -> None:
    hooked = [kind.name for kind in NATIVE_KINDS if kind.convert_output is not None]

    assert all(
        kind.convert_output is convert_prediction_output for kind in GEOMETRY_KINDS
    )
    assert CLASSIFICATION_PREDICTION_KIND.convert_output is convert_prediction_output
    assert sorted(hooked) == sorted(
        [kind.name for kind in GEOMETRY_KINDS] + ["classification_prediction"]
    )


@pytest.mark.parametrize(("kind", "factory"), NATIVE_PAYLOADS, ids=PAYLOAD_IDS)
def test_native_ingress_returns_the_same_object(kind: Kind, factory: Callable) -> None:
    # given
    payload = factory()
    storage = [tensor.data_ptr() for tensor in _tensors_of(payload)]

    # when
    ingested = kind.to_payload(payload)

    # then: neither hook swapped a tensor inside the object for a copy
    kind.check(ingested)
    assert ingested is payload
    assert [tensor.data_ptr() for tensor in _tensors_of(ingested)] == storage


@pytest.mark.parametrize(
    ("kind", "factory"),
    [
        (OBJECT_DETECTION_PREDICTION_KIND, _detections),
        (INSTANCE_SEGMENTATION_PREDICTION_KIND, _dense_instances),
        (KEYPOINT_DETECTION_PREDICTION_KIND, _key_point_prediction),
        (DETECTION_KIND, lambda device: next(iter(_dense_instances(device=device)))),
    ],
)
def test_validation_reads_no_tensor_values(kind: Kind, factory: Callable) -> None:
    # given: meta tensors carry shape, dtype and device but no data
    payload = factory(device="meta")

    # when
    kind.check(payload)
    ingested = kind.to_payload(payload)

    # then
    assert ingested is payload
    assert {tensor.device.type for tensor in _tensors_of(payload)} == {"meta"}


@pytest.mark.parametrize(("kind", "factory"), NATIVE_PAYLOADS, ids=PAYLOAD_IDS)
def test_serialized_payload_round_trips_to_the_same_native_value(
    kind: Kind, factory: Callable
) -> None:
    # given
    payload = factory()

    # when
    serialized = kind.to_serialized(payload)
    decoded = _wire_round_trip(kind, payload)

    # then: the codec itself is checked field by field in test_native_codecs.py
    assert serialized[NATIVE_ENVELOPE_KEY] == 1
    kind.check(decoded)
    assert type(decoded) is type(payload)
    assert encode_native(decoded) == encode_native(payload)
    assert {tensor.device.type for tensor in _tensors_of(decoded)} <= {"cpu"}


def test_empty_tensor_keeps_dtype_and_zero_sized_shape() -> None:
    # when
    decoded = _wire_round_trip(TENSOR_KIND, torch.zeros(0, 8, dtype=torch.float16))

    # then
    assert (decoded.dtype, tuple(decoded.shape)) == (torch.float16, (0, 8))


def test_empty_detections_keep_provenance_and_anisotropic_scale() -> None:
    # when
    decoded = _wire_round_trip(OBJECT_DETECTION_PREDICTION_KIND, _empty_detections())

    # then
    assert decoded.image_metadata == CROP_METADATA
    assert decoded.image_metadata["class_names"] == {0: "car", 1: "truck"}
    assert decoded.image_metadata["scaling_relative_to_root_parent"] == [0.5, 0.25]
    assert (decoded.xyxy.dtype, tuple(decoded.xyxy.shape)) == (torch.float16, (0, 4))
    assert decoded.class_id.dtype == torch.int32
    assert decoded.bboxes_metadata == []


def test_key_points_keep_covariance_confidences_and_absent_boxes() -> None:
    # given
    key_points = _key_points()

    # when
    decoded_key_points, decoded_boxes = _wire_round_trip(
        KEYPOINT_DETECTION_PREDICTION_KIND, (key_points, None)
    )

    # then: hidden keypoint slots (confidence 0) keep their positions
    assert decoded_boxes is None
    assert torch.equal(decoded_key_points.xy, key_points.xy)
    assert torch.equal(decoded_key_points.confidence, key_points.confidence)
    assert torch.equal(decoded_key_points.covariance, key_points.covariance)
    assert torch.equal(
        decoded_key_points.detection_confidence, key_points.detection_confidence
    )
    assert decoded_key_points.key_points_metadata == key_points.key_points_metadata


def test_semantic_rle_keeps_disconnected_regions() -> None:
    # given
    prediction = _semantic_segmentation()

    # when
    decoded = _wire_round_trip(SEMANTIC_SEGMENTATION_PREDICTION_KIND, prediction)

    # then
    assert decoded.mask.masks == prediction.mask.masks
    rle = {"size": list(decoded.mask.image_size), "counts": decoded.mask.masks[0]}
    pixels = mask_utils.decode(rle)
    assert pixels[0:4, 0:4].all() and pixels[15:20, 25:30].all()
    assert int(pixels.sum()) == 16 + 25


def test_detection_row_keeps_rle_mask_scalars_and_tracker_id() -> None:
    # given
    xyxy, mask, class_id, confidence, _, data, metadata = _detection_tuple()
    row = (xyxy, mask, class_id, confidence, np.int64(5), data, metadata)

    # when
    decoded = _wire_round_trip(DETECTION_KIND, row)

    # then
    assert decoded[1] == mask
    assert (decoded[2].dtype, decoded[2].shape) == (class_id.dtype, torch.Size([]))
    assert type(decoded[4]) is np.int64 and decoded[4] == 5
    assert decoded[6] == metadata


def test_code_strings_ids_labels_and_custom_metadata_survive() -> None:
    # given: custom values within the documented grammar, with tag-like keys
    detections = _code_detections()
    detections.bboxes_metadata[0].update(
        {
            "polygon": np.array([[1, 2], [3, 4]], dtype=np.int32),
            "velocity": torch.tensor([0.5, -1.0]),
            "counts": b"\x01\x02",
            "span": (3, 4),
            "tensor": {"not": "a tag"},
            NATIVE_ENVELOPE_KEY: "just a key",
        }
    )

    # when
    decoded = _wire_round_trip(QR_CODE_DETECTION_KIND, detections)

    # then
    first, second = decoded.bboxes_metadata
    assert first["data"] == "https://roboflow.com" and second["data"] == "second code"
    assert first["polygon"].dtype == np.int32
    assert torch.equal(first["velocity"], torch.tensor([0.5, -1.0]))
    assert first["counts"] == b"\x01\x02" and first["span"] == (3, 4)
    assert first["tensor"] == {"not": "a tag"}
    assert first[NATIVE_ENVELOPE_KEY] == "just a key"


def test_unsupported_metadata_fails_serialization_with_its_path() -> None:
    # given
    detections = _detections()
    detections.image_metadata["video"] = object()

    # when / then
    with pytest.raises(ContractError, match=r"image_metadata\['video'\]: unsupported"):
        OBJECT_DETECTION_PREDICTION_KIND.to_serialized(detections)


def test_family_serializers_reject_other_families() -> None:
    # given: an RLE payload reaching a kind union [object_detection, rle]
    rle = _rle_instances()

    # when / then
    with pytest.raises(ContractError, match="Expected inference_models.Detections"):
        OBJECT_DETECTION_PREDICTION_KIND.to_serialized(rle)
    with pytest.raises(ContractError, match="must be InstancesRLEMasks"):
        RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND.to_serialized(_dense_instances())
    with pytest.raises(ContractError, match="Expected inference_models.Detections"):
        OBJECT_DETECTION_PREDICTION_KIND.to_payload(
            RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND.to_serialized(rle)
        )


def test_legacy_v1_json_is_imported_where_declared() -> None:
    # given
    legacy_detections = {
        "image": {"width": WIDTH, "height": HEIGHT},
        "predictions": [
            {
                "x": 4.0,
                "y": 6.0,
                "width": 4.0,
                "height": 4.0,
                "confidence": 0.9,
                "class_id": 0,
                "class": "car",
                "detection_id": "a",
            }
        ],
    }
    legacy_rle = {
        "image": {"width": WIDTH, "height": HEIGHT},
        "predictions": [
            {
                **legacy_detections["predictions"][0],
                "rle_mask": {"size": [HEIGHT, WIDTH], "counts": "PP1"},
            }
        ],
    }

    # when
    boxes = OBJECT_DETECTION_PREDICTION_KIND.to_payload(legacy_detections)
    masks = RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND.to_payload(legacy_rle)
    tensor = TENSOR_KIND.to_payload([[1, 2], [3, 4]])

    # then
    assert boxes.xyxy.tolist() == [[2.0, 4.0, 6.0, 8.0]]
    assert boxes.bboxes_metadata[0]["detection_id"] == "a"
    assert isinstance(masks.mask, InstancesRLEMasks)
    assert (tensor.dtype, tuple(tensor.shape)) == (torch.float32, (2, 2))


INVALID_WIRE_VALUES = [
    (OBJECT_DETECTION_PREDICTION_KIND, {"predictions": []}, "importable legacy V1"),
    (OBJECT_DETECTION_PREDICTION_KIND, "boxes", "importable legacy V1"),
    (
        KEYPOINT_DETECTION_PREDICTION_KIND,
        {"image": {}, "predictions": []},
        "Expected tuple or a native envelope, got dict",
    ),
    (DETECTION_KIND, [[0, 0, 1, 1], None, 0], "Expected tuple or a native envelope"),
    (EMBEDDING_KIND, "0.5, 0.5", "legacy nested list of numbers, got str"),
    (
        TENSOR_KIND,
        {NATIVE_ENVELOPE_KEY: 1, "value": {"tuple": []}},
        "Expected a torch.Tensor, got tuple",
    ),
    (
        TENSOR_KIND,
        {NATIVE_ENVELOPE_KEY: 3, "value": None},
        "Unsupported native envelope version 3",
    ),
]


@pytest.mark.parametrize(("kind", "value", "message"), INVALID_WIRE_VALUES)
def test_invalid_wire_values_are_rejected_with_the_reason(
    kind: Kind, value: Any, message: str
) -> None:
    # when / then
    with pytest.raises(ContractError, match=message):
        kind.to_payload(value)


def _with(factory: Callable, **changes: Any) -> Callable[[], Any]:
    def build() -> Any:
        payload = factory()
        for name, value in changes.items():
            setattr(payload, name, value)
        return payload

    return build


def _key_points_with_one_box() -> Tuple[KeyPoints, Detections]:
    one_box = Detections(
        xyxy=torch.zeros(1, 4),
        class_id=torch.zeros(1, dtype=torch.long),
        confidence=torch.zeros(1),
    )

    return _key_points(), one_box


INVALID_PAYLOADS = [
    (EMBEDDING_KIND, lambda: torch.zeros(2, 3), r"shape \(\*\), got \(2, 3\)"),
    (EMBEDDING_KIND, lambda: torch.tensor([1, 2]), "dtype must be floating"),
    (EMBEDDING_KIND, lambda: [0.5, 0.5], "must be a torch.Tensor, got list"),
    (TENSOR_KIND, lambda: np.zeros(3), "Expected a torch.Tensor, got ndarray"),
    (
        TENSOR_KIND,
        lambda: torch.zeros(2, dtype=torch.complex64),
        "complex64 is not supported",
    ),
    (
        TENSOR_KIND,
        lambda: torch.zeros(2, dtype=torch.uint16),
        "uint16 is not supported",
    ),
    (
        OBJECT_DETECTION_PREDICTION_KIND,
        _key_point_prediction,
        "Expected inference_models.Detections, got tuple",
    ),
    (
        OBJECT_DETECTION_PREDICTION_KIND,
        _with(_detections, class_id=torch.tensor([0.0, 1.0])),
        "class_id dtype must be integer",
    ),
    (
        OBJECT_DETECTION_PREDICTION_KIND,
        _with(_detections, confidence=torch.tensor([0.9])),
        r"confidence must have shape \(2\), got \(1,\)",
    ),
    (
        OBJECT_DETECTION_PREDICTION_KIND,
        _with(_detections, bboxes_metadata=[{"detection_id": "a"}]),
        "bboxes_metadata has 1 entries for 2 rows",
    ),
    (
        OBJECT_DETECTION_PREDICTION_KIND,
        _with(_detections, class_id=torch.tensor([0, 1], device="meta")),
        "class_id is on meta, but the rest of the prediction is on cpu",
    ),
    (
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        _with(_dense_instances, mask=torch.zeros(2, HEIGHT, WIDTH)),
        "mask dtype must be binary",
    ),
    (
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        _with(_dense_instances, mask=torch.zeros(1, HEIGHT, WIDTH, dtype=torch.bool)),
        r"mask must have shape \(2, \*, \*\)",
    ),
    (
        RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        _with(
            _rle_instances,
            mask=InstancesRLEMasks(image_size=(HEIGHT, WIDTH), masks=[b"x"]),
        ),
        "holds 1 masks for 2 rows",
    ),
    (
        SEMANTIC_SEGMENTATION_PREDICTION_KIND,
        _dense_instances,
        "must be InstancesRLEMasks for this kind, got Tensor",
    ),
    (
        KEYPOINT_DETECTION_PREDICTION_KIND,
        _key_points_with_one_box,
        "KeyPoints has 2 instances but its Detections has 1 boxes",
    ),
    (
        KEYPOINT_DETECTION_PREDICTION_KIND,
        lambda: (_with(_key_points, covariance=torch.zeros(2, 2, 2))(), None),
        r"covariance must have shape \(2, 2, 2, 2\)",
    ),
    (
        KEYPOINT_DETECTION_PREDICTION_KIND,
        _detections,
        r"Expected a \(KeyPoints, Detections \| None\) tuple",
    ),
    (
        CLASSIFICATION_PREDICTION_KIND,
        _with(
            _classification, class_id=torch.tensor([1, 0]), confidence=torch.zeros(2, 2)
        ),
        r"class_id must have shape \(1\), got \(2,\)",
    ),
    (
        CLASSIFICATION_PREDICTION_KIND,
        _detections,
        "Expected inference_models.ClassificationPrediction",
    ),
    (DETECTION_KIND, lambda: _detection_tuple()[:6], "Expected the tuple"),
    (
        DETECTION_KIND,
        lambda: (torch.zeros(4), None, 1.5, 0.5, None, {}, {}),
        "class_id must be a torch.Tensor, got float",
    ),
]


@pytest.mark.parametrize(("kind", "factory", "message"), INVALID_PAYLOADS)
def test_invalid_native_payloads_are_rejected_with_the_reason(
    kind: Kind, factory: Callable, message: str
) -> None:
    # given
    payload = factory()

    # when / then
    with pytest.raises(ContractError, match=message):
        kind.check(payload)


def test_classification_output_hook_checks_options_and_keeps_the_payload() -> None:
    # given
    prediction = _classification()

    # when
    converted = CLASSIFICATION_PREDICTION_KIND.to_output(
        prediction, options={"coordinates_system": "root"}
    )

    # then: classifications carry no spatial field
    assert converted is prediction
    with pytest.raises(ValueError, match="Unknown coordinates_system 'sideways'"):
        CLASSIFICATION_PREDICTION_KIND.to_output(
            prediction, options={"coordinates_system": "sideways"}
        )


class _PassPredictions(Block):
    """Return the received detections unchanged."""

    type = "test/pass_predictions"
    outputs = {"predictions": Output(OBJECT_DETECTION_PREDICTION_KIND)}

    class Params(BlockParams):
        predictions: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description="Detections to return."
        )

    def run(self, *, predictions: Detections) -> Dict[str, Any]:
        return {"predictions": predictions}


def _pass_through_plan() -> Any:
    selector = "$steps.pass.predictions"
    definition = {
        "version": "2.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            }
        ],
        "steps": [
            {
                "type": "test/pass_predictions",
                "name": "pass",
                "predictions": "$inputs.predictions",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "own",
                "selector": selector,
                "coordinates_system": "own",
            },
            {
                "type": "JsonField",
                "name": "root",
                "selector": selector,
                "coordinates_system": "root",
            },
            {
                "type": "JsonField",
                "name": "parent",
                "selector": selector,
                "coordinates_system": "parent",
            },
            # Decision 036: no option means parent, the workflow root (V1).
            {"type": "JsonField", "name": "default", "selector": selector},
        ],
    }
    catalogue = Catalogue([_PassPredictions], kinds=NATIVE_KINDS)
    plan = compile_workflow(definition, catalogue=catalogue)

    return plan


def test_workflow_outputs_own_and_root_coordinates_without_mutating_siblings() -> None:
    # given
    detections = _detections()
    plan = _pass_through_plan()

    # when
    result = plan.create_session().run({"predictions": [detections]})
    (row,) = result.rows()

    # then
    assert row["own"] is detections
    assert detections.xyxy.tolist() == BOXES
    assert detections.image_metadata == CROP_METADATA
    for name in ("root", "parent", "default"):
        assert row[name].xyxy.tolist() == ROOT_BOXES
        assert row[name].image_metadata["image_dimensions"] == [300, 400]
        assert row[name].image_metadata["scaling_relative_to_root_parent"] == 1.0
        assert row[name].bboxes_metadata == detections.bboxes_metadata
        assert row[name].class_id is detections.class_id


def test_workflow_serialized_rows_are_strict_json_and_decode_back() -> None:
    # given
    plan = _pass_through_plan()

    # when
    result = plan.create_session().run({"predictions": [_detections()]})
    (row,) = result.rows()
    (serialized,) = result.rows(serialize=True)
    wire = json.loads(json.dumps(serialized, allow_nan=False))

    # then
    for name in ("own", "root"):
        decoded = OBJECT_DETECTION_PREDICTION_KIND.to_payload(wire[name])
        assert encode_native(decoded) == encode_native(row[name])


def test_workflow_accepts_envelopes_and_legacy_json_and_rejects_garbage() -> None:
    # given
    plan = _pass_through_plan()
    envelope = OBJECT_DETECTION_PREDICTION_KIND.to_serialized(_detections())
    legacy = {
        "image": {"width": WIDTH, "height": HEIGHT},
        "predictions": [
            {"x": 4, "y": 6, "width": 4, "height": 4, "class": "car", "class_id": 0}
        ],
    }

    # when
    (from_envelope,) = plan.create_session().run({"predictions": [envelope]}).rows()
    (from_legacy,) = plan.create_session().run({"predictions": [legacy]}).rows()

    # then
    assert from_envelope["own"].image_metadata == CROP_METADATA
    assert from_envelope["root"].xyxy.tolist() == ROOT_BOXES
    assert from_legacy["own"].xyxy.tolist() == [[2.0, 4.0, 6.0, 8.0]]
    with pytest.raises(WorkflowInputError, match="importable legacy V1"):
        plan.create_session().run({"predictions": ["not detections"]})
