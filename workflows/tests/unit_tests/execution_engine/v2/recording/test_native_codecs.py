"""Native recording codecs: tensors, ImageData and prediction carriers as raw binary."""

import dataclasses
import datetime
import json
from typing import Any

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.execution_engine.v2.blocks import (
    NATIVE_RECORDING_CODECS,
    create_catalogue,
)
from roboflow_workflows.execution_engine.v2.blocks.image_data import (
    CompositeSource,
    FrameMapping,
    ImageData,
)
from roboflow_workflows.execution_engine.v2.blocks.payload_codec import (
    NATIVE_CARRIERS,
    SUPPORTED_TENSOR_DTYPES,
)
from roboflow_workflows.execution_engine.v2.recording import (
    CodecRegistry,
    RecordingCodecError,
    RecordingCorruptError,
)

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from tests.unit_tests.execution_engine.v2.recording.test_codecs import MemoryBlobs

REGISTRY = CodecRegistry(create_catalogue().codecs.values())


def encode(value: Any):
    blobs = MemoryBlobs()
    node = REGISTRY.encoder(blobs, location="g.f").encode(value)
    return json.loads(json.dumps(node, allow_nan=False)), blobs


def round_trip(value: Any) -> Any:
    node, blobs = encode(value)
    return REGISTRY.decoder(blobs, location="g.f").decode(node)


def assert_same_tensor(actual: Any, expected: torch.Tensor) -> None:
    assert type(actual) is torch.Tensor
    assert actual.device.type == "cpu"
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected.detach().cpu())


def assert_same_carrier(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    for field in dataclasses.fields(expected):
        mine, theirs = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(theirs, torch.Tensor):
            assert_same_tensor(mine, theirs)
        elif isinstance(theirs, InstancesRLEMasks):
            assert mine == theirs
        else:
            assert mine == theirs, field.name


# Tensors -------------------------------------------------------------------------


@pytest.mark.parametrize("dtype_name", list(SUPPORTED_TENSOR_DTYPES))
def test_every_supported_tensor_dtype_round_trips_exactly(dtype_name: str) -> None:
    dtype = SUPPORTED_TENSOR_DTYPES[dtype_name]
    tensor = (torch.arange(24) % 2 if dtype is torch.bool else torch.arange(24) - 7).to(
        dtype
    )
    tensor = tensor.reshape(2, 3, 4)

    assert_same_tensor(round_trip(tensor), tensor)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.arange(60, dtype=torch.float32).reshape(3, 4, 5).transpose(0, 2),
        torch.zeros(1, 3, 4, 4).to(memory_format=torch.channels_last),
        torch.empty(0, 4),
        torch.tensor(3.5, requires_grad=True),
        torch.tensor([float("nan"), float("inf"), -0.0]),
    ],
    ids=["transposed", "channels-last", "empty", "grad", "non-finite"],
)
def test_tensor_layouts_keep_logical_values(tensor: torch.Tensor) -> None:
    decoded = round_trip(tensor)

    assert decoded.shape == tensor.shape
    assert decoded.is_contiguous()
    assert not decoded.requires_grad
    assert torch.equal(decoded.nan_to_num(), tensor.detach().nan_to_num())
    assert torch.equal(torch.signbit(decoded), torch.signbit(tensor.detach()))


@pytest.mark.parametrize("dtype", list(SUPPORTED_TENSOR_DTYPES.values()))
@pytest.mark.parametrize(
    "shape,strides",
    [((1,), (38,)), ((0,), (38,)), ((1, 1), (76, 38)), ((1,), (0,))],
    ids=["singleton-column", "empty-column", "singleton-matrix", "zero-stride"],
)
def test_contiguous_views_with_nonunit_strides_round_trip(dtype, shape, strides):
    storage = torch.arange(76).to(dtype)
    tensor = storage.as_strided(shape, strides, storage_offset=3)
    original = storage.clone()
    assert tensor.is_contiguous()
    assert tensor.stride(-1) != 1

    decoded = round_trip(tensor)

    assert decoded.dtype == dtype
    assert decoded.shape == tensor.shape
    assert torch.equal(decoded, tensor)
    assert torch.equal(storage, original)


def test_tensor_bytes_are_raw_in_the_blob_not_base64_in_json() -> None:
    tensor = torch.arange(4096, dtype=torch.int32)

    node, blobs = encode(tensor)

    assert blobs.puts == [tensor.numpy().tobytes()]
    assert len(json.dumps(node)) < 200


def test_single_instance_with_strided_class_id_round_trips():
    predictions = InstanceDetections(
        xyxy=torch.tensor([[1, 2, 5, 6]], dtype=torch.int32),
        class_id=torch.arange(76, dtype=torch.int32)[3:4:38],
        confidence=torch.linspace(0.0, 1.0, 76)[30:31:38],
        mask=torch.ones((1, 8, 8), dtype=torch.bool),
        image_metadata={"class_names": {3: "person"}},
    )
    assert predictions.class_id.stride() == (38,)

    decoded = round_trip(predictions)

    assert_same_carrier(decoded, predictions)


def test_decoded_tensors_own_writable_memory() -> None:
    node, blobs = encode(torch.zeros(4))

    first = REGISTRY.decoder(blobs).decode(node)
    first += 1
    second = REGISTRY.decoder(blobs).decode(node)

    assert torch.equal(second, torch.zeros(4))


def test_unsupported_tensors_fail_explicitly() -> None:
    with pytest.raises(RecordingCodecError, match=r"g\.f\[0\].*complex64"):
        encode([torch.zeros(2, dtype=torch.complex64)])
    with pytest.raises(
        RecordingCodecError, match="no recording codec for torch.nn.parameter.Parameter"
    ):
        encode(torch.nn.Parameter(torch.zeros(2)))


def test_tensor_data_that_do_not_match_the_shape_are_corruption() -> None:
    node, blobs = encode(torch.zeros(4))
    node["codec"]["value"]["shape"] = [5]

    with pytest.raises(RecordingCorruptError, match="needs 20 bytes, got 16"):
        REGISTRY.decoder(blobs, location="g[0].f").decode(node)


def test_bool_tensor_bytes_other_than_0_and_1_are_corruption() -> None:
    node, blobs = encode(torch.tensor([True, False, True]))
    blobs.data[2] = 5

    with pytest.raises(RecordingCorruptError, match="only bytes 0 and 1"):
        REGISTRY.decoder(blobs, location="g[0].f").decode(node)

    empty, empty_blobs = encode(torch.zeros(0, 4, dtype=torch.bool))
    decoded = REGISTRY.decoder(empty_blobs).decode(empty)
    assert decoded.dtype is torch.bool and decoded.shape == (0, 4)


# ImageData -----------------------------------------------------------------------


def frame() -> torch.Tensor:
    return (
        torch.arange(3 * 6 * 8, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(3, 6, 8)
    )


def test_image_data_keeps_pixels_identity_geometry_and_video_metadata() -> None:
    video = VideoMetadata(
        video_identifier="camera-7",
        frame_number=42,
        frame_timestamp=datetime.datetime(
            2026, 10, 6, 12, 0, 0, 123456, tzinfo=datetime.timezone.utc
        ),
        fps=29.97,
        measured_fps=None,
        comes_from_video_file=True,
    )
    root = ImageData.from_tensor(frame(), image_id="root", video_metadata=video)
    crop = root.crop((1, 2, 7, 5))
    image = ImageData(
        tensor_image=crop.tensor_image,
        image_id="crop-1",
        parent=FrameMapping(
            "root", (6, 8), scale_xy=(0.5, 1 / 3), offset_xy=(1.25, -2.0)
        ),
        root=crop.root,
        video_metadata=video,
    )

    decoded = round_trip(image)

    assert type(decoded) is ImageData
    assert_same_tensor(decoded.tensor_image, image.tensor_image)
    assert decoded.image_id == "crop-1"
    assert decoded.parent == image.parent
    assert decoded.root == image.root
    assert decoded.video_metadata == video
    assert decoded.composite_sources is None


def test_grayscale_and_composite_images_round_trip() -> None:
    tile = ImageData.from_tensor(frame()[:1], image_id="tile")
    canvas = ImageData(
        tensor_image=torch.zeros(1, 12, 16, dtype=torch.uint8),
        image_id="canvas",
        parent=FrameMapping.identity("canvas", (12, 16)),
        root=FrameMapping.identity("canvas", (12, 16)),
        composite_sources=(
            CompositeSource.place(tile, index=(0, 3), canvas_xyxy=(0, 0, 8, 6)),
        ),
    )

    decoded = round_trip(canvas)

    assert decoded.composite_sources == canvas.composite_sources
    assert decoded.tensor_image.shape == (1, 12, 16)


def test_image_pixels_are_one_raw_blob() -> None:
    image = ImageData.from_tensor(frame(), image_id="root")

    node, blobs = encode(image)

    assert blobs.puts == [image.tensor_image.numpy().tobytes()]
    assert node["codec"]["name"] == "v2/image_data@1"


def test_invalid_recorded_image_is_corruption() -> None:
    node, blobs = encode(ImageData.from_tensor(frame(), image_id="root"))
    node["codec"]["value"]["image_id"] = ""

    with pytest.raises(RecordingCorruptError, match="invalid image"):
        REGISTRY.decoder(blobs).decode(node)


# Prediction carriers ---------------------------------------------------------------------


def boxes(rows: int) -> dict:
    return {
        "xyxy": torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4),
        "class_id": torch.arange(rows, dtype=torch.int64),
        "confidence": torch.linspace(0.1, 0.9, rows),
    }


CARRIERS = {
    "Detections": Detections(
        **boxes(3),
        image_metadata={"image_id": "root"},
        bboxes_metadata=[{}, {"a": 1}, {}],
    ),
    "EmptyDetections": Detections(**boxes(0)),
    "InstanceDetectionsBitmask": InstanceDetections(
        **boxes(2), mask=torch.rand(2, 64, 48) > 0.5, image_metadata={"k": (1, 2)}
    ),
    "InstanceDetectionsRle": InstanceDetections(
        **boxes(2),
        mask=InstancesRLEMasks(image_size=(64, 48), masks=[b"\x01\x02rle", "text-rle"]),
    ),
    "KeyPoints": KeyPoints(
        xy=torch.rand(2, 5, 2),
        class_id=torch.tensor([0, 1]),
        confidence=torch.rand(2, 5),
        covariance=None,
        detection_confidence=torch.tensor([0.5, 0.75]),
    ),
    "Classification": ClassificationPrediction(
        class_id=torch.tensor([2]),
        confidence=torch.rand(1, 4),
        images_metadata=[{"x": None}],
    ),
    "MultiLabel": MultiLabelClassificationPrediction(
        class_ids=torch.tensor([0, 3]), confidence=torch.rand(4)
    ),
}


@pytest.mark.parametrize("name", list(CARRIERS))
def test_native_carriers_round_trip_every_field(name: str) -> None:
    carrier = CARRIERS[name]

    assert_same_carrier(round_trip(carrier), carrier)


def test_carrier_tuples_round_trip() -> None:
    value = (CARRIERS["Detections"], CARRIERS["KeyPoints"])

    decoded = round_trip(value)

    assert type(decoded) is tuple
    assert_same_carrier(decoded[1], value[1])


def test_masks_are_stored_as_raw_bytes() -> None:
    carrier = CARRIERS["InstanceDetectionsBitmask"]

    node, blobs = encode(carrier)

    assert carrier.mask.numpy().tobytes() in blobs.puts
    assert len(json.dumps(node)) < 2_000


def test_rle_mask_bytes_are_stored_as_raw_bytes() -> None:
    node, blobs = encode(CARRIERS["InstanceDetectionsRle"])

    assert b"\x01\x02rle" in blobs.puts


def test_every_native_carrier_has_a_codec() -> None:
    names = {codec.name for codec in NATIVE_RECORDING_CODECS}

    assert {f"v2/{name}@1" for name in NATIVE_CARRIERS} <= names
    assert {"v2/torch_tensor@1", "v2/image_data@1"} <= names
    assert set(create_catalogue().codecs) == names


def test_carrier_with_wrong_fields_is_corruption() -> None:
    node, blobs = encode(CARRIERS["Detections"])
    del node["codec"]["value"]["fields"]["xyxy"]

    with pytest.raises(RecordingCorruptError, match="needs exactly the fields"):
        REGISTRY.decoder(blobs).decode(node)
