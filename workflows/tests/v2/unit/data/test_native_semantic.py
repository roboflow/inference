"""Native semantic segmentation kind: labels and winning confidence, unchanged."""

import json

import pytest
import torch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.predictions import (
    NATIVE_KINDS,
    NATIVE_SEMANTIC_SEGMENTATION_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.recording import (
    CodecRegistry,
    RecordingCorruptError,
)
from roboflow_workflows.v2.recording_codecs import NATIVE_RECORDING_CODECS

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.base.types import InstancesRLEMasks
from tests.v2.support.recording.codecs import MemoryBlobs

REGISTRY = CodecRegistry(NATIVE_RECORDING_CODECS)
KIND = NATIVE_SEMANTIC_SEGMENTATION_KIND


def _root():
    return ImageData.from_tensor(
        torch.zeros((3, 4, 6), dtype=torch.uint8), image_id="root"
    )


def _result(
    *,
    label_dtype=torch.int64,
    confidence_dtype=torch.float32,
    image=None,
):
    # Class 2 covers two disconnected regions; every pixel has its own score.
    labels = torch.tensor(
        [
            [2, 2, 0, 0, 2, 2],
            [2, 2, 0, 1, 1, 2],
            [0, 0, 0, 1, 1, 0],
            [2, 0, 0, 0, 0, 0],
        ],
        dtype=label_dtype,
    )
    confidence = (torch.arange(24, dtype=torch.float32) / 32 + 0.125).reshape(4, 6)
    image = image if image is not None else _root()
    return SemanticSegmentationResult(
        segmentation_map=labels,
        confidence=confidence.to(confidence_dtype),
        image_metadata=FrameRef.from_image(image).prediction_metadata(),
    )


def _rle_semantic():
    return InstanceDetections(
        xyxy=torch.zeros((1, 4)),
        class_id=torch.tensor([2]),
        confidence=torch.tensor([1.0]),
        mask=InstancesRLEMasks(image_size=(4, 6), masks=[b"0"]),
    )


@pytest.mark.parametrize(
    "label_dtype, confidence_dtype",
    [
        (torch.int64, torch.float32),
        (torch.int32, torch.float16),
        (torch.uint8, torch.bfloat16),
    ],
)
def test_kind_accepts_native_label_and_confidence_dtypes(label_dtype, confidence_dtype):
    result = _result(label_dtype=label_dtype, confidence_dtype=confidence_dtype)

    KIND.check(result)

    assert KIND.deserialize(result) is result


@pytest.mark.parametrize(
    "field, value, match",
    [
        (
            "segmentation_map",
            torch.zeros((4, 6)),
            "segmentation_map dtype must be integer",
        ),
        ("segmentation_map", torch.zeros((1, 4, 6), dtype=torch.int64), "shape"),
        ("confidence", torch.zeros((4, 6), dtype=torch.int64), "confidence dtype"),
        ("confidence", torch.zeros((4, 5)), r"shape \(4, 6\)"),
        ("confidence", torch.zeros((4, 6), device="meta"), "is on meta"),
        ("image_metadata", ["not", "a", "dict"], "image_metadata"),
    ],
)
def test_kind_rejects_misaligned_or_mistyped_fields(field, value, match):
    result = _result()
    setattr(result, field, value)

    with pytest.raises(ContractError, match=match):
        KIND.check(result)


def test_old_per_class_rle_kind_is_unchanged_and_distinct():
    assert (
        SEMANTIC_SEGMENTATION_PREDICTION_KIND.name == "semantic_segmentation_prediction"
    )
    assert KIND.name == "native_semantic_segmentation"
    SEMANTIC_SEGMENTATION_PREDICTION_KIND.check(_rle_semantic())
    with pytest.raises(ContractError):
        SEMANTIC_SEGMENTATION_PREDICTION_KIND.check(_result())
    with pytest.raises(ContractError):
        KIND.check(_rle_semantic())
    assert KIND in NATIVE_KINDS
    assert len(NATIVE_KINDS) == 12


@pytest.mark.parametrize("confidence_dtype", [torch.float32, torch.float16])
def test_envelope_round_trip_keeps_both_fields_dtypes_and_frame(confidence_dtype):
    result = _result(label_dtype=torch.int32, confidence_dtype=confidence_dtype)

    wire = json.loads(json.dumps(KIND.serialize(result)))
    restored = KIND.deserialize(wire)

    assert torch.equal(restored.segmentation_map, result.segmentation_map)
    assert restored.segmentation_map.dtype == torch.int32
    assert torch.equal(restored.confidence, result.confidence)
    assert restored.confidence.dtype == confidence_dtype
    assert frame_of(restored) == frame_of(result)


def test_recording_round_trip_keeps_fields_dtypes_and_records_source_device():
    result = _result(label_dtype=torch.int16, confidence_dtype=torch.float16)
    blobs = MemoryBlobs()

    node = json.loads(
        json.dumps(REGISTRY.encoder(blobs, location="g.f").encode(result))
    )
    restored = REGISTRY.decoder(blobs, location="g.f").decode(node)

    assert node["codec"]["name"] == "v2/SemanticSegmentationResult@1"
    assert node["codec"]["value"]["device"] == "cpu"
    assert set(node["codec"]["value"]["fields"]) == {
        "segmentation_map",
        "confidence",
        "image_metadata",
    }
    assert restored.segmentation_map.dtype == torch.int16
    assert restored.confidence.dtype == torch.float16
    assert torch.equal(restored.segmentation_map, result.segmentation_map)
    assert torch.equal(restored.confidence, result.confidence)
    assert restored.image_metadata == result.image_metadata


def test_recording_rejects_a_corrupt_or_misaligned_semantic_node():
    blobs = MemoryBlobs()
    encoder = REGISTRY.encoder(blobs, location="g.f")
    node = encoder.encode(_result())
    misaligned = json.loads(json.dumps(node))
    misaligned["codec"]["value"]["fields"]["confidence"] = encoder.encode(
        torch.zeros((2, 3))
    )
    missing_device = json.loads(json.dumps(node))
    del missing_device["codec"]["value"]["device"]
    decoder = REGISTRY.decoder(blobs, location="g.f")

    with pytest.raises(RecordingCorruptError, match="invalid semantic result"):
        decoder.decode(misaligned)
    with pytest.raises(RecordingCorruptError, match="'device'"):
        decoder.decode(missing_device)


def test_root_output_passes_root_grids_and_refuses_resampled_grids():
    at_root = _result()
    resized = _result(image=_root().resize((2, 3), image_id="half"))
    resized.segmentation_map = resized.segmentation_map[:2, :3]
    resized.confidence = resized.confidence[:2, :3]
    bare = _result()
    bare.image_metadata = None

    assert KIND.convert_output(at_root, {}) is at_root
    assert KIND.convert_output(bare, {"coordinates_system": "root"}) is bare
    assert KIND.convert_output(resized, {"coordinates_system": "own"}) is resized
    with pytest.raises(ValueError, match="no lossless workflow-root form"):
        KIND.convert_output(resized, {"coordinates_system": "parent"})
