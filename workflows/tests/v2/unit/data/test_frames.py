"""Typed frames of native predictions: geometry, provenance and conflicts."""

import datetime
import json
import math

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.data.coordinates import prediction_to_root
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import (
    FRAME_RECORD_KEY,
    FrameRef,
    frame_of,
)
from roboflow_workflows.v2.data.image import (
    CompositeSource,
    ImageData,
    legacy_prediction_metadata,
)
from roboflow_workflows.v2.data.payload_codec import decode_native, encode_native
from roboflow_workflows.v2.engine.recording import CodecRegistry
from roboflow_workflows.v2.recording_codecs import NATIVE_RECORDING_CODECS

from inference_models.models.base.classification import ClassificationPrediction
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from tests.v2.support.recording.codecs import MemoryBlobs

VIDEO = VideoMetadata(
    video_identifier="camera-3",
    frame_number=41,
    frame_timestamp=datetime.datetime(
        2026, 10, 12, 8, 30, tzinfo=datetime.timezone.utc
    ),
    fps=25.0,
)


def _root(*, video_metadata=None):
    return ImageData.from_tensor(
        torch.zeros((3, 29, 47), dtype=torch.uint8),
        image_id="root",
        video_metadata=video_metadata,
    )


def _crop_then_resize(root):
    # Root 47x29; crop x 7..27, y 3..13 (20x10); resize to 10x5: scale 2.
    crop = root.crop((7, 3, 27, 13), image_id="crop")
    resized = crop.resize((5, 10), image_id="small")

    return resized


def _legacy_metadata(image):
    # The V1 keys alone, without the typed record.
    return legacy_prediction_metadata(
        image_id=image.image_id,
        size_hw=image.size_hw,
        parent=image.parent,
        root=image.root,
        composite_sources=image.composite_sources,
    )


def _boxes(image_metadata):
    return Detections(
        xyxy=torch.tensor([[2.0, 1.0, 4.0, 3.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.5]),
        image_metadata=image_metadata,
    )


def test_crop_then_resize_maps_local_point_to_root_and_parent():
    frame = FrameRef.from_image(_crop_then_resize(_root()))

    assert frame.root.map_xy(2, 1) == (11.0, 5.0)
    assert frame.parent.map_xy(2, 1) == (4.0, 2.0)
    assert frame.image_id == "small"
    assert frame.size_hw == (5, 10)
    assert frame.parent.frame_id == "crop"
    assert frame.parent.frame_size_hw == (10, 20)
    assert frame.root.frame_id == "root"
    assert frame.root.frame_size_hw == (29, 47)
    assert frame.video_metadata is None
    assert not frame.is_composite


def test_attached_frame_survives_on_the_prediction_and_keeps_legacy_keys():
    image = _crop_then_resize(_root())
    frame = FrameRef.from_image(image)

    metadata = frame.prediction_metadata()

    assert {k: v for k, v in metadata.items() if k != FRAME_RECORD_KEY} == (
        _legacy_metadata(image)
    )
    assert image.prediction_metadata() == metadata
    assert metadata["scaling_relative_to_root_parent"] == 0.5
    assert frame_of(_boxes(metadata)) == frame


def test_legacy_keys_alone_rebuild_the_frame_inverting_scale_once():
    image = _crop_then_resize(_root())

    frame = frame_of(_boxes(_legacy_metadata(image)))

    assert frame == FrameRef.from_image(image)
    assert frame.root.scale_xy == (2.0, 2.0)
    assert frame.root.map_xy(2, 1) == (11.0, 5.0)


def test_odd_aspect_ratio_keeps_per_axis_scales():
    crop = _root().crop((7, 3, 27, 13), image_id="crop")
    odd = crop.resize((3, 7), image_id="odd")
    frame = FrameRef.from_image(odd)

    from_record = frame_of(_boxes(frame.prediction_metadata()))
    from_legacy = frame_of(_boxes(_legacy_metadata(odd)))

    assert from_record == frame
    assert frame.root.scale_xy == (20 / 7, 10 / 3)
    assert all(
        math.isclose(a, e, rel_tol=1e-12)
        for a, e in zip(from_legacy.root.scale_xy, frame.root.scale_xy)
    )
    assert _legacy_metadata(odd)["scaling_relative_to_root_parent"] == [
        7 / 20,
        3 / 10,
    ]


def test_source_and_time_survive_only_where_known():
    image = _crop_then_resize(_root(video_metadata=VIDEO))
    frame = FrameRef.from_image(image)

    recorded = frame_of(_boxes(json.loads(json.dumps(frame.prediction_metadata()))))
    legacy_only = frame_of(_boxes(_legacy_metadata(image)))
    from_image = frame_of(_boxes(image.prediction_metadata()))

    assert recorded.video_metadata == VIDEO
    assert recorded.video_metadata.frame_timestamp == VIDEO.frame_timestamp
    assert legacy_only.video_metadata is None
    assert from_image.video_metadata == VIDEO


@pytest.mark.parametrize(
    "key, value",
    [
        ("root_parent_coordinates", [7.0, 4.0]),
        ("scaling_relative_to_root_parent", 0.25),
        ("root_parent_id", "other-root"),
        ("parent_id", "other-image"),
        ("image_dimensions", [5, 11]),
        ("is_composite", True),
    ],
)
def test_record_and_legacy_keys_that_disagree_are_rejected(key, value):
    metadata = FrameRef.from_image(_crop_then_resize(_root())).prediction_metadata()
    metadata[key] = value
    prediction = _boxes(metadata)

    with pytest.raises(ContractError, match="Conflicting frame"):
        frame_of(prediction)
    with pytest.raises(ValueError, match="Conflicting frame"):
        prediction_to_root(prediction)


def test_record_alone_is_accepted_without_legacy_keys():
    frame = FrameRef.from_image(_crop_then_resize(_root()))

    assert frame_of(_boxes({FRAME_RECORD_KEY: frame.to_record()})) == frame


@pytest.mark.parametrize(
    "metadata, match",
    [
        (None, "has no image_metadata"),
        ({"parent_id": "x"}, "lacks frame geometry"),
        ({FRAME_RECORD_KEY: {"version": 2}}, "version must be 1"),
        ({FRAME_RECORD_KEY: {"version": True}}, "version must be 1"),
        ({FRAME_RECORD_KEY: []}, "must be a dict"),
    ],
)
def test_missing_or_malformed_geometry_fails_usefully(metadata, match):
    with pytest.raises(ContractError, match=match):
        frame_of(_boxes(metadata))


def test_malformed_record_values_and_keys_fail_usefully():
    record = FrameRef.from_image(_root()).to_record()
    extra = {**record, "guess": 1}
    bad_size = {**record, "size_hw": [0, 4]}

    with pytest.raises(ContractError, match="unexpected \\['guess'\\]"):
        FrameRef.from_record(extra)
    with pytest.raises(ContractError, match="Invalid frame record"):
        FrameRef.from_record(bad_size)
    with pytest.raises(ContractError, match="legacy scaling"):
        frame_of(_boxes({**_legacy_metadata(_root()), "scaling_relative_to_parent": 0}))


def test_unknown_imported_ids_stay_unknown():
    metadata = _legacy_metadata(_crop_then_resize(_root()))
    for key in ("parent_id", "parent_frame_id", "root_parent_id"):
        del metadata[key]

    frame = frame_of(_boxes(metadata))

    assert frame.image_id is None
    assert frame.parent.frame_id is None
    assert frame.root.frame_id is None
    assert frame.root.map_xy(2, 1) == (11.0, 5.0)


def test_composite_root_survives_and_refuses_single_source_restoration():
    tile = _root()
    canvas = ImageData.composite(
        torch.zeros((3, 20, 40), dtype=torch.uint8),
        sources=[CompositeSource.place(tile, index=(0,), canvas_xyxy=(0, 0, 20, 20))],
        image_id="mosaic",
    )
    crop = canvas.crop((2, 2, 12, 12), image_id="mosaic-crop")
    frame = FrameRef.from_image(crop)
    prediction = _boxes(frame.prediction_metadata())

    restored = frame_of(prediction)

    assert restored.is_composite
    assert restored == frame
    assert restored.composite_sources[0].image_id == "root"
    assert restored.root.frame_id == "mosaic"
    with pytest.raises(ValueError, match="composite"):
        prediction_to_root(prediction)


def test_root_conversion_moves_the_record_with_the_legacy_keys():
    image = _crop_then_resize(_root(video_metadata=VIDEO))
    prediction = _boxes(FrameRef.from_image(image).prediction_metadata())

    converted = prediction_to_root(prediction)
    frame = frame_of(converted)

    assert converted.xyxy.tolist() == [[11.0, 5.0, 15.0, 9.0]]
    assert frame.image_id == "root"
    assert frame.size_hw == (29, 47)
    assert frame.root.map_xy(11, 5) == (11.0, 5.0)
    assert frame.video_metadata == VIDEO
    assert prediction.image_metadata[FRAME_RECORD_KEY]["image_id"] == "small"


def test_record_survives_the_native_envelope_and_recording_codecs():
    prediction = _boxes(
        FrameRef.from_image(
            _crop_then_resize(_root(video_metadata=VIDEO))
        ).prediction_metadata()
    )
    registry = CodecRegistry(NATIVE_RECORDING_CODECS)
    blobs = MemoryBlobs()
    node = json.loads(
        json.dumps(registry.encoder(blobs, location="g.f").encode(prediction))
    )

    from_envelope = decode_native(json.loads(json.dumps(encode_native(prediction))))
    from_recording = registry.decoder(blobs, location="g.f").decode(node)

    assert frame_of(from_envelope) == frame_of(prediction)
    assert frame_of(from_recording) == frame_of(prediction)


def test_frames_read_no_tensor_values():
    meta = torch.device("meta")
    image_metadata = FrameRef.from_image(
        _crop_then_resize(_root())
    ).prediction_metadata()
    prediction = Detections(
        xyxy=torch.empty((2, 4), device=meta),
        class_id=torch.empty(2, dtype=torch.int64, device=meta),
        confidence=torch.empty(2, device=meta),
        image_metadata=image_metadata,
    )

    assert frame_of(prediction).image_id == "small"


def test_other_native_families_expose_their_frame():
    image = _crop_then_resize(_root())
    metadata = FrameRef.from_image(image).prediction_metadata()
    points = KeyPoints(
        xy=torch.zeros((1, 2, 2)),
        class_id=torch.tensor([0]),
        confidence=torch.ones((1, 2)),
        image_metadata=metadata,
    )
    classification = ClassificationPrediction(
        class_id=torch.tensor([1]),
        confidence=torch.tensor([[0.2, 0.8]]),
        images_metadata=[metadata],
    )
    other = _boxes(FrameRef.from_image(_root()).prediction_metadata())

    assert frame_of(points) == frame_of((points, _boxes(metadata)))
    assert frame_of(classification).image_id == "small"
    with pytest.raises(ContractError, match="different frames"):
        frame_of((points, other))
    with pytest.raises(ContractError, match="does not support"):
        frame_of(torch.zeros(3))
