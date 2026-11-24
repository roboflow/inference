"""Root conversion and ``frame_of`` read typed, dual and legacy frames alike.

A complete typed record converts without legacy keys. Both entry points share
one reader of the legacy keys, so they accept the same numbers and differ only
in the documented completeness rule. Every case builds the same frame: a 29x47 root, cropped at x=7..27, y=3..13
(a 10x20 crop) and resized to 5x10. Local pixels per root pixel are 0.5 on
both axes, so ``root_xy = 2 * local_xy + (7, 3)``. The expected values below
are computed by hand from that equation, not by the code under test.
"""

import datetime

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.data.coordinates import (
    convert_prediction_output,
    prediction_to_root,
)
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import FRAME_RECORD_KEY, FrameRef, frame_of
from roboflow_workflows.v2.data.image import (
    CompositeSource,
    FrameMapping,
    ImageData,
    legacy_prediction_metadata,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)

VIDEO = VideoMetadata(
    video_identifier="camera-7",
    frame_number=12,
    frame_timestamp=datetime.datetime(
        2026, 10, 12, 9, 15, tzinfo=datetime.timezone.utc
    ),
    fps=30.0,
)
ROOT_HW = (29, 47)
LOCAL_BOX = [2.0, 1.0, 4.0, 3.0]
ROOT_BOX = [11.0, 5.0, 15.0, 9.0]


def _root(*, video_metadata=None):
    return ImageData.from_tensor(
        torch.zeros((3, *ROOT_HW), dtype=torch.uint8),
        image_id="root",
        video_metadata=video_metadata,
    )


def _small(*, video_metadata=None, size_hw=(5, 10)):
    crop = _root(video_metadata=video_metadata).crop((7, 3, 27, 13), image_id="crop")
    return crop.resize(size_hw, image_id="small")


def _typed_only(image):
    return {FRAME_RECORD_KEY: FrameRef.from_image(image).to_record()}


def _dual(image):
    return FrameRef.from_image(image).prediction_metadata()


def _legacy_only(image):
    return legacy_prediction_metadata(
        image_id=image.image_id,
        size_hw=image.size_hw,
        parent=image.parent,
        root=image.root,
        composite_sources=image.composite_sources,
    )


ALL_FORMATS = pytest.mark.parametrize(
    "metadata_of", [_typed_only, _dual, _legacy_only], ids=["typed", "dual", "legacy"]
)
TYPED_FORMATS = pytest.mark.parametrize(
    "metadata_of", [_typed_only, _dual], ids=["typed", "dual"]
)


def _boxes(metadata, xyxy=(LOCAL_BOX,)):
    return Detections(
        xyxy=torch.tensor(xyxy),
        class_id=torch.zeros(len(xyxy), dtype=torch.int64),
        confidence=torch.full((len(xyxy),), 0.9),
        image_metadata=metadata,
    )


def _expected_root_frame(video_metadata=None):
    identity = FrameMapping(frame_id="root", frame_size_hw=ROOT_HW)
    return FrameRef(
        image_id="root",
        size_hw=ROOT_HW,
        parent=identity,
        root=identity,
        video_metadata=video_metadata,
    )


@ALL_FORMATS
def test_crop_resize_boxes_reach_the_root_in_every_metadata_format(metadata_of):
    boxes = _boxes(metadata_of(_small()))

    converted = prediction_to_root(boxes)

    assert converted.xyxy.tolist() == [ROOT_BOX]
    assert boxes.xyxy.tolist() == [LOCAL_BOX]
    assert frame_of(converted) == _expected_root_frame()


@TYPED_FORMATS
def test_root_record_keeps_source_and_time_and_names_the_root_as_parent(metadata_of):
    small = _small(video_metadata=VIDEO)
    boxes = _boxes({**metadata_of(small), "user_key": "kept"})

    converted = prediction_to_root(boxes)

    record = converted.image_metadata[FRAME_RECORD_KEY]
    assert record == _expected_root_frame(VIDEO).to_record()
    assert record["parent"]["frame_id"] == "root"
    assert record["video_metadata"]["video_identifier"] == "camera-7"
    assert record["video_metadata"]["frame_number"] == 12
    assert converted.image_metadata["user_key"] == "kept"
    assert boxes.image_metadata[FRAME_RECORD_KEY]["image_id"] == "small"


def test_typed_only_input_stays_typed_only_after_root_conversion():
    converted = prediction_to_root(_boxes(_typed_only(_small())))

    assert set(converted.image_metadata) == {FRAME_RECORD_KEY}


def test_dual_input_rewrites_legacy_keys_to_agree_with_the_root_record():
    converted = prediction_to_root(_boxes(_dual(_small())))

    metadata = converted.image_metadata
    assert metadata["parent_id"] == "root"
    assert metadata["root_parent_coordinates"] == [0, 0]
    assert metadata["image_dimensions"] == list(ROOT_HW)
    assert frame_of(converted) == _expected_root_frame()


@ALL_FORMATS
def test_repeat_conversion_returns_the_root_prediction_unchanged(metadata_of):
    converted = prediction_to_root(_boxes(metadata_of(_small(video_metadata=VIDEO))))

    assert prediction_to_root(converted) is converted


@ALL_FORMATS
def test_prediction_on_the_root_itself_is_returned_unchanged(metadata_of):
    boxes = _boxes(metadata_of(_root(video_metadata=VIDEO)))

    assert prediction_to_root(boxes) is boxes
    assert convert_prediction_output(boxes, {"coordinates_system": "root"}) is boxes


@TYPED_FORMATS
def test_non_uniform_keypoints_and_covariance_reach_the_root(metadata_of):
    # Resized to 5x4: 0.2 local px per root px on x, 0.5 on y.
    small = _small(size_hw=(5, 4))
    key_points = KeyPoints(
        xy=torch.tensor([[[1.0, 1.0], [2.0, 3.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.8, 0.0]]),
        image_metadata=metadata_of(small),
        covariance=torch.ones((1, 2, 2, 2)),
    )

    converted = prediction_to_root(key_points)

    assert converted.xy.tolist() == [[[12.0, 5.0], [17.0, 9.0]]]
    # A C A^T with A = diag(5, 2).
    assert converted.covariance[0, 0].tolist() == [[25.0, 10.0], [10.0, 4.0]]
    assert frame_of(converted) == _expected_root_frame()


def test_typed_only_keypoint_tuple_converts_points_and_boxes_together():
    metadata = _typed_only(_small())
    key_points = KeyPoints(
        xy=torch.tensor([[[1.0, 2.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.8]]),
        image_metadata=metadata,
    )

    converted_points, converted_boxes = prediction_to_root(
        (key_points, _boxes(metadata))
    )

    assert converted_points.xy.tolist() == [[[9.0, 7.0]]]
    assert converted_boxes.xyxy.tolist() == [ROOT_BOX]
    assert frame_of((converted_points, converted_boxes)) == _expected_root_frame()


def test_typed_only_detection_row_converts_box_and_metadata():
    metadata = _typed_only(_small(video_metadata=VIDEO))
    row = (torch.tensor(LOCAL_BOX), None, 0, 0.9, 5, {"name": "a"}, metadata)

    converted = prediction_to_root(row)

    assert converted[0].tolist() == ROOT_BOX
    assert converted[2:5] == (0, 0.9, 5)
    assert converted[6][FRAME_RECORD_KEY] == _expected_root_frame(VIDEO).to_record()


def test_typed_only_dense_mask_is_resampled_onto_the_root_canvas():
    local_mask = torch.zeros((1, 5, 10), dtype=torch.bool)
    local_mask[0, 1, 2] = True
    instances = InstanceDetections(
        xyxy=torch.tensor([LOCAL_BOX]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=local_mask,
        image_metadata=_typed_only(_small()),
    )

    converted = prediction_to_root(instances)

    # Root pixel (X, Y) samples local (floor((X + .5 - 7) / 2), floor((Y + .5 - 3) / 2)),
    # so local (x=2, y=1) covers root x in {11, 12} and y in {5, 6}.
    expected = torch.zeros((1, *ROOT_HW), dtype=torch.bool)
    expected[0, 5:7, 11:13] = True
    assert torch.equal(converted.mask, expected)


def test_typed_only_mask_canvas_must_match_the_record_size():
    instances = InstanceDetections(
        xyxy=torch.tensor([LOCAL_BOX]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=torch.zeros((1, 6, 10), dtype=torch.bool),
        image_metadata=_typed_only(_small()),
    )

    with pytest.raises(ValueError, match="Mask canvas"):
        prediction_to_root(instances)


@ALL_FORMATS
def test_transformed_semantic_grid_refuses_root_coordinates(metadata_of):
    grid = SemanticSegmentationResult(
        segmentation_map=torch.zeros((5, 10), dtype=torch.int64),
        confidence=torch.ones((5, 10)),
        image_metadata=metadata_of(_small()),
    )

    with pytest.raises(ValueError, match="no lossless"):
        prediction_to_root(grid)


@TYPED_FORMATS
def test_semantic_grid_on_the_root_converts_unchanged(metadata_of):
    grid = SemanticSegmentationResult(
        segmentation_map=torch.zeros(ROOT_HW, dtype=torch.int64),
        confidence=torch.ones(ROOT_HW),
        image_metadata=metadata_of(_root()),
    )

    assert prediction_to_root(grid) is grid


@TYPED_FORMATS
@pytest.mark.parametrize("crop_canvas", [False, True], ids=["canvas", "canvas-crop"])
def test_composite_frame_refuses_root_coordinates(metadata_of, crop_canvas):
    canvas = ImageData.composite(
        torch.zeros((3, 20, 40), dtype=torch.uint8),
        sources=[
            CompositeSource.place(_root(), index=(0,), canvas_xyxy=(0, 0, 20, 20))
        ],
        image_id="mosaic",
    )
    image = (
        canvas.crop((2, 2, 12, 12), image_id="mosaic-crop") if crop_canvas else canvas
    )
    boxes = _boxes(metadata_of(image))

    with pytest.raises(ValueError, match="composite"):
        prediction_to_root(boxes)
    assert convert_prediction_output(boxes, {"coordinates_system": "own"}) is boxes


def test_typed_record_that_conflicts_with_a_legacy_key_is_rejected():
    metadata = {**_dual(_small()), "root_parent_coordinates": [8, 3]}

    with pytest.raises(ContractError, match="Conflicting frame"):
        prediction_to_root(_boxes(metadata))


def test_partial_legacy_keys_must_agree_with_the_typed_record():
    agreeing = {**_typed_only(_small()), "parent_id": "small"}
    conflicting = {**_typed_only(_small()), "parent_id": "other"}

    assert prediction_to_root(_boxes(agreeing)).xyxy.tolist() == [ROOT_BOX]
    with pytest.raises(ContractError, match="Conflicting frame"):
        prediction_to_root(_boxes(conflicting))


def _without(metadata, *keys):
    return {key: value for key, value in metadata.items() if key not in keys}


@pytest.mark.parametrize(
    "scaling",
    [np.array([0.5, 0.5]), np.float32(0.5), np.float64(0.5), [0.5, 0.5]],
    ids=["numpy-pair", "float32", "float64", "list"],
)
def test_both_readers_accept_the_same_legacy_numbers(scaling):
    metadata = {
        **_legacy_only(_small()),
        "scaling_relative_to_root_parent": scaling,
        "scaling_relative_to_parent": scaling,
        "root_parent_coordinates": np.array([7, 3]),
    }
    boxes = _boxes(metadata)

    assert frame_of(boxes).root.scale_xy == (2.0, 2.0)
    assert frame_of(boxes).root.offset_xy == (7.0, 3.0)
    assert prediction_to_root(boxes).xyxy.tolist() == [ROOT_BOX]


def test_numpy_legacy_numbers_beside_a_record_agree_with_it():
    metadata = {
        **_dual(_small()),
        "scaling_relative_to_root_parent": np.float32(0.5),
        "root_parent_coordinates": np.array([7.0, 3.0]),
    }

    assert frame_of(_boxes(metadata)) == FrameRef.from_image(_small())
    assert prediction_to_root(_boxes(metadata)).xyxy.tolist() == [ROOT_BOX]


@pytest.mark.parametrize(
    "scaling, message",
    [
        (True, "real numbers"),
        (0, "positive"),
        (float("inf"), "finite"),
        ([0.5, 0.5, 0.5], "pair"),
    ],
)
def test_both_readers_reject_the_same_malformed_scale(scaling, message):
    boxes = _boxes(
        {
            **_legacy_only(_small()),
            "scaling_relative_to_root_parent": scaling,
        }
    )

    with pytest.raises(ContractError, match=message):
        frame_of(boxes)
    with pytest.raises(ContractError, match=message):
        prediction_to_root(boxes)


@pytest.mark.parametrize(
    "dropped",
    [
        ("image_dimensions",),
        ("parent_dimensions", "parent_coordinates", "scaling_relative_to_parent"),
    ],
    ids=["no-local-size", "no-parent-keys"],
)
def test_root_conversion_needs_only_the_root_keys_but_frame_of_needs_all(dropped):
    boxes = _boxes(_without(_legacy_only(_small()), *dropped))

    assert prediction_to_root(boxes).xyxy.tolist() == [ROOT_BOX]
    with pytest.raises(ContractError, match="lacks frame geometry"):
        frame_of(boxes)


def test_root_conversion_needs_the_root_id_key_but_frame_of_does_not():
    boxes = _boxes(_without(_legacy_only(_small()), "root_parent_id"))

    assert frame_of(boxes).root.frame_id is None
    with pytest.raises(ContractError, match="'root_parent_id' is missing"):
        prediction_to_root(boxes)


def test_author_metadata_from_image_keeps_source_and_time_through_frame_of():
    # Written the way example 03 attaches metadata to a model result.
    small = _small(video_metadata=VIDEO)
    boxes = _boxes({**small.prediction_metadata(), "class_names": {0: "marker"}})

    frame = frame_of(boxes)

    assert frame.video_metadata == VIDEO
    assert frame == FrameRef.from_image(small)
    assert small.prediction_metadata() == _dual(small)
    assert prediction_to_root(boxes).image_metadata[FRAME_RECORD_KEY] == (
        _expected_root_frame(VIDEO).to_record()
    )


def test_explicit_legacy_metadata_has_no_source_and_time():
    metadata = _legacy_only(_small(video_metadata=VIDEO))

    assert FRAME_RECORD_KEY not in metadata
    assert frame_of(_boxes(metadata)).video_metadata is None


def test_both_readers_accept_numpy_and_integral_float_sizes():
    metadata = {
        **_legacy_only(_small()),
        "image_dimensions": np.array([5, 10]),
        "root_parent_dimensions": [29.0, 47.0],
        "parent_dimensions": (np.int64(10), np.int64(20)),
    }
    boxes = _boxes(metadata)

    frame = frame_of(boxes)

    assert (frame.size_hw, frame.root.frame_size_hw) == ((5, 10), ROOT_HW)
    assert frame.parent.frame_size_hw == (10, 20)
    assert prediction_to_root(boxes).xyxy.tolist() == [ROOT_BOX]


def test_explicit_none_scale_means_unit_scale_in_both_readers():
    # Crop without resize, so the true scale is 1 on both axes.
    crop = _root().crop((7, 3, 27, 13), image_id="crop")
    metadata = {
        **_legacy_only(crop),
        "scaling_relative_to_root_parent": None,
        "scaling_relative_to_parent": None,
    }
    boxes = _boxes(metadata)

    assert frame_of(boxes).root.scale_xy == (1.0, 1.0)
    assert prediction_to_root(boxes).xyxy.tolist() == [[9.0, 4.0, 11.0, 6.0]]


def test_explicit_none_root_id_is_an_unknown_root_in_both_readers():
    boxes = _boxes({**_legacy_only(_small()), "root_parent_id": None})

    converted = prediction_to_root(boxes)

    assert frame_of(boxes).root.frame_id is None
    assert converted.xyxy.tolist() == [ROOT_BOX]
    assert converted.image_metadata["parent_id"] is None


@pytest.mark.parametrize("root_id", [5, ""], ids=["int", "empty"])
def test_both_readers_reject_a_root_id_that_is_not_a_name(root_id):
    boxes = _boxes({**_legacy_only(_small()), "root_parent_id": root_id})

    with pytest.raises(ValueError, match="frame_id"):
        frame_of(boxes)
    with pytest.raises(ValueError, match="frame_id"):
        prediction_to_root(boxes)


def test_null_record_means_no_record_in_both_readers():
    legacy = {**_legacy_only(_small()), FRAME_RECORD_KEY: None}
    bare = _boxes({FRAME_RECORD_KEY: None, "class_names": {}})

    assert frame_of(_boxes(legacy)) == frame_of(_boxes(_legacy_only(_small())))
    assert prediction_to_root(_boxes(legacy)).xyxy.tolist() == [ROOT_BOX]
    assert prediction_to_root(bare) is bare
    with pytest.raises(ContractError, match="lacks frame geometry"):
        frame_of(bare)


def test_explicit_none_image_id_on_the_root_mapping_is_already_root():
    boxes = _boxes({**_legacy_only(_root()), "parent_id": None})

    assert prediction_to_root(boxes) is boxes
