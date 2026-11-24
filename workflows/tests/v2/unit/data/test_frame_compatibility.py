"""Whether a prediction's metadata may describe the image a block holds."""

import copy
import dataclasses
import datetime
import json

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import (
    FRAME_RECORD_KEY,
    FrameRef,
    frame_from_metadata,
    require_compatible_frame,
)
from roboflow_workflows.v2.data.image import (
    CompositeSource,
    ImageData,
    legacy_prediction_metadata,
)

FRAME_1 = VideoMetadata(
    video_identifier="camera-3",
    frame_number=1,
    frame_timestamp=datetime.datetime(
        2026, 10, 12, 8, 30, 1, tzinfo=datetime.timezone.utc
    ),
    fps=25.0,
)
FRAME_0 = FRAME_1.model_copy(
    update={
        "frame_number": 0,
        "frame_timestamp": datetime.datetime(
            2026, 10, 12, 8, 30, tzinfo=datetime.timezone.utc
        ),
    }
)


LEGACY_FRAME_KEYS = [
    "parent_id",
    "parent_frame_id",
    "root_parent_id",
    "image_dimensions",
    "parent_dimensions",
    "root_parent_dimensions",
    "parent_coordinates",
    "root_parent_coordinates",
    "scaling_relative_to_parent",
    "scaling_relative_to_root_parent",
]


def _root(image_id="root", *, video_metadata=None):
    return ImageData.from_tensor(
        torch.zeros((3, 29, 47), dtype=torch.uint8),
        image_id=image_id,
        video_metadata=video_metadata,
    )


def _small(root):
    # Crop 20x10 at (7, 3), then resize to an odd 7x3: per-axis scales.
    return root.crop((7, 3, 27, 13), image_id="crop").resize((3, 7), image_id="small")


def _legacy_metadata(image):
    return legacy_prediction_metadata(
        image_id=image.image_id,
        size_hw=image.size_hw,
        parent=image.parent,
        root=image.root,
        composite_sources=image.composite_sources,
    )


def _mosaic():
    # Two 47x29 tiles side by side on a 40x10 canvas.
    tiles = [_root("left"), _root("right")]
    sources = [
        CompositeSource.place(tile, index=[i], canvas_xyxy=(20 * i, 0, 20 * i + 20, 10))
        for i, tile in enumerate(tiles)
    ]

    return ImageData.composite(
        torch.zeros((3, 10, 40), dtype=torch.uint8), sources=sources, image_id="mosaic"
    )


# --- healthy and unknown provenance -------------------------------------------


@pytest.mark.parametrize("describe", ["typed", "legacy", "json"])
def test_metadata_of_the_same_image_is_compatible(describe):
    image = _small(_root(video_metadata=FRAME_1))
    metadata = {
        "typed": image.prediction_metadata(),
        "legacy": _legacy_metadata(image),
        "json": json.loads(json.dumps(image.prediction_metadata())),
    }[describe]

    require_compatible_frame(metadata, image=image)


@pytest.mark.parametrize("describe", ["typed", "legacy"])
def test_composite_metadata_of_the_same_canvas_is_compatible(describe):
    mosaic = _mosaic()
    metadata = (
        mosaic.prediction_metadata()
        if describe == "typed"
        else _legacy_metadata(mosaic)
    )

    require_compatible_frame(metadata, image=mosaic)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"class_names": {0: "car"}, "prediction_type": "object-detection"},
        {FRAME_RECORD_KEY: None, "parent_id": None, "image_dimensions": None},
    ],
    ids=["none", "empty", "unrelated-keys", "explicit-nones"],
)
def test_absent_provenance_is_unknown_not_a_conflict(metadata):
    require_compatible_frame(metadata, image=_small(_root()))


@pytest.mark.parametrize(
    "key, value",
    [
        ("image_dimensions", [3, 7]),
        ("image_dimensions", np.array([3, 7])),
        ("parent_id", "small"),
        ("parent_coordinates", [0, 0]),
        ("root_parent_coordinates", (np.float64(7.0), np.int64(3))),
        ("scaling_relative_to_root_parent", [7 / 20, 3 / 10]),
        ("is_composite", np.bool_(False)),
    ],
)
def test_valid_partial_legacy_field_that_agrees_is_compatible(key, value):
    # The image is a 7x3 resize of a crop at (7, 3) of the 47x29 root.
    require_compatible_frame({key: value}, image=_small(_root()))


def test_missing_legacy_scale_is_unknown_not_one():
    image = _small(_root())
    metadata = _legacy_metadata(image)
    del metadata["scaling_relative_to_parent"]
    del metadata["scaling_relative_to_root_parent"]

    require_compatible_frame(metadata, image=image)


def test_record_without_image_id_or_video_is_compatible():
    image = _small(_root(video_metadata=FRAME_1))
    anonymous = dataclasses.replace(
        FrameRef.from_image(image), image_id=None, video_metadata=None
    )

    require_compatible_frame(anonymous.prediction_metadata(), image=image)


@pytest.mark.parametrize("key", LEGACY_FRAME_KEYS)
def test_none_mirror_next_to_a_record_is_unknown(key):
    # As if the mirror were absent; the full-frame reader still rejects it.
    image = _small(_root())
    metadata = {**image.prediction_metadata(), key: None}

    require_compatible_frame(metadata, image=image)
    with pytest.raises(ContractError, match="Conflicting frame"):
        frame_from_metadata(metadata)


def test_none_mirror_does_not_hide_a_known_difference():
    image = _small(_root())
    other_root = {
        **_small(_root("other-root")).prediction_metadata(),
        "parent_id": None,
    }
    contradicted = {
        **image.prediction_metadata(),
        "parent_id": None,
        "root_parent_id": "other-root",
    }

    with pytest.raises(ContractError, match="Frame mismatch"):
        require_compatible_frame(other_root, image=image)
    with pytest.raises(ContractError, match="Conflicting frame"):
        require_compatible_frame(contradicted, image=image)


def test_image_without_video_or_parent_id_accepts_known_metadata():
    # A V1 crop names no parent frame and may have no video metadata.
    image = _small(_root(video_metadata=FRAME_1))
    metadata = image.prediction_metadata()
    v1_like = dataclasses.replace(
        image,
        parent=dataclasses.replace(image.parent, frame_id=None),
        video_metadata=None,
    )

    require_compatible_frame(metadata, image=v1_like)


def test_legacy_without_composite_markers_leaves_composite_unknown():
    mosaic = _mosaic()
    metadata = _legacy_metadata(mosaic)
    del metadata["is_composite"]
    del metadata["composite_sources"]

    require_compatible_frame(metadata, image=mosaic)


def test_pixels_are_not_read():
    # Meta tensors have shapes but no data: reading values would raise.
    image = ImageData.from_tensor(
        torch.empty((3, 29, 47), dtype=torch.uint8, device="meta"), image_id="root"
    )

    require_compatible_frame(image.prediction_metadata(), image=image)
    require_compatible_frame(_legacy_metadata(image), image=image)


# --- malformed supplied provenance ---------------------------------------------


@pytest.mark.parametrize(
    "valid, malformed",
    [
        ({"image_dimensions": [3, 7]}, {"image_dimensions": "not-a-size"}),
        ({"parent_id": "small"}, {"parent_id": 123}),
        ({"parent_coordinates": [0, 0]}, {"parent_coordinates": "bad"}),
    ],
    ids=["size", "id", "parent-offset"],
)
def test_malformed_partial_legacy_field_differs_from_valid_one(valid, malformed):
    # The C004-01 pairs that frame_of and root placement cannot tell apart.
    image = _small(_root())

    require_compatible_frame(valid, image=image)
    with pytest.raises(ContractError):
        require_compatible_frame(malformed, image=image)


@pytest.mark.parametrize(
    "metadata",
    [
        {"image_dimensions": [3, 7.5]},
        {"parent_dimensions": [0, 20]},
        {"root_parent_coordinates": [float("nan"), 3]},
        {"scaling_relative_to_parent": -1},
        {"scaling_relative_to_root_parent": True},
        {"parent_frame_id": ""},
        {"root_parent_id": 7},
        {"is_composite": "yes"},
        {"is_composite": False, "composite_sources": []},
        {"composite_sources": "tiles"},
        {"composite_sources": [{"index": [0]}]},
    ],
)
def test_malformed_partial_legacy_field_is_rejected(metadata):
    with pytest.raises(ContractError):
        require_compatible_frame(metadata, image=_small(_root()))


@pytest.mark.parametrize(
    "record",
    ["frame", {"version": 2}, {"version": 1, "image_id": "small"}],
)
def test_malformed_record_is_rejected(record):
    with pytest.raises(ContractError):
        require_compatible_frame({FRAME_RECORD_KEY: record}, image=_small(_root()))


def test_metadata_that_is_not_a_dict_is_rejected():
    with pytest.raises(ContractError):
        require_compatible_frame([("parent_id", "small")], image=_small(_root()))


@pytest.mark.parametrize(
    "key, value",
    [
        ("parent_id", "other-camera:0"),
        ("root_parent_id", "other-root"),
        ("root_parent_coordinates", [0.0, 0.0]),
        ("is_composite", True),
    ],
)
def test_record_and_contradicting_legacy_mirror_are_rejected(key, value):
    image = _small(_root())
    metadata = {**image.prediction_metadata(), key: value}

    with pytest.raises(ContractError, match="Conflicting frame"):
        require_compatible_frame(metadata, image=image)


def _tile_member_as_float(*path):
    # Placements whose first tile has an integer written as a float: 0 -> 0.0.
    def make(metadata):
        sources = copy.deepcopy(metadata["composite_sources"])
        *parents, last = path
        member = sources[0]
        for key in parents:
            member = member[key]
        member[last] = [float(member[last][0]), *member[last][1:]]
        return sources

    return make


MALFORMED_MIRRORS = [
    pytest.param("parent_id", lambda metadata: 123, id="id"),
    pytest.param("root_parent_id", lambda metadata: "", id="root-id"),
    pytest.param(
        "image_dimensions", lambda metadata: [3.0000000001, 7], id="fractional-size"
    ),
    pytest.param(
        "parent_dimensions",
        lambda metadata: [10.0000000001, 20],
        id="fractional-parent-size",
    ),
    pytest.param(
        "root_parent_dimensions",
        lambda metadata: [29.0000000001, 47],
        id="fractional-root-size",
    ),
    pytest.param(
        "parent_coordinates", lambda metadata: [float("nan"), 0], id="nan-offset"
    ),
    pytest.param(
        "scaling_relative_to_root_parent", lambda metadata: True, id="bool-scale"
    ),
    pytest.param("is_composite", lambda metadata: "yes", id="text-flag"),
    pytest.param(
        "composite_sources", _tile_member_as_float("index"), id="float-tile-index"
    ),
    pytest.param(
        "composite_sources",
        _tile_member_as_float("canvas_xyxy"),
        id="float-tile-canvas",
    ),
    pytest.param(
        "composite_sources",
        _tile_member_as_float("parent", "size_hw"),
        id="float-tile-parent-size",
    ),
    pytest.param(
        "composite_sources",
        _tile_member_as_float("root", "size_hw"),
        id="float-tile-root-size",
    ),
]


@pytest.mark.parametrize("key, malformed", MALFORMED_MIRRORS)
@pytest.mark.parametrize("describe", ["typed", "legacy"])
def test_malformed_mirror_is_rejected_with_or_without_a_record(
    describe, key, malformed
):
    # A record that matches the image must not hide a malformed mirror.
    image = _mosaic() if key == "composite_sources" else _small(_root())
    metadata = image.prediction_metadata()
    supplied = metadata if describe == "typed" else {}
    supplied[key] = malformed(metadata)

    with pytest.raises(ContractError, match="Invalid frame metadata"):
        require_compatible_frame(supplied, image=image)


@pytest.mark.parametrize("describe", ["typed", "legacy"])
def test_adjacent_large_size_mirror_is_rejected(describe):
    # Sizes are discrete: no relative tolerance at any magnitude.
    image = _small(_root())
    image = dataclasses.replace(
        image,
        parent=dataclasses.replace(image.parent, frame_size_hw=(1_000_000_000, 20)),
    )
    supplied = image.prediction_metadata() if describe == "typed" else {}
    metadata = {**supplied, "parent_dimensions": [1_000_000_001, 20]}

    with pytest.raises(ContractError, match="parent.size_hw"):
        require_compatible_frame(metadata, image=image)


@pytest.mark.parametrize("describe", ["typed", "legacy"])
def test_integral_float_or_numpy_legacy_sizes_are_valid(describe):
    # Legacy sizes are numbers with integer values; only typed sizes are ints.
    image = _small(_root())
    supplied = image.prediction_metadata() if describe == "typed" else {}
    metadata = {
        **supplied,
        "image_dimensions": [3.0, 7.0],
        "parent_dimensions": np.array([10, 20]),
        "root_parent_dimensions": (np.float32(29), 47),
    }

    require_compatible_frame(metadata, image=image)


ZERO_D_KEYS = ["image_dimensions", "parent_coordinates", "scaling_relative_to_parent"]


@pytest.mark.parametrize("key", ZERO_D_KEYS)
@pytest.mark.parametrize("describe", ["typed", "legacy"])
def test_zero_dimensional_array_field_is_rejected(describe, key):
    # A 0-d numpy array has no length; it must not escape as TypeError.
    image = _small(_root())
    supplied = image.prediction_metadata() if describe == "typed" else {}
    metadata = {**supplied, key: np.array(3)}

    with pytest.raises(ContractError):
        require_compatible_frame(metadata, image=image)


@pytest.mark.parametrize("key", ZERO_D_KEYS)
def test_frame_reader_rejects_zero_dimensional_array_mirror(key):
    metadata = {**_small(_root()).prediction_metadata(), key: np.array(3)}

    with pytest.raises(ContractError, match="Conflicting frame"):
        frame_from_metadata(metadata)


# --- known different frames ----------------------------------------------------


@pytest.mark.parametrize("describe", ["typed", "legacy"])
def test_other_image_of_the_same_size_is_rejected(describe):
    target = _root("target:1")
    other = _root("other-camera:0")
    metadata = (
        other.prediction_metadata() if describe == "typed" else _legacy_metadata(other)
    )

    with pytest.raises(ContractError, match="image_id"):
        require_compatible_frame(metadata, image=target)


@pytest.mark.parametrize("describe", ["typed", "legacy"])
def test_same_id_and_size_at_another_crop_position_is_rejected(describe):
    root = _root()
    predicted_on = root.crop((0, 0, 10, 5), image_id="crop")
    target = root.crop((5, 0, 15, 5), image_id="crop")
    metadata = (
        predicted_on.prediction_metadata()
        if describe == "typed"
        else _legacy_metadata(predicted_on)
    )

    with pytest.raises(ContractError, match="parent.offset_xy"):
        require_compatible_frame(metadata, image=target)


def test_same_id_and_size_at_another_scale_is_rejected():
    root = _root()
    predicted_on = root.resize((10, 20), image_id="small")
    target = root.crop((0, 0, 20, 10), image_id="small")

    with pytest.raises(ContractError, match="parent.scale_xy"):
        require_compatible_frame(_legacy_metadata(predicted_on), image=target)


@pytest.mark.parametrize(
    "prompt_video, field",
    [
        (FRAME_0, "frame_number"),
        (
            FRAME_1.model_copy(update={"video_identifier": "camera-4"}),
            "video_identifier",
        ),
        (
            FRAME_1.model_copy(
                update={
                    "frame_timestamp": FRAME_1.frame_timestamp
                    + datetime.timedelta(microseconds=1)
                }
            ),
            "frame_timestamp",
        ),
    ],
)
def test_same_id_and_size_from_another_source_or_time_is_rejected(prompt_video, field):
    # Caller-supplied image ids can repeat across video frames.
    prompt_image = _root("frame", video_metadata=prompt_video)
    current = _root("frame", video_metadata=FRAME_1)

    with pytest.raises(ContractError, match=field):
        require_compatible_frame(prompt_image.prediction_metadata(), image=current)


@pytest.mark.parametrize(
    "prompt_number, current_number",
    [(1_000_000_000, 1_000_000_001), (2**53, 2**53 + 1)],
)
def test_adjacent_large_frame_numbers_are_rejected(prompt_number, current_number):
    # Frame numbers are discrete: no relative tolerance at any magnitude.
    prompt_image = _root(
        "frame",
        video_metadata=FRAME_1.model_copy(update={"frame_number": prompt_number}),
    )
    current = _root(
        "frame",
        video_metadata=FRAME_1.model_copy(update={"frame_number": current_number}),
    )
    metadata = json.loads(json.dumps(prompt_image.prediction_metadata()))

    with pytest.raises(ContractError, match="frame_number"):
        require_compatible_frame(metadata, image=current)


@pytest.mark.parametrize(
    "metadata",
    [{"image_dimensions": [5, 10]}, {"parent_id": "other"}, {"root_parent_id": "x"}],
)
def test_partial_legacy_field_that_disagrees_is_rejected(metadata):
    with pytest.raises(ContractError, match="Frame mismatch"):
        require_compatible_frame(metadata, image=_small(_root()))


def _plain_canvas():
    # An ordinary image with the mosaic's id and size.
    return ImageData.from_tensor(
        torch.zeros((3, 10, 40), dtype=torch.uint8), image_id="mosaic"
    )


@pytest.mark.parametrize(
    "metadata",
    [_plain_canvas().prediction_metadata(), {"is_composite": False}],
    ids=["typed", "legacy"],
)
def test_ordinary_metadata_for_a_composite_image_is_rejected(metadata):
    with pytest.raises(ContractError, match="is_composite"):
        require_compatible_frame(metadata, image=_mosaic())


@pytest.mark.parametrize("describe", ["typed", "legacy"])
def test_composite_metadata_for_an_ordinary_image_is_rejected(describe):
    mosaic = _mosaic()
    metadata = (
        mosaic.prediction_metadata() if describe == "typed" else {"is_composite": True}
    )

    with pytest.raises(ContractError, match="is_composite"):
        require_compatible_frame(metadata, image=_plain_canvas())


@pytest.mark.parametrize("describe", ["typed", "legacy"])
def test_same_canvas_id_with_other_placements_is_rejected(describe):
    mosaic = _mosaic()
    one_tile = dataclasses.replace(
        mosaic, composite_sources=mosaic.composite_sources[:1]
    )
    metadata = (
        mosaic.prediction_metadata()
        if describe == "typed"
        else _legacy_metadata(mosaic)
    )

    with pytest.raises(ContractError, match="composite_sources"):
        require_compatible_frame(metadata, image=one_tile)


def test_supplied_id_mirror_cannot_fill_an_anonymous_record():
    image = _small(_root())
    anonymous = dataclasses.replace(FrameRef.from_image(image), image_id=None)
    metadata = {**anonymous.prediction_metadata(), "parent_id": image.image_id}

    with pytest.raises(ContractError, match="Conflicting frame.*image_id"):
        require_compatible_frame(metadata, image=image)


def test_valid_tile_mirror_cannot_disagree_with_the_composite_record():
    image = _mosaic()
    metadata = image.prediction_metadata()
    metadata["composite_sources"] = copy.deepcopy(metadata["composite_sources"][:1])

    with pytest.raises(ContractError, match="Conflicting frame.*composite_sources"):
        require_compatible_frame(metadata, image=image)
