"""The legacy ``is_composite`` flag and ``composite_sources`` across frame readers."""

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.data.coordinates import prediction_to_root
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import (
    FRAME_RECORD_KEY,
    FrameRef,
    frame_of,
    root_placement_from_metadata,
)
from roboflow_workflows.v2.data.image import (
    CompositeSource,
    ImageData,
    legacy_prediction_metadata,
)

from inference_models.models.base.object_detection import Detections

COMPOSITE_ROOT = "composite image \\(mosaic\\)"
ROOT_BOX = [[11.0, 5.0, 15.0, 9.0]]
ABSENT = object()
ORDINARY_MARKERS = {
    "absent": ABSENT,
    "none": None,
    "false": False,
    "numpy-false": np.False_,
}
COMPOSITE_MARKERS = {"true": True, "numpy-true": np.True_}
MALFORMED_MARKERS = {"one": 1, "zero": 0, "string": "false"}
# Historical V1 root keys only: no parent keys, so no full frame.
PARTIAL_LEGACY = {
    "image_dimensions": [5, 10],
    "root_parent_dimensions": [29, 47],
    "root_parent_coordinates": [7, 3],
    "root_parent_id": "root",
    "scaling_relative_to_root_parent": 0.5,
}


def _root():
    return ImageData.from_tensor(
        torch.zeros((3, 29, 47), dtype=torch.uint8), image_id="root"
    )


def _small():
    # Root 47x29; crop x 7..27, y 3..13 (20x10); resize to 10x5: scale 2.
    crop = _root().crop((7, 3, 27, 13), image_id="crop")
    small = crop.resize((5, 10), image_id="small")

    return small


def _mosaic(sources):
    canvas = ImageData.composite(
        torch.zeros((3, 20, 40), dtype=torch.uint8),
        sources=sources,
        image_id="mosaic",
    )

    return canvas


MOSAICS = {
    "one-tile": lambda: _mosaic(
        [CompositeSource.place(_root(), index=(0,), canvas_xyxy=(0, 0, 20, 20))]
    ),
    "no-tiles": lambda: _mosaic([]),
}


def _legacy(image):
    metadata = legacy_prediction_metadata(
        image_id=image.image_id,
        size_hw=image.size_hw,
        parent=image.parent,
        root=image.root,
        composite_sources=image.composite_sources,
    )

    return metadata


def _dual(image):
    return FrameRef.from_image(image).prediction_metadata()


def _typed_only(image):
    return {FRAME_RECORD_KEY: FrameRef.from_image(image).to_record()}


def _with(metadata, **changes):
    changed = dict(metadata)
    for key, value in changes.items():
        changed.pop(key, None)
        if value is not ABSENT:
            changed[key] = value

    return changed


def _boxes(image_metadata):
    return Detections(
        xyxy=torch.tensor([[2.0, 1.0, 4.0, 3.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.5]),
        image_metadata=image_metadata,
    )


def _assert_maps_to_root(metadata):
    placement = root_placement_from_metadata(metadata)

    assert placement.root.frame_size_hw == (29, 47)
    assert placement.root.map_xy(2, 1) == (11.0, 5.0)
    assert prediction_to_root(_boxes(metadata)).xyxy.tolist() == ROOT_BOX


def _assert_refuses_root(metadata, match=COMPOSITE_ROOT):
    with pytest.raises(ContractError, match=match):
        root_placement_from_metadata(metadata)
    with pytest.raises(ContractError, match=match):
        prediction_to_root(_boxes(metadata))


@pytest.mark.parametrize("marker", ORDINARY_MARKERS.values(), ids=ORDINARY_MARKERS)
def test_ordinary_marker_keeps_partial_legacy_metadata_ordinary(marker):
    metadata = _with(PARTIAL_LEGACY, is_composite=marker)

    _assert_maps_to_root(metadata)
    assert root_placement_from_metadata(metadata).root.frame_id == "root"


@pytest.mark.parametrize("marker", ORDINARY_MARKERS.values(), ids=ORDINARY_MARKERS)
def test_ordinary_marker_keeps_a_full_legacy_frame_ordinary(marker):
    unmarked = _legacy(_small())
    metadata = _with(unmarked, is_composite=marker)

    frame = frame_of(_boxes(metadata))

    assert not frame.is_composite
    assert frame == frame_of(_boxes(unmarked))
    _assert_maps_to_root(metadata)


@pytest.mark.parametrize("marker", COMPOSITE_MARKERS.values(), ids=COMPOSITE_MARKERS)
@pytest.mark.parametrize("legacy", ["partial", "full"])
def test_true_marker_without_sources_refuses_both_readers(marker, legacy):
    base = PARTIAL_LEGACY if legacy == "partial" else _legacy(_small())
    metadata = _with(base, is_composite=marker)

    _assert_refuses_root(metadata)
    if legacy == "full":
        with pytest.raises(ContractError, match="'composite_sources' list"):
            frame_of(_boxes(metadata))


@pytest.mark.parametrize("mosaic", MOSAICS.values(), ids=MOSAICS)
@pytest.mark.parametrize("marker", [True, ABSENT, None], ids=["true", "absent", "none"])
def test_legacy_composite_sources_mark_a_composite(mosaic, marker):
    image = mosaic()
    metadata = _with(_legacy(image), is_composite=marker)

    frame = frame_of(_boxes(metadata))

    assert frame.is_composite
    assert frame.composite_sources == image.composite_sources
    _assert_refuses_root(metadata)


@pytest.mark.parametrize("sources", [[], [{"unread": True}]], ids=["empty", "tile"])
@pytest.mark.parametrize("legacy", ["partial", "full"])
def test_composite_sources_on_ordinary_root_keys_refuse_root(sources, legacy):
    base = PARTIAL_LEGACY if legacy == "partial" else _legacy(_small())

    _assert_refuses_root(_with(base, composite_sources=sources))


@pytest.mark.parametrize("legacy", ["partial", "full", "mosaic"])
def test_false_marker_with_composite_sources_is_contradictory(legacy):
    if legacy == "mosaic":
        metadata = _with(_legacy(MOSAICS["no-tiles"]()), is_composite=False)
    else:
        base = PARTIAL_LEGACY if legacy == "partial" else _legacy(_small())
        metadata = _with(base, is_composite=False, composite_sources=[])
    match = "'is_composite' to False but also sets 'composite_sources'"

    _assert_refuses_root(metadata, match=match)
    if legacy != "partial":
        with pytest.raises(ContractError, match=match):
            frame_of(_boxes(metadata))


@pytest.mark.parametrize("marker", MALFORMED_MARKERS.values(), ids=MALFORMED_MARKERS)
@pytest.mark.parametrize("legacy", ["partial", "full", "dual"])
def test_non_boolean_marker_is_rejected(marker, legacy):
    base = {
        "partial": PARTIAL_LEGACY,
        "full": _legacy(_small()),
        "dual": _dual(_small()),
    }[legacy]
    metadata = _with(base, is_composite=marker)

    _assert_refuses_root(metadata, match="'is_composite' must be a boolean")
    if legacy != "partial":
        with pytest.raises(ContractError, match="'is_composite' must be a boolean"):
            frame_of(_boxes(metadata))


def test_partial_legacy_metadata_still_has_no_full_frame():
    with pytest.raises(ContractError, match="lacks frame geometry"):
        frame_of(_boxes(_with(PARTIAL_LEGACY, is_composite=False)))


@pytest.mark.parametrize("marker", ORDINARY_MARKERS.values(), ids=ORDINARY_MARKERS)
def test_ordinary_record_agrees_with_an_ordinary_marker(marker):
    metadata = _with(_dual(_small()), is_composite=marker)

    frame = frame_of(_boxes(metadata))

    assert frame == FrameRef.from_image(_small())
    _assert_maps_to_root(metadata)


@pytest.mark.parametrize(
    "changes",
    [
        {"is_composite": True},
        {"is_composite": np.True_},
        {"composite_sources": []},
        {"is_composite": False, "composite_sources": []},
    ],
    ids=["true", "numpy-true", "empty-sources", "false-with-sources"],
)
def test_ordinary_record_conflicts_with_composite_legacy_keys(changes):
    metadata = _with(_dual(_small()), **changes)

    _assert_refuses_root(metadata, match="Conflicting frame")
    with pytest.raises(ContractError, match="Conflicting frame"):
        frame_of(_boxes(metadata))


@pytest.mark.parametrize("mosaic", MOSAICS.values(), ids=MOSAICS)
@pytest.mark.parametrize(
    "marker",
    [True, np.True_, ABSENT, None],
    ids=["true", "numpy-true", "absent", "none"],
)
def test_composite_record_agrees_with_a_composite_or_unset_marker(mosaic, marker):
    image = mosaic()
    metadata = _with(_dual(image), is_composite=marker)

    frame = frame_of(_boxes(metadata))

    assert frame == FrameRef.from_image(image)
    assert frame.is_composite
    _assert_refuses_root(metadata)


@pytest.mark.parametrize("mosaic", MOSAICS.values(), ids=MOSAICS)
@pytest.mark.parametrize(
    "changes",
    [{"is_composite": False}, {"is_composite": False, "composite_sources": ABSENT}],
    ids=["false", "false-without-sources"],
)
def test_composite_record_conflicts_with_a_false_marker(mosaic, changes):
    metadata = _with(_dual(mosaic()), **changes)

    _assert_refuses_root(metadata, match="Conflicting frame")
    with pytest.raises(ContractError, match="Conflicting frame"):
        frame_of(_boxes(metadata))


def test_typed_only_ordinary_frame_maps_to_root():
    metadata = _typed_only(_small())

    assert not frame_of(_boxes(metadata)).is_composite
    _assert_maps_to_root(metadata)


@pytest.mark.parametrize("mosaic", MOSAICS.values(), ids=MOSAICS)
def test_typed_only_composite_frame_refuses_root(mosaic):
    image = mosaic()
    metadata = _typed_only(image)

    assert frame_of(_boxes(metadata)) == FrameRef.from_image(image)
    _assert_refuses_root(metadata)
