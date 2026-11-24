"""Image parent rows: kept by derivation, cleared by composites, persisted.

A crop made for one detection row refers to it with ``ImageData.parent_row``.
A classifier copies the reference into its prediction, and a keyed join maps
the label back to the original row after filtering, sorting or repeating.
"""

import json

import pytest
import torch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image import CompositeSource, ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.row_identity import (
    RowRef,
    classification_parent,
    require_row_record,
    with_classification_parent,
    with_row_identity,
)
from roboflow_workflows.v2.engine.recording import (
    CodecRegistry,
    RecordingCorruptError,
)
from roboflow_workflows.v2.recording_codecs import NATIVE_RECORDING_CODECS

from inference_models.models.base.classification import ClassificationPrediction
from inference_models.models.base.object_detection import Detections
from tests.v2.support.recording.codecs import MemoryBlobs

REGISTRY = CodecRegistry(NATIVE_RECORDING_CODECS)
# Exact key set of a v2/image_data@1 node written before parent_row existed.
PRE_PARENT_ROW_IMAGE_KEYS = {
    "pixels",
    "image_id",
    "parent",
    "root",
    "video_metadata",
    "composite_sources",
}


def _frame():
    pixels = torch.arange(3 * 12 * 16, dtype=torch.int64).reshape(3, 12, 16) % 251
    return ImageData.from_tensor(pixels.to(torch.uint8), image_id="root")


def _encode(image):
    blobs = MemoryBlobs()
    node = REGISTRY.encoder(blobs, location="g.f").encode(image)
    return json.loads(json.dumps(node, allow_nan=False)), blobs


def _decode(node, blobs):
    return REGISTRY.decoder(blobs, location="g.f").decode(node)


# ImageData ------------------------------------------------------------------------


def test_new_images_have_no_parent_row():
    frame = _frame()

    assert frame.parent_row is None
    assert (
        ImageData.from_numpy_rgb(frame.tensor_image.permute(1, 2, 0).numpy()).parent_row
        is None
    )


@pytest.mark.parametrize("interpolation", ["bilinear", "nearest", "area"])
def test_derivation_keeps_the_parent_row(interpolation):
    reference = RowRef("r1", 2)
    crop = _frame().crop((2, 1, 10, 9)).with_parent_row(reference)

    resized = crop.resize((4, 4), interpolation=interpolation)
    recolored = crop.with_pixels(255 - crop.tensor_image)
    nested = crop.crop((1, 1, 5, 5))

    for image in (resized, recolored, nested):
        assert image.parent_row == reference
    assert recolored.image_id == crop.image_id
    assert resized.root == crop.resize((4, 4), interpolation=interpolation).root


def test_crop_of_a_crop_refers_to_its_immediate_detection_row():
    outer = _frame().crop((0, 0, 12, 12)).with_parent_row(RowRef("outer", 0))

    inner = outer.crop((2, 2, 6, 6)).with_parent_row(RowRef("inner", 3))

    assert inner.parent_row == RowRef("inner", 3)
    assert inner.root.frame_id == "root"
    assert outer.parent_row == RowRef("outer", 0)


def test_repeated_crops_of_one_row_share_the_reference_not_the_id():
    reference = RowRef("r1", 1)
    frame = _frame()

    first = frame.crop((2, 1, 10, 9)).with_parent_row(reference)
    second = frame.crop((2, 1, 10, 9)).with_parent_row(reference)

    assert first.parent_row == second.parent_row
    assert first.image_id != second.image_id


def test_composite_has_no_singular_parent_row():
    tiles = [
        _frame().crop((0, 0, 8, 8)).with_parent_row(RowRef("r1", row)) for row in (0, 1)
    ]
    canvas = torch.zeros((3, 8, 16), dtype=torch.uint8)

    composite = ImageData.composite(
        canvas,
        sources=[
            CompositeSource.place(
                tile,
                index=(position,),
                canvas_xyxy=(8 * position, 0, 8 * position + 8, 8),
            )
            for position, tile in enumerate(tiles)
        ],
    )

    assert composite.parent_row is None


def test_parent_row_must_be_a_row_ref():
    with pytest.raises(TypeError, match="parent_row must be a RowRef"):
        _frame().with_parent_row({"namespace": "r1", "id": 0})


# Image JSON -----------------------------------------------------------------------


def test_image_json_round_trips_the_parent_row():
    crop = _frame().crop((2, 1, 10, 9)).with_parent_row(RowRef("r1", 2))

    wire = json.loads(json.dumps(IMAGE_KIND.to_serialized(crop)))
    restored = IMAGE_KIND.to_payload(wire)

    assert wire["parent_row"] == {"namespace": "r1", "id": 2}
    assert restored.parent_row == RowRef("r1", 2)
    assert restored.parent == crop.parent


def test_image_json_without_parent_row_is_unchanged():
    wire = IMAGE_KIND.to_serialized(_frame())

    assert "parent_row" not in wire
    assert IMAGE_KIND.to_payload(wire).parent_row is None


def test_malformed_image_json_parent_row_is_rejected():
    wire = IMAGE_KIND.to_serialized(_frame())
    wire["parent_row"] = {"namespace": "r1", "id": -1}

    with pytest.raises(ContractError, match="non-negative integer"):
        IMAGE_KIND.to_payload(wire)


def test_v1_crop_ids_never_become_parent_rows():
    wire = IMAGE_KIND.to_serialized(_frame())
    v1 = {
        "type": "base64",
        "value": wire["value"],
        "parent_id": "crop-of-detection-7",
        "root_parent_id": "root",
    }

    assert IMAGE_KIND.to_payload(v1).parent_row is None


# Recording ------------------------------------------------------------------------


def test_recording_round_trips_the_parent_row():
    crop = _frame().crop((2, 1, 10, 9)).with_parent_row(RowRef("r1", 2))

    node, blobs = _encode(crop)
    decoded = _decode(node, blobs)

    assert node["codec"]["name"] == "v2/image_data@1"
    assert node["codec"]["value"]["parent_row"] == {"namespace": "r1", "id": 2}
    assert decoded.parent_row == RowRef("r1", 2)


def test_images_without_parent_row_keep_the_original_node():
    node, blobs = _encode(_frame())

    assert set(node["codec"]["value"]) == PRE_PARENT_ROW_IMAGE_KEYS
    assert _decode(node, blobs).parent_row is None


def test_pre_parent_row_recording_node_decodes():
    # Literal node of the original codec; pixels are one 1x1 grey blob.
    blobs = MemoryBlobs()
    pixels = REGISTRY.encoder(blobs, location="g.f").encode(
        torch.full((1, 1, 1), 7, dtype=torch.uint8)
    )
    frame = {
        "frame_id": "old",
        "size_hw": [1, 1],
        "scale_xy": [1.0, 1.0],
        "offset_xy": [0.0, 0.0],
    }
    node = {
        "codec": {
            "name": "v2/image_data@1",
            "value": {
                "pixels": pixels,
                "image_id": "old",
                "parent": frame,
                "root": frame,
                "video_metadata": None,
                "composite_sources": None,
            },
        }
    }

    decoded = _decode(json.loads(json.dumps(node)), blobs)

    assert decoded.image_id == "old"
    assert decoded.parent_row is None


@pytest.mark.parametrize(
    "change, match",
    [
        ({"parent_row": {"namespace": "", "id": 0}}, "invalid image"),
        ({"parent_row": None}, "invalid image"),
        ({"surprise": 1}, "exactly the keys"),
    ],
)
def test_malformed_recorded_parent_row_is_corruption(change, match):
    node, blobs = _encode(_frame())
    node["codec"]["value"].update(change)

    with pytest.raises(RecordingCorruptError, match=match):
        _decode(node, blobs)


# Crop -> classify -> join ---------------------------------------------------------

LABELS = ("A", "B", "C")


def _original():
    detections = Detections(
        xyxy=torch.tensor(
            [[0.0, 0.0, 4.0, 4.0], [4.0, 0.0, 8.0, 4.0], [8.0, 0.0, 12.0, 4.0]]
        ),
        class_id=torch.arange(3),
        confidence=torch.full((3,), 0.5),
        image_metadata={},
    )
    return with_row_identity(detections)


def _crop_and_classify(frame, predictions):
    """Stand-in for dynamic crop + classifier: one label per crop."""
    if not len(predictions):
        return []
    record = require_row_record(predictions, consumer="test/crop")
    ids, rectangles = record.host_ids_with(predictions.xyxy.round().to(torch.int64))
    classifications = []
    for row_id, rectangle in zip(ids, rectangles):
        crop = frame.crop(rectangle).with_parent_row(RowRef(record.namespace, row_id))
        # The "classifier" reads which column the crop came from.
        label = LABELS[int(crop.root.map_xy(0, 0)[0]) // 4]
        prediction = ClassificationPrediction(
            class_id=torch.tensor([LABELS.index(label)]),
            confidence=torch.ones(1, 3),
        )
        classifications.append(with_classification_parent(prediction, crop.parent_row))
    return classifications


def _join(original, classifications):
    """Stand-in for keyed replacement, including the duplicate rule."""
    targets = require_row_record(original, consumer="test/replace")
    by_parent = {}
    for classification in classifications:
        parent = classification_parent(classification)
        if parent is None:
            raise ContractError("classification without parent row")
        if parent.namespace != targets.namespace:
            raise ContractError("foreign namespace")
        if parent in by_parent:
            raise ContractError(f"duplicate classification for {parent}")
        by_parent[parent] = LABELS[int(classification.class_id[0])]
    labels = [by_parent.get(reference, "fallback") for reference in targets.refs()]
    return labels


@pytest.mark.parametrize(
    "indices, expected",
    [
        ([1], ["fallback", "B", "fallback"]),
        ([2, 0, 1], ["A", "B", "C"]),
        ([], ["fallback", "fallback", "fallback"]),
    ],
)
def test_labels_return_to_their_original_rows(indices, expected):
    frame = _frame()
    original = _original()
    selected = select_predictions(original, indices=indices)

    labels = _join(original, _crop_and_classify(frame, selected))

    assert labels == expected


def test_repeated_rows_give_duplicate_classifications_that_are_rejected():
    frame = _frame()
    original = _original()
    repeated = select_predictions(original, indices=[1, 1])
    classifications = _crop_and_classify(frame, repeated)

    assert [classification_parent(c).id for c in classifications] == [1, 1]
    with pytest.raises(ContractError, match="duplicate classification"):
        _join(original, classifications)
    assert _join(original, classifications[:1]) == ["fallback", "B", "fallback"]


def test_independently_identified_branches_cannot_be_joined():
    frame = _frame()
    plain = _original()
    plain.image_metadata = {}
    branch = with_row_identity(select_predictions(plain, indices=[1]))

    with pytest.raises(ContractError, match="foreign namespace"):
        _join(with_row_identity(plain), _crop_and_classify(frame, branch))
