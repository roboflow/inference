"""Unit tests of ``roboflow_core/detections_consensus@v1`` on native predictions.

Every source is one stand-in model's prediction of the 40x60 ``street`` image.
Rows are written as ``(class_id, confidence, box)``; the default class map is
``{0: person, 1: car}``. Dense masks fill their box, RLE masks encode the same
pixels, and views store them on a stride-2 logit grid.
"""

import dataclasses
import itertools
import logging

import numpy as np
import pycocotools.mask as mask_utils
import pytest
import torch
from roboflow_workflows.v2.blocks.fusion import BLOCKS
from roboflow_workflows.v2.blocks.fusion.detections_consensus import v1 as consensus
from roboflow_workflows.v2.blocks.fusion.detections_consensus.v1 import BLOCK_TYPE, DetectionsConsensusBlock, detections_consensus
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import FrameMapping
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.masks.providers import mask_provider_of
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

SIZE_HW = (40, 60)
STRIDE = 2
NAMES = {0: "person", 1: "car"}
PERSON = (0, 0.9, (10, 10, 30, 30))
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]


def frame(image_id="street", *, size_hw=SIZE_HW, offset_xy=(0.0, 0.0)) -> FrameRef:
    """A frame; a non-zero offset makes it a crop of a 100x100 parent."""
    if offset_xy == (0.0, 0.0):
        mapping = FrameMapping.identity(image_id, size_hw)
    else:
        mapping = FrameMapping(
            frame_id="parent", frame_size_hw=(100, 100), offset_xy=offset_xy
        )
    return FrameRef(image_id=image_id, size_hw=size_hw, parent=mapping, root=mapping)


def boxes(
    rows,
    *,
    names=NAMES,
    namespace="model",
    frame_ref=None,
    row_metadata=None,
    device="cpu",
) -> Detections:
    frame_ref = frame_ref or frame()
    metadata = frame_ref.prediction_metadata()
    if names is not None:
        metadata["class_names"] = dict(names)
    detections = Detections(
        xyxy=torch.tensor(
            [box for _, _, box in rows], dtype=torch.float32, device=device
        ).reshape(-1, 4),
        class_id=torch.tensor(
            [class_id for class_id, _, _ in rows], dtype=torch.int64, device=device
        ),
        confidence=torch.tensor(
            [score for _, score, _ in rows], dtype=torch.float32, device=device
        ),
        image_metadata=metadata,
        bboxes_metadata=row_metadata,
    )
    if namespace is None:
        return detections
    return with_row_identity(detections, namespace=namespace)


def box_masks(detections) -> torch.Tensor:
    masks = torch.zeros(
        (len(detections), *SIZE_HW), dtype=torch.bool, device=detections.xyxy.device
    )
    for row, (x0, y0, x1, y1) in enumerate(detections.xyxy.int().tolist()):
        masks[row, y0:y1, x0:x1] = True
    return masks


def dense(rows, *, masks=None, **kwargs) -> InstanceDetections:
    detections = boxes(rows, **kwargs)
    return InstanceDetections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        mask=box_masks(detections) if masks is None else masks,
        image_metadata=detections.image_metadata,
        bboxes_metadata=detections.bboxes_metadata,
    )


def rle(rows, **kwargs) -> InstanceDetections:
    instances = dense(rows, **kwargs)
    encoded = [
        mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))["counts"]
        for mask in instances.mask.cpu()
    ]
    return dataclasses.replace(
        instances, mask=InstancesRLEMasks(image_size=SIZE_HW, masks=encoded)
    )


def view(rows, **kwargs) -> SegmentationView:
    detections = boxes(rows, **kwargs)
    cells = torch.nn.functional.max_pool2d(
        box_masks(detections).float().unsqueeze(1), STRIDE
    ).squeeze(1)
    height, width = SIZE_HW
    geometry = MaskGridGeometry(
        grid_size_hw=(height // STRIDE, width // STRIDE),
        inference_size_hw=SIZE_HW,
        padding_ltrb=(0, 0, 0, 0),
        scale_xy=(1.0, 1.0),
        crop_size_hw=SIZE_HW,
        original_size_hw=SIZE_HW,
        static_crop_xywh=(0, 0, width, height),
    )
    return SegmentationView(
        detections=detections,
        scores=cells * 16.0 - 8.0,
        score_type="logits",
        threshold=0.0,
        geometry=geometry,
    )


def agree(*sources, required_votes=2, **options) -> dict:
    return detections_consensus(list(sources), required_votes=required_votes, **options)


def row_classes(predictions) -> list:
    return [row["class"] for row in predictions.bboxes_metadata]


# ---------------------------------------------------------------------------
# Block contract
# ---------------------------------------------------------------------------


def test_block_is_exported_under_its_v2_identity_only() -> None:
    assert BLOCKS.count(DetectionsConsensusBlock) == 1
    assert DetectionsConsensusBlock.type == BLOCK_TYPE == "roboflow_core/detections_consensus@v1"
    assert not getattr(DetectionsConsensusBlock, "aliases", ())


def run_block(*sources, **options) -> dict:
    block = DetectionsConsensusBlock.__new__(DetectionsConsensusBlock)
    defaults = {
        "required_votes": 2,
        "class_aware": True,
        "iou_threshold": 0.3,
        "confidence": 0.0,
        "classes_to_consider": None,
        "required_objects": None,
        "presence_confidence_aggregation": "max",
        "detections_merge_confidence_aggregation": "average",
        "detections_merge_coordinates_aggregation": "average",
        "detections_merge_mask_aggregation": "union",
        "raise_on_class_name_conflict": False,
    }
    result = block.run(predictions_batches=list(sources), **{**defaults, **options})

    return result


def test_run_gives_each_renamed_merge_option_its_own_effect() -> None:
    # given: a confident wide person and a weak narrow car on one spot, and
    # two overlapping person masks
    wide = boxes([(0, 0.9, (0, 0, 20, 30))], namespace="a")
    narrow = boxes([(1, 0.5, (10, 10, 20, 20))], namespace="b")
    left = dense([(0, 0.9, (2, 4, 10, 20))], namespace="a")
    right = dense([(0, 0.8, (4, 4, 12, 20))], namespace="b")

    # when: every mode differs from its default, and from the other option's
    boxes_result = run_block(
        wide,
        narrow,
        class_aware=False,
        iou_threshold=0.1,
        detections_merge_confidence_aggregation="min",
        detections_merge_coordinates_aggregation="max",
    )
    masks_result = run_block(
        left, right, detections_merge_mask_aggregation="intersection"
    )

    # then: min confidence picks the car; max coordinates the wide box; the
    # intersection keeps 6x16 pixels (the union would keep 10x16)
    merged = boxes_result["predictions"]
    assert row_classes(merged) == ["car"]
    assert merged.confidence.tolist() == pytest.approx([0.5])
    assert merged.xyxy.tolist() == [[0.0, 0.0, 20.0, 30.0]]
    assert int(masks_result["predictions"].mask.sum()) == 96
    with pytest.raises(ValueError, match="conflicting class names"):
        run_block(
            boxes([PERSON], namespace="a"),
            boxes([PERSON], names={0: "car"}, namespace="b"),
            class_aware=False,
            raise_on_class_name_conflict=True,
        )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_two_models_merge_one_person_and_drop_a_lone_row(device: str) -> None:
    # given
    first = boxes(
        [(0, 0.9, (10, 10, 50, 90)), (1, 0.8, (0, 0, 4, 4))],
        namespace="a",
        row_metadata=[{"detection_id": "a0", "source": "a"}, {"detection_id": "a1"}],
        device=device,
    )
    second = boxes([(0, 0.7, (12, 8, 52, 88))], namespace="b", device=device)

    # when
    result = agree(first, second)

    # then
    merged = result["predictions"]
    assert isinstance(merged, Detections) and not isinstance(merged, InstanceDetections)
    assert merged.xyxy.device.type == device
    assert merged.xyxy.tolist() == [[11.0, 9.0, 51.0, 89.0]]
    assert merged.confidence.tolist() == pytest.approx([0.8])
    assert merged.class_id.tolist() == [0]
    assert merged.bboxes_metadata == [{"source": "a", "class": "person"}]
    assert merged.image_metadata["class_names"] == {0: "person"}
    assert merged.image_metadata["prediction_type"] == "object-detection"
    assert frame_of(merged) == frame()
    record = row_record(merged)
    assert record is not None
    assert record.namespace not in ("a", "b")
    assert result["object_present"] is True
    assert result["presence_confidence"] == {"person": pytest.approx(0.8)}


def test_one_required_vote_keeps_every_row_once() -> None:
    # given
    first = boxes([PERSON, (1, 0.5, (40, 0, 50, 10))], namespace="a")
    second = boxes([(0, 0.7, (10, 10, 30, 30))], namespace="b")

    # when
    merged = agree(first, second, required_votes=1)["predictions"]

    # then
    assert merged.xyxy.tolist() == [[10, 10, 30, 30], [40, 0, 50, 10]]
    assert merged.confidence.tolist() == pytest.approx([0.8, 0.5])


def test_more_votes_than_sources_keep_nothing() -> None:
    result = agree(boxes([PERSON], namespace="a"), boxes([PERSON]), required_votes=3)

    assert len(result["predictions"]) == 0
    assert result["object_present"] is False
    assert result["presence_confidence"] == {}


def test_equal_overlap_matches_the_earlier_row_and_frees_the_later_one() -> None:
    # given: second has the same box twice; the later copy has no partner left
    first = boxes([(0, 0.9, (10, 10, 30, 30))], namespace="a")
    second = boxes(
        [(0, 0.5, (10, 10, 30, 30)), (0, 0.3, (10, 10, 30, 30))], namespace="b"
    )

    # when
    merged = agree(first, second, required_votes=1)["predictions"]

    # then: anchor a0 takes b0 (mean 0.7); b1 anchors alone afterwards
    assert merged.confidence.tolist() == pytest.approx([0.7, 0.3])


def test_highest_overlap_wins_over_row_order() -> None:
    # given
    first = boxes([(0, 0.9, (10, 10, 30, 30))], namespace="a")
    second = boxes(
        [(0, 0.1, (10, 10, 30, 40)), (0, 0.5, (10, 10, 30, 31))], namespace="b"
    )

    # when
    merged = agree(first, second)["predictions"]

    # then
    assert merged.confidence.tolist() == pytest.approx([0.7])


@pytest.mark.parametrize("threshold, matches", [(0.5, 0), (0.49, 1)])
def test_overlap_must_exceed_the_threshold(threshold: float, matches: int) -> None:
    # given: IoU is exactly 0.5
    first = boxes([(0, 0.9, (0, 0, 10, 10))], namespace="a")
    second = boxes([(0, 0.9, (0, 0, 10, 5))], namespace="b")

    # when
    merged = agree(first, second, iou_threshold=threshold)["predictions"]

    # then
    assert len(merged) == matches


def test_class_aware_matching_compares_names_not_ids() -> None:
    # given: id 0 is "person" in both maps, id 3 means person in the second
    first = boxes([(0, 0.9, (10, 10, 30, 30))], namespace="a")
    second = boxes([(3, 0.7, (10, 10, 30, 30))], names={3: "person"}, namespace="b")
    car = boxes([(1, 0.7, (10, 10, 30, 30))], namespace="c")

    # when
    same_name = agree(first, second)
    other_class = agree(first, car)
    agnostic = agree(first, car, class_aware=False)

    # then
    assert row_classes(same_name["predictions"]) == ["person"]
    assert len(other_class["predictions"]) == 0
    assert row_classes(agnostic["predictions"]) == ["person"]
    assert agnostic["presence_confidence"] == {"any_object": pytest.approx(0.8)}


def test_a_dropped_group_leaves_its_rows_for_later_anchors() -> None:
    # given: a0 overlaps b0 but is too weak; a1 overlaps b0 better
    first = boxes([(0, 0.1, (0, 0, 10, 14)), (0, 0.9, (0, 0, 10, 11))], namespace="a")
    second = boxes([(0, 0.9, (0, 0, 10, 10))], namespace="b")

    # when
    merged = agree(first, second, confidence=0.6)["predictions"]

    # then: a0 + b0 averaged 0.5 < 0.6; a1 + b0 kept
    assert merged.confidence.tolist() == pytest.approx([0.9])
    assert merged.xyxy.tolist() == [[0.0, 0.0, 10.0, 10.5]]


def test_classes_to_consider_drop_other_and_unnamed_rows() -> None:
    # given
    first = boxes([(0, 0.9, (0, 0, 10, 10)), (1, 0.8, (20, 20, 30, 30))], namespace="a")
    second = boxes(
        [(0, 0.7, (0, 0, 10, 10)), (1, 0.6, (20, 20, 30, 30))], namespace="b"
    )
    unnamed = boxes([(0, 0.9, (0, 0, 10, 10))], names=None, namespace="c")

    # when
    cars = agree(first, second, classes_to_consider=["car"])
    everything = agree(first, second, classes_to_consider=[])
    without_names = agree(unnamed, unnamed, classes_to_consider=["person"])

    # then
    assert row_classes(cars["predictions"]) == ["car"]
    assert cars["presence_confidence"] == {"car": pytest.approx(0.7)}
    assert row_classes(everything["predictions"]) == ["person", "car"]
    assert len(without_names["predictions"]) == 0


def test_unnamed_rows_match_and_report_as_class_id() -> None:
    # given
    unnamed = boxes([(4, 0.9, (0, 0, 10, 10))], names=None, namespace="a")

    # when
    result = agree(unnamed, boxes([(4, 0.5, (0, 0, 10, 10))], names=None))

    # then
    assert row_classes(result["predictions"]) == ["class_4"]
    assert result["predictions"].image_metadata["class_names"] == {4: "class_4"}
    assert result["presence_confidence"] == {"class_4": pytest.approx(0.7)}


@pytest.mark.parametrize(
    "metadata, namespace",
    [
        (None, "a"),
        ([{"detection_id": "x"}], None),
        (None, None),
    ],
    ids=["row-identity", "detection-id", "position"],
)
def test_rows_without_row_identity_are_told_apart(metadata, namespace) -> None:
    # given: each source has one row overlapping the anchor of the other
    first = boxes([PERSON], namespace=namespace, row_metadata=metadata)
    second = boxes([PERSON, PERSON], namespace=None)

    # when
    merged = agree(first, second, required_votes=1)["predictions"]

    # then: a0 + b0, then b1 alone
    assert len(merged) == 2


def test_shared_detection_id_marks_both_rows_used_as_in_v1() -> None:
    # given: both rows carry the same detection_id and no row identity
    shared = [{"detection_id": "same"}]
    first = boxes([PERSON], namespace=None, row_metadata=shared)
    second = boxes([(0, 0.5, (40, 0, 50, 10))], namespace=None, row_metadata=shared)

    # when
    merged = agree(first, second, required_votes=1)["predictions"]

    # then
    assert merged.xyxy.tolist() == [[10.0, 10.0, 30.0, 30.0]]


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, confidence, class_name",
    [("average", 0.6, "car"), ("max", 0.9, "person"), ("min", 0.4, "car")],
)
def test_confidence_aggregation_also_picks_the_class(
    mode: str, confidence: float, class_name: str
) -> None:
    # given: one person and two cars on the same spot
    first = boxes([(0, 0.9, (0, 0, 10, 10))], namespace="a")
    second = boxes([(1, 0.5, (0, 0, 10, 10))], namespace="b")
    third = boxes([(1, 0.4, (0, 0, 10, 10))], namespace="c")

    # when
    result = agree(
        first,
        second,
        third,
        required_votes=3,
        class_aware=False,
        confidence_aggregation=mode,
    )

    # then
    merged = result["predictions"]
    assert merged.confidence.tolist() == pytest.approx([confidence])
    assert row_classes(merged) == [class_name]
    assert merged.class_id.tolist() == [0 if class_name == "person" else 1]


def test_average_class_vote_tie_goes_to_the_earlier_row() -> None:
    first = boxes([(1, 0.5, (0, 0, 10, 10))], namespace="a")
    second = boxes([(0, 0.9, (0, 0, 10, 10))], namespace="b")

    merged = agree(first, second, class_aware=False)["predictions"]

    assert row_classes(merged) == ["car"]


def test_selected_row_metadata_follows_the_chosen_class() -> None:
    # given
    first = boxes(
        [(0, 0.9, (0, 0, 10, 10))],
        namespace="a",
        row_metadata=[{"detection_id": "a0", "model": "a"}],
    )
    second = boxes(
        [(1, 0.5, (0, 0, 10, 10))],
        namespace="b",
        row_metadata=[{"detection_id": "b0", "model": "b"}],
    )

    # when
    merged = agree(first, second, class_aware=False, confidence_aggregation="min")

    # then
    assert merged["predictions"].bboxes_metadata == [{"model": "b", "class": "car"}]


@pytest.mark.parametrize(
    "mode, box",
    [
        ("average", [5.0, 5.0, 20.0, 25.0]),
        ("max", [0.0, 0.0, 20.0, 30.0]),
        ("min", [10.0, 10.0, 20.0, 20.0]),
    ],
)
def test_coordinate_aggregation_without_masks(mode: str, box: list) -> None:
    first = boxes([(0, 0.9, (0, 0, 20, 30))], namespace="a")
    second = boxes([(0, 0.9, (10, 10, 20, 20))], namespace="b")

    merged = agree(first, second, iou_threshold=0.1, coordinates_aggregation=mode)

    assert merged["predictions"].xyxy.tolist() == [box]


def test_class_name_conflict_between_merged_rows_warns_and_keeps_own_names(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # given: id 0 is person in one model and car in the other; each row is
    # its own group, so only the class map of the result sees the conflict
    first = boxes([(0, 0.9, (0, 0, 10, 10))], namespace="a")
    second = boxes([(0, 0.8, (0, 0, 10, 10))], names={0: "car"}, namespace="b")

    # when
    with caplog.at_level(logging.WARNING, logger=consensus.__name__):
        result = agree(first, second, required_votes=1)

    # then
    assert "conflicting class names for class_id=0" in caplog.text
    merged = result["predictions"]
    assert row_classes(merged) == ["person", "car"]
    assert merged.image_metadata["class_names"] == {0: "car"}
    assert result["presence_confidence"] == {
        "person": pytest.approx(0.9),
        "car": pytest.approx(0.8),
    }


def test_class_name_conflict_inside_a_group_raises_when_asked() -> None:
    first = boxes([(0, 0.9, (0, 0, 10, 10))], namespace="a")
    second = boxes([(0, 0.8, (0, 0, 10, 10))], names={0: "car"}, namespace="b")

    with pytest.raises(ValueError, match="conflicting class names"):
        agree(first, second, class_aware=False, raise_on_class_name_conflict=True)


# ---------------------------------------------------------------------------
# Presence
# ---------------------------------------------------------------------------


def people_and_cars(**options) -> dict:
    rows = [
        (0, 0.9, (0, 0, 10, 10)),
        (0, 0.5, (20, 0, 30, 10)),
        (1, 0.7, (40, 0, 50, 10)),
    ]
    return agree(boxes(rows, namespace="a"), boxes(rows, namespace="b"), **options)


@pytest.mark.parametrize(
    "required, class_aware, present",
    [
        (None, True, True),
        (3, True, True),
        (4, True, False),
        ({"person": 2, "car": 1}, True, True),
        ({"person": 2, "car": 2}, True, False),
        ({"bus": 1}, True, False),
        ({"person": 2, "car": 1}, False, True),
        ({"person": 2, "car": 2}, False, False),
    ],
)
def test_required_objects(required, class_aware: bool, present: bool) -> None:
    result = people_and_cars(required_objects=required, class_aware=class_aware)

    assert result["object_present"] is present
    assert bool(result["presence_confidence"]) is present


@pytest.mark.parametrize("mode, person", [("max", 0.9), ("min", 0.5), ("average", 0.7)])
def test_presence_confidence_aggregation_per_class(mode: str, person: float) -> None:
    result = people_and_cars(presence_confidence_aggregation=mode)

    assert result["presence_confidence"] == {
        "person": pytest.approx(person),
        "car": pytest.approx(0.7),
    }


# ---------------------------------------------------------------------------
# Empty inputs and frames
# ---------------------------------------------------------------------------


def test_all_empty_sources_describe_the_frame_and_report_nothing() -> None:
    # when
    result = agree(boxes([], namespace="a"), view([], namespace="b"))

    # then
    merged = result["predictions"]
    assert len(merged) == 0 and type(merged) is Detections
    assert frame_of(merged) == frame()
    assert merged.image_metadata["class_names"] == {}
    assert row_record(merged) is not None
    assert result["object_present"] is False
    assert result["presence_confidence"] == {}


def test_rows_filtered_away_still_describe_the_common_frame() -> None:
    result = agree(boxes([PERSON], namespace="a"), boxes([PERSON]),
                   classes_to_consider=["car"])  # fmt: skip

    merged = result["predictions"]
    assert len(merged) == 0
    assert frame_of(merged) == frame()
    assert merged.image_metadata["prediction_type"] == "object-detection"
    assert result["object_present"] is False


def test_predictions_of_different_images_are_rejected() -> None:
    other = boxes([PERSON], frame_ref=frame("other"))

    with pytest.raises(ValueError, match="different images or coordinate frames"):
        agree(boxes([PERSON], namespace="a"), other)


def test_crop_predictions_are_not_matched_with_the_parent() -> None:
    crop = boxes([PERSON], frame_ref=frame("street", offset_xy=(10.0, 10.0)))

    with pytest.raises(ValueError, match="map crop predictions into their parent"):
        agree(boxes([PERSON], namespace="a"), crop)


def test_an_empty_source_of_another_frame_is_ignored() -> None:
    empty_crop = boxes([], frame_ref=frame("crop", offset_xy=(10.0, 10.0)))

    result = agree(boxes([PERSON], namespace="a"), empty_crop, required_votes=1)

    assert len(result["predictions"]) == 1


def street_frame(image_id) -> FrameRef:
    """The ``street`` geometry under a local image id that may be unknown."""
    mapping = FrameMapping.identity("street", SIZE_HW)
    return FrameRef(image_id=image_id, size_hw=SIZE_HW, parent=mapping, root=mapping)


def sources_named(image_ids, *, make) -> list:
    sources = [
        make([PERSON], namespace=f"m{index}", frame_ref=street_frame(image_id))
        for index, image_id in enumerate(image_ids)
    ]
    return sources


SOURCE_MAKERS = pytest.mark.parametrize(
    "make", [boxes, dense, view], ids=["boxes", "dense", "view"]
)


@SOURCE_MAKERS
@pytest.mark.parametrize(
    "image_ids", list(itertools.permutations([None, "A", "B"])), ids=str
)
def test_an_unknown_image_id_does_not_bridge_two_known_ids(
    image_ids, make, monkeypatch
) -> None:
    # given: the check must reject before any mask is asked for
    def no_masks(payload):
        raise AssertionError("the frame check asked for masks")

    monkeypatch.setattr(consensus, "mask_provider_of", no_masks)
    sources = sources_named(image_ids, make=make)

    # when / then
    with pytest.raises(ValueError, match="different images or coordinate frames"):
        agree(*sources, required_votes=3)


@SOURCE_MAKERS
@pytest.mark.parametrize(
    "image_ids, image_id",
    [
        ((None, "A", "A"), "A"),
        (("A", None, "A"), "A"),
        (("A", "A", None), "A"),
        ((None, None, None), None),
    ],
)
def test_unknown_image_ids_agree_with_one_known_id(image_ids, image_id, make) -> None:
    result = agree(*sources_named(image_ids, make=make), required_votes=3)

    merged = result["predictions"]
    assert len(merged) == 1
    assert frame_of(merged).image_id == image_id


def test_an_unknown_image_id_still_needs_the_same_geometry() -> None:
    crop = frame(None, offset_xy=(10.0, 10.0))
    sources = [boxes([PERSON], frame_ref=crop), *sources_named(["A"], make=boxes)]

    with pytest.raises(ValueError, match="map crop predictions into their parent"):
        agree(*sources)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"required_votes": 0}, "required_votes"),
        ({"required_votes": True}, "required_votes"),
        ({"classes_to_consider": "person"}, "classes_to_consider"),
        ({"required_objects": 0}, "required_objects"),
        ({"required_objects": {"person": 0}}, "required_objects"),
        ({"confidence_aggregation": "median"}, "aggregation mode"),
        ({"mask_aggregation": "average"}, "mask aggregation"),
    ],
)
def test_invalid_options_are_rejected(options: dict, message: str) -> None:
    options = {"required_votes": 2, **options}

    with pytest.raises(ValueError, match=message):
        detections_consensus([boxes([PERSON])], **options)


def test_sources_must_be_predictions() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        detections_consensus([], required_votes=1)
    with pytest.raises(ValueError, match="prediction 1 must be Detections"):
        detections_consensus([boxes([PERSON]), {"xyxy": []}], required_votes=1)


@CUDA
def test_sources_on_different_devices_are_rejected() -> None:
    with pytest.raises(ValueError, match="is on cuda"):
        agree(boxes([PERSON], namespace="a"), boxes([PERSON], device="cuda"))


# ---------------------------------------------------------------------------
# Keypoints
# ---------------------------------------------------------------------------


def key_points(rows, *, with_boxes=True):
    detections = boxes(rows, namespace=None)
    points = KeyPoints(
        xy=torch.zeros((len(rows), 2, 2)),
        class_id=detections.class_id,
        confidence=torch.full((len(rows), 2), 0.5),
        image_metadata=detections.image_metadata,
    )
    return with_row_identity(
        (points, detections if with_boxes else None), namespace="pose"
    )


def test_keypoint_predictions_vote_through_their_boxes() -> None:
    result = agree(key_points([PERSON]), boxes([PERSON], namespace="a"))

    assert result["predictions"].xyxy.tolist() == [[10.0, 10.0, 30.0, 30.0]]


def test_keypoints_without_boxes_are_rejected() -> None:
    with pytest.raises(ValueError, match="has no bounding-box Detections"):
        agree(key_points([PERSON], with_boxes=False), boxes([PERSON]))


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def test_box_only_consensus_never_reads_masks(monkeypatch) -> None:
    # given
    def no_masks(payload):
        raise AssertionError("box-only consensus asked for masks")

    monkeypatch.setattr(consensus, "mask_provider_of", no_masks)

    # when
    merged = agree(boxes([PERSON], namespace="a"), boxes([PERSON]))["predictions"]

    # then
    assert type(merged) is Detections


def test_masks_are_read_only_for_rows_of_masked_groups(monkeypatch) -> None:
    # given: dense rows 0 and 2 match; row 1 matches nothing
    selected = []

    class Spy:
        def __init__(self, payload):
            self._provider = mask_provider_of(payload)

        def select(self, *, indices):
            selected.append(list(indices))
            return self._provider.select(indices=indices)

    monkeypatch.setattr(consensus, "mask_provider_of", Spy)
    masked = dense(
        [
            (0, 0.9, (0, 0, 10, 10)),
            (0, 0.9, (20, 0, 30, 10)),
            (0, 0.9, (40, 0, 50, 10)),
        ],
        namespace="a",
    )
    plain = boxes([(0, 0.5, (0, 0, 10, 10)), (0, 0.5, (40, 0, 50, 10))])

    # when
    merged = agree(masked, plain)["predictions"]

    # then
    assert selected == [[0, 2]]
    assert isinstance(merged, InstanceDetections)
    assert merged.mask.shape == (2, *SIZE_HW)


@pytest.mark.parametrize(
    "mode, box, pixels",
    [
        ("union", [2.0, 4.0, 10.0, 20.0], 128),
        ("max", [2.0, 4.0, 10.0, 20.0], 128),
        ("intersection", [0.0, 0.0, 0.0, 0.0], 0),
        ("min", [0.0, 0.0, 0.0, 0.0], 0),
    ],
)
def test_box_only_rows_count_as_empty_masks(mode, box, pixels) -> None:
    # given
    masked = dense([(0, 0.9, (2, 4, 10, 20))], namespace="a")
    plain = boxes([(0, 0.7, (2, 4, 10, 20))])

    # when
    result = agree(masked, plain, mask_aggregation=mode)

    # then: the box follows the mask; an empty mask keeps the row at (0, 0, 0, 0)
    merged = result["predictions"]
    assert merged.image_metadata["prediction_type"] == "instance-segmentation"
    assert merged.xyxy.tolist() == [box]
    assert int(merged.mask.sum()) == pixels
    assert result["object_present"] is True


def test_box_only_groups_get_an_empty_mask_beside_masked_groups() -> None:
    # given: the person group has a mask, the car group has none
    masked = dense([(0, 0.9, (2, 4, 10, 20))], namespace="a")
    plain = boxes([(0, 0.7, (2, 4, 10, 20)), (1, 0.6, (40, 0, 50, 10))])
    cars = boxes([(1, 0.6, (40, 0, 50, 10))], namespace="c")

    # when
    merged = agree(masked, plain, cars)["predictions"]

    # then
    assert merged.xyxy.tolist() == [[2.0, 4.0, 10.0, 20.0], [40.0, 0.0, 50.0, 10.0]]
    assert merged.mask.flatten(1).sum(dim=1).tolist() == [128, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_dense_rle_and_grid_masks_of_one_image_combine(device: str) -> None:
    # given: three representations of overlapping person masks
    sources = [
        dense([(0, 0.9, (2, 4, 10, 20))], namespace="dense", device=device),
        rle([(0, 0.8, (4, 4, 12, 20))], namespace="rle", device=device),
        view([(0, 0.7, (2, 6, 12, 20))], namespace="view", device=device),
    ]

    # when
    union = agree(*sources, required_votes=3)["predictions"]
    intersection = agree(*sources, required_votes=3, mask_aggregation="intersection")

    # then
    assert isinstance(union, InstanceDetections)
    assert union.mask.device.type == device
    expected = torch.zeros(SIZE_HW, dtype=torch.bool)
    expected[4:20, 2:12] = True
    assert torch.equal(union.mask[0].cpu(), expected)
    assert union.xyxy.tolist() == [[2.0, 4.0, 12.0, 20.0]]
    assert intersection["predictions"].xyxy.tolist() == [[4.0, 6.0, 10.0, 20.0]]


@pytest.mark.parametrize("mode, pixels", [("max", 160), ("min", 128)])
def test_max_and_min_masks_pick_by_pixel_count(mode: str, pixels: int) -> None:
    first = dense([(0, 0.9, (2, 4, 10, 20))], namespace="a")
    second = rle([(0, 0.8, (2, 4, 12, 20))], namespace="b")

    merged = agree(first, second, mask_aggregation=mode)["predictions"]

    assert int(merged.mask.sum()) == pixels


@pytest.mark.parametrize("make_source", [dense, rle], ids=["dense", "rle"])
@pytest.mark.parametrize(
    "extent",
    [(7, 11, 8, 12), (3, 5, 18, 15), (58, 38, 60, 40), None],
    ids=["single-pixel", "rectangle", "bottom-right-border", "empty-mask"],
)
def test_mask_boxes_cover_the_full_raster_extent(make_source, extent) -> None:
    masks = torch.zeros((1, *SIZE_HW), dtype=torch.bool)
    if extent is not None:
        x0, y0, x1, y1 = extent
        masks[0, y0:y1, x0:x1] = True
    source = make_source([PERSON], masks=masks)

    merged = agree(source, required_votes=1)["predictions"]

    expected_box = [0.0] * 4 if extent is None else list(extent)
    assert merged.xyxy.tolist() == [expected_box]
    assert torch.equal(merged.mask, masks)


@pytest.mark.parametrize(
    "make_source", [dense, rle, view], ids=["dense", "rle", "view"]
)
def test_zero_instance_masks_remain_an_empty_detection_result(make_source) -> None:
    merged = agree(make_source([]), required_votes=1)["predictions"]

    assert type(merged) is Detections
    assert merged.xyxy.shape == (0, 4)
    assert frame_of(merged) == frame()


@pytest.mark.parametrize("mode", ["average", "max", "min"])
def test_box_only_continuous_bounds_are_not_expanded(mode: str) -> None:
    box = (2.25, 4.5, 10.75, 20.125)
    first = boxes([(0, 0.9, box)], namespace="a")
    second = boxes([(0, 0.8, box)], namespace="b")

    merged = agree(first, second, coordinates_aggregation=mode)["predictions"]

    assert merged.xyxy.tolist() == [list(box)]
