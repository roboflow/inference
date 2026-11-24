"""Overlap filter: both overlap rules, edges, references, classes and carriers."""

import pytest
import torch
from roboflow_workflows.v2.blocks.analytics.overlap.v1 import OverlapFilterBlock
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

BICYCLE = 1


def _rows(*rows, rows_metadata=None):
    """Detections from (class_id, x1, y1, x2, y2) tuples."""
    detections = Detections(
        xyxy=torch.tensor([list(row[1:]) for row in rows], dtype=torch.float32).reshape(
            -1, 4
        ),
        class_id=torch.tensor([row[0] for row in rows], dtype=torch.int64),
        confidence=torch.full((len(rows),), 0.9),
        image_metadata={"class_names": {0: "person", 1: "bicycle"}},
        bboxes_metadata=rows_metadata or [{"row": i} for i in range(len(rows))] or None,
    )

    return detections


def _filter(prediction, overlap_type="Center Overlap", name="bicycle"):
    output = OverlapFilterBlock().run(
        predictions=prediction, overlap_class_name=name, overlap_type=overlap_type
    )["overlaps"]

    return output


def _kept(output):
    return [row["row"] for row in output.bboxes_metadata or []]


def test_people_on_a_bicycle_are_kept_and_the_bicycle_dropped():
    prediction = _rows(
        (0, 12, 0, 28, 30),
        (BICYCLE, 0, 10, 40, 40),
        (0, 60, 0, 80, 30),
    )

    assert _kept(_filter(prediction)) == [0]
    assert _kept(_filter(prediction, "Any Overlap")) == [0]


def test_centre_on_a_reference_edge_is_outside_but_touching_boxes_overlap():
    prediction = _rows(
        (BICYCLE, 0, 0, 10, 10),
        (0, 5, 8, 15, 12),  # centre (10, 10): on the corner
        (0, 10, 2, 20, 4),  # touches the right edge; centre (15, 3)
        (0, 10, 10, 20, 20),  # touches the corner only
        (0, 11, 0, 20, 10),  # 1 px apart
        (0, 4, 4, 6, 6),  # inside
    )

    assert _kept(_filter(prediction)) == [5]
    assert _kept(_filter(prediction, "Any Overlap")) == [1, 2, 3, 5]


def test_any_of_several_references_keeps_a_row_once_in_input_order():
    prediction = _rows(
        (0, 52, 2, 58, 8),
        (BICYCLE, 0, 0, 10, 10),
        (0, 2, 2, 8, 8),
        (BICYCLE, 50, 0, 60, 10),
        (BICYCLE, 1, 1, 9, 9),  # a reference inside a reference is dropped
    )

    assert _kept(_filter(prediction)) == [0, 2]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [(0, 0, 0, 10, 10)],
        [(BICYCLE, 0, 0, 10, 10), (BICYCLE, 2, 2, 4, 4)],
    ],
)
def test_without_references_or_other_rows_nothing_is_kept(rows):
    output = _filter(_rows(*rows))

    assert len(output) == 0
    assert isinstance(output, Detections)


def test_class_names_come_from_the_row_the_map_or_the_class_id():
    prediction = _rows(
        (7, 0, 0, 10, 10),
        (0, 2, 2, 8, 8),
        (BICYCLE, 2, 2, 8, 8),
        rows_metadata=[{"row": 0}, {"row": 1, "class": "cart"}, {"row": 2}],
    )

    assert _kept(_filter(prediction, name="class_7")) == [1, 2]
    assert _kept(_filter(prediction, name="cart")) == [0, 2]
    # Row 2 is a bicycle by the map; row 1 is a cart by its own class.
    assert _kept(_filter(prediction, name="bicycle")) == [0, 1]


def test_masks_and_row_identity_follow_kept_rows():
    masks = torch.zeros((3, 50, 50), dtype=torch.bool)
    masks[0, 1, 1] = masks[2, 2, 2] = True
    prediction = with_row_identity(
        InstanceDetections(
            xyxy=torch.tensor([[2.0, 2, 8, 8], [0, 0, 10, 10], [30, 30, 40, 40]]),
            class_id=torch.tensor([0, BICYCLE, 0]),
            confidence=torch.tensor([0.5, 0.6, 0.7]),
            mask=masks,
            image_metadata={"class_names": {0: "person", 1: "bicycle"}},
        ),
        namespace="rows",
    )

    output = _filter(prediction)

    assert isinstance(output, InstanceDetections)
    assert torch.equal(output.mask, masks[:1])
    assert output.confidence.tolist() == [0.5]
    assert row_record(output).namespace == "rows"
    assert len(prediction) == 3


def test_view_stays_lazy_and_selects_its_scores(monkeypatch):
    def forbidden(self):
        raise AssertionError("full_res must not run")

    monkeypatch.setattr(SegmentationView, "full_res", forbidden)
    scores = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
    view = SegmentationView(
        detections=_rows((BICYCLE, 0, 0, 10, 10), (0, 30, 30, 40, 40), (0, 2, 2, 8, 8)),
        scores=scores,
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.identity((4, 4)),
    )

    output = _filter(view)

    assert isinstance(output, SegmentationView)
    assert torch.equal(output.scores, scores[2:])
    assert _kept(output.detections) == [2]
    assert view.materialization_count == 0


def test_float64_boxes_are_compared_without_float32_rounding():
    # Centre x = 1 - 2**-40 is inside the reference edge at 1; in float32 it
    # would round onto the edge and be outside.
    shift = 2.0**-40
    prediction = Detections(
        xyxy=torch.tensor(
            [[0.0, 0.0, 1.0, 10.0], [0.5 - shift, 0.0, 1.5 - shift, 10.0]],
            dtype=torch.float64,
        ),
        class_id=torch.tensor([BICYCLE, 0]),
        confidence=torch.ones(2),
        image_metadata={"class_names": {1: "bicycle"}},
        bboxes_metadata=[{"row": 0}, {"row": 1}],
    )

    assert _kept(_filter(prediction)) == [1]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"overlap_type": "Partial"}, "overlap_type must be one"),
        ({"name": 3}, "overlap_class_name must be a string"),
    ],
)
def test_invalid_direct_calls_fail(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _filter(_rows((0, 0, 0, 1, 1)), **kwargs)
