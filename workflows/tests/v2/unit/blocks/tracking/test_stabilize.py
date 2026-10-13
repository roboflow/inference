"""Detections stabilizer: smoothing, Kalman gap fill, order, masks and sources."""

from datetime import datetime

import numpy as np
import pytest
import torch
from pycocotools import mask as coco_mask
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.tracking.base import TrackerCapacityError
from roboflow_workflows.v2.blocks.tracking.stabilize import (
    DEFAULT_SOURCE,
    StabilizeDetectionsBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

# V1 filter constants: process noise Q = 0.001 I, measurement noise R = 0.01 I.
Q, R = 0.001, 0.01


def _image(source="a"):
    image = ImageData.from_tensor(
        torch.zeros((3, 40, 60), dtype=torch.uint8),
        video_metadata=VideoMetadata(
            video_identifier=source,
            fps=30,
            frame_number=0,
            frame_timestamp=datetime(2026, 10, 13),
        ),
    )

    return image


def _rows(*rows, dtype=torch.float32, frame="f"):
    """Detections from (tracker_id, x1[, width]) tuples; boxes are 10 px tall."""
    widths = [row[2] if len(row) > 2 else 10.0 for row in rows]
    boxes = [[row[1], 0.0, row[1] + width, 10.0] for row, width in zip(rows, widths)]
    detections = Detections(
        xyxy=torch.tensor(boxes, dtype=dtype).reshape(-1, 4),
        class_id=torch.arange(len(rows), dtype=torch.int64),
        confidence=torch.full((len(rows),), 0.9),
        image_metadata={"class_names": {0: "car"}, "frame": frame},
        bboxes_metadata=[
            {"tracker_id": row[0], "detection_id": f"{frame}-{row[0]}"} for row in rows
        ]
        or None,
    )

    return detections


def _run(block, prediction, source="a", *, window=3, alpha=0.5, **kwargs):
    output = block.run(
        image=_image(source),
        predictions=prediction,
        smoothing_window_size=window,
        bbox_smoothing_coefficient=alpha,
        **kwargs,
    )["tracked_detections"]

    return output


def _x1(output):
    values = [round(x, 6) for x in output.xyxy[:, 0].tolist()]

    return values


def _ids(output):
    ids = [row["tracker_id"] for row in output.bboxes_metadata or []]

    return ids


def test_constant_motion_smooths_then_fills_gaps_until_the_window_empties():
    block = StabilizeDetectionsBlock()
    seen = [_x1(_run(block, _rows((1, float(x))))) for x in (0, 10, 20)]
    gaps = [_run(block, _rows()) for _ in range(3)]

    # a = 0.5: seen rows sit halfway back; gap rows advance a * 10 px.
    assert seen == [[0.0], [5.0], [15.0]]
    assert [_x1(gap) for gap in gaps] == [[25.0], [30.0], []]
    assert gaps[0].xyxy.tolist() == [[25.0, 0.0, 35.0, 10.0]]
    assert block._sources["a"].rows == {}


def test_kalman_correction_weights_the_mean_shift_by_the_gain():
    block = StabilizeDetectionsBlock()
    for x in (0.0, 10.0, 30.0):
        _run(block, _rows((1, x)))
    gap = _run(block, _rows())

    # Frame 2 starts v = 10 with P = I, predicted to P = (1 + Q) I. Frame 3
    # shifts 20: the window mean is 15, the gain (1 + Q) / (1 + Q + R).
    gain = (1 + Q) / (1 + Q + R)
    velocity = 10 + gain * (15 - 10)
    # Boxes return in float32.
    assert _x1(gap) == [pytest.approx(30 + 0.5 * velocity, abs=1e-5)]


def test_shift_is_measured_at_the_far_corner_as_in_v1():
    block = StabilizeDetectionsBlock()
    _run(block, _rows((1, 0.0, 10.0)))
    _run(block, _rows((1, 0.0, 20.0)))
    gap = _run(block, _rows())

    # x1 stays, x2 grows 10: the far corner moves 10 px, the centre only 5.
    # The gap box starts from the raw cached box [0, 0, 20, 10].
    assert gap.xyxy.tolist() == [[5.0, 0.0, 25.0, 10.0]]


def test_window_is_fixed_at_the_first_frame_and_coefficient_follows_calls():
    block = StabilizeDetectionsBlock()
    _run(block, _rows((1, 0.0)), window=1)
    _run(block, _rows((1, 10.0)), window=5)
    gap = _run(block, _rows(), window=5, alpha=1.0)
    after = _run(block, _rows(), window=5)

    # alpha 1.0 applies at once; window 1 (not 5) ends the track after one gap.
    assert _x1(gap) == [20.0]
    assert _x1(after) == []


def test_output_order_is_frame_rows_then_gap_rows_and_duplicates_keep_last_row():
    block = StabilizeDetectionsBlock()
    _run(block, _rows((1, 0.0), (2, 100.0), (3, 200.0)))
    _run(block, _rows((1, 2.0), (2, 102.0), (3, 202.0)))
    output = _run(block, _rows((3, 204.0), (5, 300.0), (3, 210.0)))

    assert _ids(output) == [3, 5, 1, 2]
    # Tracker 3 keeps its first position with its last row (x 210), smoothed
    # against the row before it in the same frame (x 204), as V1 does.
    assert _x1(output) == [207.0, 300.0, 3.0, 103.0]
    assert output.bboxes_metadata[0]["detection_id"] == "f-3"


def test_gap_rows_keep_their_frame_metadata_and_output_gets_fresh_identity():
    block = StabilizeDetectionsBlock()
    first = with_row_identity(_rows((1, 0.0), frame="one"), namespace="model-1")
    second = with_row_identity(_rows((1, 10.0), frame="two"), namespace="model-2")
    _run(block, first)
    _run(block, second)
    current = with_row_identity(_rows((9, 50.0), frame="three"), namespace="m3")
    output = _run(block, current)

    assert output.image_metadata["frame"] == "three"
    assert output.bboxes_metadata[1]["detection_id"] == "two-1"
    record = row_record(output)
    assert record.rows == 2 and record.namespace not in {"model-2", "m3"}
    assert row_record(current).namespace == "m3"


def _segmentations(*xs, rle=False, size=(40, 60)):
    rows = _rows(*[(index + 1, x) for index, x in enumerate(xs)])
    masks = torch.zeros((len(xs),) + size, dtype=torch.bool)
    for index, x in enumerate(xs):
        masks[index, :10, int(x) : int(x) + 10] = True
    mask = masks
    if rle:
        mask = InstancesRLEMasks(
            image_size=size,
            masks=[
                coco_mask.encode(np.asfortranarray(m.numpy().astype(np.uint8)))[
                    "counts"
                ]
                for m in masks
            ],
        )
    predictions = InstanceDetections(
        xyxy=rows.xyxy,
        class_id=rows.class_id,
        confidence=rows.confidence,
        mask=mask,
        image_metadata=rows.image_metadata,
        bboxes_metadata=rows.bboxes_metadata,
    )

    return predictions


@pytest.mark.parametrize("rle", [False, True])
def test_gap_rows_carry_their_earlier_masks(rle):
    block = StabilizeDetectionsBlock()
    _run(block, _segmentations(0.0, 30.0, rle=rle))
    _run(block, _segmentations(2.0, 32.0, rle=rle))
    current = _segmentations(4.0, rle=rle)
    output = _run(block, current)

    assert _ids(output) == [1, 2]
    if rle:
        assert output.mask.masks == current.mask.masks + [
            _segmentations(2.0, 32.0, rle=True).mask.masks[1]
        ]
    else:
        assert torch.equal(output.mask[0], current.mask[0])
        assert torch.equal(output.mask[1], _segmentations(2.0, 32.0).mask[1])


def test_gap_rows_of_another_mask_size_fail_clearly():
    block = StabilizeDetectionsBlock()
    _run(block, _segmentations(0.0, 30.0))
    _run(block, _segmentations(2.0, 32.0))

    with pytest.raises(ValueError, match="layout"):
        _run(block, _segmentations(4.0, size=(20, 30)))


def test_untracked_rows_fail_and_change_no_state():
    block = StabilizeDetectionsBlock()
    untracked = _rows((1, 0.0))
    untracked.bboxes_metadata = [{}]

    with pytest.raises(ValueError, match="every row needs a tracker_id"):
        _run(block, untracked)
    assert block._sources == {}


def test_sources_are_isolated_and_bounded():
    block = StabilizeDetectionsBlock()
    for x in (0.0, 10.0):
        _run(block, _rows((1, x)), "a", max_sources=2)
        _run(block, _rows((1, 100.0 - x)), "b", max_sources=2)

    gap_a = _run(block, _rows(), "a", max_sources=2)
    gap_b = _run(block, _rows(), "b", max_sources=2)

    assert (_x1(gap_a), _x1(gap_b)) == ([15.0], [85.0])
    with pytest.raises(TrackerCapacityError, match="max_sources=2"):
        _run(block, _rows((1, 0.0)), "c", max_sources=2)


def test_plain_call_without_video_identifier_uses_the_default_source():
    block = StabilizeDetectionsBlock()
    _run(block, _rows((1, 0.0)), "")
    _run(block, _rows((1, 10.0)), "")
    gap = _run(block, _rows(), "")

    assert list(block._sources) == [DEFAULT_SOURCE]
    assert _x1(gap) == [15.0]


def test_other_context_failures_are_not_taken_for_a_plain_call(monkeypatch):
    def broken(self):
        raise RuntimeError("context broke")

    monkeypatch.setattr(StabilizeDetectionsBlock, "execution_context", property(broken))

    with pytest.raises(RuntimeError, match="context broke"):
        _run(StabilizeDetectionsBlock(), _rows((1, 0.0)), "")


def test_input_is_not_changed_and_box_dtype_is_kept():
    block = StabilizeDetectionsBlock()
    _run(block, _rows((1, 0.0), dtype=torch.float16))
    current = _rows((1, 10.0), dtype=torch.float16)
    current.bboxes_metadata[0]["_host_xyxy"] = [10.0, 0.0, 20.0, 10.0]
    output = _run(block, current)

    assert output.xyxy.dtype == torch.float16
    assert current.xyxy.tolist() == [[10.0, 0.0, 20.0, 10.0]]
    assert "_host_xyxy" in current.bboxes_metadata[0]
    assert "_host_xyxy" not in output.bboxes_metadata[0]
