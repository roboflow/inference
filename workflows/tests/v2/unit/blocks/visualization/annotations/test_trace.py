"""``v2/trace_visualization@v1``: paths of tracks over frames of each source.

The V1 oracle is the latest V1 block class itself, one instance kept across
frames, fed the same rows as ``sv.Detections``; V1 test modules are not
imported. Analytic cases use a flat grey image and a red ``CUSTOM`` palette.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.core_steps.visualizations.trace.v1 import (
    TraceVisualizationBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from roboflow_workflows.v2.blocks.visualization.annotations.sources import (
    SourceCapacityError,
    covers_root,
)
from roboflow_workflows.v2.blocks.visualization.annotations.trace import (
    TraceVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import spec_of

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from tests.v2.unit.blocks.visualization.annotations.support import (
    detections,
    pixels_of,
    scene,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
SIZE = (40, 64)
GREY = (100, 100, 100)
RED = (255, 0, 0)
RED_STYLE = {"color_palette": "CUSTOM", "custom_colors": ["#FF0000"]}
# (x1, y1, x2, y2, tracker id)
Row = Tuple[float, float, float, float, int]


def frame(
    number: int,
    *,
    video: str = "cam-a",
    size: Tuple[int, int] = SIZE,
    grey: bool = False,
) -> ImageData:
    pixels = (
        np.full(size + (3,), GREY, dtype=np.uint8)
        if grey
        else scene(*size, seed=number)
    )
    metadata = VideoMetadata(
        video_identifier=video,
        frame_number=number,
        frame_timestamp=T0 + timedelta(seconds=number / 30),
    )
    image = ImageData.from_numpy_rgb(
        pixels, image_id=f"{video}-{number}", video_metadata=metadata
    )

    return image


def tracked(image: ImageData, rows: Sequence[Row], *, class_id: int = 0) -> Detections:
    objects = [(x1, y1, x2, y2, class_id, 0.9) for x1, y1, x2, y2, _ in rows]
    predictions = detections(
        image, objects, rows=[{"tracker_id": row[4]} for row in rows]
    )

    return predictions


def centred(x: float, y: float, tracker_id: int, *, half: float = 4.0) -> Row:
    return (x - half, y - half, x + half, y + half, tracker_id)


class V2Trace:
    """One block instance across frames, parameters validated like the engine."""

    def __init__(self, **parameters: Any) -> None:
        self.block = TraceVisualizationBlock()
        params = spec_of(TraceVisualizationBlock).validate_params(
            {
                **parameters,
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
            }
        )
        self.arguments = dict(params)

    def __call__(
        self, image: ImageData, predictions: Any, **selected: Any
    ) -> ImageData:
        arguments = {
            **self.arguments,
            **selected,
            "image": image,
            "predictions": predictions,
        }
        output = self.block.run(**arguments)["image"]

        return output


class V1Trace:
    """The latest V1 block, one instance across frames."""

    def __init__(self, **parameters: Any) -> None:
        self.block = TraceVisualizationBlockV1()
        self.parameters = {
            "color_palette": "DEFAULT",
            "palette_size": 10,
            "custom_colors": [],
            "color_axis": "CLASS",
            "position": "CENTER",
            "trace_length": 30,
            "thickness": 1,
            **parameters,
        }

    def __call__(self, image: ImageData, rows: Sequence[Row]) -> np.ndarray:
        bgr = np.ascontiguousarray(pixels_of(image)[:, :, ::-1])
        v1_image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="frame"),
            numpy_image=bgr,
        )
        v1_rows = sv.Detections(
            xyxy=np.array([row[:4] for row in rows], dtype=np.float32).reshape(-1, 4),
            class_id=np.zeros(len(rows), dtype=int),
            confidence=np.full(len(rows), 0.9, dtype=np.float32),
            tracker_id=np.array([row[4] for row in rows], dtype=int),
        )
        result = self.block.run(
            image=v1_image, predictions=v1_rows, copy_image=True, **self.parameters
        )
        output = result["image"].numpy_image[:, :, ::-1]

        return output


def red_pixels(image: ImageData) -> np.ndarray:
    mask = np.all(pixels_of(image) == RED, axis=-1)

    return mask


# Two tracks crossing on fractional positions; track 7 joins on frame 2.
MOVES = (
    [centred(8.3, 6.6, 1), centred(55.5, 30.2, 2)],
    [centred(14.9, 9.1, 1), centred(49.2, 27.7, 2)],
    [centred(21.6, 13.8, 1), centred(42.0, 23.4, 2), centred(30.0, 33.0, 7)],
    [centred(27.2, 18.4, 1), centred(36.8, 20.9, 2), centred(33.5, 30.1, 7)],
    [centred(33.7, 22.0, 1), centred(29.1, 17.5, 2), centred(37.0, 27.3, 7)],
)


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"trace_length": 3, "thickness": 2, "color_axis": "TRACK"},
        {"position": "BOTTOM_RIGHT", "trace_length": 2, "color_axis": "INDEX"},
    ],
)
def test_every_frame_equals_v1_when_each_frame_has_rows(parameters: Dict) -> None:
    v2, v1 = V2Trace(**parameters), V1Trace(**parameters)

    for number, rows in enumerate(MOVES):
        image = frame(number)
        output = v2(image, tracked(image, rows))
        np.testing.assert_array_equal(pixels_of(output), v1(image, rows))


def test_a_path_joins_the_last_trace_length_frames() -> None:
    trace = V2Trace(trace_length=3, **RED_STYLE)
    outputs = []
    for number, x in enumerate((10, 20, 30, 40)):
        image = frame(number, grey=True)
        outputs.append(trace(image, tracked(image, [centred(x, 16, 1)])))

    assert not red_pixels(outputs[0]).any()
    assert np.flatnonzero(red_pixels(outputs[1])[16]).tolist() == list(range(10, 21))
    assert np.flatnonzero(red_pixels(outputs[3])[16]).tolist() == list(range(20, 41))
    assert red_pixels(outputs[3]).sum() == 21


def test_a_missing_track_is_not_drawn_and_keeps_its_points_until_they_age() -> None:
    trace = V2Trace(trace_length=3, **RED_STYLE)
    plan = (
        [centred(10, 10, 1), centred(50, 30, 2)],
        [centred(20, 10, 1), centred(50, 34, 2)],
        [centred(50, 38, 2)],  # track 1 lost for a frame
        [centred(30, 10, 1)],  # back: joins its points of frames 1 and 3
        [],
        [],
        [centred(40, 10, 1)],  # its older points have aged out
    )
    outputs = []
    for number, rows in enumerate(plan):
        image = frame(number, grey=True)
        outputs.append(trace(image, tracked(image, rows)))

    assert not red_pixels(outputs[2])[10].any()
    assert red_pixels(outputs[2])[30:39, 50].all()
    assert np.flatnonzero(red_pixels(outputs[3])[10]).tolist() == list(range(20, 31))
    assert not red_pixels(outputs[4]).any()
    assert not red_pixels(outputs[6]).any()


def test_frames_without_rows_age_points_unlike_v1() -> None:
    v2, v1 = V2Trace(trace_length=3, **RED_STYLE), V1Trace(trace_length=3, **RED_STYLE)
    plan = ([centred(10, 10, 1)], [], [], [], [centred(40, 10, 1)])
    for number, rows in enumerate(plan):
        image = frame(number, grey=True)
        output, reference = v2(image, tracked(image, rows)), v1(image, rows)

    # V1 joined points four frames apart; V2 drops the frame-0 point.
    assert np.all(reference[10, 10:41] == RED)
    assert not red_pixels(output).any()


def test_interleaved_sources_never_join_each_others_points() -> None:
    shared = V2Trace(trace_length=4, **RED_STYLE)
    alone = {
        "cam-a": V2Trace(trace_length=4, **RED_STYLE),
        "cam-b": V2Trace(trace_length=4, **RED_STYLE),
    }
    x_of = {"cam-a": (8, 16, 24), "cam-b": (56, 48, 40)}
    for number in range(3):
        for video in ("cam-a", "cam-b"):
            image = frame(number, video=video, grey=True)
            rows = [centred(x_of[video][number], 20, 1)]  # same tracker id in both
            output = shared(image, tracked(image, rows))
            expected = alone[video](image, tracked(image, rows))
            np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))

    assert np.flatnonzero(red_pixels(output)[20]).tolist() == list(range(40, 57))


def test_a_lower_frame_number_restarts_only_that_source() -> None:
    trace = V2Trace(**RED_STYLE)
    for number, video in ((5, "cam-a"), (5, "cam-b"), (6, "cam-b")):
        image = frame(number, video=video, grey=True)
        trace(image, tracked(image, [centred(10 + number, 20, 1)]))

    replayed = frame(0, video="cam-a", grey=True)
    restarted = trace(replayed, tracked(replayed, [centred(30, 20, 1)]))
    later = frame(7, video="cam-b", grey=True)
    continued = trace(later, tracked(later, [centred(30, 20, 1)]))

    assert not red_pixels(restarted).any()
    assert np.flatnonzero(red_pixels(continued)[20]).tolist() == list(range(15, 31))


def test_reset_state_forgets_every_source() -> None:
    trace = V2Trace(**RED_STYLE)
    first = frame(0, grey=True)
    trace(first, tracked(first, [centred(10, 20, 1)]))

    trace.block.reset_state()
    second = frame(1, grey=True)
    output = trace(second, tracked(second, [centred(30, 20, 1)]))

    assert not red_pixels(output).any()


def test_one_source_too_many_fails_and_changes_no_history() -> None:
    trace = V2Trace(max_sources=2, **RED_STYLE)
    for video in ("cam-a", "cam-b"):
        image = frame(0, video=video, grey=True)
        trace(image, tracked(image, [centred(10, 20, 1)]))

    extra = frame(0, video="cam-c", grey=True)
    with pytest.raises(SourceCapacityError, match="max_sources=2"):
        trace(extra, tracked(extra, [centred(10, 20, 1)]))
    image = frame(1, video="cam-a", grey=True)
    output = trace(image, tracked(image, [centred(20, 20, 1)]))

    assert np.flatnonzero(red_pixels(output)[20]).tolist() == list(range(10, 21))


def test_a_crop_is_refused_with_the_remedy_and_changes_no_history() -> None:
    trace = V2Trace(**RED_STYLE)
    first = frame(0, grey=True)
    trace(first, tracked(first, [centred(10, 20, 1)]))

    crop = frame(1, grey=True).crop((0, 0, 32, 40))
    with pytest.raises(ValueError, match="Draw before cropping"):
        trace(crop, tracked(crop, [centred(15, 20, 1)]))
    second = frame(1, grey=True)
    output = trace(second, tracked(second, [centred(20, 20, 1)]))

    assert np.flatnonzero(red_pixels(output)[20]).tolist() == list(range(10, 21))


def test_a_resized_whole_frame_is_drawn_and_a_new_size_restarts_the_paths() -> None:
    trace = V2Trace(**RED_STYLE)
    halves = []
    for number, x in enumerate((5, 10)):
        half = frame(number, grey=True).resize((20, 32), interpolation="nearest")
        halves.append(trace(half, tracked(half, [centred(x, 10, 1, half=2)])))
    full = frame(2, grey=True)
    restarted = trace(full, tracked(full, [centred(30, 20, 1)]))

    assert halves[1].size_hw == (20, 32)
    assert np.flatnonzero(red_pixels(halves[1])[10]).tolist() == list(range(5, 11))
    assert not red_pixels(restarted).any()


@pytest.mark.parametrize(
    ("derive", "covers"),
    [
        (lambda image: image, True),
        (lambda image: image.resize((17, 23)), True),
        (lambda image: image.resize((80, 128)).resize((13, 7)), True),
        (lambda image: image.crop((0, 0, 64, 40)), True),
        (lambda image: image.crop((0, 0, 63, 40)), False),
        (lambda image: image.crop((1, 0, 64, 40)).resize((40, 64)), False),
    ],
)
def test_only_images_of_the_whole_input_frame_cover_it(
    derive: Any, covers: bool
) -> None:
    assert covers_root(derive(frame(0))) is covers


def test_pending_rows_are_skipped_and_untracked_rows_are_refused() -> None:
    trace = V2Trace(**RED_STYLE)
    for number, x in enumerate((10, 20)):
        image = frame(number, grey=True)
        output = trace(image, tracked(image, [centred(x, 10, -1), centred(x, 30, 3)]))

    assert not red_pixels(output)[10].any()
    assert red_pixels(output)[30, 10:21].all()
    untracked = detections(image, [(4, 4, 12, 12, 0, 0.9)])
    with pytest.raises(ValueError, match="needs tracker_id on every row"):
        trace(image, untracked)


def test_a_new_position_restarts_the_paths() -> None:
    trace = V2Trace(**RED_STYLE)
    first = frame(0, grey=True)
    trace(first, tracked(first, [centred(10, 20, 1)]))

    second = frame(1, grey=True)
    output = trace(second, tracked(second, [centred(20, 20, 1)]), position="TOP_LEFT")

    assert not red_pixels(output).any()


def test_centre_of_mass_of_filled_boxes_draws_like_center() -> None:
    by_mass, by_box = V2Trace(position="CENTER_OF_MASS", **RED_STYLE), V2Trace(
        **RED_STYLE
    )
    for number, x in enumerate((12, 22, 32)):
        image = frame(number, grey=True)
        boxes = tracked(image, [(x - 4, 16, x + 4, 24, 1)])
        masks = torch.zeros((1,) + SIZE, dtype=torch.bool)
        masks[0, 16:24, x - 4 : x + 4] = True
        segmented = InstanceDetections(
            xyxy=boxes.xyxy,
            class_id=boxes.class_id,
            confidence=boxes.confidence,
            mask=masks,
            image_metadata=boxes.image_metadata,
            bboxes_metadata=boxes.bboxes_metadata,
        )
        output = by_mass(image, segmented)
        expected = by_box(image, boxes)

    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))
    assert red_pixels(output)[20, 12:33].all()
    with pytest.raises(ValueError, match="CENTER_OF_MASS needs instance masks"):
        by_mass(image, boxes)


def test_empty_frames_return_an_unchanged_copy_and_keep_the_input() -> None:
    trace = V2Trace(**RED_STYLE)
    image = frame(0)
    before = pixels_of(image).copy()
    output = trace(image, tracked(image, []))

    assert output is not image
    assert output.tensor_image.dtype == torch.uint8
    assert output.size_hw == image.size_hw
    np.testing.assert_array_equal(pixels_of(output), before)
    np.testing.assert_array_equal(pixels_of(image), before)


@pytest.mark.parametrize(
    ("field", "value", "literal_message", "selected_message"),
    [
        ("trace_length", 0, "trace_length must be an integer >= 1", None),
        (
            "thickness",
            1.5,
            "thickness: Input should be a valid integer",
            "thickness must be an integer >= 1",
        ),
        (
            "position",
            "MIDDLE",
            "position: Input should be 'CENTER'",
            "position must be one of",
        ),
    ],
)
def test_a_selected_value_is_checked_when_it_arrives(
    field: str, value: Any, literal_message: str, selected_message: Any
) -> None:
    with pytest.raises(Exception, match=literal_message):
        V2Trace(**{field: value})
    trace = V2Trace(**{field: f"$inputs.{field}"})
    image = frame(0)

    with pytest.raises(ValueError, match=selected_message or literal_message):
        trace(image, tracked(image, []), **{field: value})


def test_v1_copy_image_is_rejected() -> None:
    with pytest.raises(Exception, match="copy_image is not a V2 parameter"):
        V2Trace(copy_image=True)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA companion: no CUDA device"
)
def test_a_cuda_image_gets_the_same_pixels_on_its_device() -> None:
    on_host, on_device = V2Trace(trace_length=3), V2Trace(trace_length=3)
    for number, rows in enumerate(MOVES):
        image = frame(number)
        cuda_image = ImageData.from_tensor(
            image.tensor_image.cuda(),
            image_id=image.image_id,
            video_metadata=image.video_metadata,
        )
        expected = on_host(image, tracked(image, rows))
        output = on_device(cuda_image, tracked(cuda_image, rows))

    assert output.tensor_image.device.type == "cuda"
    np.testing.assert_array_equal(pixels_of(output), pixels_of(expected))
