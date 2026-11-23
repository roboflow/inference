import base64
import math
import weakref
from typing import Any, List, Optional, Sequence, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from roboflow_workflows.core_steps.common import vlm_reference_examples
from roboflow_workflows.core_steps.common.vlm_reference_examples import (
    CROP_MAX_EDGE,
    CROP_MIN_EDGE,
    REFERENCE_FRONT_MARKER_RULE,
    REFERENCE_NUMBERS_LEGEND_INTRO,
    REFERENCE_ORIENTED_BOXED_INTRO,
    REFERENCE_ORIENTED_CROPS_INTRO,
    REFERENCE_ORIENTED_NUMBERED_INTRO,
    ReferencePartsCache,
    gemini_reference_parts_for,
    load_reference_example,
    openai_reference_parts_for,
    render_boxed_examples,
    render_example_crops,
    validate_reference_examples,
)

ORIENTED = "oriented-object-detection"
BACKGROUND = (160, 90, 20)  # BGR
STRIPE = (0, 0, 255)  # BGR
BOAT_COLOR = (75, 25, 230)  # BGR of the first label color
PIER_COLOR = (75, 180, 60)  # BGR of the second label color

Box = Tuple[float, float, float, float, float]


def _ramp_image(width: int, height: int, box: Box) -> np.ndarray:
    """An image with one box that is brighter toward its front and has a red stripe
    along its +height side, so a crop shows how it was turned."""
    center_x, center_y, box_width, box_height, angle = box
    ys, xs = np.mgrid[0:height, 0:width]
    dx, dy = xs + 0.5 - center_x, ys + 0.5 - center_y
    along = dx * math.cos(angle) + dy * math.sin(angle)
    across = -dx * math.sin(angle) + dy * math.cos(angle)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = BACKGROUND
    inside = (np.abs(along) <= box_width / 2) & (np.abs(across) <= box_height / 2)
    ramp = (40 + 200 * (along + box_width / 2) / box_width).clip(0, 255)
    image[inside] = ramp[inside, None].astype(np.uint8)
    image[inside & (across > box_height / 2 - 10)] = STRIPE
    return image


def _numpy(pixels: np.ndarray) -> dict:
    return {"type": "numpy_object", "value": pixels}


def _base64(pixels: np.ndarray) -> dict:
    _, encoded = cv2.imencode(".png", pixels)
    return {"type": "base64", "value": base64.b64encode(encoded.tobytes()).decode()}


def _white(width: int = 200, height: int = 100) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _example(
    pixels: Optional[np.ndarray] = None,
    annotations: Optional[Sequence[dict]] = None,
    image: Any = None,
) -> dict:
    if image is None:
        image = _numpy(_white() if pixels is None else pixels)
    return {
        "image": image,
        "annotations": (
            list(annotations)
            if annotations is not None
            else [{"class": "boat", "obb": [100, 50, 60, 20, 0.5]}]
        ),
    }


def _decode(data: str) -> np.ndarray:
    return cv2.imdecode(
        np.frombuffer(base64.b64decode(data), dtype=np.uint8), cv2.IMREAD_COLOR
    )


def _crops_of(
    pixels: np.ndarray,
    box: Box,
    *,
    has_front: bool,
    label: str = "boat",
) -> np.ndarray:
    examples = validate_reference_examples(
        [_example(pixels, [{"class": label, "obb": list(box)}])],
        task_type=ORIENTED,
        labels=[label],
    )
    groups = render_example_crops(
        examples,
        labels=[label],
        task_type=ORIENTED,
        class_heading={label: has_front},
    )
    return _decode(groups[0][1][0])


def _has_color_near(
    image: np.ndarray,
    point: Tuple[float, float],
    color: Tuple[int, int, int],
    *,
    radius: int = 3,
    tolerance: int = 70,
) -> bool:
    x, y = round(point[0]), round(point[1])
    window = image[
        max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1
    ]
    close = np.abs(window.astype(int) - np.array(color)).max(axis=2) <= tolerance
    return bool(close.any())


def _is_white(image: np.ndarray, point: Tuple[float, float]) -> bool:
    x, y = round(point[0]), round(point[1])
    return int(image[y, x].min()) >= 235


def test_validate_reference_examples_fills_the_axis_aligned_box_of_an_obb() -> None:
    # given
    annotation = {"class": "boat", "obb": [100, 50, 60, 20, math.radians(30)]}

    # when
    result = validate_reference_examples(
        [_example(annotations=[annotation])], task_type=ORIENTED, labels=["boat"]
    )

    # then
    box = result[0].annotations[0]
    assert box.obb == pytest.approx([100, 50, 60, 20, math.radians(30)])
    assert (box.x, box.y) == (100, 50)
    assert box.width == pytest.approx(60 * math.cos(math.radians(30)) + 10)
    assert box.height == pytest.approx(30 + 20 * math.cos(math.radians(30)))


def test_validate_reference_examples_keeps_the_axis_aligned_box_it_is_given() -> None:
    # given
    annotation = {
        "class": "boat",
        "x": 90,
        "y": 40,
        "width": 70,
        "height": 30,
        "obb": [100, 50, 60, 20, 0.5],
    }

    # when
    result = validate_reference_examples(
        [_example(annotations=[annotation])], task_type=ORIENTED, labels=["boat"]
    )

    # then
    box = result[0].annotations[0]
    assert (box.x, box.y, box.width, box.height) == (90, 40, 70, 30)
    assert box.obb == [100, 50, 60, 20, 0.5]


def test_validate_reference_examples_reads_an_oriented_prediction() -> None:
    # given
    prediction = {
        "x": 100,
        "y": 50,
        "width": 62,
        "height": 47,
        "class": "boat",
        "class_id": 0,
        "confidence": 0.9,
        "obb": [100, 50, 60, 20, 0.5],
    }

    # when
    result = validate_reference_examples(
        [_example(annotations=[prediction])], task_type=ORIENTED, labels=["boat"]
    )

    # then
    assert result[0].annotations[0].obb == [100, 50, 60, 20, 0.5]


def test_validate_reference_examples_requires_an_obb_for_the_oriented_task() -> None:
    # given
    annotations = [
        {"class": "boat", "obb": [100, 50, 60, 20, 0.5]},
        {"class": "boat", "x": 50, "y": 50, "width": 40, "height": 20},
    ]

    # when
    with pytest.raises(ValueError) as error:
        validate_reference_examples(
            [_example(annotations=annotations)], task_type=ORIENTED, labels=["boat"]
        )

    # then
    assert "reference_examples[0]" in str(error.value)
    assert "needs an `obb` on every annotation" in str(error.value)


def test_validate_reference_examples_does_not_need_an_obb_for_detection() -> None:
    # given
    annotation = {"class": "boat", "x": 50, "y": 50, "width": 40, "height": 20}

    # when
    result = validate_reference_examples(
        [_example(annotations=[annotation])],
        task_type="object-detection",
        labels=["boat"],
    )

    # then
    assert result[0].annotations[0].obb is None


@pytest.mark.parametrize(
    "obb",
    [
        [100, 50, 60, 20],
        [100, 50, 60, 20, 0.5, 1],
        "abcde",
        {"cx": 1},
        [100, 50, 0, 20, 0.5],
        [100, 50, 60, -1, 0.5],
        [100, 50, float("nan"), 20, 0.5],
        [100, 50, 60, 20, float("inf")],
        [100, 50, 60, 20, "wide"],
    ],
)
@pytest.mark.parametrize("task_type", [ORIENTED, "object-detection"])
def test_validate_reference_examples_names_a_malformed_obb(
    obb: Any, task_type: str
) -> None:
    # given
    annotation = {"class": "boat", "obb": obb}

    # when
    with pytest.raises(ValueError) as error:
        validate_reference_examples(
            [_example(annotations=[annotation])], task_type=task_type, labels=["boat"]
        )

    # then
    message = str(error.value)
    assert "reference_examples[0]" in message
    assert "annotations.0.obb" in message
    assert "Field required" not in message


def test_validate_reference_examples_rejects_labels_the_block_does_not_have() -> None:
    with pytest.raises(ValueError, match=r"does not have: \['boat'\]"):
        validate_reference_examples([_example()], task_type=ORIENTED, labels=["pier"])


def test_validate_reference_examples_names_both_supported_tasks() -> None:
    with pytest.raises(
        ValueError, match="object-detection and oriented-object-detection tasks"
    ):
        validate_reference_examples(
            [_example()], task_type="instance-segmentation", labels=["boat"]
        )


def test_load_reference_example_keeps_an_oriented_box_whole() -> None:
    # given
    examples = validate_reference_examples(
        [_example(annotations=[{"class": "boat", "obb": [190, 10, 60, 20, 0.5]}])],
        task_type=ORIENTED,
        labels=["boat"],
    )

    # when
    loaded = load_reference_example(examples[0], index=0, task_type=ORIENTED)

    # then
    assert loaded.image.shape == (100, 200, 3)
    assert loaded.boxes == ()
    assert loaded.oriented_boxes == (("boat", 190.0, 10.0, 60.0, 20.0, 0.5),)


def test_load_reference_example_reads_the_axis_aligned_box_for_detection() -> None:
    # given
    annotation = {
        "class": "boat",
        "x": 50,
        "y": 50,
        "width": 40,
        "height": 20,
        "obb": [150, 50, 60, 20, 0.5],
    }
    examples = validate_reference_examples(
        [_example(annotations=[annotation])],
        task_type="object-detection",
        labels=["boat"],
    )

    # when
    loaded = load_reference_example(examples[0], index=0)

    # then
    assert loaded.boxes == (("boat", 30.0, 40.0, 70.0, 60.0),)
    assert loaded.oriented_boxes == ()


@pytest.mark.parametrize(
    "obb",
    [
        [500, 50, 40, 20, 0.5],
        [-30, 50, 40, 20, 0.0],
        [100, 160, 40, 20, 1.0],
        # Inside the corner of its axis-aligned box, apart from the image.
        [-12, -12, 120, 6, math.radians(45) + math.pi / 2],
    ],
)
def test_load_reference_example_rejects_an_oriented_box_outside_its_image(
    obb: List[float],
) -> None:
    # given
    examples = validate_reference_examples(
        [_example(annotations=[{"class": "boat", "obb": obb}])],
        task_type=ORIENTED,
        labels=["boat"],
    )

    # when / then
    with pytest.raises(ValueError, match="outside its 200x100 image"):
        load_reference_example(examples[0], index=0, task_type=ORIENTED)


def test_load_reference_example_accepts_a_box_that_only_crosses_the_image() -> None:
    # given
    examples = validate_reference_examples(
        [_example(annotations=[{"class": "boat", "obb": [-20, 50, 100, 8, 0.0]}])],
        task_type=ORIENTED,
        labels=["boat"],
    )

    # when
    loaded = load_reference_example(examples[0], index=0, task_type=ORIENTED)

    # then
    assert len(loaded.oriented_boxes) == 1


@pytest.mark.parametrize("angle", [0, 30, 90, 135, 180, 225, -60, -90, 270, 720 + 15])
def test_render_example_crops_turns_an_oriented_box_upright_with_its_front_right(
    angle: float,
) -> None:
    # given
    box = (200.0, 150.0, 160.0, 60.0, math.radians(angle))
    reference = _crops_of(
        _ramp_image(400, 300, (200.0, 150.0, 160.0, 60.0, 0.0)),
        (200.0, 150.0, 160.0, 60.0, 0.0),
        has_front=True,
    )

    # when
    crop = _crops_of(_ramp_image(400, 300, box), box, has_front=True)

    # then
    assert crop.shape == reference.shape == (78, 208, 3)
    assert np.abs(crop.astype(int) - reference.astype(int)).mean() < 6
    # Darker on the left, brighter on the right, and the stripe along the bottom.
    row = 20
    assert crop[row, 178].mean() > crop[row, 30].mean() + 100
    assert tuple(int(v) for v in crop[66, 100]) == pytest.approx(STRIPE, abs=40)
    assert tuple(int(v) for v in crop[3, 3]) == pytest.approx(BACKGROUND, abs=12)


def test_render_example_crops_turns_the_long_side_of_a_class_without_a_front() -> None:
    # given
    box = (200.0, 150.0, 60.0, 160.0, math.radians(30))
    pixels = _ramp_image(400, 300, box)

    # when
    without_front = _crops_of(pixels, box, has_front=False)
    with_front = _crops_of(pixels, box, has_front=True)

    # then
    assert without_front.shape == (78, 208, 3)
    assert with_front.shape == (208, 78, 3)


def test_render_example_crops_fills_what_is_outside_the_image_with_gray() -> None:
    # given
    box = (20.0, 30.0, 120.0, 60.0, math.radians(-20))
    pixels = _ramp_image(200, 150, box)

    # when
    crop = _crops_of(pixels, box, has_front=True)

    # then
    assert crop.shape == (78, 156, 3)
    assert tuple(int(v) for v in crop[2, 2]) == pytest.approx((114,) * 3, abs=6)
    # The inside of the image is still there, with its own background.
    inside_corner = crop[crop.shape[0] - 3, crop.shape[1] - 3]
    assert tuple(int(v) for v in inside_corner) == pytest.approx(BACKGROUND, abs=12)


@pytest.mark.parametrize(
    "box, expected_longest_edge",
    [
        ((100.0, 80.0, 24.0, 10.0, math.radians(75)), CROP_MIN_EDGE),
        ((1500.0, 1000.0, 1600.0, 900.0, math.radians(-35)), CROP_MAX_EDGE),
    ],
)
def test_render_example_crops_brings_an_oriented_crop_within_the_edge_limits(
    box: Box, expected_longest_edge: int
) -> None:
    # given
    pixels = np.zeros((2000, 3000, 3), dtype=np.uint8)

    # when
    crop = _crops_of(pixels, box, has_front=True)

    # then
    assert max(crop.shape[:2]) == expected_longest_edge


def test_render_example_crops_reads_only_the_part_of_the_image_around_the_box() -> None:
    # given
    pixels = np.zeros((2000, 3000, 3), dtype=np.uint8)
    small = (1500.0, 1000.0, 30.0, 12.0, math.radians(20))
    large = (1500.0, 1000.0, 1600.0, 900.0, math.radians(-35))

    # when
    with patch.object(
        vlm_reference_examples.cv2, "warpAffine", wraps=cv2.warpAffine
    ) as warp:
        _crops_of(pixels, small, has_front=True)
        _crops_of(pixels, large, has_front=True)

    # then
    small_source, large_source = (call.args[0] for call in warp.call_args_list)
    assert max(small_source.shape[:2]) < 100
    assert max(large_source.shape[:2]) < 1000


def test_render_example_crops_ranks_oriented_boxes_by_their_own_area() -> None:
    # given
    # The thin diagonal box has the largest axis-aligned box and the smallest area.
    annotations = [
        {"class": "boat", "obb": [60, 150, 30, 30, 0.0]},
        {"class": "boat", "obb": [120, 150, 34, 34, 0.0]},
        {"class": "boat", "obb": [180, 150, 38, 38, 0.0]},
        {"class": "boat", "obb": [200, 100, 200, 4, math.radians(45)]},
    ]
    examples = validate_reference_examples(
        [_example(_white(400, 300), annotations)], task_type=ORIENTED, labels=["boat"]
    )

    # when
    groups = render_example_crops(examples, labels=["boat"], task_type=ORIENTED)

    # then
    assert [label for label, _ in groups] == ["boat"]
    shapes = [_decode(crop).shape[:2] for crop in groups[0][1]]
    assert shapes == [(CROP_MIN_EDGE, CROP_MIN_EDGE)] * 3


def test_render_example_crops_puts_the_oriented_boxes_of_the_largest_area_first() -> (
    None
):
    # given
    annotations = [
        {"class": "boat", "obb": [100, 100, 40, 40, 0.0]},
        {"class": "boat", "obb": [300, 100, 160, 100, 0.0]},
        {"class": "boat", "obb": [100, 250, 100, 40, 0.0]},
    ]
    examples = validate_reference_examples(
        [_example(_white(400, 300), annotations)], task_type=ORIENTED, labels=["boat"]
    )

    # when
    groups = render_example_crops(examples, labels=["boat"], task_type=ORIENTED)

    # then
    shapes = [_decode(crop).shape[:2] for crop in groups[0][1]]
    # 160x100, 100x40 and 40x40 with margins of 15 % and at least 8 pixels each side;
    # the smallest is scaled up to the minimum edge.
    assert shapes == [(130, 208), (56, 130), (CROP_MIN_EDGE, CROP_MIN_EDGE)]


@pytest.mark.parametrize("render", [render_boxed_examples, render_example_crops])
def test_oriented_rendering_releases_each_example_image_before_the_next_one_loads(
    render: Any,
) -> None:
    # given
    examples = validate_reference_examples(
        [_example(image=_base64(_white())) for _ in range(3)],
        task_type=ORIENTED,
        labels=["boat"],
    )
    loaded = []
    alive_at_each_load = []
    load = vlm_reference_examples.load_reference_example

    def counting_load(example: Any, **kwargs: Any) -> Any:
        alive_at_each_load.append(sum(image() is not None for image in loaded))
        result = load(example, **kwargs)
        loaded.append(weakref.ref(result.image))
        return result

    # when
    with patch.object(
        vlm_reference_examples, "load_reference_example", new=counting_load
    ):
        render(examples, labels=["boat"], task_type=ORIENTED)

    # then
    assert alive_at_each_load == [0, 0, 0]


def _corner_points(box: Box) -> List[Tuple[float, float]]:
    center_x, center_y, width, height, angle = box
    heading = (math.cos(angle), math.sin(angle))
    across = (-math.sin(angle), math.cos(angle))
    return [
        (
            center_x + a * width / 2 * heading[0] + b * height / 2 * across[0],
            center_y + a * width / 2 * heading[1] + b * height / 2 * across[1],
        )
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]


def test_render_boxed_examples_draws_the_oriented_outline_in_the_label_color() -> None:
    # given
    boat = (130.0, 150.0, 140.0, 50.0, math.radians(30))
    pier = (300.0, 80.0, 90.0, 30.0, math.radians(-70))
    annotations = [
        {"class": "boat", "obb": list(boat)},
        {"class": "pier", "obb": list(pier)},
    ]
    examples = validate_reference_examples(
        [_example(_white(400, 300), annotations)],
        task_type=ORIENTED,
        labels=["boat", "pier"],
    )

    # when
    rendered = render_boxed_examples(
        examples, labels=["boat", "pier"], task_type=ORIENTED
    )

    # then
    image = _decode(rendered[0])
    assert image.shape == (300, 400, 3)
    for box, color in ((boat, BOAT_COLOR), (pier, PIER_COLOR)):
        corners = _corner_points(box)
        for start, end in zip(corners, corners[1:] + corners[:1]):
            middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            assert _has_color_near(image, middle, color)
        # The inside of the box and the corners of its axis-aligned box stay clear.
        assert _is_white(image, (box[0], box[1]))
        left = min(x for x, _ in corners)
        top = min(y for _, y in corners)
        right = max(x for x, _ in corners)
        bottom = max(y for _, y in corners)
        for corner in ((left, top), (right, top), (left, bottom), (right, bottom)):
            inward_x = 3 if corner[0] == left else -3
            inward_y = 3 if corner[1] == top else -3
            assert _is_white(image, (corner[0] + inward_x, corner[1] + inward_y))


def test_render_boxed_examples_marks_the_front_of_a_class_with_a_front() -> None:
    # given
    boat = (150.0, 150.0, 140.0, 50.0, math.radians(30))
    examples = validate_reference_examples(
        [_example(_white(400, 300), [{"class": "boat", "obb": list(boat)}])],
        task_type=ORIENTED,
        labels=["boat"],
    )
    heading = (math.cos(boat[4]), math.sin(boat[4]))
    beyond_the_front = (
        boat[0] + (boat[2] / 2 + 5) * heading[0],
        boat[1] + (boat[2] / 2 + 5) * heading[1],
    )
    beyond_the_back = (
        boat[0] - (boat[2] / 2 + 5) * heading[0],
        boat[1] - (boat[2] / 2 + 5) * heading[1],
    )

    # when
    with_front = _decode(
        render_boxed_examples(
            examples,
            labels=["boat"],
            task_type=ORIENTED,
            class_heading={"boat": True},
        )[0]
    )
    without_front = _decode(
        render_boxed_examples(
            examples,
            labels=["boat"],
            task_type=ORIENTED,
            class_heading={"boat": False},
        )[0]
    )
    unknown = _decode(
        render_boxed_examples(examples, labels=["boat"], task_type=ORIENTED)[0]
    )

    # then
    assert _has_color_near(with_front, beyond_the_front, BOAT_COLOR, radius=1)
    assert _is_white(with_front, beyond_the_back)
    assert _is_white(without_front, beyond_the_front)
    assert _is_white(unknown, beyond_the_front)


def test_gemini_reference_parts_describe_oriented_boxes() -> None:
    # when
    parts = gemini_reference_parts_for(
        [_example()], labels=["boat"], task_type=ORIENTED
    )

    # then
    assert parts[0] == {"text": REFERENCE_ORIENTED_BOXED_INTRO}
    assert parts[1] == {"text": "Reference image 1:"}
    assert parts[2]["inline_data"]["mime_type"] == "image/jpeg"
    assert len(parts) == 3


def test_gemini_reference_parts_explain_the_front_marker_only_when_one_is_drawn() -> (
    None
):
    # given
    annotations = [{"class": "boat", "obb": [100, 50, 60, 20, 0.5]}]
    labels = ["boat", "pier"]

    # when
    marked = gemini_reference_parts_for(
        [_example(annotations=annotations)],
        labels=labels,
        task_type=ORIENTED,
        class_heading={"boat": True, "pier": False},
    )
    unmarked = gemini_reference_parts_for(
        [_example(annotations=annotations)],
        labels=labels,
        task_type=ORIENTED,
        class_heading={"boat": False, "pier": True},
    )

    # then
    assert marked[0] == {
        "text": REFERENCE_ORIENTED_BOXED_INTRO + REFERENCE_FRONT_MARKER_RULE
    }
    assert unmarked[0] == {"text": REFERENCE_ORIENTED_BOXED_INTRO}


def test_gemini_reference_parts_number_the_oriented_labels_outside_ascii() -> None:
    # given
    annotations = [
        {"class": "猫", "obb": [100, 50, 60, 20, 0.5]},
        {"class": "cat", "obb": [50, 40, 30, 10, 0.1]},
    ]

    # when
    parts = gemini_reference_parts_for(
        [_example(annotations=annotations)],
        labels=["cat", "猫"],
        task_type=ORIENTED,
        class_heading={"猫": True},
    )

    # then
    assert parts[0] == {
        "text": (
            f"{REFERENCE_ORIENTED_NUMBERED_INTRO}{REFERENCE_FRONT_MARKER_RULE}"
            f'{REFERENCE_NUMBERS_LEGEND_INTRO}\n1: "cat"\n2: "猫"'
        )
    }


def test_openai_reference_parts_describe_the_upright_oriented_crops() -> None:
    # given
    annotations = [
        {"class": "boat", "obb": [100, 50, 60, 20, 0.5]},
        {"class": "pier", "obb": [50, 40, 30, 10, 0.1]},
    ]

    # when
    parts = openai_reference_parts_for(
        [_example(annotations=annotations)],
        labels=["boat", "pier"],
        task_type=ORIENTED,
        class_heading={"boat": True},
    )

    # then
    assert parts[0] == {"type": "input_text", "text": REFERENCE_ORIENTED_CROPS_INTRO}
    assert parts[1] == {"type": "input_text", "text": 'Examples of "boat":'}
    assert parts[2]["type"] == "input_image"
    assert parts[3] == {"type": "input_text", "text": 'Examples of "pier":'}
    assert len(parts) == 5


def _oriented_examples() -> List[dict]:
    pixels = _white(300, 200)
    return [
        {
            "image": _base64(pixels),
            "annotations": [
                {
                    "class": "boat",
                    "x": 100,
                    "y": 80,
                    "width": 60,
                    "height": 40,
                    "obb": [100, 80, 60, 20, 0.5],
                }
            ],
        }
    ]


@pytest.mark.parametrize(
    "parts_for", [gemini_reference_parts_for, openai_reference_parts_for]
)
def test_oriented_reference_parts_are_cached_apart_from_detection_parts(
    parts_for: Any,
) -> None:
    # given
    examples = _oriented_examples()
    cache = ReferencePartsCache()

    # when
    with patch.object(
        vlm_reference_examples,
        "load_image",
        side_effect=vlm_reference_examples.load_image,
    ) as load_image:
        detection = parts_for(examples, labels=["boat"], cache=cache)
        oriented = parts_for(examples, labels=["boat"], task_type=ORIENTED, cache=cache)
        again = parts_for(examples, labels=["boat"], task_type=ORIENTED, cache=cache)
        detection_again = parts_for(examples, labels=["boat"], cache=cache)

    # then
    assert load_image.call_count == 2
    assert oriented is again
    assert detection is detection_again
    assert oriented[0] != detection[0]


@pytest.mark.parametrize(
    "parts_for", [gemini_reference_parts_for, openai_reference_parts_for]
)
def test_oriented_reference_parts_are_built_again_for_another_class_heading(
    parts_for: Any,
) -> None:
    # given
    examples = _oriented_examples()
    cache = ReferencePartsCache()

    # when
    with patch.object(
        vlm_reference_examples,
        "load_image",
        side_effect=vlm_reference_examples.load_image,
    ) as load_image:
        first = parts_for(
            examples,
            labels=["boat"],
            task_type=ORIENTED,
            class_heading={"boat": True},
            cache=cache,
        )
        same = parts_for(
            examples,
            labels=["boat"],
            task_type=ORIENTED,
            class_heading={"boat": True},
            cache=cache,
        )
        other = parts_for(
            examples,
            labels=["boat"],
            task_type=ORIENTED,
            class_heading={"boat": False},
            cache=cache,
        )

    # then
    assert load_image.call_count == 2
    assert same is first
    assert other is not first


def test_oriented_reference_parts_are_built_again_for_another_box() -> None:
    # given
    examples = _oriented_examples()
    turned = _oriented_examples()
    turned[0]["annotations"][0]["obb"] = [100, 80, 60, 20, 1.5]
    cache = ReferencePartsCache()

    # when
    with patch.object(
        vlm_reference_examples,
        "load_image",
        side_effect=vlm_reference_examples.load_image,
    ) as load_image:
        gemini_reference_parts_for(
            examples, labels=["boat"], task_type=ORIENTED, cache=cache
        )
        gemini_reference_parts_for(
            turned, labels=["boat"], task_type=ORIENTED, cache=cache
        )

    # then
    assert load_image.call_count == 2


def test_the_class_heading_does_not_change_the_cache_key_of_detection() -> None:
    # given
    examples = _oriented_examples()

    # when
    plain = vlm_reference_examples._cache_key(
        examples, labels=["boat"], rendering="gemini"
    )
    with_heading = vlm_reference_examples._cache_key(
        examples,
        labels=["boat"],
        rendering="gemini",
        task_type="object-detection",
        class_heading={"boat": True},
    )
    oriented = vlm_reference_examples._cache_key(
        examples,
        labels=["boat"],
        rendering="gemini",
        task_type=ORIENTED,
        class_heading={"boat": True},
    )

    # then
    assert plain == with_heading
    assert oriented != plain
