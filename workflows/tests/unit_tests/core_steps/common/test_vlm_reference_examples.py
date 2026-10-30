import base64
from typing import List, Optional, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from roboflow_workflows.core_steps.common import vlm_reference_examples
from roboflow_workflows.core_steps.common.vlm_reference_examples import (
    CROP_MAX_EDGE,
    CROP_MIN_EDGE,
    REFERENCE_BOXED_INTRO,
    REFERENCE_CROPS_INTRO,
    TARGET_IMAGE_INTRO,
    TARGET_ONLY_RULE,
    gemini_reference_parts_for,
    load_reference_examples,
    openai_reference_parts_for,
    render_boxed_examples,
    render_example_crops,
    validate_reference_examples,
    with_reference_parts,
)


@pytest.fixture(autouse=True)
def clear_reference_parts_cache() -> None:
    vlm_reference_examples._cached_parts.cache_clear()


def _image_base64(
    width: int, height: int, color: Tuple[int, int, int] = (255, 255, 255)
) -> str:
    image = np.full((height, width, 3), color, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", image)
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _example(
    width: int = 200,
    height: int = 100,
    boxes: Optional[List[dict]] = None,
) -> dict:
    return {
        "image": {"type": "base64", "value": _image_base64(width, height)},
        "boxes": (
            boxes
            if boxes is not None
            else [{"x": 50, "y": 50, "width": 40, "height": 20, "class": "cat"}]
        ),
    }


def _decode(data: str) -> np.ndarray:
    return cv2.imdecode(
        np.frombuffer(base64.b64decode(data), dtype=np.uint8), cv2.IMREAD_COLOR
    )


@pytest.mark.parametrize("value", [None, []])
def test_validate_reference_examples_treats_missing_examples_as_none(value) -> None:
    # when
    result = validate_reference_examples(
        value, task_type="object-detection", labels=["cat"]
    )

    # then
    assert result == []


def test_validate_reference_examples_reads_roboflow_prediction_boxes() -> None:
    # when
    result = validate_reference_examples(
        [_example()], task_type="object-detection", labels=["cat"]
    )

    # then
    assert len(result) == 1
    box = result[0].boxes[0]
    assert (box.x, box.y, box.width, box.height, box.class_name) == (
        50,
        50,
        40,
        20,
        "cat",
    )


def test_validate_reference_examples_rejects_more_than_ten_examples() -> None:
    with pytest.raises(ValueError, match="at most 10 examples, got 11"):
        validate_reference_examples(
            [_example()] * 11, task_type="object-detection", labels=["cat"]
        )


def test_validate_reference_examples_rejects_other_tasks() -> None:
    with pytest.raises(ValueError, match="only supported for the object-detection"):
        validate_reference_examples(
            [_example()], task_type="classification", labels=["cat"]
        )


def test_validate_reference_examples_rejects_labels_the_block_does_not_have() -> None:
    with pytest.raises(ValueError, match=r"does not have: \['cat'\]"):
        validate_reference_examples(
            [_example()], task_type="object-detection", labels=["dog"]
        )


def test_validate_reference_examples_skips_the_label_check_for_unknown_labels() -> None:
    # when
    result = validate_reference_examples(
        [_example()], task_type="object-detection", labels=None
    )

    # then
    assert len(result) == 1


def test_validate_reference_examples_names_the_malformed_field() -> None:
    # given
    example = _example(boxes=[{"x": 50, "y": 50, "height": 20, "class": "cat"}])

    # when
    with pytest.raises(ValueError) as error:
        validate_reference_examples(
            [example], task_type="object-detection", labels=["cat"]
        )

    # then
    assert "reference_examples[0]" in str(error.value)
    assert "boxes.0.width" in str(error.value)


def test_load_reference_examples_clips_boxes_to_the_image() -> None:
    # given
    examples = validate_reference_examples(
        [
            _example(
                boxes=[{"x": 190, "y": 10, "width": 40, "height": 40, "class": "cat"}]
            )
        ],
        task_type="object-detection",
        labels=["cat"],
    )

    # when
    loaded = load_reference_examples(examples)

    # then
    assert loaded[0].image.shape == (100, 200, 3)
    assert loaded[0].boxes == (("cat", 170.0, 0.0, 200.0, 30.0),)


def test_load_reference_examples_rejects_a_box_outside_its_image() -> None:
    # given
    examples = validate_reference_examples(
        [
            _example(
                boxes=[{"x": 500, "y": 50, "width": 40, "height": 20, "class": "cat"}]
            )
        ],
        task_type="object-detection",
        labels=["cat"],
    )

    # when / then
    with pytest.raises(ValueError, match="outside its 200x100 image"):
        load_reference_examples(examples)


def test_render_example_crops_keeps_the_largest_boxes_in_label_order() -> None:
    # given
    boxes = [
        {"x": 30, "y": 30, "width": 10, "height": 10, "class": "cat"},
        {"x": 100, "y": 50, "width": 60, "height": 40, "class": "cat"},
        {"x": 150, "y": 50, "width": 30, "height": 30, "class": "cat"},
        {"x": 50, "y": 70, "width": 20, "height": 20, "class": "cat"},
        {"x": 180, "y": 20, "width": 20, "height": 10, "class": "dog"},
    ]
    examples = load_reference_examples(
        validate_reference_examples(
            [_example(boxes=boxes)],
            task_type="object-detection",
            labels=["dog", "cat", "bird"],
        )
    )

    # when
    groups = render_example_crops(examples, labels=["dog", "cat", "bird"])

    # then
    assert [label for label, _ in groups] == ["dog", "cat"]
    cat_crops = [_decode(crop) for crop in groups[1][1]]
    assert len(cat_crops) == 3
    # The 60x40 box with its margin is the widest crop, upscaled to the minimum edge.
    assert cat_crops[0].shape[1] == CROP_MIN_EDGE
    for crop in cat_crops:
        assert CROP_MIN_EDGE <= max(crop.shape[:2]) <= CROP_MAX_EDGE


def test_render_boxed_examples_downscales_and_draws_in_the_label_color() -> None:
    # given
    examples = load_reference_examples(
        validate_reference_examples(
            [
                _example(
                    width=2048,
                    height=1024,
                    boxes=[
                        {
                            "x": 1024,
                            "y": 512,
                            "width": 800,
                            "height": 400,
                            "class": "dog",
                        }
                    ],
                )
            ],
            task_type="object-detection",
            labels=["cat", "dog"],
        )
    )

    # when
    rendered = render_boxed_examples(examples, labels=["cat", "dog"])

    # then
    image = _decode(rendered[0])
    assert image.shape == (512, 1024, 3)
    # The second label takes the second color, green (60, 180, 75) in RGB; the
    # right edge of the box sits at x = 1424 / 2 in the downscaled image.
    blue, green, red = (int(v) for v in image[256, 711])
    assert green > 150 and red < 110 and blue < 110
    assert min(int(v) for v in image[256, 512]) >= 250


def test_gemini_reference_parts_number_the_boxed_images() -> None:
    # when
    parts = gemini_reference_parts_for([_example(), _example()], labels=["cat"])

    # then
    assert parts[0] == {"text": REFERENCE_BOXED_INTRO}
    assert parts[1] == {"text": "Reference image 1:"}
    assert parts[2]["inline_data"]["mime_type"] == "image/jpeg"
    assert parts[3] == {"text": "Reference image 2:"}
    assert len(parts) == 5


def test_openai_reference_parts_group_crops_by_label() -> None:
    # given
    boxes = [
        {"x": 50, "y": 50, "width": 40, "height": 20, "class": "cat"},
        {"x": 150, "y": 50, "width": 40, "height": 20, "class": "dog"},
    ]

    # when
    parts = openai_reference_parts_for([_example(boxes=boxes)], labels=["cat", "dog"])

    # then
    assert parts[0] == {"type": "input_text", "text": REFERENCE_CROPS_INTRO}
    assert parts[1] == {"type": "input_text", "text": 'Examples of "cat":'}
    assert parts[2]["type"] == "input_image"
    assert parts[2]["image_url"].startswith("data:image/jpeg;base64,")
    assert parts[3] == {"type": "input_text", "text": 'Examples of "dog":'}
    assert len(parts) == 5


def test_reference_parts_are_built_once_for_the_same_examples() -> None:
    # given
    examples = [_example()]
    real_load_image = vlm_reference_examples.load_image

    # when
    with patch.object(
        vlm_reference_examples, "load_image", side_effect=real_load_image
    ) as load_image:
        first = gemini_reference_parts_for(examples, labels=["cat"])
        second = gemini_reference_parts_for(examples, labels=["cat"])

    # then
    assert first == second
    assert load_image.call_count == 1


def test_with_reference_parts_puts_references_first_and_points_at_the_target() -> None:
    # given
    image_part = {"inline_data": {"mime_type": "image/jpeg", "data": "target"}}
    parts = [
        image_part,
        {"text": "Detect all objects in this image. Only use these labels: cat"},
    ]
    reference_parts = [{"text": "references"}]

    # when
    result = with_reference_parts(
        parts,
        reference_parts=reference_parts,
        target_intro={"text": TARGET_IMAGE_INTRO},
        is_image_part=lambda part: "inline_data" in part,
    )

    # then
    assert result == [
        {"text": "references"},
        {"text": TARGET_IMAGE_INTRO},
        image_part,
        {
            "text": "Detect all objects in the target image. Only use these labels: cat"
            + TARGET_ONLY_RULE
        },
    ]
    assert parts[1]["text"].startswith("Detect all objects in this image.")


def test_with_reference_parts_keeps_text_before_the_image_in_place() -> None:
    # given
    parts = [
        {"type": "input_text", "text": "List of all classes: cat"},
        {"type": "input_image", "image_url": "data:image/png;base64,target"},
    ]

    # when
    result = with_reference_parts(
        parts,
        reference_parts=[],
        target_intro={"type": "input_text", "text": TARGET_IMAGE_INTRO},
        is_image_part=lambda part: part.get("type") == "input_image",
    )

    # then
    assert result == [
        {"type": "input_text", "text": "List of all classes: cat" + TARGET_ONLY_RULE},
        {"type": "input_text", "text": TARGET_IMAGE_INTRO},
        parts[1],
    ]
