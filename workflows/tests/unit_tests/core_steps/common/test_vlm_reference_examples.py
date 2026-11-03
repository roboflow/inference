import base64
from pathlib import Path
from typing import Any, List, Optional, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from roboflow_workflows.core_steps.common import vlm_reference_examples
from roboflow_workflows.core_steps.common.vlm_reference_examples import (
    CROP_MAX_EDGE,
    CROP_MIN_EDGE,
    MAX_REFERENCE_EXAMPLES,
    REFERENCE_BOXED_INTRO,
    REFERENCE_CROPS_INTRO,
    TARGET_IMAGE_INTRO,
    TARGET_ONLY_RULE,
    ReferencePartsCache,
    gemini_reference_parts_for,
    load_reference_examples,
    openai_reference_parts_for,
    render_boxed_examples,
    render_example_crops,
    validate_reference_examples,
    with_reference_parts,
)
from roboflow_workflows.core_steps.models.foundation.google_gemini.v6 import (
    BlockManifest as GeminiManifest,
)
from roboflow_workflows.core_steps.models.foundation.openai.v7 import (
    BlockManifest as OpenAIManifest,
)
from roboflow_workflows.errors import RuntimeInputError
from roboflow_workflows.prototypes.image_codec import (
    WorkflowsLocalImageCodec,
    reset_image_codec,
    set_image_codec,
)


class _ServerLikeCodec(WorkflowsLocalImageCodec):
    """Reads local files and serves URLs, like the server's codec with its defaults."""

    def __init__(self) -> None:
        self.url_images: List[np.ndarray] = []
        self.fetched_urls: List[str] = []
        self.local_reads: List[str] = []

    def load_image(
        self, value: Any, disable_preproc_auto_orient: bool = False
    ) -> Tuple[np.ndarray, bool]:
        path = value["value"] if isinstance(value, dict) else value
        is_file_type = isinstance(value, dict) and value.get("type") == "file"
        if is_file_type or (isinstance(value, str) and Path(path).is_file()):
            self.local_reads.append(path)
            return cv2.imread(path), True
        return super().load_image(
            value, disable_preproc_auto_orient=disable_preproc_auto_orient
        )

    def fetch_url(self, value: str, cv_imread_flags: int = cv2.IMREAD_COLOR):
        self.fetched_urls.append(value)
        return self.url_images.pop(0)

    def ensure_local_file_load_allowed(self, path: str) -> None:
        self.local_reads.append(path)


@pytest.fixture
def server_like_codec():
    codec = _ServerLikeCodec()
    reset_image_codec()
    set_image_codec(codec)
    yield codec
    reset_image_codec()


def _image_base64(
    width: int, height: int, color: Tuple[int, int, int] = (255, 255, 255)
) -> str:
    image = np.full((height, width, 3), color, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", image)
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _example(
    width: int = 200,
    height: int = 100,
    annotations: Optional[List[dict]] = None,
) -> dict:
    return {
        "image": {"type": "base64", "value": _image_base64(width, height)},
        "annotations": (
            annotations
            if annotations is not None
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


def test_validate_reference_examples_reads_a_whole_roboflow_prediction() -> None:
    # given
    prediction = {
        "x": 50,
        "y": 50,
        "width": 40,
        "height": 20,
        "class": "cat",
        "class_id": 0,
        "confidence": 0.9,
        "detection_id": "first",
        "points": [{"x": 30, "y": 40}, {"x": 70, "y": 40}, {"x": 50, "y": 60}],
    }

    # when
    result = validate_reference_examples(
        [_example(annotations=[prediction])],
        task_type="object-detection",
        labels=["cat"],
    )

    # then
    assert len(result) == 1
    box = result[0].annotations[0]
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


@pytest.mark.parametrize(
    "task_type, annotations",
    [
        ("classification", [{"class": "cat"}]),
        ("multi-label-classification", [{"class": "cat"}, {"class": "dog"}]),
        (
            "instance-segmentation",
            [
                {
                    "class": "cat",
                    "points": [
                        {"x": 30, "y": 40},
                        {"x": 70, "y": 40},
                        {"x": 50, "y": 60},
                    ],
                }
            ],
        ),
        (
            "instance-segmentation",
            [{"class": "cat", "rle_mask": {"size": [100, 200], "counts": "abc"}}],
        ),
    ],
)
def test_validate_reference_examples_rejects_other_tasks(
    task_type, annotations
) -> None:
    with pytest.raises(ValueError, match="only supported for the object-detection"):
        validate_reference_examples(
            [_example(annotations=annotations)],
            task_type=task_type,
            labels=["cat", "dog"],
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
    example = _example(annotations=[{"x": 50, "y": 50, "height": 20, "class": "cat"}])

    # when
    with pytest.raises(ValueError) as error:
        validate_reference_examples(
            [example], task_type="object-detection", labels=["cat"]
        )

    # then
    assert "reference_examples[0]" in str(error.value)
    assert "annotations.0.width" in str(error.value)


@pytest.mark.parametrize(
    "field, value",
    [
        ("image", "$inputs.reference_image"),
        ("image", "$steps.crop.crops"),
        ("image", {"type": "url", "value": "$inputs.reference_url"}),
        ("annotations", "$steps.detector.predictions"),
    ],
)
def test_validate_reference_examples_refuses_selectors_inside_an_example(
    field, value
) -> None:
    with pytest.raises(ValueError, match="selectors are not resolved inside"):
        validate_reference_examples(
            [{**_example(), field: value}],
            task_type="object-detection",
            labels=["cat"],
        )


@pytest.mark.parametrize("manifest", [GeminiManifest, OpenAIManifest])
def test_manifest_schema_describes_the_examples_and_their_limit(manifest) -> None:
    # when
    schema = manifest.model_json_schema()

    # then
    options = schema["properties"]["reference_examples"]["anyOf"]
    literal = next(option for option in options if option.get("type") == "array")
    assert literal["items"] == {"$ref": "#/$defs/ReferenceExample"}
    assert literal["maxItems"] == MAX_REFERENCE_EXAMPLES
    example = schema["$defs"]["ReferenceExample"]
    assert sorted(example["required"]) == ["annotations", "image"]
    annotation = schema["$defs"]["ReferenceAnnotation"]
    assert sorted(annotation["required"]) == ["class", "height", "width", "x", "y"]


@pytest.mark.parametrize(
    "manifest, block_type",
    [
        (GeminiManifest, "roboflow_core/google_gemini@v6"),
        (OpenAIManifest, "roboflow_core/open_ai@v7"),
    ],
)
def test_manifest_rejects_more_literal_examples_than_the_limit(
    manifest, block_type
) -> None:
    with pytest.raises(ValidationError, match="at most 10 items"):
        manifest.model_validate(
            {
                "type": block_type,
                "name": "vlm",
                "images": "$inputs.image",
                "task_type": "object-detection",
                "classes": ["cat"],
                "reference_examples": [_example()] * (MAX_REFERENCE_EXAMPLES + 1),
            }
        )


def test_load_reference_examples_clips_boxes_to_the_image() -> None:
    # given
    examples = validate_reference_examples(
        [
            _example(
                annotations=[
                    {"x": 190, "y": 10, "width": 40, "height": 40, "class": "cat"}
                ]
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
                annotations=[
                    {"x": 500, "y": 50, "width": 40, "height": 20, "class": "cat"}
                ]
            )
        ],
        task_type="object-detection",
        labels=["cat"],
    )

    # when / then
    with pytest.raises(ValueError, match="outside its 200x100 image"):
        load_reference_examples(examples)


@pytest.mark.parametrize(
    "as_image",
    [lambda path: path, lambda path: {"type": "file", "value": path}],
    ids=["bare path", "file dict"],
)
def test_load_reference_examples_never_reads_a_local_file(
    server_like_codec: _ServerLikeCodec, tmp_path: Path, as_image
) -> None:
    # given
    path = tmp_path / "server-file.png"
    cv2.imwrite(str(path), np.full((100, 200, 3), 255, dtype=np.uint8))
    examples = validate_reference_examples(
        [{**_example(), "image": as_image(str(path))}],
        task_type="object-detection",
        labels=["cat"],
    )

    # when / then
    with pytest.raises(RuntimeInputError, match=r"reference_examples\[0\]\.image"):
        load_reference_examples(examples)
    assert server_like_codec.local_reads == []


def test_load_reference_examples_fetches_urls_through_the_codec(
    server_like_codec: _ServerLikeCodec,
) -> None:
    # given
    server_like_codec.url_images = [np.full((100, 200, 3), 7, dtype=np.uint8)]
    url = "https://example.com/labeled.jpg"
    examples = validate_reference_examples(
        [{**_example(), "image": {"type": "url", "value": url}}],
        task_type="object-detection",
        labels=["cat"],
    )

    # when
    loaded = load_reference_examples(examples)

    # then
    assert server_like_codec.fetched_urls == [url]
    assert loaded[0].image[0, 0].tolist() == [7, 7, 7]


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
            [_example(annotations=boxes)],
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
                    annotations=[
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
    parts = openai_reference_parts_for(
        [_example(annotations=boxes)], labels=["cat", "dog"]
    )

    # then
    assert parts[0] == {"type": "input_text", "text": REFERENCE_CROPS_INTRO}
    assert parts[1] == {"type": "input_text", "text": 'Examples of "cat":'}
    assert parts[2]["type"] == "input_image"
    assert parts[2]["image_url"].startswith("data:image/jpeg;base64,")
    assert parts[3] == {"type": "input_text", "text": 'Examples of "dog":'}
    assert len(parts) == 5


@pytest.mark.parametrize(
    "parts_for", [gemini_reference_parts_for, openai_reference_parts_for]
)
def test_reference_parts_are_built_once_for_the_same_examples(parts_for) -> None:
    # given
    examples = [_example()]
    cache = ReferencePartsCache()

    # when
    with patch.object(
        vlm_reference_examples,
        "load_image",
        side_effect=vlm_reference_examples.load_image,
    ) as load_image:
        first = parts_for(examples, labels=["cat"], cache=cache)
        second = parts_for(examples, labels=["cat"], cache=cache)

    # then
    assert first == second
    assert load_image.call_count == 1


def test_reference_parts_are_built_again_for_other_labels_or_examples() -> None:
    # given
    cache = ReferencePartsCache()
    other_labels = ["dog", "cat"]
    other_example = _example(width=300)

    # when
    with patch.object(
        vlm_reference_examples,
        "load_image",
        side_effect=vlm_reference_examples.load_image,
    ) as load_image:
        gemini_reference_parts_for([_example()], labels=["cat"], cache=cache)
        gemini_reference_parts_for([_example()], labels=other_labels, cache=cache)
        gemini_reference_parts_for([other_example], labels=["cat"], cache=cache)

    # then
    assert load_image.call_count == 3


def test_reference_parts_load_on_every_call_without_a_cache() -> None:
    # when
    with patch.object(
        vlm_reference_examples,
        "load_image",
        side_effect=vlm_reference_examples.load_image,
    ) as load_image:
        gemini_reference_parts_for([_example()], labels=["cat"])
        gemini_reference_parts_for([_example()], labels=["cat"])

    # then
    assert load_image.call_count == 2


def test_reference_parts_cache_keeps_nothing_from_a_failed_build(
    server_like_codec: _ServerLikeCodec,
) -> None:
    # given
    url = {"type": "url", "value": "https://example.com/labeled.jpg"}
    examples = [{**_example(), "image": url}]
    cache = ReferencePartsCache()
    server_like_codec.url_images = [np.full((100, 200, 3), 7, dtype=np.uint8)]

    # when
    with patch.object(
        server_like_codec, "fetch_url", side_effect=OSError("connection reset")
    ):
        with pytest.raises(RuntimeInputError, match="connection reset"):
            gemini_reference_parts_for(examples, labels=["cat"], cache=cache)
    parts = gemini_reference_parts_for(examples, labels=["cat"], cache=cache)

    # then
    assert parts[0] == {"text": REFERENCE_BOXED_INTRO}
    assert server_like_codec.fetched_urls == [url["value"]]


def test_reference_parts_of_a_new_cache_see_the_image_a_url_serves_now(
    server_like_codec: _ServerLikeCodec,
) -> None:
    # given
    url = {"type": "url", "value": "https://example.com/labeled.jpg"}
    examples = [{**_example(), "image": url}]
    server_like_codec.url_images = [
        np.full((100, 200, 3), 30, dtype=np.uint8),
        np.full((100, 200, 3), 220, dtype=np.uint8),
    ]
    first_job, second_job = ReferencePartsCache(), ReferencePartsCache()

    # when
    first = gemini_reference_parts_for(examples, labels=["cat"], cache=first_job)
    repeated = gemini_reference_parts_for(examples, labels=["cat"], cache=first_job)
    second = gemini_reference_parts_for(examples, labels=["cat"], cache=second_job)

    # then
    assert repeated == first
    assert len(server_like_codec.fetched_urls) == 2
    corner = (2, 2)
    assert abs(int(_decode(first[2]["inline_data"]["data"])[corner][0]) - 30) < 8
    assert abs(int(_decode(second[2]["inline_data"]["data"])[corner][0]) - 220) < 8


@pytest.mark.parametrize(
    "parts_for", [gemini_reference_parts_for, openai_reference_parts_for]
)
def test_reference_parts_of_a_numpy_image_are_rendered_on_every_call(
    parts_for,
) -> None:
    # given
    pixels = np.full((100, 200, 3), 30, dtype=np.uint8)
    example = {**_example(), "image": {"type": "numpy_object", "value": pixels}}
    cache = ReferencePartsCache()

    # when
    first = parts_for([example], labels=["cat"], cache=cache)
    pixels[:] = 220
    second = parts_for([example], labels=["cat"], cache=cache)

    # then
    assert first != second


def test_numpy_image_of_a_reference_example_keeps_its_pixels() -> None:
    # given
    pixels = np.full((100, 200, 3), 30, dtype=np.uint8)
    examples = validate_reference_examples(
        [{**_example(), "image": {"type": "numpy_object", "value": pixels}}],
        task_type="object-detection",
        labels=["cat"],
    )

    # when
    loaded = load_reference_examples(examples)

    # then
    assert np.array_equal(loaded[0].image, pixels)


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
    assert "labels: cat\nThe reference images" in result[-1]["text"]
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
