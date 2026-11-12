import base64
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import ImageDraw
from pydantic import ValidationError
from roboflow_workflows.core_steps.common import vlm_reference_examples
from roboflow_workflows.core_steps.common.vlm_reference_examples import (
    CROP_MAX_EDGE,
    CROP_MIN_EDGE,
    MAX_REFERENCE_EXAMPLES,
    REFERENCE_BOXED_INTRO,
    REFERENCE_CROPS_INTRO,
    REFERENCE_NUMBERED_INTRO,
    TARGET_IMAGE_INTRO,
    TARGET_ONLY_RULE,
    ReferencePartsCache,
    gemini_reference_parts_for,
    load_reference_example,
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


@pytest.fixture
def waiting_callers() -> Iterator[threading.Semaphore]:
    """Released once by every caller that starts waiting for a build in progress."""
    waiting = threading.Semaphore(0)
    wait = vlm_reference_examples._PendingBuild.wait

    def counted_wait(pending: Any, **kwargs: Any) -> List[dict]:
        waiting.release()
        return wait(pending, **kwargs)

    with patch.object(vlm_reference_examples._PendingBuild, "wait", new=counted_wait):
        yield waiting


def _recording_build(text: str, builds: List[str]) -> Callable[[], List[dict]]:
    def build() -> List[dict]:
        builds.append(text)
        return [{"text": text}]

    return build


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
    color: Tuple[int, int, int] = (255, 255, 255),
) -> dict:
    return {
        "image": {"type": "base64", "value": _image_base64(width, height, color)},
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


def test_load_reference_example_clips_boxes_to_the_image() -> None:
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
    loaded = load_reference_example(examples[0], index=0)

    # then
    assert loaded.image.shape == (100, 200, 3)
    assert loaded.boxes == (("cat", 170.0, 0.0, 200.0, 30.0),)


def test_load_reference_example_rejects_a_box_outside_its_image() -> None:
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
        load_reference_example(examples[0], index=0)


@pytest.mark.parametrize(
    "as_image",
    [lambda path: path, lambda path: {"type": "file", "value": path}],
    ids=["bare path", "file dict"],
)
def test_load_reference_example_never_reads_a_local_file(
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
        load_reference_example(examples[0], index=0)
    assert server_like_codec.local_reads == []


def test_load_reference_example_fetches_urls_through_the_codec(
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
    loaded = load_reference_example(examples[0], index=0)

    # then
    assert server_like_codec.fetched_urls == [url]
    assert loaded.image[0, 0].tolist() == [7, 7, 7]


@pytest.mark.parametrize("render", [render_boxed_examples, render_example_crops])
def test_rendering_releases_each_example_image_before_the_next_one_loads(
    server_like_codec: _ServerLikeCodec, render
) -> None:
    # given
    url = {"type": "url", "value": "https://example.com/labeled.jpg"}
    examples = validate_reference_examples(
        [{**_example(), "image": url}] * 3,
        task_type="object-detection",
        labels=["cat"],
    )
    fetched_images = []
    alive_at_each_fetch = []

    def fetch_url(value: str, cv_imread_flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
        alive_at_each_fetch.append(sum(image() is not None for image in fetched_images))
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        fetched_images.append(weakref.ref(image))
        return image

    # when
    with patch.object(server_like_codec, "fetch_url", new=fetch_url):
        render(examples, labels=["cat"])

    # then
    assert alive_at_each_fetch == [0, 0, 0]


def test_render_example_crops_keeps_the_largest_boxes_in_label_order() -> None:
    # given
    boxes = [
        {"x": 30, "y": 30, "width": 10, "height": 10, "class": "cat"},
        {"x": 100, "y": 50, "width": 60, "height": 40, "class": "cat"},
        {"x": 150, "y": 50, "width": 30, "height": 30, "class": "cat"},
        {"x": 50, "y": 70, "width": 20, "height": 20, "class": "cat"},
        {"x": 180, "y": 20, "width": 20, "height": 10, "class": "dog"},
    ]
    examples = validate_reference_examples(
        [_example(annotations=boxes)],
        task_type="object-detection",
        labels=["dog", "cat", "bird"],
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


@pytest.mark.parametrize("labels", [["cat"], ["dog", "cat"]])
@pytest.mark.parametrize("equal_area", [False, True], ids=["increasing", "equal"])
def test_render_example_crops_encodes_only_three_boxes_per_label_per_image(
    labels: List[str], equal_area: bool
) -> None:
    # given
    pixels = np.random.default_rng(0).integers(
        0, 256, size=(100, 200, 3), dtype=np.uint8
    )
    _, encoded = cv2.imencode(".png", pixels)
    image = {"type": "base64", "value": base64.b64encode(encoded).decode("ascii")}
    boxes = [
        {
            "class": label,
            "x": 60 + index % 80,
            "y": 50,
            "width": 20 if equal_area else 20 + index / 10,
            "height": 20,
        }
        for index in range(1000)
        for label in reversed(labels)
    ]
    selected = []
    for label in labels:
        label_boxes = [box for box in boxes if box["class"] == label]
        selected.extend(label_boxes[:3] if equal_area else label_boxes[-3:][::-1])

    examples = validate_reference_examples(
        [{"image": image, "annotations": boxes}],
        task_type="object-detection",
        labels=labels,
    )
    expected_examples = validate_reference_examples(
        [{"image": image, "annotations": selected}],
        task_type="object-detection",
        labels=labels,
    )
    expected = render_example_crops(expected_examples, labels=labels)

    # when
    with patch.object(
        vlm_reference_examples,
        "_crop_base64",
        wraps=vlm_reference_examples._crop_base64,
    ) as encode_crop:
        groups = render_example_crops(examples, labels=labels)

    # then
    assert groups == expected
    assert encode_crop.call_count == 3 * len(labels)


def test_render_example_crops_keeps_the_largest_boxes_across_examples() -> None:
    # given
    examples = validate_reference_examples(
        [
            _example(
                annotations=[
                    {"x": 30, "y": 30, "width": 40, "height": 40, "class": "cat"},
                    {"x": 120, "y": 50, "width": 20, "height": 20, "class": "cat"},
                ],
                color=(10, 10, 10),
            ),
            _example(
                annotations=[
                    {"x": 50, "y": 50, "width": 40, "height": 40, "class": "cat"}
                ],
                color=(100, 100, 100),
            ),
            _example(
                annotations=[
                    {"x": 50, "y": 50, "width": 30, "height": 30, "class": "cat"},
                    {"x": 130, "y": 50, "width": 50, "height": 50, "class": "cat"},
                ],
                color=(200, 200, 200),
            ),
        ],
        task_type="object-detection",
        labels=["cat"],
    )

    # when
    groups = render_example_crops(examples, labels=["cat"])

    # then
    # The 50x50 box of the last example, then both 40x40 boxes in the order given.
    shades = [int(_decode(crop)[0, 0, 0]) for crop in groups[0][1]]
    assert len(shades) == 3
    for shade, expected in zip(shades, [200, 10, 100]):
        assert abs(shade - expected) < 8


def test_render_boxed_examples_downscales_and_draws_in_the_label_color() -> None:
    # given
    examples = validate_reference_examples(
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


def test_gemini_reference_parts_number_the_labels_when_one_is_outside_ascii() -> None:
    # given
    boxes = [
        {"x": 30, "y": 50, "width": 30, "height": 20, "class": "猫"},
        {"x": 100, "y": 50, "width": 30, "height": 20, "class": "狗"},
        {"x": 170, "y": 50, "width": 30, "height": 20, "class": "cat"},
    ]

    # when
    with patch.object(
        ImageDraw.ImageDraw,
        "text",
        autospec=True,
        side_effect=ImageDraw.ImageDraw.text,
    ) as draw_text:
        parts = gemini_reference_parts_for(
            [_example(annotations=boxes)], labels=["cat", "猫", "鸟", "狗"]
        )

    # then
    assert parts[0] == {
        "text": f'{REFERENCE_NUMBERED_INTRO}\n1: "cat"\n2: "猫"\n4: "狗"'
    }
    assert [call.args[2] for call in draw_text.call_args_list] == ["2", "4", "1"]


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


def test_reference_parts_see_the_image_a_url_serves_once_they_expire(
    server_like_codec: _ServerLikeCodec,
) -> None:
    # given
    url = {"type": "url", "value": "https://example.com/labeled.jpg"}
    examples = [{**_example(), "image": url}]
    server_like_codec.url_images = [
        np.full((100, 200, 3), 30, dtype=np.uint8),
        np.full((100, 200, 3), 220, dtype=np.uint8),
    ]
    now = [0.0]
    cache = ReferencePartsCache(ttl_seconds=60, clock=lambda: now[0])

    # when
    first = gemini_reference_parts_for(examples, labels=["cat"], cache=cache)
    now[0] = 59.0
    before_expiry = gemini_reference_parts_for(examples, labels=["cat"], cache=cache)
    now[0] = 60.0
    after_expiry = gemini_reference_parts_for(examples, labels=["cat"], cache=cache)

    # then
    assert before_expiry == first
    assert len(server_like_codec.fetched_urls) == 2
    corner = (2, 2)
    assert abs(int(_decode(first[2]["inline_data"]["data"])[corner][0]) - 30) < 8
    assert (
        abs(int(_decode(after_expiry[2]["inline_data"]["data"])[corner][0]) - 220) < 8
    )


def test_reference_parts_stay_cached_for_several_example_sets() -> None:
    # given
    cache = ReferencePartsCache()
    first_set, second_set = [_example()], [_example(width=300)]

    # when
    with patch.object(
        vlm_reference_examples,
        "load_image",
        side_effect=vlm_reference_examples.load_image,
    ) as load_image:
        for examples in (first_set, second_set, first_set, second_set):
            gemini_reference_parts_for(examples, labels=["cat"], cache=cache)

    # then
    assert load_image.call_count == 2


def test_gemini_and_openai_parts_of_the_same_examples_are_cached_apart() -> None:
    # given
    cache = ReferencePartsCache()

    # when
    gemini = gemini_reference_parts_for([_example()], labels=["cat"], cache=cache)
    openai = openai_reference_parts_for([_example()], labels=["cat"], cache=cache)

    # then
    assert gemini[0] == {"text": REFERENCE_BOXED_INTRO}
    assert openai[0] == {"type": "input_text", "text": REFERENCE_CROPS_INTRO}


def test_reference_parts_cache_drops_the_least_recently_used_entry_first() -> None:
    # given
    cache = ReferencePartsCache(max_entries=2)
    builds: List[str] = []

    # when
    for key in ["a", "b", "a", "c", "a", "b"]:
        cache.get_or_build(key, build=_recording_build(key, builds))

    # then
    assert builds == ["a", "b", "c", "b"]


def test_reference_parts_cache_keeps_its_parts_within_max_bytes() -> None:
    # given
    cache = ReferencePartsCache(max_bytes=2 * sys.getsizeof("aaaa"))
    builds: List[str] = []
    too_large = "x" * 200

    # when
    for key in ["aaaa", "bbbb", "cccc", too_large, too_large, "bbbb", "aaaa"]:
        cache.get_or_build(key, build=_recording_build(key, builds))

    # then
    assert builds == ["aaaa", "bbbb", "cccc", too_large, too_large, "aaaa"]


@pytest.mark.parametrize("text", ["a" * 100, "é" * 100, "漢" * 100, "😀" * 100])
def test_reference_parts_cache_accounts_for_unicode_storage(text: str) -> None:
    cache = ReferencePartsCache(max_bytes=sys.getsizeof(text) - 1)
    builds: List[str] = []
    for _ in range(2):
        assert cache.get_or_build("key", build=_recording_build(text, builds)) == [
            {"text": text}
        ]
    assert builds == [text, text]
    assert not cache._entries
    assert cache._size == 0


@pytest.mark.parametrize(
    "parts_for", [gemini_reference_parts_for, openai_reference_parts_for]
)
def test_rendered_unicode_labels_cannot_exceed_cache_byte_budget(parts_for) -> None:
    label = "😀" * 2000
    cache = ReferencePartsCache(max_bytes=6000)
    example = _example(
        width=120,
        height=80,
        annotations=[{"class": label, "x": 60, "y": 40, "width": 40, "height": 20}],
    )
    with patch.object(
        vlm_reference_examples,
        "load_reference_example",
        wraps=load_reference_example,
    ) as load:
        first = parts_for([example], labels=[label], cache=cache)
        second = parts_for([example], labels=[label], cache=cache)
    assert first == second
    assert load.call_count == 2
    assert not cache._entries
    assert cache._size == 0


def test_reference_parts_cache_evicts_by_unicode_bytes_and_releases_on_clear() -> None:
    text = "😀" * 100
    cache = ReferencePartsCache(max_bytes=sys.getsizeof(text))
    builds: List[str] = []
    for key in ["a", "b", "a"]:
        cache.get_or_build(key, build=_recording_build(text, builds))
        assert cache._size == sys.getsizeof(text)
        assert len(cache._entries) == 1
        cache.get_or_build(key, build=lambda: pytest.fail("newest entry was evicted"))
    assert len(builds) == 3
    cache.clear()
    assert cache._size == 0
    assert not cache._entries


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_reference_parts_cache_requires_a_finite_positive_pending_timeout(
    timeout,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ReferencePartsCache(pending_timeout_seconds=timeout)


def test_waiter_times_out_and_retry_succeeds_before_original_build_finishes() -> None:
    # A fixed clock keeps the pending lease alive until the real Event wait times out.
    cache = ReferencePartsCache(pending_timeout_seconds=0.05, clock=lambda: 0.0)
    started, release = threading.Event(), threading.Event()

    def stalled_build() -> List[dict]:
        started.set()
        assert release.wait(timeout=10)
        return [{"text": "stale"}]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.get_or_build, "key", build=stalled_build)
        try:
            assert started.wait(timeout=5)
            waiter = pool.submit(
                cache.get_or_build, "key", build=lambda: pytest.fail("duplicate build")
            )
            with pytest.raises(TimeoutError, match="Timed out waiting"):
                waiter.result(timeout=5)
            fresh = cache.get_or_build("key", build=lambda: [{"text": "fresh"}])
            assert not first.done()
        finally:
            release.set()
        assert first.result(timeout=5) == [{"text": "stale"}]

    assert (
        cache.get_or_build("key", build=lambda: pytest.fail("lost replacement"))
        is fresh
    )
    assert cache._size == sys.getsizeof("fresh")


@pytest.mark.parametrize("old_fails", [False, True])
def test_expired_build_cannot_remove_or_overwrite_replacement(
    old_fails: bool, waiting_callers: threading.Semaphore
) -> None:
    now = [0.0]
    cache = ReferencePartsCache(clock=lambda: now[0])
    old_started, new_started = threading.Event(), threading.Event()
    release_old, release_new = threading.Event(), threading.Event()

    def old_build() -> List[dict]:
        old_started.set()
        assert release_old.wait(timeout=10)
        if old_fails:
            raise OSError("old failure")
        return [{"text": "old"}]

    def new_build() -> List[dict]:
        new_started.set()
        assert release_new.wait(timeout=10)
        return [{"text": "new"}]

    with ThreadPoolExecutor(max_workers=4) as pool:
        old = pool.submit(cache.get_or_build, "key", build=old_build)
        try:
            assert old_started.wait(timeout=5)
            old_waiter = pool.submit(cache.get_or_build, "key", build=new_build)
            assert waiting_callers.acquire(timeout=5)
            now[0] = 61.0
            new = pool.submit(cache.get_or_build, "key", build=new_build)
            assert new_started.wait(timeout=5)
            with pytest.raises(TimeoutError, match="Timed out waiting"):
                old_waiter.result(timeout=5)
            new_waiter = pool.submit(
                cache.get_or_build, "key", build=lambda: pytest.fail("duplicate retry")
            )
            assert waiting_callers.acquire(timeout=5)
            release_old.set()
            if old_fails:
                with pytest.raises(OSError, match="old failure"):
                    old.result(timeout=5)
            else:
                assert old.result(timeout=5) == [{"text": "old"}]
            assert not new_waiter.done()
            release_new.set()
            assert new.result(timeout=5) == [{"text": "new"}]
            assert new_waiter.result(timeout=5) is new.result()
        finally:
            release_old.set()
            release_new.set()
    assert (
        cache.get_or_build("key", build=lambda: pytest.fail("lost retry"))
        is new.result()
    )


def test_other_keys_expire_abandoned_pending_builds(
    waiting_callers: threading.Semaphore,
) -> None:
    now = [0.0]
    cache = ReferencePartsCache(clock=lambda: now[0])
    started, release = threading.Event(), threading.Event()

    def stalled_build() -> List[dict]:
        started.set()
        assert release.wait(timeout=10)
        return [{"text": "expired"}]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.get_or_build, "stalled", build=stalled_build)
        try:
            assert started.wait(timeout=5)
            waiter = pool.submit(cache.get_or_build, "stalled", build=stalled_build)
            assert waiting_callers.acquire(timeout=5)
            now[0] = 61.0
            cache.get_or_build("unrelated", build=lambda: [{"text": "healthy"}])
            with pytest.raises(TimeoutError, match="Timed out waiting"):
                waiter.result(timeout=5)
            assert not cache._pending
        finally:
            release.set()
        first.result(timeout=5)
    assert len(cache._entries) == 1
    assert cache.get_or_build(
        "unrelated", build=lambda: pytest.fail("healthy entry was evicted")
    ) == [{"text": "healthy"}]


def test_reference_parts_cache_builds_a_key_once_for_concurrent_callers(
    waiting_callers: threading.Semaphore,
) -> None:
    # given
    cache = ReferencePartsCache()
    builds: List[str] = []

    def build() -> List[dict]:
        builds.append("parts")
        # Finish only once the three other callers wait for this build.
        for _ in range(3):
            assert waiting_callers.acquire(timeout=10)
        return [{"text": "parts"}]

    # when
    with ThreadPoolExecutor(max_workers=4) as pool:
        calls = [pool.submit(cache.get_or_build, "key", build=build) for _ in range(4)]
        results = [call.result(timeout=20) for call in calls]

    # then
    assert builds == ["parts"]
    assert all(result is results[0] for result in results)


def test_reference_parts_cache_builds_other_keys_while_one_is_building() -> None:
    # given
    cache = ReferencePartsCache()
    building, release = threading.Event(), threading.Event()

    def slow_build() -> List[dict]:
        building.set()
        release.wait(timeout=10)
        return [{"text": "slow"}]

    # when
    with ThreadPoolExecutor(max_workers=1) as pool:
        slow = pool.submit(cache.get_or_build, "slow", build=slow_build)
        assert building.wait(timeout=10)
        fast = cache.get_or_build("fast", build=lambda: [{"text": "fast"}])
        slow_was_building = not slow.done()
        release.set()
        slow_parts = slow.result(timeout=10)

    # then
    assert fast == [{"text": "fast"}]
    assert slow_was_building
    assert slow_parts == [{"text": "slow"}]


def test_reference_parts_cache_gives_waiting_callers_the_error_and_keeps_nothing(
    waiting_callers: threading.Semaphore,
) -> None:
    # given
    cache = ReferencePartsCache()
    builds: List[str] = []

    def failing_build() -> List[dict]:
        builds.append("failed")
        assert waiting_callers.acquire(timeout=10)
        raise OSError("connection reset")

    # when
    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [
            pool.submit(cache.get_or_build, "key", build=failing_build)
            for _ in range(2)
        ]
        errors = [call.exception(timeout=20) for call in calls]
    parts = cache.get_or_build("key", build=_recording_build("parts", builds))

    # then
    assert [str(error) for error in errors] == ["connection reset"] * 2
    assert builds == ["failed", "parts"]
    assert parts == [{"text": "parts"}]


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
    loaded = load_reference_example(examples[0], index=0)

    # then
    assert np.array_equal(loaded.image, pixels)


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
