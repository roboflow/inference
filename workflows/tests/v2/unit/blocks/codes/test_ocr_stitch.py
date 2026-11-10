"""Unit tests of ``roboflow_core/stitch_ocr_detections@v2``, called directly.

Layouts are small and analytic. ``HORIZONTAL`` holds two lines, "AB" over
"CD", of boxes 20 wide and 10 tall with a 4-pixel gap. ``VERTICAL`` is the
same layout with x and y swapped: column "AB" left of column "CD". Rows are
given out of reading order so that every algorithm has to sort them.
"""

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch
from roboflow_workflows.v2.blocks.codes.stitch_ocr_detections.v2 import StitchOCRDetectionsBlock

from inference_models.models.base.object_detection import Detections

Box = Tuple[float, float, float, float]

HORIZONTAL: Dict[str, Box] = {
    "D": (24, 30, 44, 40),
    "A": (0, 0, 20, 10),
    "C": (0, 30, 20, 40),
    "B": (24, 0, 44, 10),
}
VERTICAL: Dict[str, Box] = {
    text: (y0, x0, y1, x1) for text, (x0, y0, x1, y1) in HORIZONTAL.items()
}
ALGORITHMS = ("tolerance", "otsu", "collimate")


def detections(
    layout: Dict[str, Box],
    *,
    text_from: str = "class_names",
    dtype: torch.dtype = torch.float32,
) -> Detections:
    # text_from: "class_names" (image map), "class" or "text" (row metadata).
    texts = list(layout)
    rows: List[dict] = [{} for _ in texts]
    if text_from != "class_names":
        rows = [{text_from: text} for text in texts]
    stitched_detections = Detections(
        xyxy=torch.tensor([layout[text] for text in texts], dtype=dtype),
        class_id=torch.arange(len(texts)),
        confidence=torch.ones(len(texts)),
        image_metadata={"class_names": dict(enumerate(texts))},
        bboxes_metadata=rows,
    )

    return stitched_detections


def read(
    predictions: Detections,
    *,
    algorithm: str = "tolerance",
    direction: str = "left_to_right",
    delimiter: str = "",
    tolerance: int = 10,
    otsu_threshold_multiplier: float = 1.0,
    collimate_tolerance: int = 10,
) -> str:
    result = StitchOCRDetectionsBlock().run(
        predictions=predictions,
        stitching_algorithm=algorithm,
        reading_direction=direction,
        tolerance=tolerance,
        delimiter=delimiter,
        otsu_threshold_multiplier=otsu_threshold_multiplier,
        collimate_tolerance=collimate_tolerance,
    )

    return result["ocr_text"]


def row_of_boxes(texts: str, starts: Sequence[float], *, top: float = 0) -> Dict:
    # One horizontal line of 10 x 10 boxes starting at the given x.
    layout = {text: (x, top, x + 10, top + 10) for text, x in zip(texts, starts)}

    return layout


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize(
    "direction, expected",
    [("left_to_right", "AB\nCD"), ("right_to_left", "BA\nDC")],
)
def test_horizontal_directions(algorithm: str, direction: str, expected: str) -> None:
    # when
    text = read(detections(HORIZONTAL), algorithm=algorithm, direction=direction)

    # then
    assert text == expected


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize(
    "direction, expected",
    [("vertical_top_to_bottom", "AB CD"), ("vertical_bottom_to_top", "DC BA")],
)
def test_vertical_directions(algorithm: str, direction: str, expected: str) -> None:
    # then: columns run left to right for top_to_bottom (V1 order) and right
    # to left for bottom_to_top; vertical lines join with a space
    text = read(detections(VERTICAL), algorithm=algorithm, direction=direction)

    assert text == expected


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_auto_follows_the_mean_box_shape_of_each_call(algorithm: str) -> None:
    # given: wide boxes read horizontally, tall boxes vertically
    block_calls = [(HORIZONTAL, "AB\nCD"), (VERTICAL, "AB CD")]

    # when
    texts = [
        read(detections(layout), algorithm=algorithm, direction="auto")
        for layout, _ in block_calls
    ]

    # then: decided per image, not carried over from the first one
    assert texts == [expected for _, expected in block_calls]


def character_boxes(lines: Sequence[str]) -> Detections:
    # Horizontal text of 12 x 16 character boxes, taller than wide like most
    # OCR characters; 2 px apart, a space adds 8 px, lines 30 px apart. Rows
    # are given in reverse reading order.
    texts: List[str] = []
    boxes: List[Box] = []
    for line_number, line in enumerate(lines):
        x = 0
        for character in line:
            if character == " ":
                x += 8
                continue
            y = 30 * line_number
            texts.insert(0, character)
            boxes.insert(0, (x, y, x + 12, y + 16))
            x += 14
    predictions = Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32),
        class_id=torch.zeros(len(texts), dtype=torch.int64),
        confidence=torch.ones(len(texts)),
        bboxes_metadata=[{"class": text} for text in texts],
    )

    return predictions


@pytest.mark.parametrize(
    "algorithm, expected",
    [
        ("tolerance", "HELLOWORLD\nV2OCR"),
        ("otsu", "HELLO WORLD\nV2 OCR"),
        ("collimate", "HELLOWORLD\nV2OCR"),
    ],
)
def test_horizontal_character_boxes_read_left_to_right(
    algorithm: str, expected: str
) -> None:
    # Regression for the removed "auto" default: tall character boxes on
    # horizontal lines read correctly once the direction is given.
    text = read(
        character_boxes(["HELLO WORLD", "V2 OCR"]),
        algorithm=algorithm,
        direction="left_to_right",
    )

    assert text == expected


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_explicit_auto_reads_tall_character_boxes_vertically(algorithm: str) -> None:
    # given: the same horizontal text; mean box 12 wide and 16 tall
    predictions = character_boxes(["HELLO WORLD", "V2 OCR"])

    # when
    auto = read(predictions, algorithm=algorithm, direction="auto")
    vertical = read(
        predictions, algorithm=algorithm, direction="vertical_top_to_bottom"
    )

    # then: auto keeps the V1 rule, so this horizontal text is read as columns
    assert auto == vertical
    assert auto != read(predictions, algorithm=algorithm, direction="left_to_right")


def test_reading_direction_is_required() -> None:
    # then: as in V1 roboflow_core/stitch_ocr_detections@v2, no default
    field = StitchOCRDetectionsBlock.Params.model_fields["reading_direction"]

    assert field.is_required()


def test_tolerance_joins_line_separators_with_the_delimiter() -> None:
    # then: preserved V1 behaviour
    assert read(detections(HORIZONTAL), delimiter=",") == "A,B,\n,C,D"


def test_tolerance_buckets_rounded_top_edges() -> None:
    # given: tops 0, 4, 8, 12, 16 fall into buckets 0, 0, 10, 10, 20
    layout = {
        text: (12 * number, 4 * number, 12 * number + 10, 4 * number + 10)
        for number, text in enumerate("abcde")
    }

    # when
    text = read(detections(layout), algorithm="tolerance", tolerance=10)

    # then
    assert text == "ab\ncd\ne"


def test_collimate_follows_skewed_text_as_one_line() -> None:
    # given: the same rising line; each step drifts 4 px, within the slack
    layout = {
        text: (12 * number, 4 * number, 12 * number + 10, 4 * number + 10)
        for number, text in enumerate("abcde")
    }

    # when
    text = read(detections(layout), algorithm="collimate", collimate_tolerance=10)

    # then
    assert text == "abcde"


def test_otsu_splits_words_at_the_wide_gap() -> None:
    # given: normalised gaps 0.2, 1.8, 0.2 are bimodal
    layout = row_of_boxes("abcd", [0, 12, 40, 52])

    # when
    text = read(detections(layout), algorithm="otsu")

    # then
    assert text == "ab cd"


def test_otsu_uses_a_non_empty_delimiter_between_words() -> None:
    # given
    layout = row_of_boxes("abcd", [0, 12, 40, 52])

    # when
    text = read(detections(layout), algorithm="otsu", delimiter="-")

    # then
    assert text == "a-b-c-d"


def test_otsu_keeps_evenly_spaced_pieces_together() -> None:
    # given: all gaps equal, so the gap distribution is unimodal
    layout = row_of_boxes("abcd", [0, 12, 24, 36])

    # when
    text = read(detections(layout), algorithm="otsu")

    # then
    assert text == "abcd"


def test_otsu_multiplier_zero_splits_every_positive_gap() -> None:
    # given
    layout = row_of_boxes("abcd", [0, 12, 40, 52])

    # when
    text = read(detections(layout), algorithm="otsu", otsu_threshold_multiplier=0.0)

    # then
    assert text == "a b c d"


@pytest.mark.parametrize("text_from", ["class_names", "class", "text"])
def test_row_text_sources(text_from: str) -> None:
    # then
    assert read(detections(HORIZONTAL, text_from=text_from)) == "AB\nCD"


def test_row_class_takes_precedence_over_text_and_class_names() -> None:
    # given
    predictions = detections(row_of_boxes("ab", [0, 12]))
    predictions.bboxes_metadata[0].update({"class": "X", "text": "Y"})
    predictions.bboxes_metadata[1].update({"text": "Z"})

    # when
    text = read(predictions)

    # then: row 0 uses its class, row 1 its text over the class_names map
    assert text == "XZ"


def test_non_string_class_falls_back_to_class_names() -> None:
    # given: a numeric class names nothing (shared row-name rule)
    predictions = detections(row_of_boxes("ab", [0, 12]))
    predictions.bboxes_metadata[0]["class"] = 7

    # when
    text = read(predictions)

    # then
    assert text == "ab"


def test_row_without_any_text_raises() -> None:
    # given
    predictions = detections(row_of_boxes("ab", [0, 12]))
    predictions.image_metadata["class_names"].pop(1)

    # when / then
    with pytest.raises(ValueError, match="row 1 has no text"):
        read(predictions)


def test_no_detections_give_empty_text() -> None:
    # given
    predictions = Detections(
        xyxy=torch.zeros((0, 4)),
        class_id=torch.zeros(0, dtype=torch.int64),
        confidence=torch.zeros(0),
    )

    # then
    assert read(predictions) == ""


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_box_dtype_does_not_change_text(dtype: torch.dtype, algorithm: str) -> None:
    # then
    assert read(detections(HORIZONTAL, dtype=dtype), algorithm=algorithm) == ("AB\nCD")


def test_noncontiguous_boxes() -> None:
    # given: xyxy as a strided view of a wider tensor
    predictions = detections(HORIZONTAL)
    wide = torch.zeros((len(predictions), 8))
    wide[:, :4] = predictions.xyxy
    predictions.xyxy = wide[:, :4]
    assert not predictions.xyxy.is_contiguous()

    # then
    assert read(predictions) == "AB\nCD"


def test_input_is_not_modified() -> None:
    # given
    predictions = detections(HORIZONTAL, text_from="text")
    boxes_before = predictions.xyxy.clone()
    rows_before: Optional[List[dict]] = copy.deepcopy(predictions.bboxes_metadata)
    image_metadata_before = copy.deepcopy(predictions.image_metadata)

    # when
    for direction in ("left_to_right", "vertical_bottom_to_top"):
        for algorithm in ALGORITHMS:
            read(predictions, algorithm=algorithm, direction=direction)

    # then
    assert torch.equal(predictions.xyxy, boxes_before)
    assert predictions.bboxes_metadata == rows_before
    assert predictions.image_metadata == image_metadata_before


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"algorithm": "fastest"}, "stitching_algorithm must be one of"),
        ({"direction": "diagonal"}, "reading_direction must be one of"),
    ],
)
def test_unknown_choices_from_references_raise(overrides: dict, message: str) -> None:
    # Literal choices are checked by the schema; reference-bound values are
    # checked again at run time.
    with pytest.raises(ValueError, match=message):
        read(detections(HORIZONTAL), **overrides)
