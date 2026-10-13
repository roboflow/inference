"""Read OCR detections as text: ``v2/stitch_ocr_detections@v2``.

Port of V1 ``roboflow_core/stitch_ocr_detections@v2``. Each detection carries
one piece of recognised text (a character or a word). The block groups the
boxes of one image into lines, orders each line and joins the pieces::

    "H" "e" "l" "l" "o"        tolerance / otsu / collimate      "Hello"
    "w" "o" "r" "l" "d"   ─▶   left_to_right, delimiter ""  ─▶   "world"
                                                                 -> "Hello\\nworld"

Vertical directions swap x and y first, so every algorithm works in one
reading frame: ``x`` runs along a line, ``y`` across lines. "Backward"
directions read a line from high to low ``x``::

    direction                 swap x/y   line order       pieces in a line
    left_to_right             no         top to bottom    left to right
    right_to_left             no         top to bottom    right to left
    vertical_top_to_bottom    yes        left to right    top to bottom
    vertical_bottom_to_top    yes        right to left    bottom to top

Lines are joined by ``"\\n"`` for horizontal text and ``" "`` for vertical
text. ``reading_direction`` has no default, as in V1. ``auto`` reads
horizontally when the mean box is wider than tall, else
``vertical_top_to_bottom``; it is decided per image. Single-character boxes
are usually taller than wide, so ``auto`` reads them vertically.

Algorithms, all as in V1:

* ``tolerance``: boxes whose rounded top edge falls into the same
  ``tolerance`` bucket form a line. Line separators are joined like pieces,
  so a delimiter also surrounds them (``"a,b,\\n,c"``).
* ``otsu``: lines from box centres within half the median box height; word
  breaks where a gap, normalised by the neighbouring box lengths, exceeds an
  Otsu threshold computed over all lines of the image. Words are separated
  by ``" "`` (or by a non-empty delimiter).
* ``collimate``: greedy chains, each box followed by the next box that is
  aligned within ``collimate_tolerance`` plus half the larger box height.
  Suited to skewed or curved text.

The text of a row is its own string ``class`` (the shared row-name rule),
else its string ``text`` metadata (the V1 OCR convention), else the
prediction's ``class_names`` entry for its class id.

CPU boundary: reading order is a sequential host algorithm producing a host
string. Each call copies the ``(N, 4)`` boxes to the host once and, only when
some row lacks its own text, the ``N`` class ids. Inputs are never modified.
"""

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pydantic import Field, StrictInt
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.data.prediction_ops import class_names_by_id, row_class_name
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.object_detection import Detections

__all__ = ["StitchOCRDetectionsBlock"]

ALGORITHMS = ("tolerance", "otsu", "collimate")
DIRECTIONS = (
    "left_to_right",
    "right_to_left",
    "vertical_top_to_bottom",
    "vertical_bottom_to_top",
    "auto",
)
VERTICAL_DIRECTIONS = ("vertical_top_to_bottom", "vertical_bottom_to_top")
BACKWARD_DIRECTIONS = ("right_to_left", "vertical_bottom_to_top")
TEXT_KEY = "text"


class StitchOCRDetectionsBlock(Block):
    """Join the OCR detections of one image into its text, in reading order.

    Example: boxes ``"b"`` at x=20 and ``"a"`` at x=0 on one line, read
    ``left_to_right`` with the ``tolerance`` algorithm, give ``"ab"``; a third
    box ``"c"`` 40 pixels lower gives ``"ab\\nc"``. An image without detections
    gives ``""``.

    Selected values of ``stitching_algorithm`` and ``reading_direction`` are
    checked like literals. The block is stateless and prunable.
    """

    type = "v2/stitch_ocr_detections@v2"
    prunable = True
    metadata = {
        "name": "Stitch OCR Detections",
        "port_of": "roboflow_core/stitch_ocr_detections@v2",
        "section": "advanced",
    }
    outputs = {
        "ocr_text": Output(
            STRING_KIND,
            source="predictions",
            description="Text of the image in reading order; empty without rows.",
        )
    }

    class Params(BlockParams):
        predictions: Ref(OBJECT_DETECTION_PREDICTION_KIND) = Field(
            description=(
                "OCR detections of one image: one character or word per row, "
                "named by its class or 'text' metadata."
            ),
            examples=["$steps.ocr_model.predictions"],
        )
        stitching_algorithm: Literal[ALGORITHMS] | Ref(STRING_KIND) = Field(
            default="tolerance",
            description=(
                "'tolerance': fixed pixel line buckets. 'otsu': line and word "
                "breaks from the image's own spacing. 'collimate': greedy "
                "chains for skewed or curved text."
            ),
            examples=["tolerance", "otsu", "collimate"],
        )
        reading_direction: Literal[DIRECTIONS] | Ref(STRING_KIND) = Field(
            description=(
                "Required, as in V1. left_to_right, right_to_left, "
                "vertical_top_to_bottom, vertical_bottom_to_top, or auto: "
                "horizontal when the mean box is wider than tall, else "
                "vertical_top_to_bottom. Single characters are usually taller "
                "than wide, so auto reads them vertically."
            ),
            examples=["left_to_right", "right_to_left", "auto"],
        )
        tolerance: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            gt=0,
            description="Line bucket size in pixels for 'tolerance'.",
            examples=[10, "$inputs.tolerance"],
        )
        delimiter: str | Ref(STRING_KIND) = Field(
            default="",
            description="Text placed between joined pieces; empty concatenates.",
            examples=["", " ", ","],
        )
        otsu_threshold_multiplier: float | Ref(FLOAT_KIND) = Field(
            default=1.0,
            ge=0.0,
            description=(
                "Scales the 'otsu' word-break threshold: above 1 splits fewer "
                "words, below 1 more."
            ),
            examples=[1.0, 1.3, 0.8],
        )
        collimate_tolerance: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            ge=0,
            description="Extra cross-line deviation in pixels for 'collimate'.",
            examples=[10, 20],
        )

    def run(
        self,
        *,
        predictions: Detections,
        stitching_algorithm: str,
        reading_direction: str,
        tolerance: int,
        delimiter: str,
        otsu_threshold_multiplier: float,
        collimate_tolerance: int,
    ) -> Dict[str, Any]:
        """Stitch the text of one image.

        Args:
            predictions: OCR detections of the image.
            stitching_algorithm: One of ``tolerance``, ``otsu``, ``collimate``.
            reading_direction: One of the four directions or ``auto``.
            tolerance: Positive line bucket size for ``tolerance``.
            delimiter: Text between joined pieces.
            otsu_threshold_multiplier: Nonnegative factor for ``otsu``.
            collimate_tolerance: Nonnegative pixel slack for ``collimate``.

        Returns:
            ``ocr_text``: the stitched text, ``""`` for no detections.

        Raises:
            ValueError: When a selected algorithm or direction is unknown, or
                a row has no text.
        """
        _check_choice(
            stitching_algorithm, name="stitching_algorithm", choices=ALGORITHMS
        )
        _check_choice(reading_direction, name="reading_direction", choices=DIRECTIONS)
        if len(predictions) == 0:
            return {"ocr_text": ""}

        texts = _row_texts(predictions)
        boxes = predictions.xyxy.detach().to("cpu").numpy()
        if reading_direction == "auto":
            reading_direction = _detect_direction(boxes)

        vertical = reading_direction in VERTICAL_DIRECTIONS
        frame = _ReadingFrame(
            boxes=boxes[:, [1, 0, 3, 2]] if vertical else boxes,
            backward=reading_direction in BACKWARD_DIRECTIONS,
            lines_reversed=reading_direction == "vertical_bottom_to_top",
            line_separator=" " if vertical else "\n",
        )
        if stitching_algorithm == "tolerance":
            text = _tolerance_text(
                frame, texts, tolerance=tolerance, delimiter=delimiter
            )
        elif stitching_algorithm == "otsu":
            text = _otsu_text(
                frame,
                texts,
                delimiter=delimiter,
                threshold_multiplier=otsu_threshold_multiplier,
            )
        else:
            text = _collimate_text(
                frame, texts, delimiter=delimiter, tolerance=collimate_tolerance
            )
        result = {"ocr_text": text}

        return result


class _ReadingFrame:
    """Boxes with ``x`` along lines and ``y`` across them, plus the order."""

    def __init__(
        self,
        *,
        boxes: np.ndarray,
        backward: bool,
        lines_reversed: bool,
        line_separator: str,
    ):
        self.boxes = boxes
        self.backward = backward
        self.lines_reversed = lines_reversed
        self.line_separator = line_separator
        self.along = (boxes[:, 0] + boxes[:, 2]) / 2
        self.across = (boxes[:, 1] + boxes[:, 3]) / 2
        self.lengths = boxes[:, 2] - boxes[:, 0]
        self.heights = boxes[:, 3] - boxes[:, 1]


def _check_choice(value: Any, *, name: str, choices: Sequence[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")


def _row_texts(predictions: Detections) -> np.ndarray:
    # Object array: the tolerance algorithm takes texts by index arrays.
    rows = predictions.bboxes_metadata or [{}] * len(predictions)
    texts: List[Optional[str]] = []
    for row in rows:
        text = row_class_name(row)
        if text is None and isinstance(row.get(TEXT_KEY), str):
            text = row[TEXT_KEY]
        texts.append(text)

    if None in texts:
        names_by_id = class_names_by_id(predictions.image_metadata) or {}
        class_ids = predictions.class_id.tolist()
        for position, text in enumerate(texts):
            if text is not None:
                continue
            name = names_by_id.get(class_ids[position])
            if name is None:
                raise ValueError(
                    f"OCR detection row {position} has no text: no string 'class' "
                    f"or 'text' metadata and class id {class_ids[position]} is not "
                    "in image_metadata['class_names']"
                )
            texts[position] = str(name)

    row_texts = np.array(texts, dtype=object)

    return row_texts


def _detect_direction(boxes: np.ndarray) -> str:
    mean_width = np.mean(boxes[:, 2] - boxes[:, 0])
    mean_height = np.mean(boxes[:, 3] - boxes[:, 1])
    direction = (
        "left_to_right" if mean_width > mean_height else "vertical_top_to_bottom"
    )

    return direction


def _tolerance_text(
    frame: _ReadingFrame, texts: np.ndarray, *, tolerance: int, delimiter: str
) -> str:
    corners = frame.boxes.round().astype(int)
    buckets = np.round(corners[:, 1] / tolerance) * tolerance
    lines: Dict[float, List[int]] = {}
    for position, bucket in enumerate(buckets):
        lines.setdefault(bucket, []).append(position)

    pieces: List[str] = []
    ordered_buckets = sorted(lines, reverse=frame.lines_reversed)
    for number, bucket in enumerate(ordered_buckets):
        members = np.array(lines[bucket])
        starts = corners[members, 0]
        order = (-starts).argsort() if frame.backward else starts.argsort()
        pieces.extend(texts[members[order]])
        if number < len(ordered_buckets) - 1:
            pieces.append(frame.line_separator)

    text = delimiter.join(pieces)

    return text


def _otsu_text(
    frame: _ReadingFrame,
    texts: np.ndarray,
    *,
    delimiter: str,
    threshold_multiplier: float,
) -> str:
    lines = _lines_by_centre(frame)
    ordered_lines = [_ordered_line(frame, line) for line in lines]
    line_gaps = [_normalized_gaps(frame, line) for line in ordered_lines]
    all_gaps = np.array([gap for gaps in line_gaps for gap in gaps.tolist()])
    threshold, is_bimodal = find_otsu_threshold(all_gaps)
    threshold *= threshold_multiplier

    word_separator = " " if delimiter == "" else delimiter
    line_texts = []
    for line, gaps in zip(ordered_lines, line_gaps):
        words = [[line[0]]]
        for member, gap in zip(line[1:], gaps):
            if is_bimodal and gap > threshold:
                words.append([member])
            else:
                words[-1].append(member)
        word_texts = [
            delimiter.join(texts[member] for member in word) for word in words
        ]
        line_texts.append(word_separator.join(word_texts))

    text = frame.line_separator.join(line_texts)

    return text


def _lines_by_centre(frame: _ReadingFrame) -> List[List[int]]:
    # Running-mean clustering of centres within half the median box height,
    # then lines in cross-line order.
    line_tolerance = np.median(frame.heights) * 0.5
    by_centre = np.argsort(frame.across)
    lines = [[by_centre[0]]]
    line_centre = frame.across[by_centre[0]]
    for member in by_centre[1:]:
        if abs(frame.across[member] - line_centre) <= line_tolerance:
            lines[-1].append(member)
            line_centre = np.mean(frame.across[lines[-1]])
        else:
            lines.append([member])
            line_centre = frame.across[member]

    centres = [np.mean(frame.across[line]) for line in lines]
    order = np.argsort(centres)
    if frame.lines_reversed:
        order = order[::-1]
    ordered = [lines[number] for number in order]

    return ordered


def _ordered_line(frame: _ReadingFrame, line: List[int]) -> List[int]:
    order = np.argsort(frame.along[line])
    if frame.backward:
        order = order[::-1]
    ordered = [line[number] for number in order]

    return ordered


def _normalized_gaps(frame: _ReadingFrame, line: List[int]) -> np.ndarray:
    # Edge gap between neighbours divided by their mean length; 0 for zero length.
    along = frame.along[line]
    lengths = frame.lengths[line]
    steps = along[:-1] - along[1:] if frame.backward else along[1:] - along[:-1]
    scales = (lengths[:-1] + lengths[1:]) / 2
    gaps = np.divide(
        steps - scales, scales, out=np.zeros_like(scales), where=scales > 0
    )

    return gaps


def find_otsu_threshold(gaps: np.ndarray) -> Tuple[float, bool]:
    """Split normalised gaps into within-word and between-word gaps.

    Otsu's method over a histogram of at most 50 bins chooses the bin centre
    maximising between-class variance. The split counts as bimodal only when
    the upper class mean exceeds 0.3 (real word gaps) and the class means are
    more than 0.3 apart. This is V1's rule unchanged.

    Args:
        gaps: Normalised gaps of all lines of one image.

    Returns:
        ``(threshold, is_bimodal)``; ``(0.0, False)`` for fewer than two gaps.
    """
    if len(gaps) < 2:
        return 0.0, False

    histogram, edges = np.histogram(gaps, bins=min(50, len(gaps)))
    centres = (edges[:-1] + edges[1:]) / 2
    best_threshold, best_variance = 0.0, 0.0
    below_mean, above_mean = 0.0, 0.0
    for candidate in centres:
        below = gaps[gaps <= candidate]
        above = gaps[gaps > candidate]
        if len(below) == 0 or len(above) == 0:
            continue
        variance = len(below) * len(above) * (below.mean() - above.mean()) ** 2
        if variance > best_variance:
            best_threshold, best_variance = candidate, variance
            below_mean, above_mean = below.mean(), above.mean()

    is_bimodal = bool(above_mean > 0.3 and abs(above_mean - below_mean) > 0.3)

    return best_threshold, is_bimodal


def _collimate_text(
    frame: _ReadingFrame, texts: np.ndarray, *, delimiter: str, tolerance: int
) -> str:
    remaining = sorted(
        range(len(texts)),
        key=lambda member: frame.along[member],
        reverse=frame.backward,
    )
    lines = [[remaining.pop(0)]]
    while remaining:
        extended = False
        for line in lines:
            for member in list(remaining):
                if _follows(frame, line[-1], member, tolerance=tolerance):
                    line.append(member)
                    remaining.remove(member)
                    extended = True
        if not extended and remaining:
            lines.append([remaining.pop(0)])

    lines.sort(
        key=lambda line: sum(frame.across[member] for member in line) / len(line),
        reverse=frame.lines_reversed,
    )
    line_texts = [delimiter.join(texts[member] for member in line) for line in lines]
    text = frame.line_separator.join(line_texts)

    return text


def _follows(frame: _ReadingFrame, parent: int, child: int, *, tolerance: int) -> bool:
    slack = max(frame.heights[parent], frame.heights[child]) / 2 + tolerance
    aligned = abs(frame.across[parent] - frame.across[child]) < slack
    if frame.backward:
        in_order = frame.along[parent] >= frame.along[child]
    else:
        in_order = frame.along[parent] <= frame.along[child]
    follows = bool(aligned and in_order)

    return follows
