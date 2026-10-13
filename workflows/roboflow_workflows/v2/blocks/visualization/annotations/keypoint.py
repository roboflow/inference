"""Keypoints: ``v2/keypoint_visualization@v1``.

Port of V1 ``roboflow_core/keypoint_visualization@v1``. Pixels equal V1::

    edge          cv2.line between skeleton pairs (sv.EdgeAnnotator)
    vertex        filled cv2.circle per keypoint (sv.VertexAnnotator)
    vertex_label  keypoint index on a rounded label (sv.VertexLabelAnnotator)

A keypoint at (0, 0) of its own image is not drawn, as in V1; models put
keypoints below their threshold there. This is decided before keypoints of
another frame are mapped into the image, so hidden points stay hidden. ``edges`` pairs are 1-based, the supervision skeleton
convention V1 used; without ``edges`` the supervision skeleton for the
keypoint count is used, and none is drawn when there is no such skeleton.

Keypoint-only predictions ``(KeyPoints, None)`` are drawn like any other.
Host boundary: one transfer of the keypoint coordinates; two when they are
mapped from another frame (local and mapped).
"""

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    AnnotationBlock,
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    parse_color,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    AnnotationParams,
    Number,
    check_literals,
    checked_choice,
    checked_int,
    checked_number,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import placed_in
from roboflow_workflows.v2.blocks.visualization.annotations.strokes import (
    Strokes,
    bounds_scan,
    stroke_reach,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text import (
    TextStyle,
    paint_labels,
    vertex_labels,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.predictions import KEYPOINT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import Ref
from supervision.key_points.skeletons import SKELETONS_BY_VERTEX_COUNT

from inference_models.models.base.keypoints_detection import KeyPoints

ANNOTATOR_TYPES = ("edge", "vertex", "vertex_label")


class KeypointVisualizationBlock(AnnotationBlock):
    """Draw keypoint skeletons, dots or index labels onto a copy of the image.

    Example: a 17-keypoint COCO pose drawn with ``annotator_type="edge"``
    connects the joints with V1's COCO skeleton in ``#A351FB``.
    """

    type = "v2/keypoint_visualization@v1"
    metadata = {
        "name": "Keypoint Visualization",
        "port_of": "roboflow_core/keypoint_visualization@v1",
        "section": "visualization",
    }

    class Params(AnnotationParams):
        predictions: Ref(KEYPOINT_DETECTION_PREDICTION_KIND) = Field(
            description="Keypoint predictions, with or without boxes.",
            examples=["$steps.keypoint_detection.predictions"],
        )
        annotator_type: Literal[ANNOTATOR_TYPES] | Ref(STRING_KIND) = Field(
            default="edge",
            description=(
                "edge: lines between connected keypoints; vertex: a dot per "
                "keypoint; vertex_label: each keypoint's index on a label."
            ),
            examples=["edge", "vertex", "$inputs.annotator_type"],
        )
        color: str | Ref(STRING_KIND) = Field(
            default="#A351FB",
            description="Color of lines, dots or label backgrounds.",
            examples=["#A351FB", "GREEN", "$inputs.color"],
        )
        text_color: str | Ref(STRING_KIND) = Field(
            default="black",
            description="Label text color (vertex_label).",
            examples=["black", "WHITE"],
        )
        text_scale: Number | Ref(FLOAT_KIND) = Field(
            default=0.5,
            description="Label font scale (vertex_label); > 0.",
            examples=[0.5],
        )
        text_thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1,
            description="Label text stroke width (vertex_label); >= 1.",
            examples=[1],
        )
        text_padding: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description="Pixels around label text (vertex_label); >= 0.",
            examples=[10],
        )
        thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=2,
            description="Line width (edge); >= 1.",
            examples=[2],
        )
        radius: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description=(
                "Dot radius (vertex) and label corner radius (vertex_label); >= 0."
            ),
            examples=[10],
        )
        edges: Optional[List[List[StrictInt]]] | Ref(LIST_OF_VALUES_KIND) = Field(
            default=None,
            description=(
                "Connected keypoints as 1-based [a, b] pairs (edge). Unset: the "
                "supervision skeleton for the keypoint count."
            ),
            examples=[[[1, 2], [2, 3]], "$inputs.edges"],
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "KeypointVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(
        self,
        *,
        image: ImageData,
        predictions: Tuple[KeyPoints, Any],
        annotator_type: str,
        color: str,
        text_color: str,
        text_scale: float,
        text_thickness: int,
        text_padding: int,
        thickness: int,
        radius: int,
        edges: Optional[Sequence[Sequence[int]]],
    ) -> Dict[str, Any]:
        """Draw the keypoints of ``predictions`` onto a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: ``(KeyPoints, Detections | None)``.
            annotator_type: ``edge``, ``vertex`` or ``vertex_label``.
            color: Line, dot or label background color.
            text_color: Label text color.
            text_scale: Label font scale.
            text_thickness: Label text stroke width.
            text_padding: Label padding.
            thickness: Line width.
            radius: Dot radius or label corner radius.
            edges: 1-based keypoint pairs, or ``None``.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, edges naming keypoints that do
                not exist, or predictions of another workflow input.
        """
        values = dict(
            annotator_type=annotator_type,
            color=color,
            text_color=text_color,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            thickness=thickness,
            radius=radius,
            edges=edges,
        )
        for name, check in _CHECKS.items():
            check(values[name])
        canvas = canvas_of(image, block=self.type)
        local_key_points = predictions[0]
        key_points, _ = placed_in(predictions, image, block=self.type)
        points = _host_points(key_points)
        local = (
            points if key_points is local_key_points else _host_points(local_key_points)
        )
        visible = _visible(local)
        rgb = parse_color(color, field="color")

        if annotator_type == "vertex_label":
            anchors, texts = [], []
            for skeleton, shown in zip(points, visible):
                for index, ((x, y), is_shown) in enumerate(zip(skeleton, shown)):
                    if is_shown:
                        anchors.append((int(x), int(y)))
                        texts.append(str(index))
            labels = vertex_labels(
                texts,
                anchors,
                style=TextStyle(text_scale, text_thickness, text_padding, radius),
                background_rgb=rgb,
                text_rgb=parse_color(text_color, field="text_color"),
            )
            paint_labels(canvas, labels)
        else:
            strokes = Strokes(image.size_hw)
            for skeleton, shown in zip(points, visible):
                if annotator_type == "edge":
                    _add_edges(
                        strokes,
                        skeleton,
                        shown,
                        edges=edges,
                        color=rgb,
                        thickness=thickness,
                    )
                else:
                    _add_vertices(strokes, skeleton, shown, color=rgb, radius=radius)
            strokes.paint(canvas)
        result = annotated(image, canvas)

        return result


def _host_points(key_points: KeyPoints) -> np.ndarray:
    # One transfer, then float64 on the host (exact for float32/int
    # coordinates; MPS has no float64).
    if len(key_points) == 0:
        return np.zeros((0, 0, 2), dtype=np.float64)

    points = key_points.xy.cpu().to(torch.float64).numpy()

    return points


def _visible(points: np.ndarray) -> np.ndarray:
    # sv annotators skip np.allclose(xy, 0), judged in the keypoints' own image.
    visible = ~np.all(np.abs(points) <= 1e-8, axis=-1)

    return visible


def _add_edges(
    strokes: Strokes,
    skeleton: np.ndarray,
    shown: np.ndarray,
    *,
    edges: Optional[Sequence[Sequence[int]]],
    color: Any,
    thickness: int,
) -> None:
    pairs = edges if edges is not None else SKELETONS_BY_VERTEX_COUNT.get(len(skeleton))
    if not pairs:
        return

    reach = stroke_reach(thickness)
    for first, second in pairs:
        if first > len(skeleton) or second > len(skeleton):
            raise ValueError(
                f"edges pair {[first, second]} names a keypoint beyond the "
                f"{len(skeleton)} keypoints of these predictions (edges are 1-based)"
            )
        if not (shown[first - 1] and shown[second - 1]):
            continue
        start, end = skeleton[first - 1], skeleton[second - 1]
        a = (int(start[0]), int(start[1]))
        b = (int(end[0]), int(end[1]))
        strokes.add(
            lambda mask, a=a, b=b: cv2.line(mask, a, b, 1, thickness),
            color=color,
            scan=bounds_scan(np.array([a, b]), reach=reach),
        )


def _add_vertices(
    strokes: Strokes,
    skeleton: np.ndarray,
    shown: np.ndarray,
    *,
    color: Any,
    radius: int,
) -> None:
    for (x, y), is_shown in zip(skeleton, shown):
        if not is_shown:
            continue
        centre = (int(x), int(y))
        strokes.add(
            lambda mask, centre=centre: cv2.circle(mask, centre, radius, 1, -1),
            color=color,
            scan=bounds_scan(np.array([centre]), reach=radius + 2),
        )


def _checked_edges(value: Any) -> Any:
    if value is None:
        return None
    pairs_ok = isinstance(value, (list, tuple)) and all(
        isinstance(pair, (list, tuple))
        and len(pair) == 2
        and all(
            isinstance(index, int) and not isinstance(index, bool) and index >= 1
            for index in pair
        )
        for pair in value
    )
    if not pairs_ok:
        raise ValueError(
            f"edges must be a list of 1-based [a, b] keypoint pairs, got {value!r}"
        )

    return value


_CHECKS = {
    "annotator_type": lambda value: checked_choice(
        value, name="annotator_type", choices=ANNOTATOR_TYPES
    ),
    "color": lambda value: parse_color(value, field="color"),
    "text_color": lambda value: parse_color(value, field="text_color"),
    "text_scale": lambda value: checked_number(
        value, name="text_scale", minimum=0.0, above_minimum=True
    ),
    "text_thickness": lambda value: checked_int(
        value, name="text_thickness", minimum=1
    ),
    "text_padding": lambda value: checked_int(value, name="text_padding", minimum=0),
    "thickness": lambda value: checked_int(value, name="thickness", minimum=1),
    "radius": lambda value: checked_int(value, name="radius", minimum=0),
    "edges": _checked_edges,
}
