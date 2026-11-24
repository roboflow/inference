"""``roboflow_core/motion_detection@v1``: moving regions of each source's frames.

A port of V1 ``roboflow_core/motion_detection@v1`` (latest: its tensor
variant). It shares the per-source MOG2 models of
``roboflow_core/background_subtraction@v1`` and adds warmup, contours and an alarm::

    mask, state = source_models.apply(image)       # CPU boundary; one record
    if suppress_first_detections and state.frames_seen <= history:
        return no motion                            # warmup
    mask = morphological_open(mask, ellipse(kernel_size))
    contours = external contours with > 2 points and area > minimum_contour_area
    if detection_zone is not None: contours = contours clipped to the region
    polygons = contours simplified by 1% of their perimeter, with >= 3 points
    motion = any polygons
    alarm = motion and not state.previous_motion
    state.previous_motion = motion
    detections, motion_zones = boxes and polygons of these same regions

``detection_zone`` sets the region where motion counts::

    None (default)                   whole image
    []                               no region: never motion
    [[x, y], ...]                    one polygon (at least 3 points)
    [polygon or None, ...]           union of the polygons; None entries
                                     (dynamic zone rows without a contour)
                                     are skipped; none left: no region
    JSON text of any of these        only from a selected string

So ``$steps.<dynamic_zone>.zones`` connects directly, and an empty zones
result watches nothing instead of the whole image.

Detections are ``Detections`` on the image's device, class ``motion``
(id 0), confidence 1, in the image's own pixel coordinates; their image
metadata carries the image's frame, so geometry maps back to its parents.
``motion_zones`` holds the same polygons as ``[[x, y], ...]`` integer lists.

Per-source state is one ``SourceState`` record of ``background_subtraction``:
model, frames seen and previous motion restart together, on a lower frame
number, a size change or ``reset_state()``. A restart starts warmup again and
forgets the previous motion, so the alarm can fire anew. Nothing is locked;
the engine calls one step one call at a time, in pulse order. Differences
from V1:

* Warmup counts frames since the source's model (re)started. V1 counted only
  suppressed frames, so turning suppression on later started a new warmup.
* A self-intersecting contour is repaired before zone clipping; V1 silently
  dropped it.
* A JSON zone is accepted from a selected string; a literal zone is a list.
* ``detection_zone=[]`` watches nothing; V1 watched the whole image. Use
  ``None`` (the default) for the whole image.
* A list of polygons is accepted and used as their union; V1 raised
  ``TypeError`` on its first frame with motion.
* Motion, alarm and previous motion count only emitted polygons, with or
  without a zone. A region whose simplified outline keeps fewer than 3
  points is not emitted and is not motion: a clipped subpixel zone, or a
  strip a few pixels wide (a 3x110 px strip passes the default area
  filter). V1 counted it: motion without polygons (an alarm if motion
  started there), then no alarm on the next usable region, which alarms
  here.
"""

import json
import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4
import cv2
import numpy as np
import torch
from pydantic import Field, StrictBool, StrictInt, field_validator
from roboflow_workflows.v2.blocks.analytics.background_subtraction.v1 import DEFAULT_HISTORY, DEFAULT_MAX_SOURCES, DEFAULT_THRESHOLD, PositiveInt, SourceModels, check_positive_int, source_key
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.prediction_ops import CLASS_NAMES_KEY
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid
from inference_models.models.base.object_detection import Detections


__all__ = [
    "DEFAULT_MINIMUM_CONTOUR_AREA",
    "DEFAULT_MORPHOLOGICAL_KERNEL_SIZE",
    "MOTION_CLASS_NAME",
    "MotionDetectionBlock",
]

DEFAULT_MINIMUM_CONTOUR_AREA = 200

DEFAULT_MORPHOLOGICAL_KERNEL_SIZE = 3

MOTION_CLASS_NAME = "motion"

_PREDICTION_TYPE_KEY = "prediction_type"

_DETECTION_ID_KEY = "detection_id"

_SIMPLIFY_FRACTION = 0.01

class MotionDetectionBlock(Block):
    """Detect moving regions against each source's learned background.

    ``motion`` is whether any region remains after filtering; ``alarm`` is
    true only on the frame where a source's motion starts; ``detections``
    and ``motion_zones`` describe the regions. OpenCV MOG2 and the contour
    search run on the CPU; GPU frames are copied to the host.
    """

    type = "roboflow_core/motion_detection@v1"
    metadata = {
        "name": "Motion Detection",
        "port_of": "roboflow_core/motion_detection@v1",
        "section": "video",
    }
    outputs = {
        "motion": Output(
            BOOLEAN_KIND,
            source="image",
            description="Whether any moving region remains after filtering.",
        ),
        "alarm": Output(
            BOOLEAN_KIND,
            source="image",
            description=(
                "True on the frame where this source's motion starts: motion "
                "now, none on its previous evaluated frame."
            ),
        ),
        "detections": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description=(
                "One box per moving region, class 'motion', confidence 1, in "
                "the image's pixel coordinates."
            ),
        ),
        "motion_zones": Output(
            LIST_OF_VALUES_KIND,
            source="image",
            description="Polygon of each region as [[x, y], ...] integer points.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "Frame of a video source. Its video_metadata names the source "
                "and gives the frame number. Images without a video identifier "
                "use their input source, else share one 'default' model."
            ),
            examples=["$inputs.image"],
        )
        minimum_contour_area: PositiveInt | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_MINIMUM_CONTOUR_AREA,
            description="Regions of this area in square pixels or less are ignored.",
            examples=[DEFAULT_MINIMUM_CONTOUR_AREA, "$inputs.minimum_area"],
        )
        morphological_kernel_size: PositiveInt | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_MORPHOLOGICAL_KERNEL_SIZE,
            description=(
                "Diameter of the elliptic opening that removes specks from the "
                "mask before regions are found."
            ),
            examples=[DEFAULT_MORPHOLOGICAL_KERNEL_SIZE],
        )
        threshold: PositiveInt | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_THRESHOLD,
            description=(
                "MOG2 varThreshold: squared colour distance from the "
                "background that counts as foreground. Lower is more sensitive."
            ),
            examples=[DEFAULT_THRESHOLD, "$inputs.threshold"],
        )
        history: PositiveInt | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_HISTORY,
            description=(
                "Processed frames that shape each source's background; also "
                "the warmup length."
            ),
            examples=[DEFAULT_HISTORY],
        )
        detection_zone: Optional[List[Any]] | Ref(LIST_OF_VALUES_KIND, STRING_KIND) = (
            Field(
                default=None,
                description=(
                    "Region where motion counts; regions are clipped to it. One "
                    "polygon of at least 3 [x, y] image points, or a list of "
                    "polygons used as their union (null entries skipped), e.g. "
                    "a dynamic zone's zones. Null watches the whole image; [] "
                    "or a list without polygons watches nothing (V1: [] was "
                    "the whole image). A selected value may also be JSON text."
                ),
                examples=[
                    None,
                    [[0, 0], [640, 0], [640, 360]],
                    "$steps.dynamic_zone.zones",
                ],
            )
        )
        suppress_first_detections: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "Report no motion until a source's model has seen history "
                "frames since it (re)started."
            ),
            examples=[True, "$inputs.warmup"],
        )
        max_sources: StrictInt = Field(
            default=DEFAULT_MAX_SOURCES,
            ge=1,
            description=(
                "Most sources with a background model (~216 MB per 1080p "
                "source). A frame of one more source fails with "
                "BackgroundModelCapacityError; no model is evicted."
            ),
            examples=[DEFAULT_MAX_SOURCES],
        )

        @field_validator("detection_zone")
        @classmethod
        def _literal_zone_is_a_region(cls, value: Any) -> Any:
            # A selector is a string here; its value is checked in run().
            if isinstance(value, list):
                zone_region(value)

            return value

    def __init__(self) -> None:
        self._sources = SourceModels(self.type)

    def run(
        self,
        *,
        image: ImageData,
        minimum_contour_area: int = DEFAULT_MINIMUM_CONTOUR_AREA,
        morphological_kernel_size: int = DEFAULT_MORPHOLOGICAL_KERNEL_SIZE,
        threshold: int = DEFAULT_THRESHOLD,
        history: int = DEFAULT_HISTORY,
        detection_zone: Any = None,
        suppress_first_detections: bool = True,
        max_sources: int = DEFAULT_MAX_SOURCES,
    ) -> Dict[str, Any]:
        """Update this source's model with one frame and find moving regions.

        Args:
            image: Frame of a video source.
            minimum_contour_area: Regions of this area or less are ignored.
            morphological_kernel_size: Diameter of the opening kernel.
            threshold: MOG2 ``varThreshold``.
            history: Frames the model remembers; also the warmup length.
            detection_zone: None (whole image), a polygon, a list of
                polygons (union), or JSON text of one of them; [] watches
                nothing.
            suppress_first_detections: Whether warmup frames report nothing.
            max_sources: Most sources with a model.

        Returns:
            ``motion``, ``alarm``, ``detections`` and ``motion_zones``.

        Raises:
            ValueError: When a number is not a positive integer,
                ``suppress_first_detections`` is not a bool, or
                ``detection_zone`` is not a valid region.
            BackgroundModelCapacityError: When this is the first frame of a
                source and ``max_sources`` sources already have a model.
        """
        check_positive_int(
            self.type,
            minimum_contour_area=minimum_contour_area,
            morphological_kernel_size=morphological_kernel_size,
            threshold=threshold,
            history=history,
            max_sources=max_sources,
        )
        if type(suppress_first_detections) is not bool:
            raise ValueError(
                f"{self.type}: suppress_first_detections must be a bool, got "
                f"{suppress_first_detections!r}"
            )
        zone = zone_region(detection_zone)
        mask, state = self._sources.apply(
            image,
            source=source_key(self, image),
            history=history,
            threshold=threshold,
            max_sources=max_sources,
        )
        if suppress_first_detections and state.frames_seen <= history:
            return _result(image, motion=False, alarm=False, polygons=[])

        contours = _moving_contours(
            mask,
            kernel_size=morphological_kernel_size,
            minimum_area=minimum_contour_area,
        )
        if zone is not None:
            contours = _clip_to_zone(contours, zone)
        polygons = _simplified_polygons(contours)
        motion = len(polygons) > 0
        alarm = motion and not state.previous_motion
        state.previous_motion = motion
        result = _result(image, motion=motion, alarm=alarm, polygons=polygons)

        return result

    def reset_state(self) -> None:
        """Forget every source's model, warmup and previous motion."""
        self._sources.clear()

def zone_region(value: Any) -> Optional[BaseGeometry]:
    """Parse a detection zone into the region where motion counts.

    Args:
        value: The ``detection_zone`` value: None, a polygon, a list of
            polygons or None entries, or JSON text of one of them.

    Returns:
        ``None`` for the whole image; otherwise the region, which is empty
        for ``[]`` or a list without polygons.

    Raises:
        ValueError: When the value is not valid JSON, not a list, or holds a
            polygon that is not at least 3 finite ``[x, y]`` points with a
            positive area.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as error:
            raise ValueError(f"detection_zone is not valid JSON: {error}") from error
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "detection_zone must be a list of at least 3 [x, y] points or a "
            f"list of such polygons, got {value!r}"
        )

    if len(value) > 0 and _is_point(value[0]):
        polygons = [_zone_polygon(value)]
    else:
        polygons = [_zone_polygon(item) for item in value if item is not None]
    if not polygons:
        return Polygon()

    # One polygon is used as given, so a single zone clips exactly as in V1.
    region = polygons[0] if len(polygons) == 1 else unary_union(polygons)

    return region

def _zone_polygon(value: Any) -> Polygon:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(
            f"detection_zone must be a list of at least 3 [x, y] points, got {value!r}"
        )
    if not all(_is_point(point) for point in value):
        raise ValueError(
            f"detection_zone points must be [x, y] pairs of finite numbers, got {value!r}"
        )

    polygon = Polygon([(float(x), float(y)) for x, y in value])
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError(
            "detection_zone must be a simple polygon with a positive area, got "
            f"{value!r}"
        )

    return polygon

def _is_point(value: Any) -> bool:
    is_pair = isinstance(value, (list, tuple)) and len(value) == 2

    return is_pair and all(_is_finite_number(coordinate) for coordinate in value)

def _is_finite_number(value: Any) -> bool:
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

    return is_number and math.isfinite(value)

@lru_cache(maxsize=32)
def _opening_kernel(size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    return kernel

def _moving_contours(
    mask: np.ndarray, *, kernel_size: int, minimum_area: int
) -> List[np.ndarray]:
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _opening_kernel(kernel_size))
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [
        contour
        for contour in contours
        if len(contour) > 2 and cv2.contourArea(contour) > minimum_area
    ]

    return kept

def _clip_to_zone(
    contours: Sequence[np.ndarray], zone: BaseGeometry
) -> List[np.ndarray]:
    # V1 rule: each overlapping polygon part becomes a contour, its float
    # corners truncated to integer pixels.
    clipped = []
    for contour in contours:
        region = make_valid(Polygon(contour.reshape(-1, 2)))
        for part in _polygons_of(region.intersection(zone)):
            corners = list(part.exterior.coords[:-1])
            if len(corners) >= 3:
                points = np.array([[int(x), int(y)] for x, y in corners], np.int32)
                clipped.append(points.reshape(-1, 1, 2))

    return clipped

def _polygons_of(geometry: Any) -> List[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]

    parts = [
        polygon
        for part in getattr(geometry, "geoms", ())
        for polygon in _polygons_of(part)
    ]

    return parts

def _simplified_polygons(contours: Sequence[np.ndarray]) -> List[np.ndarray]:
    polygons = []
    for contour in contours:
        epsilon = _SIMPLIFY_FRACTION * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) >= 3:
            polygons.append(simplified.reshape(-1, 2))

    return polygons

def _result(
    image: ImageData, *, motion: bool, alarm: bool, polygons: Sequence[np.ndarray]
) -> Dict[str, Any]:
    result = {
        "motion": motion,
        "alarm": alarm,
        "detections": _detections(image, polygons),
        "motion_zones": [polygon.tolist() for polygon in polygons],
    }

    return result

def _detections(image: ImageData, polygons: Sequence[np.ndarray]) -> Detections:
    # V1 boxes: cv2.boundingRect, so x2 = x + width (one past the last pixel).
    boxes = []
    for polygon in polygons:
        x, y, width, height = cv2.boundingRect(polygon)
        boxes.append([x, y, x + width, y + height])
    rows = len(boxes)
    device = image.device
    image_metadata = image.prediction_metadata()
    image_metadata[CLASS_NAMES_KEY] = {0: MOTION_CLASS_NAME}
    image_metadata[_PREDICTION_TYPE_KEY] = "object-detection"
    detections = Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(rows, 4).to(device),
        class_id=torch.zeros((rows,), dtype=torch.int64, device=device),
        confidence=torch.ones((rows,), dtype=torch.float32, device=device),
        image_metadata=image_metadata,
        bboxes_metadata=[{_DETECTION_ID_KEY: str(uuid4())} for _ in range(rows)],
    )

    return with_row_identity(detections)
