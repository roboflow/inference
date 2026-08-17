from collections import Counter, defaultdict, deque
from math import sqrt
from typing import DefaultDict, Deque, Optional, Sequence, Tuple, Union

import numpy as np
import supervision as sv

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks


def empty_detections_like(
    detections: Union[Detections, InstanceDetections],
) -> Union[Detections, InstanceDetections]:
    bboxes_metadata = [] if isinstance(detections.bboxes_metadata, list) else None
    if isinstance(detections, InstanceDetections):
        mask = detections.mask
        if isinstance(mask, InstancesRLEMasks):
            mask = InstancesRLEMasks(image_size=mask.image_size, masks=[])
        else:
            mask = mask[0:0]
        return InstanceDetections(
            xyxy=detections.xyxy[0:0],
            class_id=detections.class_id[0:0],
            confidence=detections.confidence[0:0],
            mask=mask,
            image_metadata=detections.image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    return Detections(
        xyxy=detections.xyxy[0:0],
        class_id=detections.class_id[0:0],
        confidence=detections.confidence[0:0],
        image_metadata=detections.image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


_DEFAULT_LINE_ANCHORS = (
    sv.Position.TOP_LEFT,
    sv.Position.TOP_RIGHT,
    sv.Position.BOTTOM_LEFT,
    sv.Position.BOTTOM_RIGHT,
)


def _stack_anchor_coordinates(
    xyxy: np.ndarray, anchors: Sequence[sv.Position]
) -> np.ndarray:
    center_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
    center_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
    result = np.empty((len(anchors), len(xyxy), 2), dtype=xyxy.dtype)
    for index, anchor in enumerate(anchors):
        if anchor == sv.Position.CENTER:
            result[index, :, 0] = center_x
            result[index, :, 1] = center_y
        elif anchor == sv.Position.CENTER_OF_MASS:
            raise ValueError(
                "Cannot use `Position.CENTER_OF_MASS` without a detection mask."
            )
        elif anchor == sv.Position.CENTER_LEFT:
            result[index, :, 0] = xyxy[:, 0]
            result[index, :, 1] = center_y
        elif anchor == sv.Position.CENTER_RIGHT:
            result[index, :, 0] = xyxy[:, 2]
            result[index, :, 1] = center_y
        elif anchor == sv.Position.BOTTOM_CENTER:
            result[index, :, 0] = center_x
            result[index, :, 1] = xyxy[:, 3]
        elif anchor == sv.Position.BOTTOM_LEFT:
            result[index, :, 0] = xyxy[:, 0]
            result[index, :, 1] = xyxy[:, 3]
        elif anchor == sv.Position.BOTTOM_RIGHT:
            result[index, :, 0] = xyxy[:, 2]
            result[index, :, 1] = xyxy[:, 3]
        elif anchor == sv.Position.TOP_CENTER:
            result[index, :, 0] = center_x
            result[index, :, 1] = xyxy[:, 1]
        elif anchor == sv.Position.TOP_LEFT:
            result[index, :, 0] = xyxy[:, 0]
            result[index, :, 1] = xyxy[:, 1]
        elif anchor == sv.Position.TOP_RIGHT:
            result[index, :, 0] = xyxy[:, 2]
            result[index, :, 1] = xyxy[:, 1]
        else:
            raise ValueError(f"{anchor} is not supported.")
    return result


def anchor_coordinates(xyxy: np.ndarray, anchor: sv.Position) -> np.ndarray:
    return _stack_anchor_coordinates(xyxy, (anchor,))[0]


class LeanLineZone:
    def __init__(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        triggering_anchors: Optional[Sequence[sv.Position]] = None,
    ) -> None:
        self.start = start
        self.end = end
        self.triggering_anchors = tuple(
            _DEFAULT_LINE_ANCHORS if triggering_anchors is None else triggering_anchors
        )
        if not self.triggering_anchors:
            raise ValueError("Triggering anchors cannot be empty.")

        minimum_crossing_threshold = 1
        self.crossing_history_length = max(2, minimum_crossing_threshold + 1)
        self.crossing_state_history: DefaultDict[
            Tuple[int, Optional[int]], Deque[bool]
        ] = defaultdict(lambda: deque(maxlen=self.crossing_history_length))
        self._in_count_per_class: Counter = Counter()
        self._out_count_per_class: Counter = Counter()

        start_x, start_y = start
        end_x, end_y = end
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        magnitude = sqrt(delta_x**2 + delta_y**2)
        if magnitude == 0:
            raise ValueError("The magnitude of the vector cannot be zero.")

        unit_vector_x = delta_x / magnitude
        unit_vector_y = delta_y / magnitude
        perpendicular_vector_x = -unit_vector_y
        perpendicular_vector_y = unit_vector_x
        self._limit_starts = np.array([start, end], dtype=float)
        self._limit_vectors = np.array(
            [
                [perpendicular_vector_x, perpendicular_vector_y],
                [-perpendicular_vector_x, -perpendicular_vector_y],
            ],
            dtype=float,
        )
        self._line_vector = np.array([delta_x, delta_y], dtype=float)

    @property
    def in_count(self) -> int:
        return sum(self._in_count_per_class.values())

    @property
    def out_count(self) -> int:
        return sum(self._out_count_per_class.values())

    def trigger(
        self, xyxy: np.ndarray, tracker_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        crossed_in = np.full(len(xyxy), False)
        crossed_out = np.full(len(xyxy), False)
        if len(xyxy) == 0:
            return crossed_in, crossed_out

        all_anchors = _stack_anchor_coordinates(xyxy, self.triggering_anchors)
        limit_deltas = all_anchors[None, :, :, :] - self._limit_starts[:, None, None, :]
        limit_cross_products = (
            self._limit_vectors[:, 0, None, None] * limit_deltas[:, :, :, 1]
            - self._limit_vectors[:, 1, None, None] * limit_deltas[:, :, :, 0]
        )
        in_limits = np.all(
            (limit_cross_products[0] > 0) == (limit_cross_products[1] > 0),
            axis=0,
        )

        line_deltas = all_anchors - np.asarray(self.start)
        triggers = (
            self._line_vector[0] * line_deltas[:, :, 1]
            - self._line_vector[1] * line_deltas[:, :, 0]
        ) < 0
        has_any_left_trigger = np.any(triggers, axis=0)
        has_any_right_trigger = np.any(~triggers, axis=0)

        class_id = None
        for index, tracker_id in enumerate(tracker_ids):
            if not in_limits[index]:
                continue
            if has_any_left_trigger[index] and has_any_right_trigger[index]:
                continue

            tracker_state = has_any_left_trigger[index]
            crossing_history = self.crossing_state_history[(tracker_id, class_id)]
            crossing_history.append(tracker_state)
            if len(crossing_history) < self.crossing_history_length:
                continue

            oldest_state = crossing_history[0]
            if crossing_history.count(oldest_state) > 1:
                continue

            if tracker_state:
                self._in_count_per_class[class_id] += 1
                crossed_in[index] = True
            else:
                self._out_count_per_class[class_id] += 1
                crossed_out[index] = True

        return crossed_in, crossed_out


class LeanPolygonZone:
    def __init__(
        self,
        polygon: np.ndarray,
        triggering_anchors: Sequence[sv.Position],
    ) -> None:
        self.polygon = polygon.astype(int)
        self.triggering_anchors = tuple(triggering_anchors)
        if not self.triggering_anchors:
            raise ValueError("Triggering anchors cannot be empty.")

        self.current_count = 0
        x_max, y_max = np.max(polygon, axis=0)
        self.mask = sv.polygon_to_mask(
            polygon=polygon, resolution_wh=(x_max + 2, y_max + 2)
        )

    def trigger(self, xyxy: np.ndarray) -> np.ndarray:
        if len(xyxy) == 0:
            self.current_count = 0
            return np.array([], dtype=bool)

        all_anchors = np.ceil(
            _stack_anchor_coordinates(xyxy, self.triggering_anchors)
        ).astype(int)
        mask_h, mask_w = self.mask.shape
        x, y = all_anchors[:, :, 0], all_anchors[:, :, 1]
        in_bounds = (x >= 0) & (y >= 0) & (x < mask_w) & (y < mask_h)
        x_safe = np.clip(x, 0, mask_w - 1)
        y_safe = np.clip(y, 0, mask_h - 1)
        is_in_zone = np.all(in_bounds & self.mask[y_safe, x_safe], axis=0)
        self.current_count = int(np.sum(is_in_zone))
        return is_in_zone.astype(bool)
