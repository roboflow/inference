"""Move native predictions between related image frames by whole pixels.

``v2/dynamic_crop@v1`` moves one detection into its crop, and
``v2/detections_stitch@v1`` moves slice predictions back into the image they
were cut from. Both are integer translations::

    target_xy = local_xy + (dx, dy)

``offset_between`` derives ``(dx, dy)`` from two frames that share a root and a
scale. A scale change is not a translation and is refused, as V1 refused to
stitch scaled predictions. ``translate_prediction`` applies the offset with
``data.coordinates.prediction_to_frame``: boxes, keypoints, per-row point
metadata and masks move; the result describes the target image.
"""

from typing import Any, Tuple

import numpy as np
from roboflow_workflows.v2.data.coordinates import prediction_to_frame
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import FrameMapping, ImageData

# Offsets closer than this to an integer count as that integer.
INTEGER_TOLERANCE = 1e-6

Offset = Tuple[int, int]


def offset_between(frame: FrameRef, *, target: ImageData) -> Offset:
    """Return the whole-pixel offset that moves ``frame`` coordinates into ``target``.

    Args:
        frame: Frame of the predictions to move.
        target: Image to move them into; shares ``frame``'s root.

    Returns:
        ``(dx, dy)`` such that ``target_xy = local_xy + (dx, dy)``.

    Raises:
        ValueError: When the frames have different roots, different scales
            (a resize lies between them) or a fractional offset.
    """
    local, outer = frame.root, target.root
    if local.frame_id != outer.frame_id:
        raise ValueError(
            f"predictions of image {frame.image_id!r} (root {local.frame_id!r}) "
            f"cannot be placed in image {target.image_id!r} (root "
            f"{outer.frame_id!r}): they come from different workflow inputs"
        )
    if not np.allclose(local.scale_xy, outer.scale_xy, rtol=0, atol=INTEGER_TOLERANCE):
        raise ValueError(
            f"predictions of image {frame.image_id!r} are scaled {local.scale_xy} "
            f"relative to the root, image {target.image_id!r} {outer.scale_xy}; "
            "only crops and slices without resizing can be moved by whole pixels"
        )

    offset = []
    for axis in range(2):
        shift = (local.offset_xy[axis] - outer.offset_xy[axis]) / outer.scale_xy[axis]
        if abs(shift - round(shift)) > INTEGER_TOLERANCE:
            raise ValueError(
                f"image {frame.image_id!r} sits at a fractional offset {shift} in "
                f"image {target.image_id!r}"
            )
        offset.append(int(round(shift)))

    return offset[0], offset[1]


def translate_prediction(prediction: Any, *, offset: Offset, target: ImageData) -> Any:
    """Move a native prediction by whole pixels into ``target``'s frame.

    Args:
        prediction: ``Detections``, ``InstanceDetections`` or a
            ``(KeyPoints, Optional[Detections])`` tuple. Not modified.
        offset: ``(dx, dy)`` added to every coordinate.
        target: Image the result describes; masks get its size.

    Returns:
        A prediction of the same type; see ``prediction_to_frame``.
    """
    mapping = FrameMapping(
        frame_id=target.image_id, frame_size_hw=target.size_hw, offset_xy=offset
    )
    moved = prediction_to_frame(
        prediction, mapping=mapping, frame=FrameRef.from_image(target)
    )

    return moved
