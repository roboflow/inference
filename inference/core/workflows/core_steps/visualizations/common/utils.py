import inspect

import numpy as np
import supervision as sv


def build_supervision_keypoints(
    *,
    xy: np.ndarray,
    confidence: np.ndarray,
    class_id: np.ndarray,
    data: dict | None = None,
) -> sv.KeyPoints:
    kwargs = {"xy": xy, "class_id": class_id}
    if data is not None:
        kwargs["data"] = data
    confidence_array = np.asarray(confidence, dtype=np.float32)
    if "keypoint_confidence" in inspect.signature(sv.KeyPoints.__init__).parameters:
        kwargs["keypoint_confidence"] = confidence_array
    else:
        kwargs["confidence"] = confidence_array
    return sv.KeyPoints(**kwargs)


def str_to_color(color: str) -> sv.Color:
    if color.startswith("#"):
        return sv.Color.from_hex(color)
    elif color.startswith("rgb"):
        r, g, b = map(int, color[4:-1].split(","))
        return sv.Color.from_rgb_tuple((r, g, b))
    elif color.startswith("bgr"):
        b, g, r = map(int, color[4:-1].split(","))
        return sv.Color.from_bgr_tuple((b, g, r))
    elif hasattr(sv.Color, color.upper()):
        return getattr(sv.Color, color.upper())
    else:
        raise ValueError(
            f"Invalid text color: {color}; valid formats are #RRGGBB, rgb(R, G, B), bgr(B, G, R), or a valid color name (like WHITE, BLACK, or BLUE)."
        )
