"""Read barcodes and QR codes in an image with a CPU decoder.

Both blocks cross to the host once per image, then decode with zxing-cpp::

    ImageData (C, H, W) uint8, any device
      └─ luminance on the image's device      (3 channels -> 1, cv2 weights)
           └─ one host copy of H x W bytes    the only pixel transfer
                ├─ zxing-cpp, every format    barcode detector
                └─ zxing-cpp, QR codes only   QR code detector
      corners (N, 4, 2) -> axis-aligned xyxy, clipped to the image
      Detections on the image's device, one row per code

Each row's decoded text is ``bboxes_metadata[i]["data"]``; UQL reads it as the
``decoded_text`` property. Rows have ``class_id`` 0, confidence 1 and a fresh
row identity; ``image_metadata`` carries the image's frame, ``class_names``
and ``prediction_type`` as in V1. The payload is plain ``Detections``, so each
output also has the object detection kind and feeds filters and painters.

Differences from V1:

* The box encloses all four corners. V1 used two corners, which shrinks or
  inverts the box of a rotated code. Unrotated codes get the same box.
* Boxes are clipped to the image; OpenCV corners can lie slightly outside.
* Decoders read the image's luminance, ``(9798 R + 19235 G + 3735 B +
  16384) >> 15``, equal to OpenCV 4.12 ``cvtColor``. V1 passed BGR pixels to
  zxing-cpp, which weighs them as RGB; black-and-white codes decode the same.
* Rows carry a row identity instead of a ``detection_id`` entry.
* The QR detector uses zxing-cpp restricted to QR codes, not V1's OpenCV
  ``QRCodeDetector``. OpenCV misses some clean codes that zxing-cpp reads.
  zxing-cpp returns decoded codes only, so the empty-text rows V1 gave for
  located but unreadable codes no longer appear. Neither decoder reads every
  code.

As in V1, the barcode detector reads every format zxing-cpp supports,
including QR codes.
"""

from typing import Any, Dict, List, Sequence, Tuple
from uuid import uuid4
import numpy as np
import torch
import zxingcpp
from pydantic import Field
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import (
    BAR_CODE_DETECTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    QR_CODE_DETECTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from inference_models.models.base.object_detection import Detections


DECODED_TEXT_KEY = "data"

_LUMINANCE_WEIGHTS = (9798, 19235, 3735)

_LUMINANCE_SHIFT = 15

Corners = Sequence[Tuple[float, float]]

__all__ = ['DECODED_TEXT_KEY']

def host_luminance(image: ImageData) -> np.ndarray:
    """Return the image's luminance as a contiguous host array.

    Three channels are reduced on the image's device first, so a CUDA image
    crosses to the host once, at one byte per pixel. A contiguous
    single-channel CPU image is returned without a copy; callers must not
    write to the result.

    Args:
        image: RGB or grayscale image on any device.

    Returns:
        ``(height, width)`` ``uint8`` array, equal to
        ``cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)`` for an RGB image.
    """
    pixels = image.tensor_image
    if pixels.shape[0] == 1:
        luminance = pixels[0]
    else:
        weights = torch.tensor(
            _LUMINANCE_WEIGHTS, dtype=torch.int32, device=pixels.device
        ).view(3, 1, 1)
        weighted = (pixels.to(torch.int32) * weights).sum(dim=0)
        rounding = 1 << (_LUMINANCE_SHIFT - 1)
        luminance = ((weighted + rounding) >> _LUMINANCE_SHIFT).to(torch.uint8)

    host = np.ascontiguousarray(luminance.cpu().numpy())

    return host

def code_detections(
    image: ImageData,
    *,
    corners: Sequence[Corners],
    texts: Sequence[str],
    class_name: str,
    prediction_type: str,
) -> Detections:
    """Build one image's code Detections from decoder corners and texts.

    Args:
        image: Image the codes were read from; sets device and frame.
        corners: Four ``(x, y)`` corners per code, in image pixels.
        texts: Decoded text per code, aligned with ``corners``.
        class_name: Name of class 0.
        prediction_type: V1 ``prediction_type`` of the result.

    Returns:
        Detections with ``float32`` xyxy enclosing each code's corners,
        clipped to the image, on the image's device, with a new row identity.
    """
    xyxy = torch.tensor(
        [_enclosing_box(code, size_hw=image.size_hw) for code in corners],
        dtype=torch.float32,
    ).reshape(-1, 4)
    rows = xyxy.shape[0]
    device = image.device
    metadata = {
        **image.prediction_metadata(),
        "class_names": {0: class_name},
        "prediction_type": prediction_type,
    }

    detections = Detections(
        xyxy=xyxy.to(device),
        class_id=torch.zeros(rows, dtype=torch.int64, device=device),
        confidence=torch.ones(rows, dtype=torch.float32, device=device),
        image_metadata=metadata,
        bboxes_metadata=[{DECODED_TEXT_KEY: text} for text in texts],
    )
    identified = with_row_identity(detections, namespace=uuid4().hex)

    return identified

def _zxing_corners(position: Any) -> List[Tuple[float, float]]:
    points = (
        position.top_left,
        position.top_right,
        position.bottom_right,
        position.bottom_left,
    )
    corners = [(float(point.x), float(point.y)) for point in points]

    return corners

def _enclosing_box(corners: Corners, *, size_hw: Tuple[int, int]) -> List[float]:
    height, width = size_hw
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    box = [
        min(max(min(xs), 0.0), width),
        min(max(min(ys), 0.0), height),
        min(max(max(xs), 0.0), width),
        min(max(max(ys), 0.0), height),
    ]

    return box
