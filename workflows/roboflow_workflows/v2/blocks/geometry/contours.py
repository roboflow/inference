"""Outer contours of the non-zero pixels of an image: ``v2/contours_detection@v1``.

Port of V1 ``roboflow_core/contours_detection@v1``. As V1::

    gray     = OpenCV RGB -> gray (a one-channel image is used as it is)
    contours, hierarchy = findContours(gray, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    image    = gray as RGB with every contour drawn in magenta, line_thickness

Every non-zero gray pixel is foreground, so the block is meant for thresholded
or otherwise binarized images. Contour points are ``(x, y)`` pixels of the
input image, in its own frame; the drawn image keeps the input's identity and
transforms (crop offsets, resize and root mapping).

CPU boundary: OpenCV finds and draws contours on the host. The gray
conversion runs on the image's device with OpenCV's fixed-point weights, so
one ``H x W`` plane goes to the host and one ``H x W`` line mask comes back;
the drawn image is composed on the image's device. Contours and hierarchy are
small host results and stay CPU tensors.

Differences from V1: ``contours`` are ``(K, 2)`` ``int32`` tensors (V1:
``(K, 1, 2)`` arrays) and ``hierarchy`` is ``(N, 4)`` ``int32`` with ``N = 0``
when nothing is found (V1: ``(1, N, 4)`` or ``None``); ``line_thickness`` must
be positive.
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from pydantic import Field, StrictInt
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, Kind
from roboflow_workflows.v2.data.predictions import TENSOR_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

# OpenCV's RGB -> gray weights for uint8 images: 0.299, 0.587, 0.114 in 15-bit
# fixed point, rounded half up (unit tests compare with cv2.cvtColor).
GRAY_WEIGHTS = (9798, 19235, 3735)
GRAY_SHIFT = 15
CONTOUR_COLOR = (255, 0, 255)
MAX_LINE_THICKNESS = 32767  # OpenCV's limit


def _is_contours(payload: Any) -> bool:
    return isinstance(payload, list) and all(
        isinstance(contour, torch.Tensor)
        and contour.dim() == 2
        and contour.shape[1] == 2
        for contour in payload
    )


def _contours_to_lists(payload: List[torch.Tensor]) -> List[List[List[int]]]:
    return [contour.tolist() for contour in payload]


CONTOURS_KIND = Kind(
    name="contours",
    description="List of (K, 2) integer tensors of (x, y) contour points.",
    validate=_is_contours,
    serialize=_contours_to_lists,
)


class ContoursDetectionBlock(Block):
    """Find, count and draw the outer contours of an image's non-zero pixels.

    Example: after a threshold block, ``number_contours`` counts the separate
    white blobs and ``image`` shows their outlines.
    """

    type = "v2/contours_detection@v1"
    metadata = {
        "name": "Image Contours",
        "port_of": "roboflow_core/contours_detection@v1",
        "section": "classical_cv",
    }
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description="Gray input as RGB with the contours drawn in magenta.",
        ),
        "contours": Output(
            CONTOURS_KIND,
            source="image",
            description="Outer contours as (K, 2) int32 (x, y) CPU tensors.",
        ),
        "hierarchy": Output(
            TENSOR_KIND,
            source="image",
            description="OpenCV hierarchy, (N, 4) int32 CPU tensor.",
        ),
        "number_contours": Output(
            INTEGER_KIND, source="image", description="Number of contours."
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Thresholded image to trace.")
        line_thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=3,
            ge=1,
            le=MAX_LINE_THICKNESS,
            description="Width of the drawn contour lines in pixels.",
            examples=[3, "$inputs.line_thickness"],
        )

    def run(self, *, image: ImageData, line_thickness: int) -> Dict[str, Any]:
        """Trace the outer contours of the non-zero pixels of ``image``.

        Args:
            image: Image to trace; non-zero gray pixels are foreground.
            line_thickness: Width of the drawn lines, at least 1.

        Returns:
            ``image``, ``contours``, ``hierarchy`` and ``number_contours``.
        """
        gray = _gray(image.tensor_image)
        found, hierarchy = cv2.findContours(
            np.ascontiguousarray(gray.cpu().numpy()),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        lines = _line_mask(found, size_hw=tuple(gray.shape), thickness=line_thickness)
        drawn = gray.expand(3, -1, -1).clone()
        color = torch.tensor(CONTOUR_COLOR, dtype=torch.uint8, device=gray.device)
        drawn[:, lines.to(gray.device)] = color.reshape(3, 1)

        result = {
            "image": image.with_pixels(drawn),
            "contours": [
                torch.from_numpy(contour.reshape(-1, 2).astype(np.int32))
                for contour in found
            ],
            "hierarchy": _hierarchy_tensor(hierarchy),
            "number_contours": len(found),
        }

        return result


def _gray(pixels: torch.Tensor) -> torch.Tensor:
    # (H, W) uint8 gray plane on the pixels' device, equal to OpenCV's.
    if pixels.shape[0] == 1:
        return pixels[0]

    weights = torch.tensor(GRAY_WEIGHTS, dtype=torch.int32, device=pixels.device)
    weighted = (pixels.to(torch.int32) * weights.reshape(3, 1, 1)).sum(dim=0)
    gray = (weighted + (1 << (GRAY_SHIFT - 1))) >> GRAY_SHIFT

    return gray.to(torch.uint8)


def _line_mask(
    contours: Tuple[np.ndarray, ...], *, size_hw: Tuple[int, int], thickness: int
) -> torch.Tensor:
    # The pixels OpenCV's drawContours paints, as an (H, W) bool CPU tensor.
    canvas = np.zeros(size_hw, dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, color=(1,), thickness=thickness)

    return torch.from_numpy(canvas.astype(bool))


def _hierarchy_tensor(hierarchy: Any) -> torch.Tensor:
    if hierarchy is None:
        return torch.zeros((0, 4), dtype=torch.int32)

    return torch.from_numpy(hierarchy.reshape(-1, 4).astype(np.int32))
