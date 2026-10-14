"""Stitch two overlapping images into one canvas: ``roboflow_core/stitch_images@v1``.

Port of V1 ``roboflow_core/stitch_images@v1``. OpenCV finds SIFT keypoints
in both images, keeps matches passing Lowe's ratio test (best distance below
0.75 of the second best) and fits a RANSAC homography. One image, the
*moving* one, is warped into the frame of the other, the *anchor*, which is
then pasted unchanged::

    image1 ─┐  SIFT + ratio test     homography        canvas
            ├─▶ matched points ──▶ moving -> anchor ──▶ warped moving image,
    image2 ─┘                                          anchor pasted on top

As in V1, ``image1`` moves when the mean x of its keypoints is smaller than
that of ``image2``'s keypoints; otherwise ``image2`` moves.

The canvas covers the warped image and the anchor. It is a composite image
(``ImageData.composite``) with its own ``stitched-...`` id, whose
``composite_sources`` hold only the anchor's exact tile: index ``(0,)`` for
``image1`` or ``(1,)`` for ``image2``. Canvas pixels outside that tile are
opaque warped content. The moving image reaches the canvas through a
projective transform, which ``CompositeSource`` cannot express, so the moving
image is not listed and no lineage maps those pixels back to it.

Predictions made on the canvas have no single workflow input to map back to.
A ``JsonField`` output in the default ``parent`` coordinates therefore
rejects them; declare ``"coordinates_system": "own"`` on that output to get
canvas pixels. V1 treated the stitched image as a new input instead, so its
default output gave canvas pixels.

When no homography can be found (too few keypoints or matches, or a
degenerate fit) the output is ``None``, as in V1, and downstream steps skip.

Size limits are checked before the canvas is allocated, and each raises
``ValueError``:

* a moving image side above 32766 pixels, the largest source OpenCV warps;
* a canvas side above 32767 pixels, the largest canvas OpenCV warps into;
* a canvas area above ``max_canvas_pixels`` (default 16_777_216, 4096 x 4096,
  about 48 MiB as RGB ``uint8``). That safety limit stops a wild homography
  from one bad fit from allocating gigabytes; raise it for genuinely large
  panoramas.

CPU boundary: SIFT, matching and warping are OpenCV host operations. Both
images are copied to host memory once and the canvas is copied back to the
device of the inputs, which must be the same.
"""

import uuid
from typing import Any, Dict, Optional, Tuple
import cv2 as cv
import numpy as np
import torch
from pydantic import Field, StrictInt
from roboflow_workflows.v2.data.image import CompositeSource, ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


__all__ = ["StitchImagesBlock"]

RATIO_TEST = 0.75

MIN_HOMOGRAPHY_POINTS = 4

MAX_WARP_SOURCE_SIDE = 32766

MAX_CANVAS_SIDE = 32767

DEFAULT_MAX_CANVAS_PIXELS = 16_777_216

STITCHED_ID_PREFIX = "stitched"

class StitchImagesBlock(Block):
    """Stitch two overlapping, textured images into one panorama.

    Example: two 120-pixel-wide views of one 200-pixel-wide scene that
    overlap by 40 pixels give one canvas about 200 pixels wide showing the
    whole scene. Images without enough common detail give ``None``.

    Grayscale and RGB inputs may be mixed; the canvas is then RGB. Inputs are
    not modified. The block is stateless and prunable.
    """

    type = "roboflow_core/stitch_images@v1"
    prunable = True
    metadata = {
        "name": "Stitch Images",
        "port_of": "roboflow_core/stitch_images@v1",
        "section": "transformation",
    }
    outputs = {
        "stitched_image": Output(
            IMAGE_KIND,
            description=(
                "Panorama of both images; None when stitching fails. The image "
                "whose keypoints have the larger mean x is pasted unchanged and "
                "is the only one mapped back. Output predictions made on the "
                "panorama with JsonField coordinates_system='own'."
            ),
        )
    }

    class Params(BlockParams):
        image1: Ref(IMAGE_KIND) = Field(
            description="First image; must overlap image2 with enough texture.",
            examples=["$inputs.image1"],
        )
        image2: Ref(IMAGE_KIND) = Field(
            description="Second image; must overlap image1 with enough texture.",
            examples=["$inputs.image2"],
        )
        max_allowed_reprojection_error: float | Ref(FLOAT_KIND) = Field(
            default=3.0,
            gt=0.0,
            description=(
                "RANSAC inlier distance in pixels (OpenCV ransacReprojThreshold). "
                "Raise it, e.g. to 5-10, for low-detail images."
            ),
            examples=[3.0, "$inputs.max_allowed_reprojection_error"],
        )
        count_of_best_matches_per_query_descriptor: StrictInt | Ref(INTEGER_KIND) = (
            Field(
                default=2,
                ge=2,
                description=(
                    "Neighbours per keypoint in matching (OpenCV knnMatch k); "
                    "the ratio test needs at least 2."
                ),
                examples=[2, "$inputs.k"],
            )
        )
        max_canvas_pixels: StrictInt | Ref(INTEGER_KIND) = Field(
            default=DEFAULT_MAX_CANVAS_PIXELS,
            ge=1,
            description=(
                "Largest allowed canvas area in pixels, checked before allocation; "
                "a larger stitch raises an error. Default 4096 x 4096."
            ),
            examples=[DEFAULT_MAX_CANVAS_PIXELS, "$inputs.max_canvas_pixels"],
        )

    def run(
        self,
        *,
        image1: ImageData,
        image2: ImageData,
        max_allowed_reprojection_error: float,
        count_of_best_matches_per_query_descriptor: int,
        max_canvas_pixels: int,
    ) -> Dict[str, Any]:
        """Stitch ``image1`` and ``image2``.

        Args:
            image1: First image.
            image2: Second image, on the same device.
            max_allowed_reprojection_error: Positive RANSAC inlier distance.
            count_of_best_matches_per_query_descriptor: Neighbours per
                keypoint, at least 2.
            max_canvas_pixels: Largest allowed canvas area, at least 1.

        Returns:
            ``stitched_image``: the composite canvas on the inputs' device, or
            ``None`` when no homography was found.

        Raises:
            ValueError: When the images are on different devices, or the
                moving image or canvas would exceed a size limit.
        """
        if image1.device != image2.device:
            raise ValueError(
                "stitch_images needs both images on one device, got "
                f"{image1.device} and {image2.device}"
            )

        channels = max(image1.channels, image2.channels)
        pixels1 = _host_pixels(image1, channels=channels)
        pixels2 = _host_pixels(image2, channels=channels)
        stitched = stitch_pixels(
            pixels1,
            pixels2,
            neighbours=count_of_best_matches_per_query_descriptor,
            reprojection_error=max_allowed_reprojection_error,
            max_canvas_pixels=max_canvas_pixels,
        )
        if stitched is None:
            return {"stitched_image": None}

        canvas, anchor_is_image1, anchor_xy = stitched
        anchor = image1 if anchor_is_image1 else image2
        x0, y0 = anchor_xy
        tile = CompositeSource.place(
            anchor,
            index=(0,) if anchor_is_image1 else (1,),
            canvas_xyxy=(x0, y0, x0 + anchor.width, y0 + anchor.height),
        )
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).contiguous()
        stitched_image = ImageData.composite(
            tensor.to(image1.device),
            sources=[tile],
            image_id=f"{STITCHED_ID_PREFIX}-{uuid.uuid4().hex}",
        )
        result = {"stitched_image": stitched_image}

        return result

def stitch_pixels(
    pixels1: np.ndarray,
    pixels2: np.ndarray,
    *,
    neighbours: int,
    reprojection_error: float,
    max_canvas_pixels: int = DEFAULT_MAX_CANVAS_PIXELS,
) -> Optional[Tuple[np.ndarray, bool, Tuple[int, int]]]:
    """Stitch two host images with SIFT matching and a RANSAC homography.

    Args:
        pixels1: ``(H, W, C)`` ``uint8`` image, ``C`` 1 or 3 (RGB).
        pixels2: Second image in the same layout.
        neighbours: Neighbours per keypoint for ``knnMatch``, at least 2.
        reprojection_error: Positive RANSAC inlier distance in pixels.
        max_canvas_pixels: Largest allowed canvas area in pixels.

    Returns:
        ``(canvas, anchor_is_image1, (x0, y0))``: the ``(H, W, C)`` canvas,
        whether ``pixels1`` is the pasted anchor, and the anchor's top-left
        corner on the canvas. ``None`` when no homography is found.

    Raises:
        ValueError: When a side of the moving image exceeds
            ``MAX_WARP_SOURCE_SIDE``, a canvas side would exceed
            ``MAX_CANVAS_SIDE`` or its area would exceed ``max_canvas_pixels``.
    """
    sift = cv.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(_gray(pixels1), None)
    keypoints2, descriptors2 = sift.detectAndCompute(_gray(pixels2), None)
    if descriptors1 is None or descriptors2 is None:
        return None

    matches = cv.BFMatcher_create().knnMatch(descriptors1, descriptors2, k=neighbours)
    good = [
        pair[0]
        for pair in matches
        if len(pair) >= 2 and pair[0].distance < RATIO_TEST * pair[1].distance
    ]
    if len(good) < MIN_HOMOGRAPHY_POINTS:
        return None

    points1 = np.float32([keypoints1[match.queryIdx].pt for match in good])
    points2 = np.float32([keypoints2[match.trainIdx].pt for match in good])
    image1_moves = np.mean([point.pt[0] for point in keypoints1]) < np.mean(
        [point.pt[0] for point in keypoints2]
    )
    if image1_moves:
        moving, anchor, moving_points, anchor_points = (
            pixels1,
            pixels2,
            points1,
            points2,
        )
    else:
        moving, anchor, moving_points, anchor_points = (
            pixels2,
            pixels1,
            points2,
            points1,
        )
    homography, _ = cv.findHomography(
        moving_points.reshape(-1, 1, 2),
        anchor_points.reshape(-1, 1, 2),
        cv.RANSAC,
        reprojection_error,
    )
    if homography is None:
        return None

    placed = _place(
        moving, anchor, homography=homography, max_canvas_pixels=max_canvas_pixels
    )
    if placed is None:
        return None

    canvas, anchor_xy = placed
    stitched = (canvas, not image1_moves, anchor_xy)

    return stitched

def _place(
    moving: np.ndarray,
    anchor: np.ndarray,
    *,
    homography: np.ndarray,
    max_canvas_pixels: int,
) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    # The canvas spans the warped moving image and the anchor at (0, 0) of the
    # anchor frame. Corners truncate to integers as in V1.
    moving_h, moving_w = moving.shape[:2]
    anchor_h, anchor_w = anchor.shape[:2]
    _check_moving_size(moving_w, moving_h)

    corners = np.float32([[0, 0], [0, moving_h], [moving_w, moving_h], [moving_w, 0]])
    warped_corners = cv.perspectiveTransform(corners.reshape(-1, 1, 2), homography)
    if not np.isfinite(warped_corners).all():
        return None

    lowest = warped_corners.min(axis=0).ravel()
    highest = warped_corners.max(axis=0).ravel()
    x_min, y_min = (min(int(value), 0) for value in lowest)
    x_max = max(int(highest[0]), anchor_w)
    y_max = max(int(highest[1]), anchor_h)
    width, height = x_max - x_min, y_max - y_min
    _check_canvas_size(width, height, max_canvas_pixels=max_canvas_pixels)

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=float)
    canvas = cv.warpPerspective(moving, translation @ homography, (width, height))
    canvas = canvas.reshape(height, width, moving.shape[2])
    canvas[-y_min : -y_min + anchor_h, -x_min : -x_min + anchor_w] = anchor
    placed = (canvas, (-x_min, -y_min))

    return placed

def _check_moving_size(width: int, height: int) -> None:
    # warpPerspective asserts on a source side of SHRT_MAX (32767) or more,
    # even when the canvas is small.
    if width > MAX_WARP_SOURCE_SIDE or height > MAX_WARP_SOURCE_SIDE:
        raise ValueError(
            f"stitch_images cannot warp a {width}x{height} image; OpenCV warps "
            f"images only up to {MAX_WARP_SOURCE_SIDE} pixels per side. "
            "Downscale the images before stitching."
        )

def _check_canvas_size(width: int, height: int, *, max_canvas_pixels: int) -> None:
    # Runs before warpPerspective allocates the canvas.
    if width > MAX_CANVAS_SIDE or height > MAX_CANVAS_SIDE:
        raise ValueError(
            f"stitch_images canvas would be {width}x{height} pixels; OpenCV "
            f"cannot warp into a side above {MAX_CANVAS_SIDE}. The images "
            "probably do not overlap enough for a usable homography."
        )

    if width * height > max_canvas_pixels:
        raise ValueError(
            f"stitch_images canvas would be {width}x{height} = "
            f"{width * height} pixels, above max_canvas_pixels="
            f"{max_canvas_pixels}. Raise max_canvas_pixels for a genuinely "
            "large panorama, or check that the images overlap."
        )

def _host_pixels(image: ImageData, *, channels: int) -> np.ndarray:
    # (H, W, channels) host copy; grayscale repeats to RGB when mixed with RGB.
    tensor = image.tensor_image.expand(channels, -1, -1)
    pixels = tensor.permute(1, 2, 0).contiguous().cpu().numpy()

    return pixels

def _gray(pixels: np.ndarray) -> np.ndarray:
    # SIFT converts BGR input with COLOR_BGR2GRAY; RGB2GRAY on RGB is the same.
    if pixels.shape[2] == 1:
        return pixels[:, :, 0]

    gray = cv.cvtColor(pixels, cv.COLOR_RGB2GRAY)

    return gray
