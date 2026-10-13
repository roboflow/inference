"""Snap instance-mask outlines to image edges: ``v2/mask_edge_snap@v1``.

Port of V1 ``roboflow_core/mask_edge_snap@v1`` (latest: ``v1_tensor``). As V1::

    magnitude = Sobel 3x3 gradient magnitude of the grayscale image
    edges     = magnitude > box_mean + sigma * box_std     (window x window)
    edges     = 3x3 closing x dilation_iterations, then one Zhang-Suen pass
                (both skipped when dilation_iterations is 0)
    edges    &= band of +/- boundary_band_width around every mask outline
    edges     = 8-connected components whose outline area >= min_contour_area
    per mask, per outline point:
        move along the outline normal up to pixel_tolerance pixels to the
        edge pixel with the best magnitude * (1 - distance / (tolerance + 1));
        inside the mask only edges at most pixel_tolerance deep count
    mask      = refined outlines, filled

``edges`` is one ``"edges"`` row holding the surviving edge pixels, or no row.
Without rows, predictions pass through; ``edges`` covers the whole image (no band).

Masks come from the representation's reference result (dense, RLE, or a
``SegmentationView``'s full-resolution masks). The result is
``InstanceDetections``; RLE masks stay RLE.

CPU boundary: Sobel, morphology, contours and distance transforms are OpenCV
host algorithms. The image and the reference masks are copied to the host
once; refined masks go back to the predictions' device.

Differences from V1:

* Boxes follow the refined masks; V1 kept the old boxes. A row whose refined
  mask is empty keeps its box. Boxes span whole pixels (``max + 1``), like
  V1's ``edges`` box.
* Masks must have the image's size. V1 resized them silently, which snaps a
  mask made for another image onto unrelated edges.
* Predictions without masks are rejected; V1 passed them through.
* RLE masks stay RLE; V1 returned dense masks.
* Host mirror keys (``_host_xyxy`` ...) are dropped from rows, since the
  boxes changed.
* Bound parameter values are range-checked like literals; V1 only checked
  literals of the exact declared type.
"""

from dataclasses import replace
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pydantic import Field, StrictInt
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.data.masks import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.masks.providers import mask_provider_of
from roboflow_workflows.v2.data.prediction_ops import CLASS_NAMES_KEY
from roboflow_workflows.v2.data.predictions import INSTANCE_SEGMENTATION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks

# Outline points on each side used to estimate the outline normal (V1 value).
TANGENT_WINDOW = 5
EDGES_CLASS_NAME = "edges"
# V1 labels the edges prediction this way; query blocks read the key.
PREDICTION_TYPE_KEY = "prediction_type"
EDGES_PREDICTION_TYPE = "object-detection"


class MaskEdgeSnapBlock(Block):
    """Move instance-mask outlines onto nearby image edges.

    Example: a segmentation model draws a part's mask a few pixels too wide;
    the refined mask follows the part's visible boundary. Blur or enhance
    contrast upstream for noisy or low-contrast images; this block does no
    preprocessing.
    """

    type = "v2/mask_edge_snap@v1"
    outputs = {
        "refined_segmentation": Output(
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            source="segmentation",
            description="Same rows; masks snapped to edges, boxes from masks.",
        ),
        "edges": Output(
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            source="image",
            description="One 'edges' row with the edge pixels used, or none.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Image whose edges are used.",
            examples=["$inputs.image"],
        )
        segmentation: Ref(
            INSTANCE_SEGMENTATION_PREDICTION_KIND, INSTANCE_SEGMENTATION_VIEW_KIND
        ) = Field(
            description="Instance segmentation of that image, with masks.",
            examples=["$steps.segmentation_model.predictions"],
        )
        pixel_tolerance: StrictInt | Ref(INTEGER_KIND) = Field(
            default=15,
            ge=1,
            le=100,
            description="Farthest an outline point may move, in pixels.",
            examples=[15, "$inputs.pixel_tolerance"],
        )
        sigma: float | Ref(FLOAT_KIND, INTEGER_KIND) = Field(
            default=1.0,
            ge=0.01,
            le=10.0,
            description="Edge strictness: local mean + sigma * local std.",
            examples=[1.0, "$inputs.sigma"],
        )
        min_contour_area: float | Ref(FLOAT_KIND, INTEGER_KIND) = Field(
            default=50.0,
            ge=0.0,
            le=10000.0,
            description="Smallest edge component outline area kept, in square "
            "pixels. Thin open edges have an area near 0.",
            examples=[50.0, 0.0],
        )
        dilation_iterations: StrictInt | Ref(INTEGER_KIND) = Field(
            default=2,
            ge=0,
            le=20,
            description="Closing passes bridging edge gaps; 0 disables closing "
            "and thinning.",
            examples=[2, 0],
        )
        boundary_band_width: StrictInt | Ref(INTEGER_KIND) = Field(
            default=15,
            ge=1,
            le=100,
            description="Edges farther than this from every mask outline are "
            "ignored, in pixels.",
            examples=[15],
        )
        adaptive_window_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=41,
            ge=3,
            le=201,
            description="Side of the local statistics window; even values grow "
            "by one.",
            examples=[41],
        )

    def run(
        self,
        *,
        image: ImageData,
        segmentation: Any,
        pixel_tolerance: int,
        sigma: float,
        min_contour_area: float,
        dilation_iterations: int,
        boundary_band_width: int,
        adaptive_window_size: int,
    ) -> Dict[str, Any]:
        """Snap every mask outline to the image's edges.

        Args:
            image: The image the segmentation was made on.
            segmentation: ``InstanceDetections`` or a ``SegmentationView``.
            pixel_tolerance: Farthest an outline point may move.
            sigma: Strictness of the adaptive edge threshold.
            min_contour_area: Smallest edge component area kept.
            dilation_iterations: Closing passes over the edge map.
            boundary_band_width: Half-width of the edge search band.
            adaptive_window_size: Side of the local statistics window.

        Returns:
            ``refined_segmentation`` and ``edges`` predictions.

        Raises:
            ValueError: On a prediction without masks, or masks of another
                size than the image. Parameter bounds are validated by the
                engine before run(); direct callers must validate them first.
        """
        # A view supplies the dense reference result, not a representation template.
        is_view = isinstance(segmentation, SegmentationView)
        source_mask = getattr(segmentation, "mask", None)
        if isinstance(segmentation, InstanceDetections) and source_mask is None:
            raise ValueError(
                "v2/mask_edge_snap@v1 needs instance masks; bind an instance "
                "segmentation prediction"
            )
        predictions = mask_provider_of(segmentation).binary().predictions

        height, width = image.size_hw
        masks = _host_masks(predictions, size_hw=(height, width))
        gray = _host_gray(image)
        magnitude, snap_region = _snap_region(
            gray,
            masks=masks,
            sigma=float(sigma),
            min_area=float(min_contour_area),
            iterations=int(dilation_iterations),
            band_radius=int(boundary_band_width),
            window=int(adaptive_window_size),
        )
        edges = _edges_prediction(snap_region, image=image)
        if not masks:
            unchanged = predictions if is_view else segmentation
            return {"refined_segmentation": unchanged, "edges": edges}

        refined = [
            _snap_mask(
                mask,
                magnitude=magnitude,
                snap_region=snap_region,
                tol=int(pixel_tolerance),
            )
            for mask in masks
        ]
        rows = predictions.bboxes_metadata or [None] * len(refined)
        refined_predictions = replace(
            predictions,
            xyxy=_boxes_of(refined, previous=predictions.xyxy),
            mask=_device_masks(
                refined, like=source_mask, device=predictions.xyxy.device
            ),
            bboxes_metadata=[without_host_mirrors(row) for row in rows],
        )
        result = {"refined_segmentation": refined_predictions, "edges": edges}

        return result


def _host_masks(
    predictions: InstanceDetections, *, size_hw: Tuple[int, int]
) -> List[np.ndarray]:
    # One (H, W) uint8 array per row; one transfer for the whole stack.
    if len(predictions.xyxy) == 0:
        return []

    mask = predictions.mask
    if tuple(mask.shape[1:]) != size_hw:
        raise ValueError(
            f"segmentation masks have size {tuple(mask.shape[1:])} but the image "
            f"has size {size_hw}; bind the image the segmentation was made on"
        )
    host = mask.to(dtype=torch.uint8).cpu().numpy()

    return list(host)


def _host_gray(image: ImageData) -> np.ndarray:
    pixels = image.tensor_image.permute(1, 2, 0).contiguous().cpu().numpy()
    if pixels.shape[2] == 1:
        return pixels[:, :, 0]

    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)

    return gray


def _snap_region(
    gray: np.ndarray,
    *,
    masks: List[np.ndarray],
    sigma: float,
    min_area: float,
    iterations: int,
    band_radius: int,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Sobel magnitude and the boolean map of edge pixels outlines may snap to.
    height, width = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)

    window += 1 - window % 2
    mean = cv2.boxFilter(magnitude, cv2.CV_32F, (window, window))
    square_mean = cv2.boxFilter(magnitude * magnitude, cv2.CV_32F, (window, window))
    std = np.sqrt(np.maximum(square_mean - mean * mean, 0.0))
    edges = (magnitude > mean + sigma * std).astype(np.uint8) * 255
    if iterations > 0:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, close_kernel, iterations=iterations
        )
        edges = _zhang_suen_one_iteration(edges)

    if masks:
        band_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (band_radius * 2 + 1, band_radius * 2 + 1)
        )
        band = np.zeros((height, width), dtype=bool)
        for mask in masks:
            band |= (cv2.dilate(mask, band_kernel) > 0) & (
                cv2.erode(mask, band_kernel) == 0
            )
        edges = edges * band.astype(np.uint8)

    count, labels = cv2.connectedComponents(edges, connectivity=8)
    kept = np.zeros(count, dtype=bool)
    for label in range(1, count):
        component = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if contours:
            kept[label] = max(cv2.contourArea(c) for c in contours) >= min_area
    snap_region = kept[labels]

    return magnitude, snap_region


def _zhang_suen_one_iteration(image: np.ndarray) -> np.ndarray:
    # One Zhang-Suen pass (two sub-iterations) over a 0/255 image, as V1.
    #   P9 P2 P3
    #   P8 P1 P4
    #   P7 P6 P5
    pixels = (image > 0).astype(np.uint8)
    if pixels.shape[0] < 3 or pixels.shape[1] < 3:
        return pixels * 255

    for first in (True, False):
        p1 = pixels[1:-1, 1:-1]
        p2, p3, p4 = pixels[:-2, 1:-1], pixels[:-2, 2:], pixels[1:-1, 2:]
        p5, p6, p7 = pixels[2:, 2:], pixels[2:, 1:-1], pixels[2:, :-2]
        p8, p9 = pixels[1:-1, :-2], pixels[:-2, :-2]
        ring = (p2, p3, p4, p5, p6, p7, p8, p9)
        neighbours = sum(p.astype(np.int16) for p in ring)
        transitions = sum(
            ((a == 0) & (b == 1)).astype(np.int16)
            for a, b in zip(ring, ring[1:] + ring[:1])
        )
        removable = (
            (p1 == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1)
        )
        if first:
            removable &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
        else:
            removable &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
        pixels[1:-1, 1:-1] = np.where(removable, 0, p1)

    return pixels * 255


def _snap_mask(
    mask: np.ndarray, *, magnitude: np.ndarray, snap_region: np.ndarray, tol: int
) -> np.ndarray:
    # Refined (H, W) bool mask; a mask without an outline is kept as it is.
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask.astype(bool)

    depth = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.dilate(mask, np.ones((3, 3), np.uint8)) == 0
    valid = snap_region & ((depth <= tol) | outside)

    refined = np.zeros((height, width), dtype=np.uint8)
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float32)
        snapped = points.copy()
        count = len(points)
        for index, (x, y) in enumerate(points):
            tangent = (
                points[(index + TANGENT_WINDOW) % count]
                - points[(index - TANGENT_WINDOW) % count]
            )
            length = np.linalg.norm(tangent)
            if length < 1e-6:
                continue
            # Preserve V1 scalar arithmetic: float32 normals; the mixed
            # Python/NumPy coordinate expression promotes to float64 on
            # NumPy 1.26 and stays float32 on NumPy 2. Outlines may differ
            # across versions; V1 parity is within the same NumPy version.
            tangent /= length
            nx, ny = -tangent[1], tangent[0]

            best_score, best_point = 0.0, None
            for sign in (1.0, -1.0):
                for step in range(1, tol + 1):
                    sx = int(round(x + sign * nx * step))
                    sy = int(round(y + sign * ny * step))
                    if 0 <= sx < width and 0 <= sy < height and valid[sy, sx]:
                        score = float(magnitude[sy, sx]) * (1.0 - step / (tol + 1))
                        if score > best_score:
                            best_score, best_point = score, (sx, sy)
            if best_point is not None:
                snapped[index] = best_point
        cv2.fillPoly(refined, [snapped.reshape(-1, 1, 2).astype(np.int32)], color=1)

    return refined.astype(bool)


def _boxes_of(masks: List[np.ndarray], *, previous: torch.Tensor) -> torch.Tensor:
    boxes = previous.clone()
    for row, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        boxes[row] = torch.tensor(box, dtype=boxes.dtype)

    return boxes


def _device_masks(masks: List[np.ndarray], *, like: Any, device: torch.device) -> Any:
    # Same representation as the input: RLE stays RLE, others become dense.
    if isinstance(like, InstancesRLEMasks):
        encoded = [
            mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"]
            for mask in masks
        ]
        return InstancesRLEMasks(image_size=like.image_size, masks=encoded)

    dense = torch.from_numpy(np.stack(masks)).to(device=device, dtype=torch.bool)

    return dense


def _edges_prediction(
    snap_region: np.ndarray, *, image: ImageData
) -> InstanceDetections:
    height, width = snap_region.shape
    device = image.device
    metadata = {
        **image.prediction_metadata(),
        CLASS_NAMES_KEY: {0: EDGES_CLASS_NAME},
        PREDICTION_TYPE_KEY: EDGES_PREDICTION_TYPE,
    }
    ys, xs = np.nonzero(snap_region)
    if ys.size == 0:
        return InstanceDetections(
            xyxy=torch.zeros((0, 4), dtype=torch.float32, device=device),
            class_id=torch.zeros((0,), dtype=torch.long, device=device),
            confidence=torch.zeros((0,), dtype=torch.float32, device=device),
            mask=torch.zeros((0, height, width), dtype=torch.bool, device=device),
            image_metadata=metadata,
            bboxes_metadata=[],
        )

    box = [[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]]
    edges = InstanceDetections(
        xyxy=torch.tensor(box, dtype=torch.float32, device=device),
        class_id=torch.zeros((1,), dtype=torch.long, device=device),
        confidence=torch.ones((1,), dtype=torch.float32, device=device),
        mask=torch.from_numpy(snap_region[None]).to(device=device),
        image_metadata=metadata,
        bboxes_metadata=[{"detection_id": str(uuid4())}],
    )

    return edges
