from typing import Annotated, List, Literal, Optional, Type, Union

import cv2
import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field, StringConstraints, field_validator

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    ANY_DATA_AS_SELECTED_ELEMENT,
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KIND_KEY,
    REFERENCE_KEY,
    SELECTED_ELEMENT_KEY,
    SELECTOR_POINTS_TO_BATCH_KEY,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = (
    "Refine instance segmentation masks by snapping edges to detected boundaries."
)
LONG_DESCRIPTION = """
Refine instance segmentation masks by snapping contour points to Sobel edges within a band around the predicted boundary. This block improves segmentation accuracy by adjusting mask edges to align with detected image features.

## How This Block Works

This block refines segmentation masks through a sophisticated multi-step pipeline:

1. **Edge Detection**: Computes Sobel gradient magnitudes from the input image to detect edges
2. **Adaptive Thresholding**: Uses per-pixel adaptive thresholding (local mean + sigma * local std) to identify significant edges
3. **Morphological Processing**: Applies closing (dilation + erosion) to bridge small gaps in edge segments
4. **Thinning**: Applies Zhang-Suen single-iteration thinning to reduce edge width to 1-2 pixels while preserving connectivity
5. **Boundary Band Creation**: Builds a search band around each predicted mask's contour
6. **Area Filtering**: Removes small edge components below a minimum area threshold
7. **Contour Snapping**: For each original mask contour point, finds the strongest nearby edge within tolerance and snaps to it

## Common Use Cases

- **Medical Image Analysis**: Refine organ/tumor segmentation masks to align with anatomical boundaries
- **Industrial Quality Control**: Improve part boundary detection for precise dimension measurement
- **Autonomous Vehicles**: Refine road/lane segmentation boundaries for improved path planning
- **Agricultural Monitoring**: Enhance crop boundary detection for yield estimation
- **Microscopy Analysis**: Refine cell/nuclei segmentation for morphological analysis
- **Document Processing**: Improve text region boundary detection for OCR

## Input Parameters

**image** : Input image (color or grayscale)
- Can be single-channel, 3-channel (BGR), or 4-channel (BGRA)
- Preprocessing (blur, contrast enhancement) should be applied upstream if needed

**segmentation** : Initial instance segmentation predictions
- Source: from object detection or instance segmentation model
- Must contain populated `mask` field; if empty, passed through unchanged

**pixel_tolerance** : Maximum perpendicular distance (pixels) for edge snapping
- Range: 5-50 typically
- 5-15: tight predictions with minimal offset
- 20-50: rough predictions needing more forgiveness

**sigma** : Strictness multiplier for adaptive Sobel threshold
- Range: 0.1-2.0 typically
- 0.1-0.5: permissive, keeps weaker edges, good for low-contrast boundaries
- 1.0-2.0: strict, only strongest edges survive, good for high-contrast images

**min_contour_area** : Minimum enclosed-polygon area for edge components
- Range: 10-1000 typically
- Small (10-50): keeps fragmented edges
- Large (200-1000): aggressive noise rejection

**dilation_iterations** : Number of morphological closing iterations
- Range: 0-10 typically
- 0: no closing, only thresholded edges
- 1-2: bridges hairline gaps
- 3-5: bridges visible dashes
- 10+: aggressive, can merge unrelated edges

**boundary_band_width** : Half-width of search band around mask contour (default: 15)
- Sets maximum distance between predicted and true boundary that can be corrected

**adaptive_window_size** : Side length of local-statistics window (default: 41)
- Should be roughly 5-10% of smaller image dimension
- Smaller (15-25): fine local contrast sensitivity, can pick up noise
- Larger (81-121): smooth threshold field, closer to global thresholding

## Outputs

**refined_segmentation** : Same detections with snapped mask contours
**edges** : Single detection containing union of all surviving edge pixels (debug/visualization)

## Preprocessing

**Preprocessing is usually critical for success.** This block does no preprocessing — what you feed in is what Sobel sees. For challenging imagery, chain Roboflow image-processing blocks upstream:

**Gaussian Blur**
    For grainy or noisy surfaces (welds, machined metal, biological tissue), blur before edge detection to suppress per-pixel noise. A 5x5 kernel with sigma 1.0 is a sensible default; increase to 7x7 or 9x9 for very noisy imagery. Don't over-blur — strong blur rounds off corners and softens real boundaries, leading to boundary positions that are biased inward.

**Bilateral Blur**
    Better than Gaussian when the image has both noise AND important sharp edges (e.g. textured fabric on a clean background). Slower, but preserves edges while denoising flat regions.

**Contrast Enhancement**
    Use when boundary contrast is genuinely too low to threshold reliably. The Contrast Enhancement block normalizes the histogram to use the full range, improving edge detection sensitivity without the noise amplification of aggressive methods. Follow with blur to suppress any remaining noise. Avoid on already-high-contrast images.

**Morphological Opening then Closing**
    Opening (erode then dilate) removes small bright specks and thin protrusions from the input before edge detection — useful when the surface has fine debris or hot pixels that would otherwise generate spurious edges. Closing (dilate then erode) fills small dark holes/gaps in bright regions; less commonly needed as preprocessing, since gap filling on the edge map itself is what the `dilation_iterations` parameter already does. Use the Morphological Transformation v2 block with the "Opening then Closing" operation for this preprocessing.

**Order matters**: Blur first, then contrast adjustment if needed. Reverse causes contrast adjustment to amplify the noise before blur can suppress it.
"""

# Custom type for segmentation that accepts both selector strings and Detections objects
# This includes the proper json_schema_extra to tell the workflow engine to resolve selectors
_segmentation_json_schema_extra = {
    REFERENCE_KEY: True,
    SELECTED_ELEMENT_KEY: ANY_DATA_AS_SELECTED_ELEMENT,
    KIND_KEY: [INSTANCE_SEGMENTATION_PREDICTION_KIND.dict()],
    SELECTOR_POINTS_TO_BATCH_KEY: "dynamic",
}


class MaskEdgeSnapManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/mask_edge_snap@v1"]
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "name": "Mask Edge Snap",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-scissors",
                "blockPriority": 12,
                "opencv": True,
            },
        },
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Input image (color or grayscale) for edge detection and snapping. Can be grayscale, single-channel, BGR, or BGRA. No preprocessing is applied internally; use upstream blocks for blur or contrast enhancement if needed.",
        examples=["$inputs.image", "$steps.preprocessing.image"],
        validation_alias=AliasChoices("image", "images"),
    )

    segmentation: Union[str, sv.Detections] = Field(
        title="Segmentation",
        description="Instance segmentation predictions with mask field populated. Each mask contour will be snapped to detected edges. If empty, segmentation is passed through unchanged. Can be a reference string like '$steps.segmentation_model.predictions' or a supervision.Detections object.",
        examples=["$steps.segmentation_model.predictions", "$inputs.segmentation"],
        json_schema_extra=_segmentation_json_schema_extra,
    )

    @field_validator("segmentation")
    @classmethod
    def validate_segmentation(
        cls, value: Union[str, sv.Detections]
    ) -> Union[str, sv.Detections]:
        if isinstance(value, str):
            if not value.startswith("$"):
                raise ValueError(
                    f"segmentation must be a workflow reference starting with '$' or a supervision.Detections object, got: {value}"
                )
        return value

    pixel_tolerance: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=15,
        description="Maximum perpendicular distance (pixels) from each contour point to candidate edges during snapping. Typical: 5-15 for tight predictions, 20-50 for rough ones. Too small: real edges outside range get missed. Too large: snap can wander to unrelated edges.",
    )

    sigma: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=1.0,
        description="Strictness multiplier for adaptive Sobel threshold (local_mean + sigma * local_std). Lower (0.1-0.5): permissive, good for low-contrast. Higher (1.0-2.0): strict, only strongest edges. Tune this AFTER other parameters.",
    )

    min_contour_area: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=50.0,
        description="Minimum enclosed-polygon area for edge components to keep. Small (10-50): keeps fragmented edges. Large (200-1000): aggressive noise rejection. Scales roughly with dilation_iterations.",
    )

    dilation_iterations: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=2,
        description="Morphological closing iterations to bridge gaps in thresholded edge map. Each iteration bridges ~2px gaps. 0: no closing. 1-2: hairline gaps. 3-5: visible dashes. 10+: aggressive merging.",
    )

    boundary_band_width: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=15,
        description="Half-width (pixels) of search band around segmentation contour. Sets maximum distance between predicted boundary and true boundary that can be corrected. Should generally be >= pixel_tolerance.",
    )

    adaptive_window_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=41,
        description="Side length of local-statistics window for adaptive threshold. Small (15-25): fine local sensitivity, can pick noise. Default 41: balanced. Large (81-121): smooth field, closer to global thresholding. Should be ~5-10% of smaller image dimension.",
    )

    @field_validator("pixel_tolerance")
    @classmethod
    def validate_pixel_tolerance(cls, value: Union[int, str]) -> Union[int, str]:
        if isinstance(value, int) and (value < 1 or value > 100):
            raise ValueError(
                "pixel_tolerance must be between 1 and 100, got: {}".format(value)
            )
        return value

    @field_validator("sigma")
    @classmethod
    def validate_sigma(cls, value: Union[float, str]) -> Union[float, str]:
        if isinstance(value, float) and (value < 0.01 or value > 10.0):
            raise ValueError(
                "sigma must be between 0.01 and 10.0, got: {}".format(value)
            )
        return value

    @field_validator("min_contour_area")
    @classmethod
    def validate_min_contour_area(cls, value: Union[float, str]) -> Union[float, str]:
        if isinstance(value, float) and (value < 0.0 or value > 10000.0):
            raise ValueError(
                "min_contour_area must be between 0.0 and 10000.0, got: {}".format(
                    value
                )
            )
        return value

    @field_validator("dilation_iterations")
    @classmethod
    def validate_dilation_iterations(cls, value: Union[int, str]) -> Union[int, str]:
        if isinstance(value, int) and (value < 0 or value > 20):
            raise ValueError(
                "dilation_iterations must be between 0 and 20, got: {}".format(value)
            )
        return value

    @field_validator("boundary_band_width")
    @classmethod
    def validate_boundary_band_width(cls, value: Union[int, str]) -> Union[int, str]:
        if isinstance(value, int) and (value < 1 or value > 100):
            raise ValueError(
                "boundary_band_width must be between 1 and 100, got: {}".format(value)
            )
        return value

    @field_validator("adaptive_window_size")
    @classmethod
    def validate_adaptive_window_size(cls, value: Union[int, str]) -> Union[int, str]:
        if isinstance(value, int) and (value < 3 or value > 201):
            raise ValueError(
                "adaptive_window_size must be between 3 and 201, got: {}".format(value)
            )
        return value

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="refined_segmentation",
                kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
            OutputDefinition(
                name="edges",
                kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class MaskEdgeSnapBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[MaskEdgeSnapManifest]:
        return MaskEdgeSnapManifest

    def run(
        self,
        image: WorkflowImageData,
        segmentation: sv.Detections,
        pixel_tolerance: int,
        sigma: float,
        min_contour_area: float,
        dilation_iterations: int,
        boundary_band_width: int,
        adaptive_window_size: int,
        *args,
        **kwargs,
    ) -> BlockResult:
        refined_segmentation, edges = refine_masks(
            image=image,
            segmentation=segmentation,
            pixel_tolerance=pixel_tolerance,
            sigma=sigma,
            min_contour_area=min_contour_area,
            dilation_iterations=dilation_iterations,
            boundary_band_width=boundary_band_width,
            adaptive_window_size=adaptive_window_size,
        )

        return {
            "refined_segmentation": refined_segmentation,
            "edges": edges,
        }


def _zhang_suen_one_iteration(image: np.ndarray) -> np.ndarray:
    """Run a single iteration of Zhang-Suen thinning.

    Removes only "simple points" — pixels whose deletion preserves local
    connectivity. Repeatedly applying this would converge to a 1-pixel
    skeleton; calling it once peels exactly one layer off thick regions
    while guaranteeing no narrow section, junction, or endpoint is broken.

    Neighborhood layout:
        P9 P2 P3
        P8 P1 P4
        P7 P6 P5
    """
    img = (image > 0).astype(np.uint8)
    h, w = img.shape
    if h < 3 or w < 3:
        return (img * 255).astype(np.uint8)

    def neighbors(arr):
        P2 = arr[:-2, 1:-1]
        P3 = arr[:-2, 2:]
        P4 = arr[1:-1, 2:]
        P5 = arr[2:, 2:]
        P6 = arr[2:, 1:-1]
        P7 = arr[2:, :-2]
        P8 = arr[1:-1, :-2]
        P9 = arr[:-2, :-2]
        return P2, P3, P4, P5, P6, P7, P8, P9

    def conditions(P1, P2, P3, P4, P5, P6, P7, P8, P9):
        # B(p): count of foreground neighbors (must be 2..6)
        B = (P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9).astype(np.int16)
        # A(p): number of 0->1 transitions in cyclic sequence
        A = (
            ((P2 == 0) & (P3 == 1)).astype(np.int16)
            + ((P3 == 0) & (P4 == 1)).astype(np.int16)
            + ((P4 == 0) & (P5 == 1)).astype(np.int16)
            + ((P5 == 0) & (P6 == 1)).astype(np.int16)
            + ((P6 == 0) & (P7 == 1)).astype(np.int16)
            + ((P7 == 0) & (P8 == 1)).astype(np.int16)
            + ((P8 == 0) & (P9 == 1)).astype(np.int16)
            + ((P9 == 0) & (P2 == 1)).astype(np.int16)
        )
        return (P1 == 1) & (B >= 2) & (B <= 6) & (A == 1)

    # Sub-iteration 1: target south-east border points
    P1 = img[1:-1, 1:-1]
    P2, P3, P4, P5, P6, P7, P8, P9 = neighbors(img)
    base = conditions(P1, P2, P3, P4, P5, P6, P7, P8, P9)
    cond1 = base & ((P2 * P4 * P6) == 0) & ((P4 * P6 * P8) == 0)
    img[1:-1, 1:-1] = np.where(cond1, 0, P1)

    # Sub-iteration 2: target north-west border points
    P1 = img[1:-1, 1:-1]
    P2, P3, P4, P5, P6, P7, P8, P9 = neighbors(img)
    base = conditions(P1, P2, P3, P4, P5, P6, P7, P8, P9)
    cond2 = base & ((P2 * P4 * P8) == 0) & ((P2 * P6 * P8) == 0)
    img[1:-1, 1:-1] = np.where(cond2, 0, P1)

    return (img * 255).astype(np.uint8)


def refine_masks(
    image: WorkflowImageData,
    segmentation: sv.Detections,
    pixel_tolerance: int,
    sigma: float,
    min_contour_area: float,
    dilation_iterations: int,
    boundary_band_width: int,
    adaptive_window_size: int,
) -> tuple:
    """Refine instance segmentation masks by snapping edges to detected boundaries."""

    np_img = image.numpy_image.copy()
    H, W = np_img.shape[:2]

    # Convert to grayscale
    if len(np_img.shape) == 2:
        gray = np_img.copy()
    elif np_img.shape[2] == 1:
        gray = np_img[:, :, 0]
    elif np_img.shape[2] == 4:
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGRA2GRAY)
    elif np_img.shape[2] == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    else:
        # For any other case, try to convert assuming BGR
        try:
            gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            # If conversion fails, take first channel
            gray = np_img[:, :, 0] if len(np_img.shape) >= 3 else np_img.copy()

    tol = int(pixel_tolerance)
    band_radius = max(1, int(boundary_band_width))
    band_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_radius * 2 + 1, band_radius * 2 + 1)
    )

    # Create boundary band from segmentation masks
    if segmentation.mask is not None and len(segmentation) > 0:
        boundary_band_pre = np.zeros((H, W), dtype=np.uint8)
        for m in segmentation.mask:
            m_uint8 = m.astype(np.uint8)
            if m_uint8.shape[:2] != (H, W):
                m_uint8 = cv2.resize(m_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
            inner_i = cv2.erode(m_uint8, band_kernel)
            outer_i = cv2.dilate(m_uint8, band_kernel)
            boundary_band_pre = np.maximum(
                boundary_band_pre, ((outer_i > 0) & (inner_i == 0)).astype(np.uint8)
            )
    else:
        boundary_band_pre = np.ones((H, W), dtype=np.uint8)

    # Compute Sobel edges
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)

    # Adaptive threshold on Sobel magnitude
    win = max(3, int(adaptive_window_size))
    if win % 2 == 0:
        win += 1
    mag_mean = cv2.boxFilter(magnitude, cv2.CV_32F, (win, win))
    mag_sq_mean = cv2.boxFilter(magnitude * magnitude, cv2.CV_32F, (win, win))
    mag_var = np.maximum(mag_sq_mean - mag_mean * mag_mean, 0.0)
    mag_std = np.sqrt(mag_var)
    threshold_field = mag_mean + float(sigma) * mag_std
    edges_adaptive = (magnitude > threshold_field).astype(np.uint8) * 255

    # Morphological closing and thinning
    iterations = max(0, int(dilation_iterations))
    if iterations > 0:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_adaptive = cv2.morphologyEx(
            edges_adaptive, cv2.MORPH_CLOSE, close_kernel, iterations=iterations
        )
        edges_adaptive = _zhang_suen_one_iteration(edges_adaptive)

    edges = edges_adaptive

    # Apply boundary band filter
    if segmentation.mask is not None and len(segmentation) > 0:
        edges_to_filter = (edges * boundary_band_pre).astype(np.uint8)
    else:
        edges_to_filter = edges.copy()

    # Filter by contour area
    min_area = max(0.0, float(min_contour_area))
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        edges_to_filter, connectivity=8
    )
    edge_filtered = np.zeros((H, W), dtype=np.uint8)
    for lbl in range(1, num_labels):
        comp_mask = (labels == lbl).astype(np.uint8)
        comp_contours, _ = cv2.findContours(
            comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not comp_contours:
            continue
        main_contour = max(comp_contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) >= min_area:
            edge_filtered[labels == lbl] = 255

    snap_region = edge_filtered > 0

    # Build edges detection output
    if snap_region.any():
        ys, xs = np.where(snap_region)
        import uuid

        edges_detections = sv.Detections(
            xyxy=np.array(
                [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=np.float32
            ),
            mask=snap_region[None, :, :],
            confidence=np.array([1.0], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            data={
                "class_name": np.array(["edges"]),
                "detection_id": np.array([str(uuid.uuid4())]),
            },
        )
    else:
        edges_detections = sv.Detections.empty()

    # If no segmentation, return early
    if segmentation.mask is None or len(segmentation) == 0:
        return segmentation, edges_detections

    # Snap contours to edges
    refined_masks = []
    TANGENT_WINDOW = 5

    for mask in segmentation.mask:
        mask_uint8 = mask.astype(np.uint8)
        if mask_uint8.shape[:2] != (H, W):
            mask_uint8 = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)

        dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        mask_dilated = cv2.dilate(mask_uint8, np.ones((3, 3), np.uint8))
        valid_snap = snap_region & ((dist <= tol) | (mask_dilated == 0))

        mask_contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not mask_contours:
            refined_masks.append(mask_uint8.astype(bool))
            continue

        new_mask = np.zeros((H, W), dtype=np.uint8)

        for mc in mask_contours:
            mc_pts = mc.reshape(-1, 2).astype(np.float32)
            n_pts = mc_pts.shape[0]
            refined_pts = mc_pts.copy()

            for i in range(n_pts):
                px, py = mc_pts[i]

                prev_pt = mc_pts[(i - TANGENT_WINDOW) % n_pts]
                next_pt = mc_pts[(i + TANGENT_WINDOW) % n_pts]
                tangent = next_pt - prev_pt
                t_len = np.linalg.norm(tangent)
                if t_len < 1e-6:
                    continue
                tangent /= t_len
                nx, ny = -tangent[1], tangent[0]

                best_score = 0.0
                best_pt = None

                for sign in (1.0, -1.0):
                    for d in range(1, tol + 1):
                        sx = int(round(px + sign * nx * d))
                        sy = int(round(py + sign * ny * d))
                        if 0 <= sx < W and 0 <= sy < H and valid_snap[sy, sx]:
                            mag = float(magnitude[sy, sx])
                            proximity = 1.0 - d / (tol + 1)
                            score = mag * proximity
                            if score > best_score:
                                best_score = score
                                best_pt = (sx, sy)

                if best_pt is not None:
                    refined_pts[i] = best_pt

            refined_contour = refined_pts.reshape(-1, 1, 2).astype(np.int32)
            cv2.fillPoly(new_mask, [refined_contour], color=1)

        refined_masks.append(new_mask.astype(bool))

    refined_masks_np = np.stack(refined_masks, axis=0)
    refined_detections = sv.Detections(
        xyxy=segmentation.xyxy.copy(),
        mask=refined_masks_np,
        confidence=segmentation.confidence,
        class_id=segmentation.class_id,
        tracker_id=segmentation.tracker_id,
        data=segmentation.data,
    )

    return refined_detections, edges_detections
