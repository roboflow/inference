from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = (
    "Refine segmentation masks by snapping boundaries to detected edges."
)

LONG_DESCRIPTION = """
Refine instance segmentation masks by detecting edges in images and snapping mask boundaries to detected edges. This block enhances segmentation accuracy by aligning mask contours with actual object boundaries detected through edge detection algorithms (Canny edge detection with preprocessing).

## How This Block Works

This block refines segmentation masks by detecting edges in the image and snapping mask boundaries to those edges. The process:

1. Receives an image and segmentation predictions with masks
2. Applies preprocessing to the image:
   - Converts to grayscale
   - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
   - Optionally applies bilateral filtering for noise reduction while preserving edges
3. Detects edges using Canny edge detection with configurable thresholds
4. Optionally dilates detected edges to make them more prominent
5. Creates a boundary band around each segmentation mask to isolate mask-specific edges
6. Filters edges by connected component size to remove isolated noise fragments
7. For each mask, finds the contour and refines it by snapping boundary points to nearby detected edges:
   - For each point on the mask contour, searches perpendicular to the contour (normal direction)
   - Snaps points to the nearest detected edge within the pixel tolerance
   - Uses local tangent direction to compute normal for robust edge snapping
8. Returns refined detections with updated masks and an edge visualization image

The edge snapping process makes mask boundaries align with detected edges, improving segmentation accuracy and boundary precision. The pixel_tolerance controls how far points can snap to edges, while min_contour_area filters out small noise edges.

## Common Use Cases

- **Crack Detection and Refinement**: Refine crack segmentation masks by snapping to detected crack edges, improving boundary accuracy for structural assessment and damage analysis
- **Surface Defect Detection**: Enhance defect segmentation by aligning mask boundaries with detected defect edges, improving accuracy for quality control and surface inspection
- **Industrial Inspection**: Refine detected anomalies (scratches, dents, corrosion) by snapping to actual edges, improving measurement accuracy and defect characterization
- **Medical Image Analysis**: Enhance lesion or tissue boundary segmentation by aligning masks to detected edges, improving clinical accuracy and feature extraction
- **Document Analysis**: Refine text or content boundary detection by snapping to document edges, improving OCR preprocessing and content extraction accuracy
- **Precision Object Detection**: Improve object boundary precision in industrial applications where exact edges are critical for downstream processing or measurement

## Connecting to Other Blocks

This block refines predictions from instance segmentation models:

- **Upstream -- Instance Segmentation Models**: Connect the output of an instance segmentation model (e.g., SAM, Mask R-CNN) that produces masks to refine
- **Upstream -- Image Preprocessing**: Connect preprocessed images to improve edge detection quality
- **Downstream -- Visualization Blocks**: Display refined masks and edge visualizations using polygon or mask visualization blocks
- **Downstream -- Measurement Blocks**: Measure refined mask areas with improved accuracy using the refined segmentation output
- **Downstream -- Storage/Export**: Save refined detections with corrected mask boundaries for archival or further analysis

## Requirements

This block requires instance segmentation predictions with masks. Input images should have clear contrast between foreground and background for effective edge detection. Canny edge detection parameters (low_threshold, high_threshold) should be tuned based on your image characteristics and lighting conditions.
"""

OUTPUT_PREDICTIONS_KEY = "refined_segmentation"
OUTPUT_EDGE_DETECTIONS_KEY = "edge_detections"


class EdgeRefinementManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/edge_refinement@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Edge Refinement",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "fas fa-vector-square",
                "blockPriority": 15,
                "opencv": True,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Input image for edge detection and boundary refinement. The image is converted to grayscale and preprocessed with contrast enhancement and optional bilateral filtering. Should have sufficient contrast between foreground objects and background for reliable edge detection. Original image metadata is preserved in the output edge visualization.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    segmentation: Selector(kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]) = Field(
        title="Segmentation Predictions",
        description="Instance segmentation predictions with masks to refine. The masks will be refined by snapping their boundaries to detected edges in the image. Input detections should contain segmentation masks (will be skipped if no masks are present).",
        examples=["$steps.segmentation.predictions"],
    )

    pixel_tolerance: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Maximum pixel distance to search for edges when snapping mask boundaries. Controls how far mask contour points can move to snap to detected edges. Larger values allow larger adjustments, smaller values keep refinement subtle. Typical values range from 1-10. Default is 5.",
        examples=[5, "$inputs.pixel_tolerance"],
    )

    low_threshold: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=50,
        description="Lower threshold for Canny edge detection. Edges with gradient magnitude below this value are not considered edges. Lower values detect fainter edges, higher values detect only strong edges. Typical range 30-100. Default is 50.",
        examples=[50, "$inputs.low_threshold"],
    )

    high_threshold: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=150,
        description="Upper threshold for Canny edge detection. Edges with gradient magnitude above this value are considered strong edges. Controls edge detection sensitivity. Typical range 100-200. Default is 150. Should be 2-3 times the low_threshold.",
        examples=[150, "$inputs.high_threshold"],
    )

    blur_kernel_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=7,
        description="Kernel size for bilateral filtering applied before edge detection (must be odd). Smooths image while preserving edges. Set to 1 to skip bilateral filtering. Typical values are 5, 7, 9, 11. Default is 7.",
        examples=[7, "$inputs.blur_kernel_size"],
    )

    min_contour_area: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=10.0,
        description="Minimum area in pixels for edge connected components to keep. Edges forming smaller regions than this are filtered as noise. Helps remove isolated edge fragments. Default is 10.0.",
        examples=[10.0, "$inputs.min_contour_area"],
    )

    dilation_iterations: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=0,
        description="Number of dilation iterations to apply to detected edges. Thickens edges to make them more prominent and easier to snap to. Set to 0 for no dilation (default). Typical values are 0-3.",
        examples=[0, "$inputs.dilation_iterations"],
    )

    band_width: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=15,
        description="Half-width of the boundary band around masks (full kernel size is 2*band_width+1). Controls the size of the elliptical morphological kernel used to create the boundary band for isolating mask-specific edges. Larger values create wider bands around masks. Typical values range from 5-25. Default is 15.",
        examples=[15, "$inputs.band_width"],
    )

    tangent_window: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Window size for computing local tangent direction at contour points. Uses points this many positions before and after each contour point to estimate the local tangent. Larger values provide smoother tangent estimates, smaller values follow finer detail. Typical values range from 2-10. Default is 5.",
        examples=[5, "$inputs.tangent_window"],
    )

    clahe_clip_limit: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=3.0,
        description="Clip limit for CLAHE (Contrast Limited Adaptive Histogram Equalization). Controls the maximum contrast enhancement in each tile. Higher values allow stronger contrast enhancement. Typical range 2.0-5.0. Default is 3.0.",
        examples=[3.0, "$inputs.clahe_clip_limit"],
    )

    clahe_tile_grid_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=8,
        description="Grid size for CLAHE tile division (uses square tiles of size clahe_tile_grid_size x clahe_tile_grid_size). Smaller tiles provide more local adaptation, larger tiles provide smoother results. Typical values range from 4-16. Default is 8.",
        examples=[8, "$inputs.clahe_tile_grid_size"],
    )

    bilateral_sigma_color: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=75.0,
        description="Sigma color parameter for bilateral filtering. Controls how much pixels with different colors are mixed. Typical range 50-100. Default is 75.0.",
        examples=[75.0, "$inputs.bilateral_sigma_color"],
    )

    bilateral_sigma_space: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=75.0,
        description="Sigma space parameter for bilateral filtering. Controls the spatial extent of the filtering kernel. Typical range 50-100. Default is 75.0.",
        examples=[75.0, "$inputs.bilateral_sigma_space"],
    )

    dilation_kernel_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=3,
        description="Kernel size for morphological dilation of edges (must be odd). Used when dilation_iterations > 0. Larger kernels produce thicker edges. Typical values are 3, 5, 7. Default is 3.",
        examples=[3, "$inputs.dilation_kernel_size"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_PREDICTIONS_KEY,
                kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
            OutputDefinition(
                name=OUTPUT_EDGE_DETECTIONS_KEY,
                kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class EdgeRefinementBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[EdgeRefinementManifest]:
        return EdgeRefinementManifest

    def run(
        self,
        image: WorkflowImageData,
        segmentation: sv.Detections,
        pixel_tolerance: int,
        low_threshold: int,
        high_threshold: int,
        blur_kernel_size: int,
        min_contour_area: float,
        dilation_iterations: int,
        band_width: int,
        tangent_window: int,
        clahe_clip_limit: float,
        clahe_tile_grid_size: int,
        bilateral_sigma_color: float,
        bilateral_sigma_space: float,
        dilation_kernel_size: int,
        *args,
        **kwargs,
    ) -> BlockResult:
        np_img = image.numpy_image.copy()
        H, W = np_img.shape[:2]

        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(int(clahe_tile_grid_size), int(clahe_tile_grid_size)),
        )
        gray = clahe.apply(gray)

        diameter = max(1, int(blur_kernel_size))
        if diameter > 1:
            gray = cv2.bilateralFilter(
                gray,
                diameter,
                sigmaColor=float(bilateral_sigma_color),
                sigmaSpace=float(bilateral_sigma_space),
            )

        edges = cv2.Canny(gray, int(low_threshold), int(high_threshold))

        iterations = max(0, int(dilation_iterations))
        if iterations > 0:
            kern_size = int(dilation_kernel_size)
            # Ensure kernel size is odd
            kern_size = kern_size if kern_size % 2 == 1 else kern_size + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kern_size, kern_size))
            edges = cv2.dilate(edges, kernel, iterations=iterations)

        tol = int(pixel_tolerance)
        band_w = int(band_width)
        band_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (band_w * 2 + 1, band_w * 2 + 1)
        )

        # Clip edges to mask boundary band FIRST — severs cracks from surface edges
        # so they become separate small components that fail the size filter
        if segmentation.mask is not None and len(segmentation) > 0:
            boundary_band_pre = np.zeros((H, W), dtype=np.uint8)
            for m in segmentation.mask:
                m_uint8 = m.astype(np.uint8)
                inner_i = cv2.erode(m_uint8, band_kernel)
                outer_i = cv2.dilate(m_uint8, band_kernel)
                boundary_band_pre = np.maximum(
                    boundary_band_pre,
                    ((outer_i > 0) & (inner_i == 0)).astype(np.uint8),
                )
            edges_to_filter = (edges * boundary_band_pre).astype(np.uint8)
        else:
            edges_to_filter = edges.copy()

        # Filter by connected component size — surface edges span the image,
        # isolated crack fragments are small and get removed
        min_area = max(0.0, float(min_contour_area))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            edges_to_filter, connectivity=8
        )
        edge_filtered = np.zeros((H, W), dtype=np.uint8)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                edge_filtered[labels == lbl] = 255

        # Convert edge contours to Detections object for debugging visualization
        edge_contours, _ = cv2.findContours(
            edge_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        edge_masks = []
        edge_xyxy = []
        for contour in edge_contours:
            edge_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(edge_mask, [contour], 0, 1, -1)
            edge_masks.append(edge_mask.astype(bool))
            if len(contour) > 0:
                x = contour[:, 0, 0]
                y = contour[:, 0, 1]
                x_min, x_max = int(np.min(x)), int(np.max(x))
                y_min, y_max = int(np.min(y)), int(np.max(y))
                edge_xyxy.append([x_min, y_min, x_max, y_max])
            else:
                edge_xyxy.append([0, 0, W - 1, H - 1])

        if len(edge_masks) > 0:
            edge_detections = sv.Detections(
                xyxy=np.array(edge_xyxy, dtype=np.float32),
                mask=np.array(edge_masks, dtype=bool),
            )
        else:
            edge_detections = sv.Detections(
                xyxy=np.array([], dtype=np.float32).reshape(0, 4),
            )

        edge_bool = edge_filtered > 0

        if segmentation.mask is None or len(segmentation) == 0:
            return {
                OUTPUT_PREDICTIONS_KEY: segmentation,
                OUTPUT_EDGE_DETECTIONS_KEY: edge_detections,
            }

        refined_masks = []
        tangent_win = int(tangent_window)

        for mask in segmentation.mask:
            mask_uint8 = mask.astype(np.uint8)

            dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            mask_dilated = cv2.dilate(mask_uint8, np.ones((3, 3), np.uint8))
            valid_snap = edge_bool & ((dist <= tol) | (mask_dilated == 0))

            mask_contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not mask_contours:
                refined_masks.append(mask)
                continue

            new_mask = np.zeros((H, W), dtype=np.uint8)

            for mc in mask_contours:
                mc_pts = mc.reshape(-1, 2).astype(np.float32)
                n_pts = mc_pts.shape[0]
                refined_pts = mc_pts.copy()

                for i in range(n_pts):
                    px, py = mc_pts[i]

                    prev_pt = mc_pts[(i - tangent_win) % n_pts]
                    next_pt = mc_pts[(i + tangent_win) % n_pts]
                    tangent = next_pt - prev_pt
                    t_len = np.linalg.norm(tangent)
                    if t_len < 1e-6:
                        continue
                    tangent /= t_len
                    nx, ny = -tangent[1], tangent[0]

                    best_dist = tol + 1
                    best_pt = None

                    for sign in (1.0, -1.0):
                        for d in range(1, tol + 1):
                            sx = int(round(px + sign * nx * d))
                            sy = int(round(py + sign * ny * d))
                            if 0 <= sx < W and 0 <= sy < H:
                                if valid_snap[sy, sx]:
                                    if d < best_dist:
                                        best_dist = d
                                        best_pt = (sx, sy)
                                    break

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
        return {
            OUTPUT_PREDICTIONS_KEY: refined_detections,
            OUTPUT_EDGE_DETECTIONS_KEY: edge_detections,
        }
