from typing import List, Literal, Optional, Tuple, Type, Union

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.constants import (
    POLYGON_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "zones"
OUTPUT_KEY_DETECTIONS: str = "predictions"
OUTPUT_KEY_SIMPLIFICATION_CONVERGED: str = "simplification_converged"
TYPE: str = "roboflow_core/dynamic_zone@v1"
SHORT_DESCRIPTION = (
    "Simplify polygons so they are geometrically convex "
    "and contain only the requested amount of vertices."
)
LONG_DESCRIPTION = """
Generate simplified polygon zones from instance segmentation detections by converting masks to contours, computing convex hulls, reducing polygon vertices to a specified count using Douglas-Peucker approximation, optionally applying least squares edge fitting, and scaling polygons to create geometric zones based on detected object shapes for zone-based analytics, spatial filtering, and region-of-interest definition workflows.

## How This Block Works

This block creates simplified polygon zones from instance segmentation detections by converting complex mask shapes into geometrically convex polygons with a specified number of vertices. The block:

1. Receives instance segmentation predictions containing masks (polygon representations) for detected objects
2. Converts masks to contours:
   - Extracts contours from each detection mask using mask-to-polygon conversion
   - Selects the largest contour from each detection (handles multiple contours per mask)
3. Computes convex hull:
   - Calculates the convex hull of the largest contour using OpenCV's convex hull algorithm
   - Ensures the resulting polygon is geometrically convex (no inward-facing angles)
   - Creates a simplified outer boundary that encompasses all points in the contour
4. Simplifies polygon to required vertex count:
   - Uses Douglas-Peucker polygon approximation algorithm to reduce vertices
   - Iteratively adjusts epsilon parameter to achieve the target number of vertices
   - Uses binary search to find the optimal epsilon value that produces the requested vertex count
   - Handles convergence: if exact vertex count cannot be achieved, pads or truncates vertices
5. Optionally applies least squares edge fitting:
   - If `apply_least_squares` is enabled, refines polygon edges by fitting lines to original contour points
   - Selects contour points between polygon vertices
   - Optionally filters to midpoint fraction (e.g., uses only central portion of each edge) to avoid edge effects
   - Fits least squares lines to selected contour points for each edge
   - Calculates intersections of fitted lines to create refined vertex positions
   - Produces a polygon that better aligns with the original contour shape
6. Scales polygon (if `scale_ratio` != 1):
   - Calculates polygon centroid (center of mass)
   - Scales polygon relative to centroid by the specified scale ratio
   - Expands or contracts polygon outward from center (scale > 1 expands, scale < 1 contracts)
   - Useful for creating buffer zones or adjusting zone boundaries
7. Updates detections with simplified polygons:
   - Stores simplified polygons in detection metadata under the polygon key
   - Regenerates masks from simplified polygons for updated detection representation
8. Returns simplified zones and updated detections:
   - `zones`: List of simplified polygons (one per detection) as coordinate lists
   - `predictions`: Updated detections with simplified polygons and masks
   - `simplification_converged`: Boolean indicating if all polygons converged to exact vertex count

The block enables creation of geometric zones from complex object shapes detected by segmentation models. It's particularly useful when zones need to be created based on detected object shapes (e.g., basketball courts, road segments, parking lots, fields) where the zone should match the object's outline but be simplified for performance and ease of use.

## Common Use Cases

- **Zone Creation from Detections**: Create polygon zones based on detected object shapes (e.g., create basketball court zones from court detections, generate road segment zones from road detections, create field zones from sports field detections), enabling detection-based zone workflows
- **Geometric Zone Simplification**: Simplify complex object shapes into geometrically convex polygons with controlled vertex counts (e.g., simplify irregular shapes to rectangles/quadrilaterals, reduce complex polygons to manageable vertex counts, create geometric zones from masks), enabling zone simplification workflows
- **Dynamic Zone Definition**: Dynamically define zones based on detected objects in images (e.g., define zones from detected regions, create zones from object shapes, generate zones from segmentation results), enabling dynamic zone workflows
- **Zone-Based Analytics Setup**: Prepare zones for zone-based analytics and filtering (e.g., prepare zones for time-in-zone analytics, create zones for zone-based filtering, set up zones for spatial analytics), enabling zone-based analytics workflows
- **Region-of-Interest Definition**: Define regions of interest based on detected object boundaries (e.g., define ROIs from object detections, create ROI zones from segmentation, generate interest regions from masks), enabling ROI definition workflows
- **Spatial Filtering and Analysis**: Create zones for spatial filtering and analysis operations (e.g., create zones for spatial filtering, prepare zones for area calculations, generate zones for spatial queries), enabling spatial analysis workflows

## Connecting to Other Blocks

This block receives instance segmentation predictions and produces simplified polygon zones:

- **After instance segmentation models** to create zones from detected object shapes (e.g., segmentation model to zones, masks to simplified polygons, detections to geometric zones), enabling segmentation-to-zone workflows
- **After detection filtering blocks** to create zones from filtered detections (e.g., filter detections then create zones, create zones from specific classes, generate zones from filtered results), enabling filter-to-zone workflows
- **Before zone-based analytics blocks** to provide simplified zones for analytics (e.g., zones for time-in-zone, zones for zone analytics, polygons for zone filtering), enabling zone-to-analytics workflows
- **Before visualization blocks** to display simplified zones (e.g., visualize zone polygons, display geometric zones, show simplified regions), enabling zone visualization workflows
- **Before spatial filtering blocks** to provide zones for spatial operations (e.g., zones for overlap filtering, polygons for spatial queries, regions for area calculations), enabling zone-to-filter workflows
- **In workflow outputs** to provide simplified zones as final output (e.g., zone generation workflows, polygon extraction workflows, geometric zone outputs), enabling zone output workflows

## Requirements

This block requires instance segmentation predictions with masks (polygon data). Input detections should be filtered to contain only the desired classes of interest before processing. The `required_number_of_vertices` parameter specifies the target vertex count for simplified polygons (e.g., 4 for rectangles/quadrilaterals, 3 for triangles). The block uses iterative Douglas-Peucker approximation with binary search to achieve the target vertex count, with a maximum of 1000 iterations. If convergence to exact vertex count fails, vertices are padded or truncated. The `scale_ratio` parameter (default 1) scales polygons relative to their centroid. The `apply_least_squares` parameter (default False) enables edge fitting to better align polygon edges with original contours. The `midpoint_fraction` parameter (0-1, default 1) controls which portion of contour points are used for least squares fitting (1 = all points, lower values use central portions of edges). The block outputs simplified polygons as lists of coordinate pairs, updated detections with simplified polygons, and a convergence flag.
"""


class DynamicZonesManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Dynamic Zone",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-square-dashed",
                "blockPriority": 3,
                "opencv": True,
            },
        }
    )
    type: Literal[f"{TYPE}", "DynamicZone"]
    predictions: Selector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Instance segmentation predictions containing masks (polygon data) for detected objects. Detections should be filtered to contain only desired classes of interest. Each detection's mask is converted to contours, and the largest contour is used to generate a simplified polygon zone. Supports instance segmentation format with mask data.",
        examples=[
            "$steps.instance_segmentation_model.predictions",
            "$segmentation.predictions",
        ],
    )
    required_number_of_vertices: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Target number of vertices for simplified polygons. The block uses Douglas-Peucker polygon approximation with iterative binary search to reduce polygon vertices to this count. Common values: 4 for rectangles/quadrilaterals, 3 for triangles, 6+ for more complex shapes. The algorithm attempts to converge to this exact count; if convergence fails (within iteration limit), vertices are padded or truncated to match the count.",
        examples=[4, 3, 6, 8],
    )
    scale_ratio: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=1,
        description="Scale factor to expand or contract resulting polygons relative to their centroid. Values > 1 expand polygons outward from center (create buffer zones), values < 1 contract polygons inward. Value of 1 (default) means no scaling. Scaling is applied after polygon simplification. Useful for creating buffer zones or adjusting zone boundaries.",
        examples=[1.0, 1.05, 1.1, 0.95],
    )
    apply_least_squares: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        default=False,
        description="If True, applies least squares line fitting to refine polygon edges by aligning them with original contour points. For each edge of the simplified polygon, fits a line to contour points between vertices, then calculates intersections of fitted lines to create refined vertex positions. Produces polygons that better match the original contour shape, especially useful when simplified polygon vertices don't align well with contour edges.",
        examples=[False, True],
    )
    midpoint_fraction: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=1,
        description="Fraction (0-1) of contour points to use for least squares fitting on each edge. Value of 1 (default) uses all contour points between vertices. Lower values use only the central portion of each edge (e.g., 0.9 uses 90% of points, centered). Useful when convex polygon vertices are not well-aligned with edges, as it focuses fitting on the central portion of edges rather than edge effects near vertices. Only applies when apply_least_squares is True.",
        examples=[1.0, 0.9, 0.8],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["predictions"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(
                name=OUTPUT_KEY_DETECTIONS, kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]
            ),
            OutputDefinition(
                name=OUTPUT_KEY_SIMPLIFICATION_CONVERGED, kind=[BOOLEAN_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def calculate_simplified_polygon(
    contours: List[np.ndarray], required_number_of_vertices: int, max_steps: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    largest_contour = max(contours, key=len)

    # https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga014b28e56cb8854c0de4a211cb2be656
    convex_contour = cv.convexHull(
        points=largest_contour,
        returnPoints=True,
        clockwise=True,
    )
    # https://docs.opencv.org/4.9.0/d3/dc0/group__imgproc__shape.html#ga8d26483c636be6b35c3ec6335798a47c
    perimeter = cv.arcLength(curve=convex_contour, closed=True)
    upper_epsilon = perimeter
    lower_epsilon = 0.0000001
    epsilon = lower_epsilon + upper_epsilon / 2
    # https://docs.opencv.org/4.9.0/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
    simplified_polygon = cv.approxPolyDP(
        curve=convex_contour, epsilon=epsilon, closed=True
    )
    for _ in range(max_steps):
        if len(simplified_polygon) == required_number_of_vertices:
            break
        if len(simplified_polygon) > required_number_of_vertices:
            lower_epsilon = epsilon
        else:
            upper_epsilon = epsilon
        epsilon = lower_epsilon + (upper_epsilon - lower_epsilon) / 2
        simplified_polygon = cv.approxPolyDP(
            curve=convex_contour, epsilon=epsilon, closed=True
        )
    while len(simplified_polygon.shape) > 2:
        simplified_polygon = np.concatenate(simplified_polygon)
    return simplified_polygon, largest_contour


def calculate_least_squares_polygon(
    contour: np.ndarray, polygon: np.ndarray, midpoint_fraction: float = 1
) -> np.ndarray:
    def find_closest_index(point: np.ndarray, contour: np.ndarray) -> int:
        dists = np.linalg.norm(contour - point, axis=1)
        return np.argmin(dists)

    def pick_contour_points_between_vertices(
        point_1: np.ndarray, point_2: np.ndarray, contour: np.ndarray
    ) -> np.ndarray:
        i1 = find_closest_index(point_1, contour)
        i2 = find_closest_index(point_2, contour)

        if i1 <= i2:
            return contour[i1 : i2 + 1]
        else:
            return np.concatenate((contour[i1:], contour[: i2 + 1]), axis=0)

    def least_squares_line(points: np.ndarray) -> Optional[Tuple[float, float]]:
        if len(points) < 2:
            return None
        x = points[:, 0]
        y = points[:, 1]
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return (a, b)

    def intersect_lines(
        line_1: Optional[Tuple[float, float]], line_2: Optional[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        if line_1 is None or line_2 is None:
            return None
        a_1, b_1 = line_1
        a_2, b_2 = line_2
        if np.isclose(a_1, a_2):
            return None
        x = (b_2 - b_1) / (a_1 - a_2)
        y = a_1 * x + b_1
        return np.array([x, y])

    pairs = [[polygon[-1], polygon[0]]] + list(zip(polygon[:-1], polygon[1:]))

    lines = []
    for point_1, point_2 in pairs:
        segment_points = pick_contour_points_between_vertices(point_1, point_2, contour)
        if midpoint_fraction < 1:
            number_of_points = int(round(len(segment_points) * midpoint_fraction))
            if number_of_points > 2:
                number_of_points_to_discard = (
                    len(segment_points) - number_of_points
                ) // 2
                segment_points = segment_points[
                    number_of_points_to_discard : len(segment_points)
                    - number_of_points_to_discard
                ]
        line_params = least_squares_line(segment_points)
        lines.append(line_params)

    intersections = []
    for i in range(len(lines)):
        line_1 = lines[i]
        line_2 = lines[(i + 1) % len(lines)]
        pt = intersect_lines(line_1, line_2)
        intersections.append(pt)

    return np.array(intersections, dtype=float).round().astype(int)


def scale_polygon(polygon: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1:
        return polygon

    M = cv.moments(polygon)

    if M["m00"] == 0:
        return polygon

    centroid_x = M["m10"] / M["m00"]
    centroid_y = M["m01"] / M["m00"]

    shifted = polygon - [centroid_x, centroid_y]
    scaled = shifted * scale
    result = scaled + [centroid_x, centroid_y]

    return result.round().astype(np.int32)


class DynamicZonesBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DynamicZonesManifest

    def run(
        self,
        predictions: Batch[sv.Detections],
        required_number_of_vertices: int,
        scale_ratio: float = 1,
        apply_least_squares: bool = False,
        midpoint_fraction: float = 1,
    ) -> BlockResult:
        result = []
        for detections in predictions:
            if detections is None:
                result.append(
                    {
                        OUTPUT_KEY: None,
                        OUTPUT_KEY_DETECTIONS: None,
                        OUTPUT_KEY_SIMPLIFICATION_CONVERGED: False,
                    }
                )
                continue
            simplified_polygons = []
            updated_detections = []
            if detections.mask is None:
                result.append(
                    {
                        OUTPUT_KEY: [],
                        OUTPUT_KEY_DETECTIONS: None,
                        OUTPUT_KEY_SIMPLIFICATION_CONVERGED: False,
                    }
                )
                continue
            all_converged = True
            for i, mask in enumerate(detections.mask):
                # copy
                updated_detection = detections[i]

                contours = sv.mask_to_polygons(mask)
                simplified_polygon, largest_contour = calculate_simplified_polygon(
                    contours=contours,
                    required_number_of_vertices=required_number_of_vertices,
                )
                if apply_least_squares:
                    simplified_polygon = calculate_least_squares_polygon(
                        contour=largest_contour,
                        polygon=simplified_polygon,
                        midpoint_fraction=midpoint_fraction,
                    )
                vertices_count, _ = simplified_polygon.shape
                if vertices_count < required_number_of_vertices:
                    all_converged = False
                    for _ in range(required_number_of_vertices - vertices_count):
                        simplified_polygon = np.append(
                            simplified_polygon,
                            [simplified_polygon[-1]],
                            axis=0,
                        )
                elif vertices_count > required_number_of_vertices:
                    all_converged = False
                    simplified_polygon = simplified_polygon[
                        :required_number_of_vertices
                    ]
                updated_detection[POLYGON_KEY_IN_SV_DETECTIONS] = np.array(
                    [simplified_polygon]
                )
                simplified_polygon = scale_polygon(
                    polygon=simplified_polygon,
                    scale=scale_ratio,
                )
                simplified_polygons.append(simplified_polygon.tolist())
                updated_detection.mask = np.array(
                    [
                        sv.polygon_to_mask(
                            polygon=simplified_polygon,
                            resolution_wh=mask.shape[::-1],
                        )
                    ]
                )
                updated_detections.append(updated_detection)
            result.append(
                {
                    OUTPUT_KEY: simplified_polygons,
                    OUTPUT_KEY_DETECTIONS: sv.Detections.merge(updated_detections),
                    OUTPUT_KEY_SIMPLIFICATION_CONVERGED: all_converged,
                }
            )
        if not result:
            result.append(
                {
                    OUTPUT_KEY: [],
                    OUTPUT_KEY_DETECTIONS: None,
                    OUTPUT_KEY_SIMPLIFICATION_CONVERGED: False,
                }
            )
        return result
