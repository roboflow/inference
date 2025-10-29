import math
from typing import List, Optional, Tuple, Union

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.constants import (
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_DETECTIONS_KEY: str = "corrected_coordinates"
OUTPUT_IMAGE_KEY: str = "warped_image"
OUTPUT_EXTENDED_TRANSFORMED_RECT_WIDTH_KEY: str = "extended_transformed_rect_width"
OUTPUT_EXTENDED_TRANSFORMED_RECT_HEIGHT_KEY: str = "extended_transformed_rect_height"
TYPE: str = "PerspectiveCorrection"
SHORT_DESCRIPTION = (
    "Adjust detection coordinates from a polygon-defined plane "
    "to a straight rectangular plane with specified width and height."
)
LONG_DESCRIPTION = """
The `PerspectiveCorrectionBlock` is a transformer block designed to correct
coordinates of detections based on transformation defined by two polygons.
This block is best suited when produced coordinates should be considered as if camera
was placed directly above the scene and was not introducing distortions.
"""
ALL_POSITIONS = "ALL"


class PerspectiveCorrectionManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Perspective Correction",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-toolbox",
                "blockPriority": 2,
                "opencv": True,
            },
        }
    )
    type: Literal["roboflow_core/perspective_correction@v1", "PerspectiveCorrection"]
    predictions: Optional[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
            ]
        )
    ] = Field(  # type: ignore
        description="Predictions",
        default=None,
        examples=["$steps.object_detection_model.predictions"],
    )
    images: Selector(kind=[IMAGE_KIND]) = Field(
        title="Image to Crop",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    perspective_polygons: Union[list, Selector(kind=[LIST_OF_VALUES_KIND])] = Field(  # type: ignore
        description="Perspective polygons (for each batch at least one must be consisting of 4 vertices)",
        examples=["$steps.perspective_wrap.zones"],
    )
    transformed_rect_width: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Transformed rect width", default=1000, examples=[1000]
    )
    transformed_rect_height: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Transformed rect height", default=1000, examples=[1000]
    )
    extend_perspective_polygon_by_detections_anchor: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description=f"If set, perspective polygons will be extended to contain all bounding boxes. Allowed values: {', '.join(sv.Position.list())}"
        + f" and {ALL_POSITIONS} to extend to contain whole bounding box",
        default="",
        examples=["CENTER"],
    )
    warp_image: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description=f"If set to True, image will be warped into transformed rect",
        default=False,
        examples=[False],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images", "predictions"]

    @classmethod
    def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
        return ["perspective_polygons"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_DETECTIONS_KEY,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    IMAGE_KIND,
                ],
            ),
            OutputDefinition(
                name=OUTPUT_EXTENDED_TRANSFORMED_RECT_WIDTH_KEY,
                kind=[
                    INTEGER_KIND,
                ],
            ),
            OutputDefinition(
                name=OUTPUT_EXTENDED_TRANSFORMED_RECT_HEIGHT_KEY,
                kind=[
                    INTEGER_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def pick_largest_perspective_polygons(
    perspective_polygons_batch: Union[
        List[np.ndarray],
        List[List[np.ndarray]],
        List[List[List[int]]],
        List[List[List[List[int]]]],
    ],
) -> List[np.ndarray]:
    if not isinstance(perspective_polygons_batch, (list, Batch)):
        raise ValueError("Unexpected type of input")
    if not perspective_polygons_batch:
        raise ValueError("Unexpected empty batch")
    if len(perspective_polygons_batch) == 4 and all(
        isinstance(p, list) and len(p) == 2 for p in perspective_polygons_batch
    ):
        perspective_polygons_batch = [perspective_polygons_batch]

    largest_perspective_polygons: List[np.ndarray] = []
    for polygons in perspective_polygons_batch:
        if polygons is None:
            continue
        if not isinstance(polygons, list) and not isinstance(polygons, np.ndarray):
            raise ValueError("Unexpected type of batch element")
        if len(polygons) == 0:
            raise ValueError("Unexpected empty batch element")
        if isinstance(polygons, np.ndarray):
            if polygons.shape != (4, 2):
                raise ValueError("Unexpected shape of batch element")
            largest_perspective_polygons.append(polygons)
            continue
        if len(polygons) == 4 and all(
            isinstance(p, list) and len(p) == 2 for p in polygons
        ):
            largest_perspective_polygons.append(np.array(polygons))
            continue
        polygons = [p if isinstance(p, np.ndarray) else np.array(p) for p in polygons]
        polygons = [p for p in polygons if p.shape == (4, 2)]
        if not polygons:
            raise ValueError("No batch element consists of 4 vertices")
        polygons = [np.around(p).astype(np.int32) for p in polygons]
        largest_polygon = max(polygons, key=lambda p: cv.contourArea(p))
        largest_perspective_polygons.append(largest_polygon)
    return largest_perspective_polygons


def sort_polygon_vertices_clockwise(polygon: np.ndarray) -> np.ndarray:
    x_center = min(polygon[:, 0]) / 2 + max(polygon[:, 0]) / 2
    y_center = min(polygon[:, 1]) / 2 + max(polygon[:, 1]) / 2
    angle = lambda p: math.atan2(x_center - p[0], y_center - p[1])
    return np.array(sorted(polygon.tolist(), key=angle, reverse=True))


def roll_polygon_vertices_to_start_from_leftmost_bottom(
    polygon: np.ndarray,
) -> np.ndarray:
    x_min = min(polygon[:, 0])
    x_max = max(polygon[:, 0])
    y_min = min(polygon[:, 1])
    y_max = max(polygon[:, 1])
    leftmost_bottom_rect = [
        [x_min, y_max],
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
    ]
    min_dist = sum(
        ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        for (x1, y1), (x2, y2) in zip(leftmost_bottom_rect, polygon)
    )
    closest = polygon
    for shift in range(4):
        rolled = np.roll(polygon, shift=(0, shift), axis=(0, 0))
        dist = sum(
            ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            for (x1, y1), (x2, y2) in zip(leftmost_bottom_rect, rolled)
        )
        if dist < min_dist or (dist == min_dist and rolled[0][0] < closest[0][0]):
            min_dist = dist
            closest = rolled
    return closest


def ccw(x1, y1, x2, y2, x3, y3):
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) > 0


def calculate_line_coeffs(
    x1: int, y1: int, x2: int, y2: int
) -> Tuple[Optional[float], float]:
    if x1 == x2:
        return None, x1
    # Solved a and b for ax + b = y
    return (y2 - y1) / (x2 - x1), (y1 * x2 - y2 * x1) / (x2 - x1)


def calculate_line_intercept_to_contain_point(
    a: Optional[float],
    x: int,
    y: int,
) -> float:
    if a is None:
        return x
    return y - a * x


def solve_line_intersection(
    a1: Optional[float],
    b1: float,
    a2: Optional[float],
    b2: float,
) -> Tuple[float, float]:
    if a1 is None and a2 is None:
        raise ValueError("Both lines are vertical")
    if a1 is None:
        x = b1
        y = a2 * x + b2
    elif a2 is None:
        x = b2
        y = a1 * x + b1
    else:
        x = (b2 - b1) / (a1 - a2)
        y = a1 * x + b1
    return x, y


def calculate_vertices_to_contain_point(
    vertex_1: np.ndarray,
    vertex_2: np.ndarray,
    vertex_3_from_1: np.ndarray,
    vertex_4_from_2: np.ndarray,
    x: int,
    y: int,
) -> Tuple[np.ndarray, np.ndarray]:
    a, _ = calculate_line_coeffs(
        x1=vertex_1[0],
        y1=vertex_1[1],
        x2=vertex_2[0],
        y2=vertex_2[1],
    )
    b = calculate_line_intercept_to_contain_point(
        a=a,
        x=x,
        y=y,
    )
    a_3_1, b_3_1 = calculate_line_coeffs(
        x1=vertex_1[0],
        y1=vertex_1[1],
        x2=vertex_3_from_1[0],
        y2=vertex_3_from_1[1],
    )
    vertex_1 = (
        np.array(
            solve_line_intersection(
                a1=a_3_1,
                b1=b_3_1,
                a2=a,
                b2=b,
            )
        )
        .round()
        .astype(int)
    )
    a_4_2, b_4_2 = calculate_line_coeffs(
        x1=vertex_2[0],
        y1=vertex_2[1],
        x2=vertex_4_from_2[0],
        y2=vertex_4_from_2[1],
    )
    vertex_2 = (
        np.array(
            solve_line_intersection(
                a1=a_4_2,
                b1=b_4_2,
                a2=a,
                b2=b,
            )
        )
        .round()
        .astype(int)
    )
    return vertex_1, vertex_2


def extend_perspective_polygon(
    polygon: List[np.ndarray],
    detections: sv.Detections,
    bbox_position: Union[sv.Position, Literal[ALL_POSITIONS]],
) -> Tuple[np.ndarray, float, float, float, float]:
    if not bbox_position:
        return polygon
    bottom_left, top_left, top_right, bottom_right = polygon
    extended_width = 0
    extended_height = 0
    original_width = max(
        (
            (bottom_left[0] - bottom_right[0]) ** 2
            + (bottom_left[1] - bottom_right[1]) ** 2
        )
        ** 0.5,
        ((top_left[0] - top_right[0]) ** 2 + (top_left[1] - top_right[1]) ** 2) ** 0.5,
    )
    original_height = max(
        ((bottom_left[0] - top_left[0]) ** 2 + (bottom_left[1] - top_left[1]) ** 2)
        ** 0.5,
        ((bottom_right[0] - top_right[0]) ** 2 + (bottom_right[1] - top_right[1]) ** 2)
        ** 0.5,
    )
    for i in range(len(detections)):
        det = detections[i]
        # extend to the left
        points = []
        if bbox_position == ALL_POSITIONS:
            points.append(
                det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_LEFT)[0]
            )
            points.append(det.get_anchors_coordinates(anchor=sv.Position.TOP_LEFT)[0])
        else:
            points.append(det.get_anchors_coordinates(anchor=bbox_position)[0])
        for x, y in points:
            if (
                cv.pointPolygonTest(
                    np.array([bottom_left, top_left, top_right, bottom_right]),
                    (x, y),
                    False,
                )
                >= 0
            ):
                continue
            if not ccw(
                x1=bottom_left[0],
                y1=bottom_left[1],
                x2=top_left[0],
                y2=top_left[1],
                x3=x,
                y3=y,
            ):
                original_bottom_left = bottom_left
                original_top_left = top_left
                bottom_left, top_left = calculate_vertices_to_contain_point(
                    vertex_1=original_bottom_left,
                    vertex_2=original_top_left,
                    vertex_3_from_1=bottom_right,
                    vertex_4_from_2=top_right,
                    x=x,
                    y=y,
                )
                extended_width += (
                    (bottom_left[0] - original_bottom_left[0]) ** 2
                    + (bottom_left[1] - original_bottom_left[1]) ** 2
                ) ** 0.5
        # extend to the right
        points = []
        if bbox_position == ALL_POSITIONS:
            points.append(
                det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_RIGHT)[0]
            )
            points.append(det.get_anchors_coordinates(anchor=sv.Position.TOP_RIGHT)[0])
        else:
            points.append(det.get_anchors_coordinates(anchor=bbox_position)[0])
        for x, y in points:
            if (
                cv.pointPolygonTest(
                    np.array([bottom_left, top_left, top_right, bottom_right]),
                    (x, y),
                    False,
                )
                >= 0
            ):
                continue
            if not ccw(
                x1=top_right[0],
                y1=top_right[1],
                x2=bottom_right[0],
                y2=bottom_right[1],
                x3=x,
                y3=y,
            ):
                original_bottom_right = bottom_right
                original_top_right = top_right
                top_right, bottom_right = calculate_vertices_to_contain_point(
                    vertex_1=original_top_right,
                    vertex_2=original_bottom_right,
                    vertex_3_from_1=top_left,
                    vertex_4_from_2=bottom_left,
                    x=x,
                    y=y,
                )
                extended_width += (
                    (bottom_right[0] - original_bottom_right[0]) ** 2
                    + (bottom_right[1] - original_bottom_right[1]) ** 2
                ) ** 0.5
        # extend to the bottom
        points = []
        if bbox_position == ALL_POSITIONS:
            points.append(
                det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_RIGHT)[0]
            )
            points.append(
                det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_LEFT)[0]
            )
        else:
            points.append(det.get_anchors_coordinates(anchor=bbox_position)[0])
        for x, y in points:
            if (
                cv.pointPolygonTest(
                    np.array([bottom_left, top_left, top_right, bottom_right]),
                    (x, y),
                    False,
                )
                >= 0
            ):
                continue
            if not ccw(
                x1=bottom_right[0],
                y1=bottom_right[1],
                x2=bottom_left[0],
                y2=bottom_left[1],
                x3=x,
                y3=y,
            ):
                original_bottom_right = bottom_right
                original_bottom_left = bottom_left
                bottom_right, bottom_left = calculate_vertices_to_contain_point(
                    vertex_1=original_bottom_right,
                    vertex_2=original_bottom_left,
                    vertex_3_from_1=top_right,
                    vertex_4_from_2=top_left,
                    x=x,
                    y=y,
                )
                extended_height += (
                    (bottom_left[0] - original_bottom_left[0]) ** 2
                    + (bottom_left[1] - original_bottom_left[1]) ** 2
                ) ** 0.5
        # extend to the top
        points = []
        if bbox_position == ALL_POSITIONS:
            points.append(det.get_anchors_coordinates(anchor=sv.Position.TOP_RIGHT)[0])
            points.append(det.get_anchors_coordinates(anchor=sv.Position.TOP_LEFT)[0])
        else:
            points.append(det.get_anchors_coordinates(anchor=bbox_position)[0])
        for x, y in points:
            if (
                cv.pointPolygonTest(
                    np.array([bottom_left, top_left, top_right, bottom_right]),
                    (x, y),
                    False,
                )
                >= 0
            ):
                continue
            if not ccw(
                x1=top_left[0],
                y1=top_left[1],
                x2=top_right[0],
                y2=top_right[1],
                x3=x,
                y3=y,
            ):
                original_top_left = top_left
                original_top_right = top_right
                top_left, top_right = calculate_vertices_to_contain_point(
                    vertex_1=original_top_left,
                    vertex_2=original_top_right,
                    vertex_3_from_1=bottom_left,
                    vertex_4_from_2=bottom_right,
                    x=x,
                    y=y,
                )
                extended_height += (
                    (top_left[0] - original_top_left[0]) ** 2
                    + (top_left[1] - original_top_left[1]) ** 2
                ) ** 0.5
    return (
        np.array(
            [
                bottom_left,
                top_left,
                top_right,
                bottom_right,
            ]
        ),
        original_width,
        original_height,
        extended_width,
        extended_height,
    )


def generate_transformation_matrix(
    src_polygon: np.ndarray,
    transformed_rect_width: int,
    transformed_rect_height: int,
    detections: Optional[sv.Detections] = None,
    detections_anchor: Optional[Union[sv.Position, Literal[ALL_POSITIONS]]] = None,
) -> Tuple[np.ndarray, float, float]:
    polygon_with_vertices_clockwise = sort_polygon_vertices_clockwise(
        polygon=src_polygon
    )
    src_polygon = roll_polygon_vertices_to_start_from_leftmost_bottom(
        polygon=polygon_with_vertices_clockwise
    )
    original_width = transformed_rect_width
    original_height = transformed_rect_height
    extended_width = 0
    extended_height = 0
    if detections and detections_anchor:
        (
            src_polygon,
            original_width,
            original_height,
            extended_width,
            extended_height,
        ) = extend_perspective_polygon(
            polygon=src_polygon,
            detections=detections,
            bbox_position=(
                sv.Position(detections_anchor)
                if detections_anchor != ALL_POSITIONS
                else detections_anchor
            ),
        )
    extended_width = extended_width * transformed_rect_width / max(original_width, 1)
    extended_height = (
        extended_height * transformed_rect_height / max(original_height, 1)
    )
    src_polygon = src_polygon.astype(np.float32)
    dst_polygon = np.array(
        [
            [0, transformed_rect_height + int(round(extended_height)) - 1],
            [0, 0],
            [transformed_rect_width + int(round(extended_width)) - 1, 0],
            [
                transformed_rect_width + int(round(extended_width)) - 1,
                transformed_rect_height + int(round(extended_height)) - 1,
            ],
        ]
    ).astype(dtype=np.float32)
    # https://docs.opencv.org/4.9.0/da/d54/group__imgproc__transform.html#ga20f62aa3235d869c9956436c870893ae
    return (
        cv.getPerspectiveTransform(
            src=src_polygon,
            dst=dst_polygon,
        ),
        extended_width,
        extended_height,
    )


def correct_detections(
    detections: sv.Detections,
    perspective_transformer: np.array,
    transformed_rect_width: float,
    transformed_rect_height: float,
) -> sv.Detections:
    corrected_detections: List[sv.Detections] = []
    for i in range(len(detections)):
        # copy
        detection = detections[i]
        mask = np.array(detection.mask)
        if (
            not np.array_equal(mask, np.array(None))
            and len(mask) > 0
            and isinstance(mask[0], np.ndarray)
        ):
            polygon = np.array(sv.mask_to_polygons(mask[0]), dtype=np.float32)
            # https://docs.opencv.org/4.9.0/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7
            corrected_polygon: np.ndarray = cv.perspectiveTransform(
                src=polygon, m=perspective_transformer
            ).reshape(-1, 2)
            detection.mask = np.array(
                [
                    sv.polygon_to_mask(
                        polygon=np.around(corrected_polygon).astype(np.int32),
                        resolution_wh=(
                            int(round(transformed_rect_width)),
                            int(round(transformed_rect_height)),
                        ),
                    ).astype(bool)
                ]
            )
            detection.xyxy = np.array(
                [
                    np.around(sv.polygon_to_xyxy(polygon=corrected_polygon)).astype(
                        np.int32
                    )
                ]
            )
        else:
            xmin, ymin, xmax, ymax = np.around(detection.xyxy[0]).tolist()
            polygon = np.array(
                [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]],
                dtype=np.float32,
            )
            # https://docs.opencv.org/4.9.0/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7
            corrected_polygon: np.ndarray = cv.perspectiveTransform(
                src=polygon, m=perspective_transformer
            ).reshape(-1, 2)
            detection.xyxy = np.array(
                [
                    np.around(sv.polygon_to_xyxy(polygon=corrected_polygon)).astype(
                        np.int32
                    )
                ]
            )
        if KEYPOINTS_XY_KEY_IN_SV_DETECTIONS in detection.data:
            corrected_key_points = cv.perspectiveTransform(
                src=np.array(
                    [detection.data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS][0]],
                    dtype=np.float32,
                ),
                m=perspective_transformer,
            ).reshape(-1, 2)
            detection[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = np.array(
                [np.around(corrected_key_points).astype(np.int32)], dtype="object"
            )
        corrected_detections.append(detection)
    return sv.Detections.merge(corrected_detections)


class PerspectiveCorrectionBlockV1(WorkflowBlock):
    def __init__(self):
        self.perspective_transformers: List[Tuple[np.ndarray, float, float]] = []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return PerspectiveCorrectionManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        predictions: Optional[Batch[sv.Detections]],
        perspective_polygons: Union[
            List[np.ndarray],
            List[List[np.ndarray]],
            List[List[List[int]]],
            List[List[List[List[int]]]],
        ],
        transformed_rect_width: List[int],
        transformed_rect_height: List[int],
        extend_perspective_polygon_by_detections_anchor: Union[
            sv.Position, Literal[ALL_POSITIONS]
        ],
        warp_image: Optional[bool],
    ) -> BlockResult:
        if not predictions and not images:
            raise ValueError(
                "Either predictions or images are required to apply perspective correction."
            )
        if warp_image and not images:
            raise ValueError(
                "images are required to warp image into requested perspective."
            )
        if not predictions:
            predictions = [None] * len(images)

        if (
            not self.perspective_transformers
            or extend_perspective_polygon_by_detections_anchor
        ):
            self.perspective_transformers = []
            largest_perspective_polygons = pick_largest_perspective_polygons(
                perspective_polygons
            )

            batch_size = len(predictions) if predictions else len(images)
            if len(largest_perspective_polygons) == 1 and batch_size > 1:
                largest_perspective_polygons = largest_perspective_polygons * batch_size

            if len(largest_perspective_polygons) != batch_size:
                raise ValueError(
                    f"Predictions batch size ({batch_size}) does not match number of perspective polygons ({largest_perspective_polygons})"
                )
            for polygon, detections, width, height in zip(largest_perspective_polygons, predictions, transformed_rect_width, transformed_rect_height):
                if polygon is None:
                    self.perspective_transformers.append(None)
                    continue
                self.perspective_transformers.append(
                    generate_transformation_matrix(
                        src_polygon=polygon,
                        detections=detections,
                        transformed_rect_width=width,
                        transformed_rect_height=height,
                        detections_anchor=extend_perspective_polygon_by_detections_anchor,
                    )
                )

        result = []
        for detections, perspective_transformer_w_h, image, width, height in zip(
            predictions, self.perspective_transformers, images, transformed_rect_width, transformed_rect_height
        ):
            perspective_transformer, extended_width, extended_height = (
                perspective_transformer_w_h
            )
            result_image = image
            if warp_image:
                # https://docs.opencv.org/4.9.0/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
                warped_image = cv.warpPerspective(
                    src=image.numpy_image,
                    M=perspective_transformer,
                    dsize=(
                        width + int(round(extended_width)),
                        height + int(round(extended_height)),
                    ),
                )
                result_image = WorkflowImageData.copy_and_replace(
                    origin_image_data=image,
                    numpy_image=warped_image,
                )

            if detections is None:
                result.append(
                    {
                        OUTPUT_DETECTIONS_KEY: None,
                        OUTPUT_IMAGE_KEY: result_image,
                        OUTPUT_EXTENDED_TRANSFORMED_RECT_WIDTH_KEY: width
                        + int(round(extended_width)),
                        OUTPUT_EXTENDED_TRANSFORMED_RECT_HEIGHT_KEY: height
                        + int(round(extended_height)),
                    }
                )
                continue

            corrected_detections = correct_detections(
                detections=detections,
                perspective_transformer=perspective_transformer,
                transformed_rect_width=width
                + int(round(extended_width)),
                transformed_rect_height=height
                + int(round(extended_height)),
            )

            result.append(
                {
                    OUTPUT_DETECTIONS_KEY: corrected_detections,
                    OUTPUT_IMAGE_KEY: result_image,
                    OUTPUT_EXTENDED_TRANSFORMED_RECT_WIDTH_KEY: width
                    + int(round(extended_width)),
                    OUTPUT_EXTENDED_TRANSFORMED_RECT_HEIGHT_KEY: height
                    + int(round(extended_height)),
                }
            )
        return result
