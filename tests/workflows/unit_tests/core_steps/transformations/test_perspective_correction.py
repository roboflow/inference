import cv2 as cv
import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.transformations.perspective_correction.v1 import (
    PerspectiveCorrectionBlockV1,
    correct_detections,
    extend_perspective_polygon,
    generate_transformation_matrix,
    pick_largest_perspective_polygons,
    roll_polygon_vertices_to_start_from_leftmost_bottom,
    sort_polygon_vertices_clockwise,
)
from inference.core.workflows.execution_engine.constants import (
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("broken_input", [1, "cat", np.array([])])
def test_pick_largest_perspective_polygons_raises_on_unexpected_type_of_input(
    broken_input,
):
    with pytest.raises(ValueError, match="Unexpected type of input"):
        pick_largest_perspective_polygons(perspective_polygons_batch=broken_input)


@pytest.mark.parametrize("broken_input", [[1], ["cat"]])
def test_pick_largest_perspective_polygons_raises_on_unexpected_type_of_batch_element(
    broken_input,
):
    with pytest.raises(ValueError, match="Unexpected type of batch element"):
        pick_largest_perspective_polygons(perspective_polygons_batch=broken_input)


@pytest.mark.parametrize("empty_batch", [[], Batch(content=[], indices=[])])
def test_pick_largest_perspective_polygons_raises_on_empty_batch(empty_batch):
    with pytest.raises(ValueError, match="Unexpected empty batch"):
        pick_largest_perspective_polygons(perspective_polygons_batch=empty_batch)


PERSPECTIVE_POLYGON_LIST = [[1, 1], [10, 1], [10, 10], [1, 10]]
PERSPECTIVE_POLYGON_NP_ARRAY = np.array(PERSPECTIVE_POLYGON_LIST)


@pytest.mark.parametrize(
    "empty_batch_elements",
    [
        [[]],
        [np.array([])],
        [[PERSPECTIVE_POLYGON_NP_ARRAY], []],
        [[PERSPECTIVE_POLYGON_LIST], []],
    ],
)
def test_pick_largest_perspective_polygons_raises_on_empty_batch_element(
    empty_batch_elements,
):
    with pytest.raises(ValueError, match="Unexpected empty batch element"):
        pick_largest_perspective_polygons(
            perspective_polygons_batch=empty_batch_elements
        )


def test_pick_largest_perspective_polygons():
    # given
    small_perspective_pollygon = [[1, 1], [10, 1], [10, 10], [1, 10]]
    large_perspective_pollygon = [[1, 1], [100, 1], [100, 100], [1, 100]]
    batch = [
        [small_perspective_pollygon, large_perspective_pollygon],
        [small_perspective_pollygon, large_perspective_pollygon],
    ]

    # when
    largest_polygons = pick_largest_perspective_polygons(
        perspective_polygons_batch=batch
    )

    # then
    assert (
        len(largest_polygons) == 2
    ), "Output batch size must be the same as input batch size"
    assert np.array_equal(
        largest_polygons[0], large_perspective_pollygon
    ), "Largest polygon must be picked"
    assert np.array_equal(
        largest_polygons[1], large_perspective_pollygon
    ), "Largest polygon must be picked"


def test_sort_polygon_vertices_clockwise():
    # when
    clockwise_polygon = sort_polygon_vertices_clockwise(
        polygon=np.array([[1, 1], [1, 10], [10, 10], [10, 1]])
    )

    # then
    assert np.array_equal(
        clockwise_polygon, np.array([[1, 10], [1, 1], [10, 1], [10, 10]])
    ), "Polygon vertices must be sorted clockwise"


def test_roll_polygon_vertices_to_start_from_leftmost_bottom():
    # when
    rolled_polygon = roll_polygon_vertices_to_start_from_leftmost_bottom(
        polygon=np.array([[1, 1], [10, 1], [10, 10], [1, 10]])
    )

    # then
    assert np.array_equal(
        rolled_polygon, np.array([[1, 10], [1, 1], [10, 1], [10, 10]])
    ), "Polygon must be reorganized so leftmost bottom "


def test_extend_rectangular_perspective_polygon_detections_within_polygon():
    # given
    polygon = np.array([[100, 110], [100, 100], [110, 100], [110, 110]])
    detections_within_polygon = sv.Detections(xyxy=np.array([[105, 105, 106, 106]]))
    detections_on_the_corners = sv.Detections(
        xyxy=np.array(
            [
                [99, 99, 101, 100],
                [109, 99, 111, 100],
                [99, 109, 101, 110],
                [109, 109, 111, 110],
            ]
        )
    )
    detections = sv.Detections.merge(
        [
            detections_within_polygon,
            detections_on_the_corners,
        ]
    )

    # when
    (
        extended_polygon,
        original_width,
        original_height,
        extended_width,
        extended_height,
    ) = extend_perspective_polygon(
        polygon=polygon,
        detections=detections,
        bbox_position=sv.Position.BOTTOM_CENTER,
    )

    # then
    assert np.array_equal(
        extended_polygon, np.array([[100, 110], [100, 100], [110, 100], [110, 110]])
    ), "Detections within polygon must not modify it"
    assert original_width == 10, "Detections within polygon must not modify it"
    assert original_height == 10, "Detections within polygon must not modify it"
    assert extended_width == 0, "Detections within polygon must not modify it"
    assert extended_height == 0, "Detections within polygon must not modify it"


def test_extend_perspective_polygon_detections_within_polygon():
    # given
    polygon = np.array([[100, 500], [50, 200], [500, 100], [400, 600]])
    detections_within_polygon = sv.Detections(xyxy=np.array([[200, 200, 201, 201]]))
    detections_on_the_corners = sv.Detections(
        xyxy=np.array(
            [
                [99, 500, 101, 500],
                [49, 200, 51, 200],
                [499, 100, 501, 100],
                [399, 600, 401, 600],
            ]
        )
    )
    detections = sv.Detections.merge(
        [
            detections_within_polygon,
            detections_on_the_corners,
        ]
    )

    # when
    extended_polygon, *_ = extend_perspective_polygon(
        polygon=polygon,
        detections=detections,
        bbox_position=sv.Position.BOTTOM_CENTER,
    )

    # then
    assert np.array_equal(
        extended_polygon, np.array([[100, 500], [50, 200], [500, 100], [400, 600]])
    ), "Detections within polygon must not modify it"


def test_extend_rectangular_perspective_polygon_detections_outside_polygon():
    # given
    polygon = np.array([[100, 110], [100, 100], [110, 100], [110, 110]])
    detections_extending_from_the_left = sv.Detections(
        xyxy=np.array([[90, 105, 92, 106]])
    )
    detections_extending_from_the_right = sv.Detections(
        xyxy=np.array([[120, 105, 122, 106]])
    )
    detections_extending_from_the_top = sv.Detections(
        xyxy=np.array([[105, 90, 106, 92]])
    )
    detections_extending_from_the_bottom = sv.Detections(
        xyxy=np.array([[105, 120, 106, 122]])
    )
    detections = sv.Detections.merge(
        [
            detections_extending_from_the_left,
            detections_extending_from_the_right,
            detections_extending_from_the_top,
            detections_extending_from_the_bottom,
        ]
    )

    # when
    (
        extended_polygon,
        original_width,
        original_height,
        extended_width,
        extended_height,
    ) = extend_perspective_polygon(
        polygon=polygon,
        detections=detections,
        bbox_position=sv.Position.BOTTOM_CENTER,
    )

    # then
    assert np.array_equal(
        extended_polygon,
        np.array([[91, 122], [91, 92], [121, 92], [121, 122]]),
    ), "Detections outside of polygon should extend it"
    assert original_width == 10, "Detections outside of polygon should extend it"
    assert original_height == 10, "Detections outside of polygon should extend it"
    assert extended_width == 20, "Detections outside of polygon should extend it"
    assert extended_height == 20, "Detections outside of polygon should extend it"


def test_extend_perspective_polygon_detections_outside_polygon_edges():
    # given
    polygon = np.array([[300, 700], [250, 400], [700, 300], [600, 800]])
    detections = sv.Detections(
        xyxy=np.array(
            [
                [199, 599, 201, 601],
                [399, 199, 401, 201],
                [799, 599, 801, 601],
                [399, 899, 401, 901],
            ]
        )
    )

    # when
    extended_polygon, *_ = extend_perspective_polygon(
        polygon=polygon,
        detections=detections,
        bbox_position=sv.Position.BOTTOM_CENTER,
    )

    # then
    assert np.array_equal(
        extended_polygon, np.array([[242, 848], [142, 258], [903, 90], [718, 1007]])
    ), "Detections within polygon must not modify it"


def test_generate_transformation_matrix():
    # given
    polygon = np.array([[100, 110], [100, 100], [110, 100], [110, 110]])

    # when
    transformation_matrix, extended_width, extended_height = (
        generate_transformation_matrix(
            src_polygon=polygon,
            detections=sv.Detections.empty(),
            transformed_rect_width=1000,
            transformed_rect_height=1000,
            detections_anchor=sv.Position.BOTTOM_CENTER,
        )
    )

    expected_transformation_matrix = np.array(
        [
            [99.9, -5.955e-15, -9990],
            [-1.6536e-14, 99.9, -9990],
            [-0, -0, 1],
        ]
    )
    assert np.allclose(
        transformation_matrix, expected_transformation_matrix
    ), "Transformation matrix must match"
    assert extended_width == 0, "Polygon must not be extended"
    assert extended_height == 0, "Polygon must not be extended"


def test_correct_detections_with_segmentation():
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20]]),
        mask=np.array(
            [
                sv.polygon_to_mask(
                    polygon=np.array([[10, 15], [15, 10], [20, 15], [15, 20]]),
                    resolution_wh=(200, 200),
                )
            ]
        ),
    )
    src_polygon = np.array([[5, 5], [25, 5], [25, 25], [5, 25]]).astype(
        dtype=np.float32
    )
    dst_polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]]).astype(
        dtype=np.float32
    )
    transformer = cv.getPerspectiveTransform(
        src=src_polygon,
        dst=dst_polygon,
    )

    # when
    corrected_detections = correct_detections(
        detections=detections,
        perspective_transformer=transformer,
        transformed_rect_width=100,
        transformed_rect_height=100,
    )

    # then
    expected_detections = sv.Detections(
        xyxy=np.array([[25, 25, 75, 75]]),
        mask=np.array(
            [
                sv.polygon_to_mask(
                    polygon=np.array([[50, 25], [25, 50], [50, 75], [75, 50]]),
                    resolution_wh=(100, 100),
                )
            ]
        ),
    )
    assert np.array_equal(
        corrected_detections.xyxy, expected_detections.xyxy
    ), "Corrected detections must match"
    assert np.array_equal(
        corrected_detections.mask, expected_detections.mask
    ), "Corrected detections must match"


def test_correct_detections_with_keypoints():
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20]]),
    )
    detections[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = np.array(
        [np.array([[10, 10], [15, 15], [10, 20], [20, 10]])], dtype="object"
    )
    src_polygon = np.array([[5, 5], [25, 5], [25, 25], [5, 25]]).astype(
        dtype=np.float32
    )
    dst_polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]]).astype(
        dtype=np.float32
    )
    transformer = cv.getPerspectiveTransform(
        src=src_polygon,
        dst=dst_polygon,
    )
    # when
    corrected_detections = correct_detections(
        detections=detections,
        perspective_transformer=transformer,
        transformed_rect_width=100,
        transformed_rect_height=100,
    )

    # then
    expected_detections = sv.Detections(
        xyxy=np.array([[25, 25, 75, 75]]),
    )
    expected_detections[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = np.array(
        [np.array([[25, 25], [50, 50], [25, 75], [75, 25]], dtype=np.int32)],
        dtype="object",
    )
    assert corrected_detections == expected_detections


def test_warp_image():
    # given
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dummy_predictions = sv.Detections(xyxy=np.array([[10, 10, 20, 20]]))
    perspective_correction_block = PerspectiveCorrectionBlockV1()

    workflow_image_data = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"), numpy_image=dummy_image
    )

    # when
    result = perspective_correction_block.run(
        images=[workflow_image_data],
        predictions=[dummy_predictions],
        perspective_polygons=[[[1, 1], [99, 1], [99, 99], [1, 99]]],
        transformed_rect_width=200,
        transformed_rect_height=200,
        extend_perspective_polygon_by_detections_anchor=None,
        warp_image=True,
    )

    # then
    assert "warped_image" in result[0], "warped_image key must be present in the result"
    assert isinstance(
        result[0]["warped_image"], WorkflowImageData
    ), f"warped_image must be of type WorkflowImageData"


def test_warp_image_batch_dims():
    # given
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    dummy_predictions = sv.Detections(xyxy=np.array([[10, 10, 20, 20]]))
    perspective_correction_block = PerspectiveCorrectionBlockV1()

    workflow_image_data = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"), numpy_image=dummy_image
    )

    # when
    result = perspective_correction_block.run(
        images=[workflow_image_data],
        predictions=[dummy_predictions],
        perspective_polygons=[[[1, 1], [99, 1], [99, 99], [1, 99]]],
        transformed_rect_width=[200],
        transformed_rect_height=[200],
        extend_perspective_polygon_by_detections_anchor=None,
        warp_image=True,
    )

    # then
    assert "warped_image" in result[0], "warped_image key must be present in the result"
    assert isinstance(
        result[0]["warped_image"], WorkflowImageData
    ), f"warped_image must be of type WorkflowImageData"


def test_batch_input():
    # given
    batch_size = 3
    dummy_images = [
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    ] * batch_size
    dummy_predictions = [sv.Detections(xyxy=np.array([[10, 10, 20, 20]]))] * batch_size
    perspective_correction_block = PerspectiveCorrectionBlockV1()

    workflow_image_data = [
        WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="test"),
            numpy_image=dummy_image,
        )
        for dummy_image in dummy_images
    ]

    # when
    result = perspective_correction_block.run(
        images=workflow_image_data,
        predictions=dummy_predictions,
        perspective_polygons=[[[1, 1], [99, 1], [99, 99], [1, 99]]],
        transformed_rect_width=[200] * batch_size,
        transformed_rect_height=[200] * batch_size,
        extend_perspective_polygon_by_detections_anchor=None,
        warp_image=True,
    )

    # then
    assert "warped_image" in result[0], "warped_image key must be present in the result"
    assert isinstance(
        result[0]["warped_image"], WorkflowImageData
    ), f"warped_image must be of type WorkflowImageData"
