"""Unit tests of ``v2/perspective_correction@v1``, called directly.

Exact cases use a polygon that maps to the rectangle by a pure shift, so the
expected boxes, masks and pixels are worked out by hand. General
homographies are compared with V1's own functions and ``cv2.warpPerspective``
(V1's call): boxes and rectangle sizes are exact; pixels may differ by 1 level
(float bilinear blending). That pixel bound is CPU evidence only.
Keypoint covariance is checked against an exact affine case and against
central-difference Jacobians of a skewed homography.
"""

from dataclasses import replace
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.core_steps.transformations.perspective_correction.v1 import (
    correct_detections as v1_correct_detections,
)
from roboflow_workflows.core_steps.transformations.perspective_correction.v1 import (
    generate_transformation_matrix as v1_transformation_matrix,
)
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.geometry import perspective
from roboflow_workflows.v2.blocks.geometry.perspective import (
    PerspectiveCorrectionBlock,
    largest_polygon,
    top_down_plane,
)
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.rle import decode_rle_masks
from roboflow_workflows.v2.data.row_identity import RowRef

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

VIDEO = VideoMetadata(
    video_identifier="cam-1",
    frame_number=7,
    frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
)
# Maps to the 400 x 300 rectangle by the shift (-100, -100).
SHIFT_POLYGON = [[100, 399], [100, 100], [499, 100], [499, 399]]
SKEWED_POLYGON = [[60, 380], [40, 30], [480, 50], [450, 360]]


def scene(channels: int = 3, height: int = 420, width: int = 560) -> ImageData:
    generator = np.random.default_rng(channels)
    pixels = generator.integers(0, 256, (channels, height, width), dtype=np.uint8)
    image = ImageData.from_tensor(
        torch.from_numpy(pixels), image_id="camera", video_metadata=VIDEO
    )

    return image


def boxes(image: ImageData, xyxy, *, mask=None, rows=None) -> Detections:
    count = len(xyxy)
    fields = dict(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.zeros(count, dtype=torch.int64),
        confidence=torch.full((count,), 0.9),
        image_metadata={**image.prediction_metadata(), "class_names": {0: "car"}},
        bboxes_metadata=rows,
    )
    if mask is not None:
        return InstanceDetections(mask=mask, **fields)

    return Detections(**fields)


def correct(image, predictions=None, **options):
    settings = dict(
        perspective_polygons=SHIFT_POLYGON,
        transformed_rect_width=400,
        transformed_rect_height=300,
        extend_perspective_polygon_by_detections_anchor="",
        warp_image=True,
    )
    settings.update(options)
    result = PerspectiveCorrectionBlock().run(
        images=image, predictions=predictions, **settings
    )

    return result


def test_pure_shift_moves_boxes_and_crops_pixels_exactly() -> None:
    image = scene()

    result = correct(image, boxes(image, [[150, 150, 200, 250]]))

    assert result["corrected_coordinates"].xyxy.tolist() == [[50, 50, 100, 150]]
    assert torch.equal(
        result["warped_image"].tensor_image, image.tensor_image[:, 100:400, 100:500]
    )
    assert result["extended_transformed_rect_width"] == 400
    assert result["extended_transformed_rect_height"] == 300


def test_warped_image_is_a_new_root_without_a_mapping_to_the_input() -> None:
    image = scene().with_parent_row(RowRef("plates", 2))

    result = correct(image, boxes(image, [[150, 150, 200, 250]]))

    warped = result["warped_image"]
    assert warped.image_id != image.image_id
    assert warped.size_hw == (300, 400)
    assert warped.root == FrameMapping.identity(warped.image_id, (300, 400))
    assert warped.parent == warped.root
    assert warped.video_metadata == VIDEO
    assert warped.parent_row == RowRef("plates", 2)
    frame = frame_of(result["corrected_coordinates"])
    assert frame.image_id == warped.image_id
    assert frame.root.frame_id == warped.image_id
    assert frame.video_metadata == VIDEO
    assert result["corrected_coordinates"].image_metadata["class_names"] == {0: "car"}


def test_without_warp_the_input_is_returned_and_predictions_leave_its_frame() -> None:
    image = scene()

    result = correct(image, boxes(image, [[150, 150, 200, 250]]), warp_image=False)

    assert result["warped_image"] is image
    frame = frame_of(result["corrected_coordinates"])
    assert frame.image_id not in (None, image.image_id)
    assert frame.root.frame_id == frame.image_id
    assert frame.size_hw == (300, 400)


def test_general_homography_matches_v1_boxes_and_opencv_pixels() -> None:
    image = scene()
    xyxy = [[100, 100, 150, 200], [300, 60, 420, 300], [0, 0, 30, 30]]
    v1_detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        class_id=np.zeros(3, dtype=int),
        confidence=np.full(3, 0.9),
    )
    v1_matrix, _, _ = v1_transformation_matrix(np.array(SKEWED_POLYGON), 200, 150)
    v1_boxes = v1_correct_detections(v1_detections, v1_matrix, 200, 150).xyxy
    bgr = np.ascontiguousarray(image.tensor_image.permute(1, 2, 0).numpy()[..., ::-1])
    v1_pixels = cv2.warpPerspective(bgr, v1_matrix, (200, 150))[..., ::-1]

    result = correct(
        image,
        boxes(image, xyxy),
        perspective_polygons=SKEWED_POLYGON,
        transformed_rect_width=200,
        transformed_rect_height=150,
    )

    assert result["corrected_coordinates"].xyxy.numpy().tolist() == v1_boxes.tolist()
    warped = result["warped_image"].tensor_image.permute(1, 2, 0).numpy()
    difference = np.abs(warped.astype(int) - v1_pixels.astype(int))
    assert difference.max() <= 1
    assert (difference > 0).mean() < 0.02


@pytest.mark.parametrize("anchor", ["BOTTOM_CENTER", "TOP_LEFT", "CENTER", "ALL"])
def test_polygon_growth_matches_v1(anchor: str) -> None:
    image = scene()
    xyxy = [[20, 300, 60, 410], [470, 10, 540, 90], [200, 200, 220, 220]]
    v1_detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        class_id=np.zeros(3, dtype=int),
        confidence=np.full(3, 0.9),
    )
    v1_matrix, extra_width, extra_height = v1_transformation_matrix(
        np.array(SKEWED_POLYGON), 200, 150, v1_detections, anchor
    )

    result = correct(
        image,
        boxes(image, xyxy),
        perspective_polygons=SKEWED_POLYGON,
        transformed_rect_width=200,
        transformed_rect_height=150,
        extend_perspective_polygon_by_detections_anchor=anchor,
        warp_image=False,
    )

    assert result["extended_transformed_rect_width"] == 200 + round(extra_width)
    assert result["extended_transformed_rect_height"] == 150 + round(extra_height)
    assert extra_width + extra_height > 0
    v1_boxes = v1_correct_detections(v1_detections, v1_matrix, 0, 0).xyxy
    assert result["corrected_coordinates"].xyxy.numpy().tolist() == v1_boxes.tolist()


@pytest.mark.parametrize("encoding", ["dense", "rle"])
def test_masks_follow_the_shift_and_set_their_boxes(encoding: str) -> None:
    image = scene()
    dense = torch.zeros((2, 420, 560), dtype=torch.bool)
    dense[0, 150:160, 120:140] = True
    dense[0, 200:205, 300:310] = True
    dense[1, 0:20, 0:20] = True
    mask = dense
    if encoding == "rle":
        mask = InstancesRLEMasks.from_coco_rle_masks(
            image_size=(420, 560), masks=[torch_mask_to_coco_rle(one) for one in dense]
        )
    detections = boxes(image, [[100, 140, 320, 210], [0, 0, 19, 19]], mask=mask)

    corrected = correct(image, detections)["corrected_coordinates"]

    if encoding == "rle":
        assert isinstance(corrected.mask, InstancesRLEMasks)
        warped = torch.from_numpy(decode_rle_masks(corrected.mask))
    else:
        warped = corrected.mask
    assert torch.equal(warped, dense[:, 100:400, 100:500])
    # Row 0 takes its two-part mask's box; row 1's mask leaves the rectangle,
    # so it keeps its shifted box corners.
    assert corrected.xyxy.tolist() == [[20, 50, 209, 104], [-100, -100, -81, -81]]


def test_key_points_and_their_boxes_move_together() -> None:
    image = scene()
    key_points = KeyPoints(
        xy=torch.tensor([[[110.0, 120.0], [150.5, 200.25]]]),
        class_id=torch.zeros(1, dtype=torch.int64),
        confidence=torch.ones((1, 2)),
        image_metadata=image.prediction_metadata(),
    )

    corrected_points, corrected_boxes = correct(
        image, (key_points, boxes(image, [[110, 120, 151, 201]]))
    )["corrected_coordinates"]

    assert torch.allclose(
        corrected_points.xy, torch.tensor([[[10.0, 20.0], [50.5, 100.25]]])
    )
    assert corrected_boxes.xyxy.tolist() == [[10, 20, 51, 101]]
    assert frame_of(corrected_points) == frame_of(corrected_boxes)


def key_points_with_covariance(image, xy, covariance, **fields) -> KeyPoints:
    xy = torch.as_tensor(xy, dtype=torch.float32)
    instances, count = xy.shape[:2]
    key_points = KeyPoints(
        xy=xy,
        class_id=torch.arange(instances, dtype=torch.int64),
        confidence=torch.linspace(0.0, 1.0, instances * count).reshape(
            instances, count
        ),
        image_metadata=image.prediction_metadata(),
        covariance=torch.as_tensor(covariance, dtype=torch.float32),
        **fields,
    )

    return key_points


def projective_oracle(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Jacobians of the homography by central differences, ``(n, 2, 2)``."""

    def apply(xy: np.ndarray) -> np.ndarray:
        homogeneous = np.c_[xy, np.ones(len(xy))] @ matrix.T
        return homogeneous[:, :2] / homogeneous[:, 2:]

    step = 1e-4
    columns = [
        (apply(points + offset) - apply(points - offset)) / (2 * step)
        for offset in (np.array([step, 0.0]), np.array([0.0, step]))
    ]

    return np.stack(columns, axis=-1)


def test_key_point_covariance_scales_with_an_affine_rectangle() -> None:
    # 30 x 20 polygon onto 59 x 58 pixels: the homography is diag(2, 3, 1).
    image = scene(height=20, width=30)
    covariance = [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.5], [0.5, 2.0]]]]
    key_points = key_points_with_covariance(
        image,
        [[[4.0, 5.0], [7.0, 9.0]]],
        covariance,
        key_points_metadata=[{"pose": "a"}],
        detection_confidence=torch.tensor([0.8]),
    )
    before = {name: getattr(key_points, name).clone() for name in ("xy", "covariance")}

    result = correct(
        image,
        (key_points, None),
        perspective_polygons=[[0, 0], [29, 0], [29, 19], [0, 19]],
        transformed_rect_width=59,
        transformed_rect_height=58,
    )
    corrected, _ = result["corrected_coordinates"]

    assert corrected.xy.tolist() == [[[8.0, 15.0], [14.0, 27.0]]]
    assert corrected.covariance.tolist() == [
        [[[4.0, 0.0], [0.0, 9.0]], [[4.0, 3.0], [3.0, 18.0]]]
    ]
    assert corrected.covariance.dtype == torch.float32
    assert torch.equal(corrected.confidence, key_points.confidence)
    assert torch.equal(corrected.class_id, key_points.class_id)
    assert corrected.key_points_metadata == [{"pose": "a"}]
    assert corrected.detection_confidence.tolist() == pytest.approx([0.8])
    assert frame_of(corrected).image_id == result["warped_image"].image_id
    assert frame_of(corrected).size_hw == (58, 59)
    assert torch.equal(key_points.xy, before["xy"])
    assert torch.equal(key_points.covariance, before["covariance"])


def test_key_point_covariance_follows_a_projective_jacobian() -> None:
    image = scene()
    generator = np.random.default_rng(3)
    xy = generator.uniform((50, 40), (470, 370), size=(3, 4, 2))
    factors = generator.normal(size=(3, 4, 2, 2))
    covariance = factors @ factors.transpose(0, 1, 3, 2) + np.eye(2)
    key_points = key_points_with_covariance(image, xy, covariance)
    plane = top_down_plane(np.array(SKEWED_POLYGON), width=300, height=200)

    corrected, _ = correct(
        image,
        (key_points, None),
        perspective_polygons=SKEWED_POLYGON,
        transformed_rect_width=300,
        transformed_rect_height=200,
    )["corrected_coordinates"]

    points = key_points.xy.double().numpy().reshape(-1, 2)
    jacobian = projective_oracle(points, plane.matrix)
    sigma = key_points.covariance.double().numpy().reshape(-1, 2, 2)
    expected = jacobian @ sigma @ jacobian.transpose(0, 2, 1)
    assert not np.allclose(jacobian[0], jacobian[-1], rtol=1e-2)
    assert np.allclose(
        corrected.covariance.numpy().reshape(-1, 2, 2), expected, rtol=1e-4
    )


def test_key_point_covariance_of_empty_and_strided_tensors() -> None:
    image = scene()
    plane_options = dict(perspective_polygons=SKEWED_POLYGON)
    for shape in ((0, 17), (2, 0)):
        empty = key_points_with_covariance(
            image, torch.zeros((*shape, 2)), torch.zeros((*shape, 2, 2))
        )
        corrected, _ = correct(image, (empty, None), **plane_options)[
            "corrected_coordinates"
        ]
        assert corrected.covariance.shape == (*shape, 2, 2)
    covariance = torch.tensor([[[[2.0, 0.3], [0.3, 1.0]], [[1.0, 0.0], [0.0, 4.0]]]])
    xy = torch.tensor([[[110.0, 120.0], [300.0, 250.0]]])
    contiguous = key_points_with_covariance(image, xy, covariance)
    strided = key_points_with_covariance(
        image,
        xy.transpose(1, 2).contiguous().transpose(1, 2),
        covariance.transpose(2, 3).contiguous().transpose(2, 3),
    )
    assert not strided.xy.is_contiguous()
    assert not strided.covariance.is_contiguous()

    from_contiguous, from_strided = (
        correct(image, (points, None), **plane_options)["corrected_coordinates"][0]
        for points in (contiguous, strided)
    )

    assert torch.equal(from_strided.covariance, from_contiguous.covariance)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs a non-CPU torch device"
)
def test_key_point_covariance_stays_on_its_device() -> None:
    image = scene()
    xy = torch.tensor([[[110.0, 120.0], [300.0, 250.0]]])
    covariance = torch.eye(2).expand(1, 2, 2, 2)
    on_cpu = key_points_with_covariance(image, xy, covariance)
    on_device = replace(
        on_cpu, xy=on_cpu.xy.to("mps"), covariance=on_cpu.covariance.to("mps")
    )

    expected, moved = (
        correct(
            image, (points, None), perspective_polygons=SKEWED_POLYGON, warp_image=False
        )["corrected_coordinates"][0]
        for points in (on_cpu, on_device)
    )

    assert moved.covariance.device.type == "mps"
    assert moved.covariance.dtype == torch.float32
    assert torch.allclose(moved.covariance.cpu(), expected.covariance, rtol=1e-4)


def test_key_point_covariance_not_shaped_like_the_points_is_refused() -> None:
    image = scene()
    key_points = key_points_with_covariance(
        image, [[[110.0, 120.0], [150.0, 200.0]]], torch.eye(2).expand(1, 1, 2, 2)
    )

    with pytest.raises(ContractError, match="covariance must have shape"):
        correct(image, (key_points, None))


def test_row_point_metadata_moves_and_inputs_stay_unchanged() -> None:
    image = scene()
    rows = [
        {
            "detection_id": "a",
            "keypoints_xy": [[110, 120], [130, 140]],
            "xyxyxyxy": np.array([[110, 120], [130, 120], [130, 140], [110, 140]]),
        }
    ]
    detections = boxes(image, [[110, 120, 130, 140]], rows=rows)
    pixels_before = image.tensor_image.clone()

    corrected = correct(image, detections)["corrected_coordinates"]

    (row,) = corrected.bboxes_metadata
    assert row["detection_id"] == "a"
    assert row["keypoints_xy"] == [[10, 20], [30, 40]]
    assert row["xyxyxyxy"].tolist() == [[10, 20], [30, 20], [30, 40], [10, 40]]
    assert detections.xyxy.tolist() == [[110, 120, 130, 140]]
    assert rows[0]["keypoints_xy"] == [[110, 120], [130, 140]]
    assert torch.equal(image.tensor_image, pixels_before)


def test_empty_detections_describe_the_rectangle() -> None:
    image = scene()

    corrected = correct(image, boxes(image, []), warp_image=False)[
        "corrected_coordinates"
    ]

    assert corrected.xyxy.shape == (0, 4)
    assert frame_of(corrected).size_hw == (300, 400)


def test_gray_and_non_contiguous_images_warp_like_contiguous_ones() -> None:
    gray = scene(channels=1)
    transposed = ImageData.from_tensor(
        scene().tensor_image.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    )
    assert not transposed.tensor_image.is_contiguous()

    warped_gray = correct(gray)["warped_image"].tensor_image
    warped_strided = correct(transposed)["warped_image"].tensor_image

    assert torch.equal(warped_gray, gray.tensor_image[:, 100:400, 100:500])
    assert torch.equal(
        warped_strided, transposed.tensor_image[:, 100:400, 100:500].contiguous()
    )


def test_missing_image_gives_missing_outputs_and_no_predictions_give_none() -> None:
    assert correct(None) == dict.fromkeys(PerspectiveCorrectionBlock.outputs)
    result = correct(scene())
    assert result["corrected_coordinates"] is None
    assert result["warped_image"].size_hw == (300, 400)


def test_the_largest_four_vertex_polygon_is_used() -> None:
    small = [[0, 0], [10, 0], [10, 10], [0, 10]]
    triangle = [[0, 0], [500, 0], [0, 500]]
    large = [[0.4, 0], [40, 0], [40, 40], [0, 40]]

    chosen = largest_polygon([small, triangle, large])

    assert chosen.tolist() == [[0, 0], [40, 0], [40, 40], [0, 40]]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"perspective_polygons": [[0, 0], [10, 10], [20, 20], [0, 30]]},
            "degenerate",
        ),
        ({"perspective_polygons": [[[0, 0], [5, 5], [9, 0]]]}, "exactly four"),
        ({"perspective_polygons": []}, "non-empty"),
        ({"extend_perspective_polygon_by_detections_anchor": "MIDDLE"}, "MIDDLE"),
        ({"transformed_rect_width": 1, "transformed_rect_height": 20}, "at least 2"),
        ({"transformed_rect_height": 1}, "transformed_rect_height=1"),
    ],
)
def test_invalid_polygons_and_anchors_are_refused(options, message) -> None:
    with pytest.raises(ValueError, match=message):
        correct(scene(), **options)


def test_predictions_of_another_image_are_refused() -> None:
    other = ImageData.from_tensor(torch.zeros((3, 10, 10), dtype=torch.uint8))

    with pytest.raises(ContractError, match="same image"):
        correct(scene(), boxes(other, [[1, 1, 2, 2]]))


def test_center_of_mass_needs_masks_and_matches_supervision() -> None:
    image = scene()
    with pytest.raises(ContractError, match="CENTER_OF_MASS"):
        correct(
            image,
            boxes(image, [[1, 1, 5, 5]]),
            extend_perspective_polygon_by_detections_anchor="CENTER_OF_MASS",
        )
    dense = torch.zeros((2, 420, 560), dtype=torch.bool)
    dense[0, 10:13, 20:24] = True
    dense[0, 30, 40] = True
    dense[1, 100:200, 300:301] = True
    detections = boxes(image, [[20, 10, 40, 30], [300, 100, 300, 199]], mask=dense)

    centroids = perspective._mask_centroids(detections)

    expected = sv.calculate_masks_centroids(dense.numpy())
    assert centroids.tolist() == expected.tolist()


def test_plane_matrix_maps_polygon_vertices_to_rectangle_corners() -> None:
    plane = top_down_plane(np.array(SKEWED_POLYGON), width=200, height=150)

    mapped = cv2.perspectiveTransform(
        np.array([SKEWED_POLYGON], dtype=np.float32), plane.matrix
    )[0]

    assert np.allclose(mapped, [[0, 149], [0, 0], [199, 0], [199, 149]], atol=1e-3)
