from contextlib import ExitStack as DoesNotRaise
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from inference.core.exceptions import PostProcessingError
from inference.core.utils.postprocess import (
    clip_boxes_coordinates,
    clip_keypoints_coordinates,
    cosine_similarity,
    crop_mask,
    get_static_crop_dimensions,
    post_process_bboxes,
    post_process_keypoints,
    post_process_polygons,
    scale_bboxes,
    scale_polygons,
    shift_bboxes,
    shift_keypoints,
    sigmoid,
    standardise_static_crop,
    stretch_bboxes,
    stretch_keypoints,
    undo_image_padding_for_predicted_boxes,
    undo_image_padding_for_predicted_keypoints,
    undo_image_padding_for_predicted_polygons,
)

Height = int
Width = int
XCoord = int
YCoord = int


def test_cosine_similarity_against_vectors() -> None:
    # given
    a = np.array([1, 0])
    b = np.array([0, 1])

    # when
    result = cosine_similarity(a=a, b=b)

    # then
    assert abs(result) < 1e-5


def test_cosine_similarity_orthogonal_vectors() -> None:
    # given
    a = np.array([1, 0])
    b = np.array([-2, 0])

    # when
    result = cosine_similarity(a=a, b=b)

    # then
    assert abs(result + 1) < 1e-5


def test_cosine_similarity_against_semi_aligned_vectors() -> None:
    # given
    a = np.array([5, 5])
    b = np.array([1, 0])

    # when
    result = cosine_similarity(a=a, b=b)

    # then
    assert abs(result - np.sqrt(2) / 2) < 1e-5


def test_cosine_similarity_against_bunch_of_vectors() -> None:
    # given
    a = np.array([[1, 0], [1, 0], [5, 5]])
    b = np.array([[0, 1], [-2, 0], [1, 0]])

    # when
    with pytest.raises(ValueError):
        # does not work against multiple vectors at once
        _ = cosine_similarity(a=a, b=b)


def test_crop_mask() -> None:
    # given
    masks = np.ones((2, 128, 128))
    boxes = np.array([[32, 32, 64, 64], [64, 64, 96, 96]])
    expected_result = np.zeros((2, 128, 128))
    expected_result[0, 32:64, 32:64] = 1
    expected_result[1, 64:96, 64:96] = 1

    # when
    result = crop_mask(masks=masks, boxes=boxes)

    # then
    assert result.shape == expected_result.shape
    assert np.allclose(result, expected_result)


def test_standardise_static_crop() -> None:
    # when
    result = standardise_static_crop(
        static_crop_config={"x_min": 15, "y_min": 20, "x_max": 50, "y_max": 75}
    )

    # then
    assert np.allclose(result, [0.15, 0.2, 0.5, 0.75])


def test_get_static_crop_dimensions_when_misconfiguration_happens() -> None:
    # when
    with pytest.raises(PostProcessingError):
        _ = get_static_crop_dimensions(
            orig_shape=(256, 256),
            preproc={"static-crop": {}},
        )


def test_get_static_crop_dimensions_when_no_crop_should_be_applied() -> None:
    # when
    result = get_static_crop_dimensions(
        orig_shape=(256, 256),
        preproc={},
    )

    # then
    assert result == ((0, 0), (256, 256))


def test_get_static_crop_dimensions_when_crop_should_be_applied() -> None:
    # when
    result = get_static_crop_dimensions(
        orig_shape=(200, 100),
        preproc={
            "static-crop": {
                "enabled": True,
                "x_min": 15,
                "y_min": 20,
                "x_max": 50,
                "y_max": 75,
            }
        },
    )

    # then
    assert result == ((15, 40), (110, 35))


def test_apply_crop_shift_to_boxes() -> None:
    # given
    predicted_bboxes = np.array(
        [
            [20, 20, 40, 40],
            [30, 60, 90, 120],
        ]
    )
    expected_result = np.array(
        [
            [30, 40, 50, 60],
            [40, 80, 100, 140],
        ]
    )

    # when
    result = shift_bboxes(
        bboxes=predicted_bboxes,
        shift_x=10,
        shift_y=20,
    )

    # then
    assert result.shape == expected_result.shape
    assert np.allclose(result, expected_result)


def test_clip_boxes_coordinates() -> None:
    # given
    predicted_bboxes = np.array(
        [
            [-0.1, 20, 40, 40.1, 0.9],
            [30, 60.1, 90, 120.1, 0.9],
        ]
    )
    expected_result = np.array(
        [
            [0, 20, 40, 40, 0.9],
            [30, 60, 90, 80, 0.9],
        ]
    )

    # when
    result = clip_boxes_coordinates(
        predicted_bboxes=predicted_bboxes,
        origin_shape=(80, 120),
    )
    assert result.shape == expected_result.shape
    assert np.allclose(result, expected_result)


def test_undo_image_padding_for_predicted_boxes() -> None:
    # given
    predicted_bboxes = np.array(
        [
            [20, 32, 40, 64],
            [30, 64, 90, 112],
        ],
        dtype=np.float32,
    )
    # scaling factor for inference is 0.5, padding of 16px along OY axis added
    # so out(OX) = in(OX) / 0.5, out(OY) = (in(OY) - 16) / 0.5
    expected_result = np.array(
        [
            [40, 32, 80, 96],
            [60, 96, 180, 192],
        ],
        dtype=np.float32,
    )

    # when
    result = undo_image_padding_for_predicted_boxes(
        predicted_bboxes=predicted_bboxes,
        infer_shape=(128, 128),
        origin_shape=(192, 256),
    )

    # then
    assert result.shape == expected_result.shape
    assert np.allclose(result, expected_result)


def test_stretch_bboxes_when_downscaling_occurred() -> None:
    # given
    predicted_bboxes = np.array(
        [
            [20, 32, 40, 64],
            [30, 64, 90, 128],
        ],
        dtype=np.float32,
    )
    # OY axis is scaled 4x for inference, OX is scaled 2x
    # so out(OX) = in(OX) / 2 and out(OY) = in(OY) / 4
    expected_result = np.array(
        [
            [10, 8, 20, 16],
            [15, 16, 45, 32],
        ],
        dtype=np.float32,
    )

    # when
    result = stretch_bboxes(
        predicted_bboxes=predicted_bboxes,
        infer_shape=(128, 128),
        origin_shape=(32, 64),
    )

    # then
    assert result.shape == expected_result.shape
    assert np.allclose(result, expected_result)


def test_stretch_bboxes_when_up_scaling_occurred() -> None:
    # given
    predicted_bboxes = np.array(
        [
            [10, 8, 20, 16],
            [15, 16, 45, 32],
        ],
        dtype=np.float32,
    )
    # OY axis is scaled 4x for inference, OX is scaled 2x
    # so out(OX) = in(OX) / 2 and out(OY) = in(OY) / 4
    expected_result = np.array(
        [
            [20, 32, 40, 64],
            [30, 64, 90, 128],
        ],
        dtype=np.float32,
    )

    # when
    result = stretch_bboxes(
        predicted_bboxes=predicted_bboxes,
        infer_shape=(32, 64),
        origin_shape=(128, 128),
    )

    # then
    assert result.shape == expected_result.shape
    assert np.allclose(result, expected_result)


def test_post_process_bboxes_when_crop_with_stretch_used() -> None:
    # given
    predicted_bboxes = [
        np.array(
            [
                [20, 32, 40, 64, 0.9],
                [30, 64, 88, 128, 0.85],
            ],
            dtype=np.float32,
        ).tolist()
    ]
    # there is a crop from (x=20, y=40) to (x=84, y=104) (box size - h=64, w=64)
    # compared to inference size (128, 128) and we stretch, such that OX scale is 2.0,
    # OY scale is 2.0, then we shift by crop (OX +20, OY +40)
    # for OX coord we take input ox / 2 + 20, for OY: input oy / 2 +  40
    expected_result = np.expand_dims(
        np.array(
            [
                [30, 56, 40, 72, 0.9],
                [35, 72, 64, 104, 0.85],
            ],
            dtype=np.float32,
        ),
        axis=0,
    )

    # when
    result = post_process_bboxes(
        predictions=predicted_bboxes,
        infer_shape=(128, 128),
        img_dims=[(200, 100)],
        preproc={
            "static-crop": {
                "enabled": True,
                "x_min": 20,
                "y_min": 20,
                "x_max": 84,
                "y_max": 52,
            }
        },
    )

    # then
    assert np.array(result).shape == (1, 2, 5)
    assert np.allclose(np.array(result), expected_result)


def test_post_process_bboxes_when_crop_with_padding_used() -> None:
    # given
    predicted_bboxes = [
        np.array(
            [
                [32, 32, 40, 64, 0.9],
                [32, 64, 88, 128, 0.85],
            ],
            dtype=np.float32,
        ).tolist()
    ]
    # there is a crop from (x=40, y=40) to (x=72, y=104) of size(h=64, w=32)
    # compared to inference size (128, 128) - crop will be given 32 padding on OX axis each side
    # we need to upscale 2x prediction coordinates, then crop shift +40, +40 in each axis
    # so out(OX) = (in(OX) - 32) / 2 + 40, out(OY) = in(OY) / 2 + 40
    expected_result = np.expand_dims(
        np.array(
            [
                [40, 56, 44, 72, 0.9],
                [40, 72, 68, 104, 0.85],
            ],
            dtype=np.float32,
        ),
        axis=0,
    )

    # when
    result = post_process_bboxes(
        predictions=predicted_bboxes,
        infer_shape=(128, 128),
        img_dims=[(200, 200)],
        preproc={
            "static-crop": {
                "enabled": True,
                "x_min": 20,
                "y_min": 20,
                "x_max": 36,
                "y_max": 52,
            }
        },
        resize_method="Fit (black edges) in",
    )

    # then
    print(np.array(result))
    assert np.array(result).shape == (1, 2, 5)
    assert np.allclose(np.array(result), expected_result)


@pytest.mark.parametrize(
    "x, expected_result",
    [
        (0.0, 0.5),
        (1.0, 0.73105),
        (-1.0, 0.26894),
        (np.array([0.0, 1.0, -1.0]), np.array([0.5, 0.73105, 0.26894])),
    ],
)
def test_sigmoid(x: float, expected_result: float) -> None:
    # when
    result = sigmoid(x)

    # then
    assert np.allclose(result, expected_result, rtol=1e-4)


@pytest.mark.parametrize(
    "bboxes, scale_x, scale_y, expected_result",
    [
        (
            np.array([[1.0, 2, 3, 4], [5, 6, 7, 8]]),
            2.0,
            4.0,
            np.array([[2.0, 8, 6, 16], [10, 24, 14, 32]]),
        ),
        (
            np.array([[1.0, 2, 3, 4], [5, 6, 7, 8]]),
            1.0,
            1.0,
            np.array([[1.0, 2, 3, 4], [5, 6, 7, 8]]),
        ),
        (
            np.array([[1.0, 2, 3, 4], [5, 6, 7, 8]]),
            0.5,
            0.5,
            np.array([[0.5, 1.0, 1.5, 2], [2.5, 3, 3.5, 4]]),
        ),
    ],
)
def test_scale_bboxes(
    bboxes: np.ndarray,
    scale_x: float,
    scale_y: float,
    expected_result: np.ndarray,
) -> None:
    # when
    result = scale_bboxes(bboxes=bboxes, scale_x=scale_x, scale_y=scale_y)

    # then
    assert np.allclose(result, expected_result)


def test_undo_image_padding_for_predicted_polygons() -> None:
    # given
    polygons = [[(32, 32), (64, 64), (48, 48)]]
    # inference size is smaller than origin and aspect ratio is mismatched
    # for inference, 32 OX padding each side was added, and we need to scale up results 4x
    # along each axis to revert back the original coordinates of input image
    # So, out(OX) = (in(OX) - 32) * 4, out(OY) = in(OY) * 4
    expected_result = np.array([[(0, 128), (128, 256), (64, 192)]])

    # when
    result = undo_image_padding_for_predicted_polygons(
        polygons=polygons,
        origin_shape=(256, 256),
        infer_shape=(64, 128),
    )

    print(np.array(result))
    # then
    assert np.allclose(np.array(result), expected_result)


def test_scale_polygons() -> None:
    # given
    polygons = [[(10, 20), (20, 30), (30, 40)], [(40, 50), (50, 60), (60, 70)]]
    expected_result = np.array(
        [[(20, 10), (40, 15), (60, 20)], [(80, 25), (100, 30), (120, 35)]]
    )

    # when
    result = scale_polygons(
        polygons=polygons,
        x_scale=2.0,
        y_scale=0.5,
    )

    # then
    assert np.allclose(np.array(result), expected_result)


def test_post_process_polygons_when_stretching_resize_used() -> None:
    # given
    polygons = [[(10, 20), (20, 30), (30, 40)], [(40, 50), (50, 60), (60, 70)]]
    # there is a crop from (x=10, y=20) to (x=90, y=180) (box size - h=160, w=80)
    # compared to inference size (100, 100) and we stretch, such that OX scale is 1.25,
    # OY scale is 0.625, then we shift by crop (OX +10, OY +20)
    # for OX coord we take input ox / 1.25 + 10, for OY: input oy / 0.625 +  20
    expected_result = np.array(
        [[(18, 52), (26, 68), (34, 84)], [(42, 100), (50, 116), (58, 132)]]
    )

    # when
    result = post_process_polygons(
        infer_shape=(100, 100),
        polys=polygons,
        origin_shape=(200, 100),
        preproc={
            "static-crop": {
                "enabled": True,
                "x_min": 10,
                "y_min": 10,
                "x_max": 90,
                "y_max": 90,
            }
        },
    )

    # then
    assert np.allclose(np.array(result), expected_result)


@pytest.mark.parametrize(
    "polygons, infer_shape, origin_shape, preproc, expected_result, expected_exception",
    [
        ([], (), (), {}, np.array([]), pytest.raises(IndexError)),  # malformed input
        ([], (100, 100), (100, 100), {}, np.array([]), DoesNotRaise()),  # no polygons
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (100, 100),
            (100, 100),
            {},
            np.array([[(10, 10), (20, 20), (10, 20)]]),
            DoesNotRaise(),
        ),  # no transformation required - square image
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (100, 100),
            (1000, 1000),
            {},
            np.array([[(100, 100), (200, 200), (100, 200)]]),
            DoesNotRaise(),
        ),  # inflate keeping aspect ratio - square image
        (
            [[(100, 100), (200, 200), (100, 200)]],
            (1000, 1000),
            (100, 100),
            {},
            np.array([[(10, 10), (20, 20), (10, 20)]]),
            DoesNotRaise(),
        ),  # shrink  keeping aspect ratio - square image
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (100, 200),
            (100, 200),
            {},
            np.array([[(10, 10), (20, 20), (10, 20)]]),
            DoesNotRaise(),
        ),  # no transformation required - rectangular image
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (200, 100),
            (200, 100),
            {},
            np.array([[(10, 10), (20, 20), (10, 20)]]),
            DoesNotRaise(),
        ),  # no transformation required - rectangular image
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (100, 200),
            (1000, 2000),
            {},
            np.array([[(100, 100), (200, 200), (100, 200)]]),
            DoesNotRaise(),
        ),  # inflate keeping aspect ratio - rectangular image
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (200, 100),
            (2000, 1000),
            {},
            np.array([[(100, 100), (200, 200), (100, 200)]]),
            DoesNotRaise(),
        ),  # inflate keeping aspect ratio - rectangular image
        (
            [[(100, 100), (200, 200), (100, 200)]],
            (1000, 2000),
            (100, 200),
            {},
            np.array([(10, 10), (20, 20), (10, 20)]),
            DoesNotRaise(),
        ),  # shrink keeping aspect ratio - rectangular image
        (
            [[(100, 100), (200, 200), (100, 200)]],
            (2000, 1000),
            (200, 100),
            {},
            np.array([(10, 10), (20, 20), (10, 20)]),
            DoesNotRaise(),
        ),  # shrink keeping aspect ratio - rectangular image
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (100, 100),
            (1000, 100),
            {},
            np.array([[(10, 100), (20, 200), (10, 200)]]),
            DoesNotRaise(),
        ),  # square -> rectangular (inflate vertically)
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (100, 100),
            (100, 1000),
            {},
            np.array([[(100, 10), (200, 20), (100, 20)]]),
            DoesNotRaise(),
        ),  # square -> rectangular (inflate horizontally)
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (100, 100),
            (10, 100),
            {},
            np.array([[(10, 1), (20, 2), (10, 2)]]),
            DoesNotRaise(),
        ),  # square -> rectangular (shrink vertically)
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (100, 100),
            (100, 10),
            {},
            np.array([[(1, 10), (2, 20), (1, 20)]]),
            DoesNotRaise(),
        ),  # square -> rectangular (shrink horizontally)
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (20, 200),
            (200, 200),
            {},
            np.array([[(10, 100), (20, 200), (10, 200)]]),
            DoesNotRaise(),
        ),  # rectangular -> square (inflate vertically)
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (200, 20),
            (200, 200),
            {},
            np.array([[(100, 10), (200, 20), (100, 20)]]),
            DoesNotRaise(),
        ),  # rectangular -> square (inflate horizontally)
        (
            [[(10, 100), (20, 200), (10, 200)]],
            (1000, 100),
            (100, 100),
            {},
            np.array([[(10, 10), (20, 20), (10, 20)]]),
            DoesNotRaise(),
        ),  # rectangular -> square (shrink vertically)
        (
            [[(100, 10), (200, 20), (100, 20)]],
            (100, 1000),
            (100, 100),
            {},
            np.array([[(10, 10), (20, 20), (10, 20)]]),
            DoesNotRaise(),
        ),  # rectangular -> square (shrink horizontally)
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (100, 200),
            (100 * 10.8, 200 * 9.6),
            {},
            np.array(
                [[(10 * 9.6, 10 * 10.8), (20 * 9.6, 20 * 10.8), (10 * 9.6, 20 * 10.8)]]
            ),
            DoesNotRaise(),
        ),  # rectangular -> rectangular (inflate to different aspect ratio)
        (
            [[(10, 10), (20, 20), (10, 20)]],
            (1080, 1920),
            (1080 / 10.8, 1920 / 9.6),
            {},
            np.array(
                [[(10 / 9.6, 10 / 10.8), (20 / 9.6, 20 / 10.8), (10 / 9.6, 20 / 10.8)]]
            ),
            DoesNotRaise(),
        ),  # rectangular -> rectangular (shrink to different aspect ratio)
    ],
)
def test_post_process_polygons(
    polygons: List[List[Tuple[XCoord, YCoord]]],
    infer_shape: Tuple[Height, Width],
    origin_shape: Tuple[Height, Width],
    preproc: Dict[str, Any],
    expected_result: np.array,
    expected_exception: Exception,
) -> None:
    with expected_exception:
        result = post_process_polygons(
            infer_shape=infer_shape,
            polys=polygons,
            origin_shape=origin_shape,
            preproc=preproc,
        )
        assert np.allclose(np.array(result), expected_result)


def test_post_process_polygons_when_fit_resize_used() -> None:
    # given
    polygons = [[(25, 20), (35, 30), (45, 40)], [(55, 50), (65, 60), (75, 70)]]
    # there is a crop from (x=10, y=20) to (x=90, y=180) (box size - h=160, w=80)
    # for inference, we scale 0.626, making 25px padding OX each side
    # then the crop shift is ox=10, oy=20
    # so effectively out(OX) = (in(OX) - 25) * 1.6 + 10, out(OY) = in(OY) * 1.6 + 20
    expected_result = np.array(
        [[(10, 52), (26, 68), (42, 84)], [(58, 100), (74, 116), (90, 132)]]
    )

    # when
    result = post_process_polygons(
        infer_shape=(100, 100),
        polys=polygons,
        origin_shape=(200, 100),
        preproc={
            "static-crop": {
                "enabled": True,
                "x_min": 10,
                "y_min": 10,
                "x_max": 90,
                "y_max": 90,
            }
        },
        resize_method="Fit (black edges) in",
    )

    # then
    assert np.allclose(np.array(result), expected_result)


def test_shift_keypoints() -> None:
    # given
    keypoints = np.array([[0, 0, 0.9, 10, 10, 0.9, 20, 25, 0.8]])
    expected_result = np.array([[5, 10, 0.9, 15, 20, 0.9, 25, 35, 0.8]])

    # when
    result = shift_keypoints(
        keypoints=keypoints,
        shift_x=5,
        shift_y=10,
    )

    # then
    assert np.allclose(np.array(result), expected_result)


def test_clip_keypoints_coordinates() -> None:
    # given
    keypoints = np.array([[-5, 0.1, 0.9, 10, 10, 0.9, 22, 25, 0.8]])
    expected_result = np.array([[0, 0, 0.9, 10, 10, 0.9, 18, 20, 0.8]])

    # when
    result = clip_keypoints_coordinates(keypoints=keypoints, origin_shape=(20, 18))

    # then
    assert np.allclose(np.array(result), expected_result)


def test_undo_image_padding_for_predicted_keypoints() -> None:
    # given
    keypoints = np.array([[32, 32, 0.8, 48, 48, 0.9, 96, 64, 0.8]])
    # For inference - image was scaled 0.25x, leaving 32px padding on OX each side
    # as a result: out(OX) = (in(OX) - 32) * 4, out(OY) = in(OY) * 4
    expected_result = np.array([[0, 128, 0.8, 64, 192, 0.9, 256, 256, 0.8]])

    # when
    result = undo_image_padding_for_predicted_keypoints(
        keypoints=keypoints,
        infer_shape=(64, 128),
        origin_shape=(256, 256),
    )

    # then
    assert np.allclose(np.array(result), expected_result)


def test_stretch_keypoints() -> None:
    # given
    keypoints = np.array([[32, 32, 0.8, 48, 48, 0.9, 96, 64, 0.8]])
    # For inference - image was scaled 0.25x OY axis and 0.5x OX axis,
    # so - out(OX) = in(OX) * 2, out(OY) = in(OY) * 4
    expected_result = np.array([[64, 128, 0.8, 96, 192, 0.9, 192, 256, 0.8]])

    # when
    result = stretch_keypoints(
        keypoints=keypoints,
        infer_shape=(64, 128),
        origin_shape=(256, 256),
    )

    # then
    assert np.allclose(np.array(result), expected_result)


def test_post_process_keypoints_when_crop_was_taken_and_stretching_method_used() -> (
    None
):
    # given
    predictions = np.array(
        [[[0, 1, 2, 3, 4, 5, 6, 32, 32, 0.8, 48, 48, 0.9, 96, 64, 0.8]]]
    ).tolist()
    # static crop was taken from (x=256, y=0) to (x=512, y=256) - of size (256, 256)
    # For inference - image was scaled 0.25x OY axis and 0.5x OX axis,
    # crop shift OX = 256, OY = 0
    # so - out(OX) = (in(OX) * 2) + 256, out(OY) = in(OY) * 4
    expected_result = np.array(
        [[[0, 1, 2, 3, 4, 5, 6, 320, 128, 0.8, 352, 192, 0.9, 448, 256, 0.8]]]
    )

    # when
    result = post_process_keypoints(
        predictions=predictions,
        keypoints_start_index=7,
        infer_shape=(64, 128),
        img_dims=[(512, 512)],
        preproc={
            "static-crop": {
                "enabled": True,
                "x_min": 50,
                "y_min": 0,
                "x_max": 100,
                "y_max": 50,
            }
        },
    )

    # then
    assert np.allclose(np.array(result), expected_result)


def test_post_process_keypoints_when_crop_was_taken_and_fit_to_padding_method_used() -> (
    None
):
    # given
    predictions = np.array(
        [[[0, 1, 2, 3, 4, 5, 6, 32, 32, 0.8, 48, 48, 0.9, 96, 64, 0.8]]]
    ).tolist()
    # static crop was taken from (x=256, y=0) to (x=512, y=256) - of size (256, 256)
    # For inference - image was scaled 0.25x, leaving 32px padding on OX each side
    # crop shift OX = 256, OY = 0
    # as a result: out(OX) = (in(OX) - 32) * 4 + 256, out(OY) = in(OY) * 4
    expected_result = np.array(
        [[[0, 1, 2, 3, 4, 5, 6, 256, 128, 0.8, 320, 192, 0.9, 512, 256, 0.8]]]
    )

    # when
    result = post_process_keypoints(
        predictions=predictions,
        keypoints_start_index=7,
        infer_shape=(64, 128),
        img_dims=[(512, 512)],
        preproc={
            "static-crop": {
                "enabled": True,
                "x_min": 50,
                "y_min": 0,
                "x_max": 100,
                "y_max": 50,
            }
        },
        resize_method="Fit (black edges) in",
    )

    # then
    assert np.allclose(np.array(result), expected_result)
