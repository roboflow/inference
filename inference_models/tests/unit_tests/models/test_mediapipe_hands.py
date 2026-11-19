import math

import numpy as np
import pytest
import torch

from inference_models.models.mediapipe_hands.geometry import (
    LANDMARKS_INPUT_SIZE,
    PalmDetection,
    RotatedRect,
    crop_to_image_matrix,
    decode_palm_detections,
    generate_palm_anchors,
    landmarks_to_hand_rect,
    letterbox_matrix,
    palm_to_hand_rect,
    project_landmarks,
    weighted_non_max_suppression,
)

try:
    import onnxruntime  # noqa: F401

    _ONNXRUNTIME_AVAILABLE = True
except ImportError:
    _ONNXRUNTIME_AVAILABLE = False

requires_onnxruntime = pytest.mark.skipif(
    not _ONNXRUNTIME_AVAILABLE,
    reason="onnxruntime is not installed (requires the onnx-* extra)",
)


def test_palm_anchors_follow_the_graph_output_layout() -> None:
    anchors = generate_palm_anchors()

    # 24 x 24 cells with 2 anchors (stride 8), then 12 x 12 cells with 6 (stride 16).
    assert anchors.shape == (2016, 2)
    np.testing.assert_allclose(anchors[0], [0.5 / 24, 0.5 / 24])
    np.testing.assert_allclose(anchors[1], anchors[0])
    np.testing.assert_allclose(anchors[2], [1.5 / 24, 0.5 / 24])
    np.testing.assert_allclose(anchors[1152:1158], [[0.5 / 12, 0.5 / 12]] * 6)
    np.testing.assert_allclose(anchors[-1], [11.5 / 12, 11.5 / 12])


def test_palm_decoding_maps_through_the_letterbox_into_image_pixels() -> None:
    # A 400 x 200 image is scaled to 192 x 96 and padded 48 px top and bottom.
    image_to_input = letterbox_matrix(image_width=400, image_height=200)
    anchors = generate_palm_anchors()
    raw_boxes = np.zeros((2016, 18), dtype=np.float32)
    raw_scores = np.full((2016, 1), -10.0, dtype=np.float32)
    anchor_index = 1152 + 6 * (12 * 6 + 6)  # stride-16 cell (6, 6), centre (104, 104)
    raw_boxes[anchor_index, :4] = [-8.0, 4.0, 48.0, 24.0]
    raw_boxes[anchor_index, 4:6] = [0.0, 10.0]
    raw_scores[anchor_index] = 4.0

    palms = decode_palm_detections(
        raw_boxes=raw_boxes,
        raw_scores=raw_scores,
        anchors=anchors,
        image_to_input=image_to_input,
        score_threshold=0.5,
    )

    assert len(palms) == 1
    # Input centre (96, 108) and size 48 x 24 -> image centre (200, 125), size 100 x 50.
    np.testing.assert_allclose(palms[0].xyxy, [150.0, 100.0, 250.0, 150.0], atol=1e-3)
    np.testing.assert_allclose(palms[0].key_points[0], [216.6667, 137.5], atol=1e-3)
    assert palms[0].score == pytest.approx(1 / (1 + math.exp(-4.0)))


def test_weighted_nms_averages_overlapping_palms_and_keeps_the_best_score() -> None:
    boxes = np.asarray(
        [[0.0, 0.0, 10.0, 10.0], [2.0, 0.0, 12.0, 10.0], [50.0, 50.0, 60.0, 60.0]],
        dtype=np.float32,
    )
    key_points = np.stack(
        [np.full((7, 2), value) for value in (0.0, 4.0, 50.0)]
    ).astype(np.float32)
    scores = np.asarray([0.9, 0.3, 0.8], dtype=np.float32)

    kept_boxes, kept_points, kept_scores = weighted_non_max_suppression(
        boxes=boxes, key_points=key_points, scores=scores, iou_threshold=0.3
    )

    np.testing.assert_allclose(kept_scores, [0.9, 0.8])
    np.testing.assert_allclose(kept_boxes[0], [0.5, 0.0, 10.5, 10.0])
    np.testing.assert_allclose(kept_points[0], np.full((7, 2), 1.0))
    np.testing.assert_allclose(kept_boxes[1], boxes[2])


def test_weighted_nms_stops_at_the_maximum_number_of_palms() -> None:
    boxes = np.asarray([[0, 0, 1, 1], [5, 5, 6, 6], [9, 9, 10, 10]], dtype=np.float32)

    kept_boxes, _, kept_scores = weighted_non_max_suppression(
        boxes=boxes,
        key_points=np.zeros((3, 7, 2), dtype=np.float32),
        scores=np.asarray([0.6, 0.9, 0.7], dtype=np.float32),
        iou_threshold=0.3,
        max_detections=2,
    )

    np.testing.assert_allclose(kept_scores, [0.9, 0.7])
    np.testing.assert_allclose(kept_boxes, boxes[[1, 2]])


@pytest.mark.parametrize(
    ("finger_direction", "expected_degrees"),
    [((0.0, -1.0), 0.0), ((1.0, 0.0), 90.0), ((0.0, 1.0), 180.0), ((-1.0, 0.0), -90.0)],
)
def test_hand_region_turns_the_wrist_to_knuckles_vector_upright(
    finger_direction, expected_degrees
) -> None:
    direction = np.asarray(finger_direction)
    key_points = np.zeros((7, 2), dtype=np.float32)
    key_points[0] = [100.0, 100.0]
    key_points[2] = key_points[0] + 30.0 * direction
    palm = PalmDetection(
        xyxy=np.asarray([80.0, 80.0, 120.0, 120.0]), key_points=key_points, score=0.9
    )

    rect = palm_to_hand_rect(palm)

    expected = math.radians(expected_degrees)
    np.testing.assert_allclose(
        [math.cos(rect.rotation), math.sin(rect.rotation)],
        [math.cos(expected), math.sin(expected)],
        atol=1e-6,
    )
    # 2.6x the palm, and moved half a palm towards the fingers.
    assert rect.width == pytest.approx(104.0)
    np.testing.assert_allclose(
        [rect.center_x, rect.center_y], [100.0, 100.0] + 20.0 * direction, atol=1e-4
    )


def test_crop_and_projection_are_inverse_for_a_rotated_region() -> None:
    rect = RotatedRect(
        center_x=300.0, center_y=200.0, width=112.0, height=112.0, rotation=math.pi / 2
    )
    crop_points = np.asarray(
        [[112.0, 112.0, 0.0], [224.0, 112.0, 10.0], [112.0, 0.0, 0.0]], dtype=np.float32
    )

    projected = project_landmarks(crop_points, rect, LANDMARKS_INPUT_SIZE)

    # Crop "right" is image "down" and crop "up" is image "right" at +90 degrees.
    np.testing.assert_allclose(projected[0, :2], [300.0, 200.0], atol=1e-4)
    np.testing.assert_allclose(projected[1, :2], [300.0, 256.0], atol=1e-4)
    np.testing.assert_allclose(projected[2, :2], [356.0, 200.0], atol=1e-4)
    assert projected[1, 2] == pytest.approx(5.0)
    matrix = crop_to_image_matrix(rect, LANDMARKS_INPUT_SIZE)
    np.testing.assert_allclose(abs(np.linalg.det(matrix[:, :2])), 0.25)


def test_region_from_landmarks_follows_the_hand_orientation() -> None:
    upright = _synthetic_landmarks()
    sideways = upright[:, ::-1] * [-1.0, 1.0] + [400.0, 0.0]  # rotated +90 degrees

    upright_rect = landmarks_to_hand_rect(upright)
    sideways_rect = landmarks_to_hand_rect(sideways)

    assert upright_rect.rotation == pytest.approx(0.0, abs=1e-6)
    assert sideways_rect.rotation == pytest.approx(math.pi / 2, abs=1e-6)
    assert sideways_rect.width == pytest.approx(upright_rect.width)


@requires_onnxruntime
@pytest.mark.parametrize(
    ("image_is_mirrored", "expected"), [(False, "left_hand"), (True, "right_hand")]
)
def test_handedness_is_reported_for_an_unmirrored_camera_by_default(
    image_is_mirrored, expected
) -> None:
    from inference_models.models.mediapipe_hands.hand_landmarks_onnx import (
        HAND_CLASS_NAMES,
        HandPrediction,
        hands_to_results,
    )

    hand = HandPrediction(
        landmarks=np.zeros((21, 3), dtype=np.float32),
        world_landmarks=np.zeros((21, 3), dtype=np.float32),
        presence=0.9,
        right_hand_score=0.8,  # MediaPipe says "Right" assuming a mirrored view
        rect=RotatedRect(0.0, 0.0, 1.0, 1.0, 0.0),
    )

    key_points, detections = hands_to_results(
        hands=[hand],
        scores=[0.9],
        image_is_mirrored=image_is_mirrored,
        device=torch.device("cpu"),
    )

    assert HAND_CLASS_NAMES[int(detections.class_id[0])] == expected
    assert key_points.key_points_metadata[0]["handedness_confidence"] == pytest.approx(
        0.8
    )


@requires_onnxruntime
def test_detector_boxes_get_a_second_landmark_pass_on_the_landmark_region() -> None:
    from inference_models.models.mediapipe_hands.hand_landmarks_onnx import (
        MediaPipeHandLandmarksONNX,
    )
    from inference_models.models.mediapipe_hands.mediapipe_hands_onnx import (
        MediaPipeHandsONNX,
    )

    landmarks_graph = _UprightHandGraph()
    model = MediaPipeHandsONNX(
        landmarks=MediaPipeHandLandmarksONNX(graph=landmarks_graph),
        hand_detector=_OneBoxDetector(),
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    [key_points], [detections] = model(
        image, confidence=0.4, hand_presence_threshold=0.5
    )

    assert landmarks_graph.calls == 2
    assert len(detections.xyxy) == 1
    # Box score 0.8 fused with presence 0.9.
    assert float(detections.confidence[0]) == pytest.approx(0.72)
    assert tuple(key_points.xy.shape) == (1, 21, 2)


def _synthetic_landmarks() -> np.ndarray:
    """An upright open hand around (100, 100): wrist at the bottom, fingers up."""
    landmarks = np.zeros((21, 2), dtype=np.float64)
    landmarks[0] = [100.0, 160.0]
    for finger, x in enumerate((70.0, 85.0, 100.0, 115.0, 130.0)):
        for joint in range(4):
            landmarks[1 + 4 * finger + joint] = [x, 120.0 - 15.0 * joint]
    return landmarks


class _UprightHandGraph:
    """Landmark graph stub: the same upright hand in every crop."""

    device = torch.device("cpu")

    def __init__(self):
        self.calls = 0
        crop_landmarks = _synthetic_landmarks()
        self._landmarks = (
            np.concatenate(
                [crop_landmarks - [100.0 - 112.0, 100.0 - 112.0], np.zeros((21, 1))],
                axis=1,
            )
            .reshape(1, 63)
            .astype(np.float32)
        )

    def run(self, batch: np.ndarray):
        self.calls += 1
        count = len(batch)
        return [
            np.repeat(self._landmarks, count, axis=0),
            np.full((count, 1), 0.9, dtype=np.float32),
            np.full((count, 1), 0.2, dtype=np.float32),
            np.zeros((count, 63), dtype=np.float32),
        ]


class _OneBoxDetector:
    def infer(self, images, **kwargs):
        from inference_models import Detections

        return [
            Detections(
                xyxy=torch.tensor([[200.0, 100.0, 300.0, 220.0]]),
                class_id=torch.tensor([0]),
                confidence=torch.tensor([0.8]),
            )
            for _ in images
        ]
