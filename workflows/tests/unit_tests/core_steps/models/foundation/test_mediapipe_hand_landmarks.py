import datetime
from collections import Counter
from typing import Callable, List, Optional, Tuple

import numpy as np
import pytest
import supervision as sv
import torch
from pydantic import ValidationError
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.foundation.mediapipe_hands.v1 import (
    BlockManifest,
    MediaPipeHandLandmarksBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

pytest.importorskip("inference_models.models.mediapipe_hands.hand_landmarks_onnx")
pytest.importorskip("onnxruntime")

from inference_models import Detections  # noqa: E402
from inference_models.models.mediapipe_hands.geometry import (  # noqa: E402
    RotatedRect,
    rect_to_list,
)
from inference_models.models.mediapipe_hands.hand_landmarks_onnx import (  # noqa: E402
    HandPrediction,
    hands_to_results,
)

PALM_ID = "palm-test"
LANDMARKS_ID = "landmarks-test"
LEFT_HAND_AT = (200.0, 200.0)
RIGHT_HAND_AT = (600.0, 200.0)


class FakeHandModels:
    """Both stage models behind the provider port.

    Hands stand still at ``hands``. The palm detector finds those in ``palms``
    (all of them by default). The landmark model finds the hand nearest a
    region's centre when it lies inside the region, with ``presence`` deciding
    how sure it is; otherwise it reports no hand.
    """

    def __init__(
        self,
        hands: List[Tuple[float, float]],
        palms: Optional[List[Tuple[float, float]]] = None,
        presence: Callable[[float, float], float] = lambda x, y: 0.9,
    ):
        self.hands = hands
        self.palms = hands if palms is None else palms
        self.presence = presence
        self.calls: Counter = Counter()
        self.landmark_regions: List[List[RotatedRect]] = []

    def add_model(self, model_id: str, api_key: Optional[str], **kwargs) -> None:
        self.calls[f"add:{model_id}"] += 1

    def run_tensor_native_inference(self, model_id: str, images, **kwargs):
        self.calls[model_id] += 1
        if model_id == PALM_ID:
            return self._palms()
        regions = kwargs["hand_rects"][0]
        self.landmark_regions.append(list(regions))
        hands = [self._hand_in(region) for region in regions]
        key_points, detections = hands_to_results(
            hands=hands,
            scores=[hand.presence for hand in hands],
            image_is_mirrored=kwargs["image_is_mirrored"],
            device=torch.device("cpu"),
        )
        return [key_points], [detections]

    def _palms(self):
        count = len(self.palms)
        boxes = [[x - 40, y - 40, x + 40, y + 40] for x, y in self.palms]
        detections = Detections(
            xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(count, 4),
            class_id=torch.zeros((count,), dtype=torch.int64),
            confidence=torch.full((count,), 0.8),
            bboxes_metadata=[
                {"hand_rect": rect_to_list(RotatedRect(x, y + 15, 150.0, 150.0, 0.0))}
                for x, y in self.palms
            ],
        )
        return [None], [detections]

    def _hand_in(self, region: RotatedRect) -> HandPrediction:
        x, y = min(
            self.hands,
            key=lambda hand: (hand[0] - region.center_x) ** 2
            + (hand[1] - region.center_y) ** 2,
            default=(region.center_x, region.center_y),
        )
        inside = (
            self.hands
            and abs(x - region.center_x) < region.width / 2
            and abs(y - region.center_y) < region.height / 2
        )
        return HandPrediction(
            landmarks=_upright_hand(x, y),
            world_landmarks=np.zeros((21, 3), dtype=np.float32),
            presence=self.presence(x, y) if inside else 0.05,
            right_hand_score=0.9 if x < 400 else 0.1,
            rect=region,
        )


def _upright_hand(center_x: float, center_y: float) -> np.ndarray:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0, :2] = [center_x, center_y + 60.0]
    for finger, dx in enumerate((-30.0, -15.0, 0.0, 15.0, 30.0)):
        for joint in range(4):
            landmarks[1 + 4 * finger + joint, :2] = [
                center_x + dx,
                center_y + 20.0 - 15.0 * joint,
            ]
    return landmarks


def _frame(number: int, video: str = "camera") -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"{video}-{number}"),
        numpy_image=np.zeros((480, 800, 3), dtype=np.uint8),
        video_metadata=VideoMetadata(
            video_identifier=video,
            frame_number=number,
            frame_timestamp=datetime.datetime(2026, 11, 19),
            fps=30,
        ),
    )


def _still() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=np.zeros((480, 800, 3), dtype=np.uint8),
    )


def _run(
    block: MediaPipeHandLandmarksBlockV1,
    images: List[WorkflowImageData],
    hand_detections: Optional[List[Optional[sv.Detections]]] = None,
    **overrides,
) -> List[sv.Detections]:
    parameters = dict(
        max_num_hands=2,
        confidence=0.5,
        hand_presence_threshold=0.5,
        image_is_mirrored=False,
        reuse_hand_regions=True,
        palm_detector_model_id=PALM_ID,
        hand_landmarks_model_id=LANDMARKS_ID,
    )
    parameters.update(overrides)
    results = block.run(images=images, hand_detections=hand_detections, **parameters)
    return [result["predictions"] for result in results]


def _block(models: FakeHandModels, mode=StepExecutionMode.LOCAL):
    return MediaPipeHandLandmarksBlockV1(
        model_manager=models, api_key=None, step_execution_mode=mode
    )


def _manifest(**fields) -> BlockManifest:
    return BlockManifest.model_validate(
        {
            "type": "roboflow_core/mediapipe_hand_landmarks@v1",
            "name": "hands",
            "images": "$inputs.image",
            **fields,
        }
    )


def test_manifest_rejects_zero_hands() -> None:
    with pytest.raises(ValidationError):
        _manifest(max_num_hands=0)


def test_palm_detector_is_not_a_dependency_when_hand_boxes_come_from_upstream() -> None:
    with_palms = _manifest().discover_dependent_resources()
    with_boxes = _manifest(
        hand_detections="$steps.detector.predictions"
    ).discover_dependent_resources()

    assert [resource.metadata.model_id for resource in with_palms] == [
        "mediapipe-palm-detector/full",
        "mediapipe-hand-landmarks/full",
    ]
    assert [resource.metadata.model_id for resource in with_boxes] == [
        "mediapipe-hand-landmarks/full"
    ]


def test_still_image_finds_hands_with_landmarks_and_handedness() -> None:
    models = FakeHandModels(hands=[LEFT_HAND_AT, RIGHT_HAND_AT])

    [hands] = _run(_block(models), [_still()])

    assert len(hands) == 2
    # MediaPipe says "Right" for the hand on the left of the image, which is
    # the person's left hand in a normal camera view.
    assert list(hands["class_name"]) == ["left_hand", "right_hand"]
    assert hands["keypoints_xy"].shape == (2, 21, 2)
    np.testing.assert_allclose(hands.confidence, [0.8 * 0.9, 0.8 * 0.9], rtol=1e-6)
    assert hands["world_landmarks"].shape == (2, 21, 3)
    assert set(hands["prediction_type"]) == {"keypoint-detection"}


def test_hands_below_presence_are_dropped() -> None:
    models = FakeHandModels(
        hands=[LEFT_HAND_AT, RIGHT_HAND_AT],
        presence=lambda x, y: 0.9 if x < 400 else 0.2,
    )

    [hands] = _run(_block(models), [_still()])

    assert list(hands["class_name"]) == ["left_hand"]


def test_no_hands_gives_empty_keypoint_predictions() -> None:
    [hands] = _run(_block(FakeHandModels(hands=[])), [_still()])

    assert len(hands) == 0
    assert hands["keypoints_xy"].shape == (0, 21, 2)


def test_upstream_boxes_replace_the_palm_detector_with_two_landmark_passes() -> None:
    models = FakeHandModels(hands=[LEFT_HAND_AT, RIGHT_HAND_AT])
    boxes = sv.Detections(
        xyxy=np.asarray([[560.0, 140.0, 640.0, 260.0]]), confidence=np.asarray([0.7])
    )

    [hands] = _run(_block(models), [_still()], hand_detections=[boxes])

    assert models.calls[PALM_ID] == 0
    assert models.calls[f"add:{PALM_ID}"] == 0
    assert models.calls[LANDMARKS_ID] == 2
    # The second pass runs on the region the first pass's landmarks give.
    first, second = models.landmark_regions
    assert first[0].width == pytest.approx(120.0 * 1.3)
    assert second[0] != first[0]
    assert list(hands["class_name"]) == ["right_hand"]
    np.testing.assert_allclose(hands.confidence, [0.7 * 0.9], rtol=1e-6)


def test_video_reuses_hand_regions_and_skips_the_palm_detector() -> None:
    models = FakeHandModels(hands=[LEFT_HAND_AT, RIGHT_HAND_AT])
    block = _block(models)

    results = [_run(block, [_frame(number)])[0] for number in range(4)]

    assert models.calls[PALM_ID] == 1
    assert [len(hands) for hands in results] == [2, 2, 2, 2]


def test_palm_detector_keeps_running_while_fewer_hands_are_tracked_than_asked_for() -> (
    None
):
    models = FakeHandModels(hands=[LEFT_HAND_AT])
    block = _block(models)

    for number in range(3):
        _run(block, [_frame(number)])

    assert models.calls[PALM_ID] == 3
    # The tracked hand is not detected twice: one region per frame.
    assert [len(regions) for regions in models.landmark_regions] == [1, 1, 1]


def test_a_lost_hand_brings_the_palm_detector_back_on_the_next_frame() -> None:
    hand_present = {"value": True}
    models = FakeHandModels(
        hands=[LEFT_HAND_AT],
        presence=lambda x, y: 0.9 if hand_present["value"] else 0.1,
    )
    block = _block(models)

    _run(block, [_frame(0)], max_num_hands=1)
    _run(block, [_frame(1)], max_num_hands=1)
    hand_present["value"] = False
    [lost] = _run(block, [_frame(2)], max_num_hands=1)
    _run(block, [_frame(3)], max_num_hands=1)

    assert len(lost) == 0
    assert models.calls[PALM_ID] == 2


def test_still_images_and_disabled_reuse_detect_on_every_image() -> None:
    models = FakeHandModels(hands=[LEFT_HAND_AT])
    block = _block(models)

    _run(block, [_still()], max_num_hands=1)
    _run(block, [_still()], max_num_hands=1)
    _run(block, [_frame(0)], max_num_hands=1, reuse_hand_regions=False)
    _run(block, [_frame(1)], max_num_hands=1, reuse_hand_regions=False)

    assert models.calls[PALM_ID] == 4


def test_a_frame_number_going_back_starts_the_video_over() -> None:
    models = FakeHandModels(hands=[LEFT_HAND_AT])
    block = _block(models)

    _run(block, [_frame(5)], max_num_hands=1)
    _run(block, [_frame(6)], max_num_hands=1)
    _run(block, [_frame(0)], max_num_hands=1)

    assert models.calls[PALM_ID] == 2


def test_each_video_keeps_its_own_hand_regions() -> None:
    models = FakeHandModels(hands=[LEFT_HAND_AT])
    block = _block(models)

    _run(block, [_frame(0, "a"), _frame(0, "b")], max_num_hands=1)
    _run(block, [_frame(1, "a"), _frame(1, "b")], max_num_hands=1)

    assert models.calls[PALM_ID] == 2


def test_tracked_hand_wins_over_an_upstream_box_on_the_same_hand() -> None:
    models = FakeHandModels(hands=[LEFT_HAND_AT], palms=[])
    block = _block(models)
    same_hand = sv.Detections(
        xyxy=np.asarray([[160.0, 140.0, 240.0, 260.0]]), confidence=np.asarray([0.7])
    )

    _run(block, [_frame(0)], hand_detections=[same_hand])
    tracked_region = models.landmark_regions[-1][0]
    [hands] = _run(block, [_frame(1)], hand_detections=[same_hand])

    assert len(hands) == 1
    # One landmark pass on the carried region; the box is not seeded again.
    assert len(models.landmark_regions[-1]) == 1
    assert models.calls[LANDMARKS_ID] == 3
    assert models.landmark_regions[-1][0].center_x == pytest.approx(
        tracked_region.center_x, abs=1.0
    )


def test_remote_step_execution_is_rejected() -> None:
    block = _block(FakeHandModels(hands=[]), mode=StepExecutionMode.REMOTE)

    with pytest.raises(NotImplementedError):
        _run(block, [_still()])


def test_tensor_variant_returns_native_keypoints_with_serialisable_metadata() -> None:
    from roboflow_workflows.core_steps.models.foundation.mediapipe_hands import (
        v1_tensor,
    )

    models = FakeHandModels(hands=[LEFT_HAND_AT, RIGHT_HAND_AT])
    block = v1_tensor.MediaPipeHandLandmarksBlockV1(
        model_manager=models, api_key=None, step_execution_mode=StepExecutionMode.LOCAL
    )
    upstream = Detections(
        xyxy=torch.tensor([[560.0, 140.0, 640.0, 260.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.7]),
    )

    [with_palms] = _run(block, [_still()])
    # A keypoint prediction tuple from upstream works as well as plain boxes.
    [with_boxes] = _run(block, [_still()], hand_detections=[(None, upstream)])

    key_points, detections = with_palms
    assert tuple(key_points.xy.shape) == (2, 21, 2)
    assert detections.bboxes_metadata[0]["keypoints_class_name"][0] == "wrist"
    assert len(detections.bboxes_metadata[1]["keypoints_xy"]) == 21
    assert detections.bboxes_metadata[0]["world_landmarks"] is not None
    key_points, detections = with_boxes
    # Only the still-image call ran the palm detector.
    assert models.calls[PALM_ID] == 1
    assert detections.class_id.tolist() == [1]  # right_hand
