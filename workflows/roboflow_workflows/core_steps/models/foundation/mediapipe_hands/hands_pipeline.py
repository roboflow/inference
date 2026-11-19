"""One frame of the MediaPipe hands pipeline, driven through the models provider.

The palm detector and the hand landmarks model are separate registered models.
This module decides which hand regions the landmark model sees on each frame:
palm detections, boxes from an upstream detector, or, on video, the regions
the previous frame's landmarks give (see
``inference_models.models.mediapipe_hands.tracking``).

``inference_models.models.mediapipe_hands`` is imported inside the functions:
older ``inference-models`` releases, which this package still allows, do not
have it, and the rest of the block library must keep importing without it.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np
from roboflow_workflows.prototypes.models_provider import ModelsProvider

DETECTOR_BOX_SCALE = 1.3


@dataclass
class HandsSession:
    """Hand regions carried from one video frame to the next."""

    hands: List[Any] = field(default_factory=list)  # List[TrackedHand]
    last_frame_number: int = -1


@dataclass(frozen=True)
class HandResult:
    """One hand in image pixels, ready to become a keypoint detection."""

    xyxy: np.ndarray  # (4,)
    key_points_xy: np.ndarray  # (21, 2)
    class_id: int
    score: float
    hand_presence: float
    handedness_confidence: float
    landmarks_z: np.ndarray  # (21,)
    world_landmarks: np.ndarray  # (21, 3)


@dataclass
class ProviderHandModels:
    """The two stage models, called through the models provider."""

    model_manager: ModelsProvider
    api_key: Optional[str]
    palm_detector_model_id: str
    hand_landmarks_model_id: str

    def detect_palms(
        self, image_bgr: np.ndarray, *, confidence: float, max_num_hands: int
    ) -> List[Any]:
        """Hand regions from the palm detector, best first."""
        from inference_models.models.mediapipe_hands.geometry import rect_from_list
        from inference_models.models.mediapipe_hands.tracking import TrackedHand

        self.model_manager.add_model(
            model_id=self.palm_detector_model_id, api_key=self.api_key
        )
        _, detections = self.model_manager.run_tensor_native_inference(
            model_id=self.palm_detector_model_id,
            images=[image_bgr],
            input_color_format="bgr",
            confidence=confidence,
            max_num_palms=max_num_hands,
        )
        palms = detections[0]
        scores = palms.confidence.detach().cpu().numpy().tolist()
        return [
            TrackedHand(rect=rect_from_list(metadata["hand_rect"]), score=float(score))
            for metadata, score in zip(palms.bboxes_metadata or [], scores)
        ]

    def predict(
        self, image_bgr: np.ndarray, rects: Sequence[Any], *, image_is_mirrored: bool
    ) -> List[dict]:
        """Landmarks for every region, one entry per region in order.

        Presence filtering is left to the caller (threshold 0), so entries stay
        aligned with ``rects``.
        """
        if not rects:
            return []
        self.model_manager.add_model(
            model_id=self.hand_landmarks_model_id, api_key=self.api_key
        )
        key_points, detections = self.model_manager.run_tensor_native_inference(
            model_id=self.hand_landmarks_model_id,
            images=[image_bgr],
            input_color_format="bgr",
            hand_rects=[list(rects)],
            hand_presence_threshold=0.0,
            image_is_mirrored=image_is_mirrored,
        )
        hands = key_points[0]
        xy = hands.xy.detach().cpu().numpy()
        class_id = hands.class_id.detach().cpu().numpy()
        xyxy = detections[0].xyxy.detach().cpu().numpy()
        return [
            {
                "xyxy": xyxy[index].astype(np.float32),
                "key_points_xy": xy[index].astype(np.float32),
                "class_id": int(class_id[index]),
                **hands.key_points_metadata[index],
            }
            for index in range(len(rects))
        ]


def run_hands_on_frame(
    image_bgr: np.ndarray,
    *,
    models: ProviderHandModels,
    session: Optional[HandsSession],
    frame_number: int,
    hand_boxes: Optional[np.ndarray],
    box_scores: Optional[np.ndarray],
    max_num_hands: int,
    confidence: float,
    hand_presence_threshold: float,
    image_is_mirrored: bool,
) -> List[HandResult]:
    """Hands on one frame; updates ``session`` with regions for the next one.

    Args:
        image_bgr: The frame.
        models: The two stage models.
        session: This video's carried regions, or None to never reuse them.
        frame_number: The frame's number in its video. Regions are reused only
            on a later frame than the session last saw; anything else resets.
        hand_boxes: (N, 4) boxes from an upstream hand detector. When given,
            the palm detector is not used.
        box_scores: (N,) scores of ``hand_boxes``.
        max_num_hands: Most hands returned.
        confidence: Minimum palm detector score.
        hand_presence_threshold: Minimum hand presence from the landmark model.
        image_is_mirrored: Report MediaPipe's mirrored handedness labels.

    Returns:
        The kept hands. A hand's score is the score of the detection that first
        found it times its current presence.
    """
    from inference_models.models.mediapipe_hands.geometry import (
        box_to_hand_rect,
        rect_from_list,
    )
    from inference_models.models.mediapipe_hands.tracking import (
        TrackedHand,
        deduplicate_hands,
        merge_tracked_and_detected,
        needs_detection,
    )

    tracked: List[TrackedHand] = []
    if session is not None and 0 <= session.last_frame_number < frame_number:
        tracked = list(session.hands)
    regions = tracked
    from_boxes: List[TrackedHand] = []
    if needs_detection(tracked, max_num_hands=max_num_hands):
        if hand_boxes is not None:
            from_boxes = _box_seeds(hand_boxes, box_scores, box_to_hand_rect)
            detected = from_boxes
        else:
            detected = models.detect_palms(
                image_bgr, confidence=confidence, max_num_hands=max_num_hands
            )
        regions = merge_tracked_and_detected(
            tracked, detected, max_num_hands=max_num_hands
        )
    regions = _refine_box_seeds(
        image_bgr,
        regions=regions,
        box_seeds=from_boxes,
        models=models,
        image_is_mirrored=image_is_mirrored,
    )
    predictions = models.predict(
        image_bgr,
        [region.rect for region in regions],
        image_is_mirrored=image_is_mirrored,
    )
    present = [
        (region, prediction)
        for region, prediction in zip(regions, predictions)
        if prediction["hand_presence"] >= hand_presence_threshold
    ]
    kept_indices = deduplicate_hands(
        np.asarray([prediction["xyxy"] for _, prediction in present]),
        [prediction["hand_presence"] for _, prediction in present],
    )
    kept = [present[index] for index in kept_indices]
    if session is not None:
        session.hands = [
            TrackedHand(
                rect=rect_from_list(prediction["next_hand_rect"]), score=region.score
            )
            for region, prediction in kept
        ]
        session.last_frame_number = frame_number
    return [_to_hand_result(region.score, prediction) for region, prediction in kept]


def _box_seeds(
    hand_boxes: np.ndarray, box_scores: Optional[np.ndarray], box_to_hand_rect
) -> List[Any]:
    from inference_models.models.mediapipe_hands.tracking import TrackedHand

    if box_scores is None:
        box_scores = np.ones((len(hand_boxes),), dtype=np.float32)
    order = np.argsort(-np.asarray(box_scores, dtype=np.float64), kind="stable")
    return [
        TrackedHand(
            rect=box_to_hand_rect(hand_boxes[index], DETECTOR_BOX_SCALE),
            score=float(box_scores[index]),
        )
        for index in order
    ]


def _refine_box_seeds(
    image_bgr: np.ndarray,
    *,
    regions: List[Any],
    box_seeds: List[Any],
    models: ProviderHandModels,
    image_is_mirrored: bool,
) -> List[Any]:
    """Swap an upright box region for the region its first landmarks give.

    A detector box says nothing about the hand's orientation, so a first
    landmark pass on the upright square finds it, and the second pass uses the
    rotated region those landmarks give, as palm detections would.
    """
    from inference_models.models.mediapipe_hands.geometry import rect_from_list
    from inference_models.models.mediapipe_hands.tracking import TrackedHand

    seed_ids = {id(seed) for seed in box_seeds}
    to_refine = [
        index for index, region in enumerate(regions) if id(region) in seed_ids
    ]
    if not to_refine:
        return regions
    first_pass = models.predict(
        image_bgr,
        [regions[index].rect for index in to_refine],
        image_is_mirrored=image_is_mirrored,
    )
    refined = list(regions)
    for index, prediction in zip(to_refine, first_pass):
        refined[index] = TrackedHand(
            rect=rect_from_list(prediction["next_hand_rect"]),
            score=regions[index].score,
        )
    return refined


def _to_hand_result(seed_score: float, prediction: dict) -> HandResult:
    presence = float(prediction["hand_presence"])
    return HandResult(
        xyxy=prediction["xyxy"],
        key_points_xy=prediction["key_points_xy"],
        class_id=prediction["class_id"],
        score=seed_score * presence,
        hand_presence=presence,
        handedness_confidence=float(prediction["handedness_confidence"]),
        landmarks_z=np.asarray(prediction["landmarks_z"], dtype=np.float32),
        world_landmarks=np.asarray(prediction["world_landmarks"], dtype=np.float32),
    )
