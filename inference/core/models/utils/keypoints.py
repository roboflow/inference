from typing import List

from inference.core.entities.responses.inference import Keypoint
from inference.core.exceptions import ModelArtefactError


def superset_keypoints_count(keypoints_metadata={}) -> int:
    """Returns the number of keypoints in the superset."""
    max_keypoints = 0
    for keypoints in keypoints_metadata.values():
        if len(keypoints) > max_keypoints:
            max_keypoints = len(keypoints)
    return max_keypoints


def model_keypoints_to_response(
    keypoints_metadata: dict,
    keypoints: List[float],
    predicted_object_class_id: int,
    keypoint_confidence_threshold: float,
) -> List[Keypoint]:
    if keypoints_metadata is None:
        raise ModelArtefactError("Keypoints metadata not available.")
    keypoint_id2name = keypoints_metadata[predicted_object_class_id]
    results = []
    for keypoint_id in range(len(keypoints) // 3):
        if keypoint_id >= len(keypoint_id2name):
            # Ultralytics only supports single class keypoint detection, so points might be padded with zeros
            break
        confidence = keypoints[3 * keypoint_id + 2]
        if confidence < keypoint_confidence_threshold:
            continue
        keypoint = Keypoint(
            x=keypoints[3 * keypoint_id],
            y=keypoints[3 * keypoint_id + 1],
            confidence=confidence,
            class_id=keypoint_id,
            class_name=keypoint_id2name[keypoint_id],
        )
        results.append(keypoint)
    return results
