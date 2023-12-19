def superset_keypoints_count(keypoints_metadata={}) -> int:
    """Returns the number of keypoints in the superset."""
    max_keypoints = 0
    for keypoints in keypoints_metadata.values():
        if len(keypoints) > max_keypoints:
            max_keypoints = len(keypoints)
    return max_keypoints
