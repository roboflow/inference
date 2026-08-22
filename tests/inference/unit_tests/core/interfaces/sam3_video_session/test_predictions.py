import numpy as np

from inference.core.interfaces.sam3_video_session.predictions import (
    class_name_for_object,
    mask_to_uncompressed_rle,
    serialize_frame_predictions,
    xyxy_to_center_bounds,
)


def test_mask_to_uncompressed_rle_uses_fortran_order() -> None:
    mask = np.zeros((4, 3), dtype=np.uint8)
    mask[1:3, 1] = 1

    encoded = mask_to_uncompressed_rle(mask)

    assert encoded["size"] == [4, 3]
    reconstructed = np.zeros(4 * 3, dtype=np.uint8)
    offset = 0
    foreground = False
    for count in encoded["counts"]:
        if foreground:
            reconstructed[offset : offset + count] = 1
        offset += count
        foreground = not foreground
    assert reconstructed.reshape((4, 3), order="F").tolist() == mask.tolist()


def test_xyxy_to_center_bounds() -> None:
    assert xyxy_to_center_bounds([10, 20, 30, 40]) == {
        "x": 20.0,
        "y": 30.0,
        "width": 20.0,
        "height": 20.0,
    }
    assert xyxy_to_center_bounds([1, 1, 1, 2]) is None


def test_serialize_frame_predictions_skips_low_score_and_empty_masks() -> None:
    masks = np.zeros((3, 8, 8), dtype=np.uint8)
    masks[0, 2:6, 2:6] = 1
    masks[2, 0:2, 0:2] = 1
    predictions, samples = serialize_frame_predictions(
        masks=masks,
        object_ids=np.array([7, 8, 9]),
        scores=np.array([0.9, 0.1, 0.8]),
        boxes=np.array(
            [
                [2.0, 2.0, 6.0, 6.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 2.0, 2.0],
            ]
        ),
        prompt_to_object_ids={"forklift": [7], "pallet": [9]},
        threshold=0.35,
        width=8,
        height=8,
    )

    assert [item["tracker_id"] for item in predictions] == [7, 9]
    assert predictions[0]["class_name"] == "forklift"
    assert predictions[0]["rle_mask"]["size"] == [8, 8]
    assert samples[0]["trackId"] == 7
    assert samples[0]["geometry"]["rleMask"]["size"] == [8, 8]
    assert class_name_for_object(7, {"forklift": [7]}) == "forklift"
