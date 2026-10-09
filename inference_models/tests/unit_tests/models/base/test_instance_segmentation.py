from dataclasses import replace

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks


def test_len_when_no_instances() -> None:
    # given
    detections = InstanceDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
        mask=torch.zeros((0, 8, 8), dtype=torch.bool),
    )

    # when / then
    assert len(detections) == 0


def test_len_when_multiple_instances_with_dense_mask() -> None:
    # given
    detections = InstanceDetections(
        xyxy=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.float32),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4], dtype=torch.float32),
        mask=torch.zeros((2, 8, 8), dtype=torch.bool),
    )

    # when / then
    assert len(detections) == 2


def test_len_counts_boxes_regardless_of_mask_representation() -> None:
    # given - len() reads xyxy.shape[0], so the RLE mask representation must not matter
    detections = InstanceDetections(
        xyxy=torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.float32
        ),
        class_id=torch.tensor([0, 1, 0], dtype=torch.long),
        confidence=torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
        mask=InstancesRLEMasks(image_size=(8, 8), masks=[b"", b"", b""]),
    )

    # when / then
    assert len(detections) == 3


def test_rle_mask_size_defaults_to_image_size() -> None:
    # given / when
    masks = InstancesRLEMasks(image_size=(1080, 1920), masks=[])

    # then
    assert masks.mask_size == (1080, 1920)


def test_rle_mask_size_can_differ_from_image_size() -> None:
    # given / when
    masks = InstancesRLEMasks(image_size=(1080, 1920), masks=[], mask_size=(270, 480))

    # then
    assert masks.mask_size == (270, 480)
    assert masks.image_size == (1080, 1920)


def test_wire_format_declares_the_encoded_size() -> None:
    # given
    # COCO `size` must describe the grid the counts were encoded on, or the
    # counts no longer sum to h*w and pycocotools decodes silently wrong
    masks = InstancesRLEMasks(
        image_size=(1080, 1920), masks=[b"abc"], mask_size=(270, 480)
    )

    # when
    encoded = masks.to_coco_rle_masks()

    # then
    assert encoded == [{"size": [270, 480], "counts": b"abc"}]


def test_wire_format_unchanged_when_sizes_agree() -> None:
    # given
    masks = InstancesRLEMasks(image_size=(1080, 1920), masks=[b"abc"])

    # when
    encoded = masks.to_coco_rle_masks()

    # then
    assert encoded == [{"size": [1080, 1920], "counts": b"abc"}]


def test_dense_mask_size_falls_back_to_the_tensor_grid() -> None:
    # given
    detections = InstanceDetections(
        xyxy=torch.zeros((2, 4)),
        class_id=torch.zeros((2,), dtype=torch.int64),
        confidence=torch.zeros((2,)),
        mask=torch.zeros((2, 540, 960), dtype=torch.bool),
    )

    # then
    assert detections.mask_size == (540, 960)


def test_dense_mask_size_honours_an_explicit_value() -> None:
    # given
    detections = InstanceDetections(
        xyxy=torch.zeros((1, 4)),
        class_id=torch.zeros((1,), dtype=torch.int64),
        confidence=torch.zeros((1,)),
        mask=torch.zeros((1, 270, 480), dtype=torch.bool),
        mask_size=(270, 480),
    )

    # then
    assert detections.mask_size == (270, 480)


def test_rle_carrier_reports_its_own_grid() -> None:
    # given
    detections = InstanceDetections(
        xyxy=torch.zeros((1, 4)),
        class_id=torch.zeros((1,), dtype=torch.int64),
        confidence=torch.zeros((1,)),
        mask=InstancesRLEMasks(
            image_size=(1080, 1920), masks=[b""], mask_size=(270, 480)
        ),
    )

    # then
    assert detections.mask_size == (270, 480)


def test_empty_detection_stack_still_reports_the_grid() -> None:
    # given
    # consumers derive the scale factor from the grid whether or not
    # anything was detected
    detections = InstanceDetections(
        xyxy=torch.zeros((0, 4)),
        class_id=torch.zeros((0,), dtype=torch.int64),
        confidence=torch.zeros((0,)),
        mask=torch.zeros((0, 270, 480), dtype=torch.bool),
    )

    # then
    assert detections.mask_size == (270, 480)


def test_to_supervision_restores_the_image_grid() -> None:
    # given
    # masks produced at a quarter of the image resolution
    detections = InstanceDetections(
        xyxy=torch.tensor([[0, 0, 20, 20]], dtype=torch.float32),
        class_id=torch.zeros((1,), dtype=torch.int64),
        confidence=torch.ones((1,), dtype=torch.float32),
        mask=InstancesRLEMasks(
            image_size=(80, 80),
            masks=[_square_rle(20, 20)],
            mask_size=(20, 20),
        ),
    )

    # when
    converted = detections.to_supervision()

    # then
    assert converted.mask.shape == (1, 80, 80)


def test_supervision_annotator_accepts_the_result() -> None:
    # given
    # the reported failure was an IndexError from boolean-index mismatch
    import numpy as np
    import supervision as sv

    scene = np.zeros((80, 80, 3), dtype=np.uint8)
    detections = InstanceDetections(
        xyxy=torch.tensor([[0, 0, 20, 20]], dtype=torch.float32),
        class_id=torch.zeros((1,), dtype=torch.int64),
        confidence=torch.ones((1,), dtype=torch.float32),
        mask=InstancesRLEMasks(
            image_size=(80, 80),
            masks=[_square_rle(20, 20)],
            mask_size=(20, 20),
        ),
    )

    # when / then
    sv.MaskAnnotator().annotate(scene.copy(), detections.to_supervision())


def test_iteration_declares_the_encoded_grid() -> None:
    # given
    detections = InstanceDetections(
        xyxy=torch.tensor([[0, 0, 20, 20]], dtype=torch.float32),
        class_id=torch.zeros((1,), dtype=torch.int64),
        confidence=torch.ones((1,), dtype=torch.float32),
        mask=InstancesRLEMasks(
            image_size=(80, 80),
            masks=[_square_rle(20, 20)],
            mask_size=(20, 20),
        ),
    )

    # when
    _, mask, *_ = next(iter(detections))

    # then
    assert mask["size"] == [20, 20]


def _square_rle(height: int, width: int) -> bytes:
    """Encode a filled square as COCO RLE counts.

    Args:
        height: Grid height.
        width: Grid width.

    Returns:
        COCO RLE counts for a square covering the middle of the grid.

    Examples:
        ```pycon
        >>> isinstance(_square_rle(20, 20), bytes)
        True

        ```
    """
    import numpy as np
    from pycocotools import mask as mask_utils

    dense = np.zeros((height, width), dtype=np.uint8)
    dense[height // 4 : height // 2, width // 4 : width // 2] = 1

    return mask_utils.encode(np.asfortranarray(dense))["counts"]


def test_reduced_dense_masks_can_annotate_original_image() -> None:
    import numpy as np
    import supervision as sv

    mask = torch.zeros((1, 20, 30), dtype=torch.bool)
    mask[:, 5:10, 8:14] = True
    detections = InstanceDetections(
        xyxy=torch.tensor([[8, 5, 14, 10]], dtype=torch.float32),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=mask,
        image_size=(200, 300),
    ).to_supervision()

    scene = np.zeros((200, 300, 3), dtype=np.uint8)
    annotated = sv.MaskAnnotator().annotate(scene=scene, detections=detections)

    assert detections.mask.shape == (1, 200, 300)
    assert detections.mask[0, 50:100, 80:140].all()
    np.testing.assert_allclose(detections.xyxy, [[80, 50, 140, 100]])
    assert detections.mask.sum() == 50 * 60
    assert annotated.any()


def test_origin_crop_is_resized_then_padded_not_stretched() -> None:
    mask = torch.zeros((1, 20, 30), dtype=torch.bool)
    mask[:, 2:10, 4:16] = True
    detections = InstanceDetections(
        xyxy=torch.tensor([[4, 2, 16, 10]], dtype=torch.float32),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=mask,
        image_size=(200, 300),
        mask_frame_size=(100, 150),
    ).to_supervision()

    assert detections.mask.shape == (1, 200, 300)
    assert detections.mask[0, 10:50, 20:80].all()
    np.testing.assert_allclose(detections.xyxy, [[20, 10, 80, 50]])
    assert detections.mask.sum() == 40 * 60


def test_empty_reduced_dense_masks_keep_original_dimensions() -> None:
    detections = InstanceDetections(
        xyxy=torch.empty((0, 4)),
        class_id=torch.empty((0,), dtype=torch.long),
        confidence=torch.empty((0,)),
        mask=torch.empty((0, 20, 30), dtype=torch.bool),
        image_size=(200, 300),
    ).to_supervision()

    assert detections.mask.shape == (0, 200, 300)


@pytest.mark.parametrize("mask_format", ["dense", "rle"])
@pytest.mark.parametrize("grid", [(200, 300), (37, 61), (400, 600)])
@pytest.mark.parametrize("empty", [False, True])
def test_image_boxes_and_masks_share_final_grid(mask_format, grid, empty) -> None:
    image_size = (200, 300)
    count = 0 if empty else 1
    image_boxes = torch.tensor([[61, 43, 142, 97]], dtype=torch.int32)[:count]
    original_boxes = image_boxes.clone()
    dense = torch.zeros((count, *grid), dtype=torch.bool)
    if mask_format == "rle":
        mask = InstancesRLEMasks.from_coco_rle_masks(
            image_size=image_size,
            mask_size=grid,
            masks=[
                mask_utils.encode(np.asfortranarray(single.numpy(), dtype=np.uint8))
                for single in dense
            ],
        )
    else:
        mask = dense

    result = InstanceDetections.from_image_coordinates(
        xyxy=image_boxes,
        class_id=torch.zeros(count, dtype=torch.int32),
        confidence=torch.ones(count),
        mask=mask,
        image_size=image_size,
    )

    scale = torch.tensor([grid[1] / 300, grid[0] / 200] * 2)
    torch.testing.assert_close(result.xyxy.float(), original_boxes.float() * scale)
    torch.testing.assert_close(image_boxes, original_boxes)
    assert result.mask_size == grid
    copied = replace(result)
    torch.testing.assert_close(copied.xyxy, result.xyxy)
    if grid == image_size:
        assert result.xyxy is image_boxes
    else:
        assert result.xyxy.is_floating_point()
    restored = result.to_supervision()
    np.testing.assert_allclose(restored.xyxy, original_boxes.numpy(), atol=1e-5)
    assert restored.mask.shape == (count, *image_size)
    if count:
        torch.testing.assert_close(next(iter(result))[0], result.xyxy[0])


def test_image_box_factory_uses_crop_extent_for_manual_local_masks() -> None:
    result = InstanceDetections.from_image_coordinates(
        xyxy=torch.tensor([[20, 10, 80, 50]], dtype=torch.int32),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=torch.zeros((1, 20, 30), dtype=torch.bool),
        image_size=(200, 300),
        mask_frame_size=(100, 150),
    )

    torch.testing.assert_close(result.xyxy, torch.tensor([[4.0, 2.0, 16.0, 10.0]]))
    np.testing.assert_allclose(result.to_supervision().xyxy, [[20, 10, 80, 50]])
