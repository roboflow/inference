import torch

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


class TestMaskSize:
    """The grid masks actually live on, carried explicitly.

    `mask.shape[1:]` and `InstancesRLEMasks.image_size` are currently the de
    facto definition of the mask canvas at roughly fifteen call sites. Once the
    resize target becomes adjustable they stop agreeing with the image, so the
    grid has to travel with the prediction rather than be inferred from it.

    Provisional: carrying this as an explicit field rather than a key in the
    image metadata is a recommendation pending maintainer ratification.
    """

    def test_rle_mask_size_defaults_to_image_size(self) -> None:
        # given / when
        masks = InstancesRLEMasks(image_size=(1080, 1920), masks=[])

        # then
        assert masks.mask_size == (1080, 1920)

    def test_rle_mask_size_can_differ_from_image_size(self) -> None:
        # given / when
        masks = InstancesRLEMasks(
            image_size=(1080, 1920), masks=[], mask_size=(270, 480)
        )

        # then
        assert masks.mask_size == (270, 480)
        assert masks.image_size == (1080, 1920)

    def test_wire_format_declares_the_encoded_size(self) -> None:
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

    def test_wire_format_unchanged_when_sizes_agree(self) -> None:
        # given
        masks = InstancesRLEMasks(image_size=(1080, 1920), masks=[b"abc"])

        # when
        encoded = masks.to_coco_rle_masks()

        # then
        assert encoded == [{"size": [1080, 1920], "counts": b"abc"}]

    def test_dense_mask_size_falls_back_to_the_tensor_grid(self) -> None:
        # given
        detections = InstanceDetections(
            xyxy=torch.zeros((2, 4)),
            class_id=torch.zeros((2,), dtype=torch.int64),
            confidence=torch.zeros((2,)),
            mask=torch.zeros((2, 540, 960), dtype=torch.bool),
        )

        # then
        assert detections.mask_size == (540, 960)

    def test_dense_mask_size_honours_an_explicit_value(self) -> None:
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

    def test_rle_carrier_reports_its_own_grid(self) -> None:
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

    def test_empty_detection_stack_still_reports_the_grid(self) -> None:
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


class TestReducedGridLeavesTheBoundaryAtImageSize:
    """`sv.Detections` and the COCO dict must describe the image, not the grid.

    `sv.Detections.mask` is documented as `(n, H, W)` matching the image and
    its annotators index the scene with it, so a reduced grid raises. The COCO
    dict is the opposite: `size` must describe the grid the counts were encoded
    on, or decoding reinterprets the runs.
    """

    def test_to_supervision_restores_the_image_grid(self) -> None:
        # given
        # masks produced at a quarter of the image resolution
        detections = InstanceDetections(
            xyxy=torch.tensor([[0, 0, 80, 80]], dtype=torch.float32),
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

    def test_supervision_annotator_accepts_the_result(self) -> None:
        # given
        # the reported failure was an IndexError from boolean-index mismatch
        import numpy as np
        import supervision as sv

        scene = np.zeros((80, 80, 3), dtype=np.uint8)
        detections = InstanceDetections(
            xyxy=torch.tensor([[0, 0, 80, 80]], dtype=torch.float32),
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

    def test_iteration_declares_the_encoded_grid(self) -> None:
        # given
        detections = InstanceDetections(
            xyxy=torch.tensor([[0, 0, 80, 80]], dtype=torch.float32),
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
