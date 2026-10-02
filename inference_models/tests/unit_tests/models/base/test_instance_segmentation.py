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
