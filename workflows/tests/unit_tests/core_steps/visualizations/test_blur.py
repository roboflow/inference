import numpy as np
import pycocotools.mask as mask_utils
import pytest
import supervision as sv
from pydantic import ValidationError
from roboflow_workflows.core_steps.visualizations.blur.v1 import (
    BlurManifest,
    BlurVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.common.annotators.blur import (
    MaskAwareBlurAnnotator,
)
from roboflow_workflows.execution_engine.constants import RLE_MASK_KEY_IN_SV_DETECTIONS
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from supervision.detection.compact_mask import CompactMask


@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/blur_visualization@v1", "BlurVisualization"]
)
@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_blur_validation_when_valid_manifest_is_given(
    type_alias: str, images_field_alias: str
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "blur1",
        "predictions": "$steps.od_model.predictions",
        images_field_alias: "$inputs.image",
        "kernel_size": 5,
    }

    # when
    result = BlurManifest.model_validate(data)

    # then
    assert result == BlurManifest(
        type=type_alias,
        name="blur1",
        images="$inputs.image",
        predictions="$steps.od_model.predictions",
        kernel_size=5,
    )


def test_blur_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "BlurVisualization",
        "name": "blur1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
        "kernel_size": 5,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlurManifest.model_validate(data)


def test_blur_visualization_block() -> None:
    # given
    block = BlurVisualizationBlockV1()

    start_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        predictions=sv.Detections(
            xyxy=np.array(
                [[0, 0, 20, 20], [80, 80, 120, 120], [450, 450, 550, 550]],
                dtype=np.float64,
            ),
            class_id=np.array([1, 1, 1]),
        ),
        copy_image=True,
        kernel_size=5,
    )

    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")

    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (1000, 1000, 3)
    # check if the image is modified
    assert not np.array_equal(output.get("image").numpy_image, start_image)


def _segmentation_predictions(
    image_size: int, box: tuple, mask_region: tuple
) -> sv.Detections:
    x1, y1, x2, y2 = box
    mask = np.zeros((1, image_size, image_size), dtype=bool)
    mx1, my1, mx2, my2 = mask_region
    mask[0, my1:my2, mx1:mx2] = True
    return sv.Detections(
        xyxy=np.array([[x1, y1, x2, y2]], dtype=np.float64),
        mask=mask,
        class_id=np.array([0]),
    )


def test_blur_visualization_block_blurs_only_inside_segmentation_mask() -> None:
    # given
    block = BlurVisualizationBlockV1()
    start_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    # the mask covers only the left half of the bounding box
    predictions = _segmentation_predictions(
        image_size=200, box=(50, 50, 150, 150), mask_region=(50, 50, 100, 150)
    )

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        predictions=predictions,
        copy_image=True,
        kernel_size=15,
    )

    # then
    result = output["image"].numpy_image
    mask = predictions.mask[0]
    assert np.array_equal(
        result[~mask], start_image[~mask]
    ), "pixels outside the mask, including the rest of the box, must stay sharp"
    assert not np.array_equal(
        result[mask], start_image[mask]
    ), "pixels inside the mask must be blurred"


def test_blur_visualization_block_blurs_whole_box_without_mask() -> None:
    # given
    block = BlurVisualizationBlockV1()
    start_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    predictions = sv.Detections(
        xyxy=np.array([[50, 50, 150, 150]], dtype=np.float64),
        class_id=np.array([0]),
    )

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        predictions=predictions,
        copy_image=True,
        kernel_size=15,
    )

    # then
    expected = sv.BlurAnnotator(kernel_size=15).annotate(
        scene=start_image.copy(), detections=predictions
    )
    assert np.array_equal(output["image"].numpy_image, expected)


def test_blur_visualization_block_decodes_rle_masks() -> None:
    # given
    block = BlurVisualizationBlockV1()
    start_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    dense = _segmentation_predictions(
        image_size=200, box=(50, 50, 150, 150), mask_region=(50, 50, 100, 150)
    )
    rle = sv.Detections(
        xyxy=dense.xyxy,
        class_id=dense.class_id,
        data={
            RLE_MASK_KEY_IN_SV_DETECTIONS: np.array(
                [mask_utils.encode(np.asfortranarray(dense.mask[0]))], dtype=object
            )
        },
    )

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=start_image,
        ),
        predictions=rle,
        copy_image=True,
        kernel_size=15,
    )

    # then
    mask = dense.mask[0]
    result = output["image"].numpy_image
    assert np.array_equal(result[~mask], start_image[~mask])
    assert not np.array_equal(result[mask], start_image[mask])


def _run_blur(image: np.ndarray, predictions: sv.Detections, padding: int = 0):
    return (
        BlurVisualizationBlockV1()
        .run(
            image=WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="some"),
                numpy_image=image,
            ),
            predictions=predictions,
            copy_image=True,
            kernel_size=15,
            padding=padding,
        )["image"]
        .numpy_image
    )


def test_blur_visualization_block_blurs_mask_pixels_outside_the_box() -> None:
    # given: the mask reaches 20 pixels past the right edge of its box, as
    # segmentation models that predict boxes and masks separately can produce
    start_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    predictions = _segmentation_predictions(
        image_size=200, box=(50, 50, 150, 150), mask_region=(50, 50, 170, 150)
    )

    # when
    result = _run_blur(start_image, predictions)

    # then
    past_box = predictions.mask[0].copy()
    past_box[:, :150] = False
    assert past_box.sum() == 2000
    changed = (result[past_box] != start_image[past_box]).any(axis=1)
    assert changed.mean() > 0.9, "mask pixels past the box must be blurred"
    outside = ~predictions.mask[0]
    assert np.array_equal(result[outside], start_image[outside])


def test_blur_visualization_block_pads_boxes() -> None:
    # given
    start_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    predictions = sv.Detections(
        xyxy=np.array([[50, 50, 150, 150]], dtype=np.float64),
        class_id=np.array([0]),
    )

    # when
    result = _run_blur(start_image, predictions, padding=10)

    # then
    expected = sv.BlurAnnotator(kernel_size=15).annotate(
        scene=start_image.copy(),
        detections=sv.Detections(xyxy=np.array([[40, 40, 160, 160]], dtype=np.float64)),
    )
    assert np.array_equal(result, expected)


def test_blur_visualization_block_pads_masks_outward() -> None:
    # given
    start_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    predictions = _segmentation_predictions(
        image_size=200, box=(50, 50, 150, 150), mask_region=(80, 80, 120, 120)
    )

    # when
    result = _run_blur(start_image, predictions, padding=5)

    # then
    changed = (result != start_image).any(axis=2)
    assert changed[100, 77], "pixels within the padding must be blurred"
    assert not changed[100, 70], "pixels beyond the padding must stay sharp"
    assert not changed[60, 60], "box pixels away from the mask must stay sharp"


def test_mask_aware_blur_reads_compact_masks_as_crops(monkeypatch) -> None:
    # given
    start_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    dense = _segmentation_predictions(
        image_size=200, box=(50, 50, 150, 150), mask_region=(60, 70, 140, 130)
    )
    compact = sv.Detections(
        xyxy=dense.xyxy,
        mask=CompactMask.from_dense(dense.mask, dense.xyxy, image_shape=(200, 200)),
        class_id=dense.class_id,
    )

    def _full_frame_read(self):
        raise AssertionError("CompactMask must not be expanded to full frames")

    monkeypatch.setattr(CompactMask, "__iter__", _full_frame_read)
    monkeypatch.setattr(CompactMask, "__array__", _full_frame_read)

    # when
    from_compact = MaskAwareBlurAnnotator(kernel_size=15).annotate(
        scene=start_image.copy(), detections=compact
    )
    from_dense = MaskAwareBlurAnnotator(kernel_size=15).annotate(
        scene=start_image.copy(), detections=dense
    )

    # then
    assert np.array_equal(from_compact, from_dense)


def test_mask_aware_blur_rejects_negative_padding() -> None:
    # when / then
    with pytest.raises(ValueError):
        MaskAwareBlurAnnotator(kernel_size=15, padding=-1)
