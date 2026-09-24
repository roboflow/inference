import numpy as np
import pycocotools.mask as mask_utils
import pytest
import supervision as sv
from pydantic import ValidationError
from roboflow_workflows.core_steps.visualizations.blur.v1 import (
    BlurManifest,
    BlurVisualizationBlockV1,
)
from roboflow_workflows.execution_engine.constants import RLE_MASK_KEY_IN_SV_DETECTIONS
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


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
