"""Public contract of the annotation blocks (decision 027).

read_rows        one packed host transfer; exact int64 class ids, own
                 box dtype; works on MPS (no float64 on the device)
placed_in        another or an unnamed workflow input -> error + remedy
mask_quality     accurate | coarse; literals fail at compile, selected
                 values at run with one message, also on box-only paths
copy_image       rejected with either value; error says delete + chain
"""

from typing import Any

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BlurVisualizationBlock,
    BoundingBoxVisualizationBlock,
    CropVisualizationBlock,
    LabelVisualizationBlock,
    MaskVisualizationBlock,
    PixelateVisualizationBlock,
    PolygonVisualizationBlock,
    SemanticSegmentationVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from inference_models.models.base.object_detection import Detections
from tests.v2.unit.blocks.visualization.annotations.support import (
    detections,
    image_of,
    instance_detections,
    pixels_of,
    run_v2,
    scene,
    semantic,
)

MASK_READERS = (
    MaskVisualizationBlock,
    PolygonVisualizationBlock,
    BlurVisualizationBlock,
    CropVisualizationBlock,
    LabelVisualizationBlock,
)
QUALITY_MESSAGE = "mask_quality must be one of ['accurate', 'coarse']"
HAS_MPS = torch.backends.mps.is_available()


def rows_with_class_ids(class_ids: list, *, device: str = "cpu") -> Detections:
    count = len(class_ids)
    boxes = Detections(
        xyxy=torch.arange(count * 4, dtype=torch.float32).reshape(count, 4).to(device),
        class_id=torch.tensor(class_ids, dtype=torch.int64).to(device),
        confidence=torch.full((count,), 0.333, dtype=torch.float32).to(device),
        image_metadata={},
    )

    return boxes


def validate(block_class: type, **parameters: Any) -> Any:
    params = spec_of(block_class).validate_params(
        {"image": "$inputs.image", "predictions": "$inputs.predictions", **parameters}
    )

    return params


def run_with_selected_quality(
    block_class: type, *, image: ImageData, predictions: Any, quality: Any
) -> Any:
    # Compile with a selector, then run with the value it delivers.
    params = validate(block_class, mask_quality="$inputs.mask_quality")
    arguments = {
        **dict(params),
        "image": image,
        "predictions": predictions,
        "mask_quality": quality,
    }
    result = block_class().run(**arguments)

    return result


def unnamed_root_image(pixels: np.ndarray, *, image_id: str) -> ImageData:
    height, width = pixels.shape[:2]
    image = ImageData(
        tensor_image=torch.from_numpy(pixels).permute(2, 0, 1).contiguous(),
        image_id=image_id,
        parent=FrameMapping(frame_id=None, frame_size_hw=(height, width)),
        root=FrameMapping(frame_id=None, frame_size_hw=(height, width)),
    )

    return image


def test_read_rows_keeps_class_ids_above_2_53_and_int64_extremes() -> None:
    # given
    extremes = [
        2**53 + 1,
        2**62 + 3,
        np.iinfo(np.int64).max,
        np.iinfo(np.int64).min,
        -1,
        0,
    ]
    boxes = rows_with_class_ids(extremes)

    # when
    rows = read_rows(boxes)

    # then
    assert rows.class_id.dtype == np.int64
    assert rows.class_id.tolist() == extremes
    assert rows.xyxy.dtype == np.float32
    assert np.array_equal(rows.xyxy, boxes.xyxy.numpy())
    assert rows.confidence.dtype == np.float64
    assert rows.confidence.tolist() == [float(np.float32(0.333))] * len(extremes)


def test_read_rows_keeps_float64_boxes_and_small_integer_class_ids() -> None:
    # given
    boxes = Detections(
        xyxy=torch.tensor([[0.1, 0.2, 1e15 + 0.5, 3.0]], dtype=torch.float64),
        class_id=torch.tensor([7], dtype=torch.int32),
        confidence=torch.tensor([0.25], dtype=torch.float16),
        image_metadata={},
    )

    # when
    rows = read_rows(boxes)

    # then
    assert rows.xyxy.dtype == np.float64
    assert rows.xyxy.tolist() == [[0.1, 0.2, 1e15 + 0.5, 3.0]]
    assert rows.class_id.tolist() == [7]
    assert rows.confidence.tolist() == [0.25]


def test_read_rows_of_no_rows_is_empty() -> None:
    # when
    rows = read_rows(rows_with_class_ids([]))

    # then
    assert len(rows) == 0
    assert rows.xyxy.shape == (0, 4)


@pytest.mark.skipif(not HAS_MPS, reason="needs a local MPS device")
def test_read_rows_on_mps_matches_cpu() -> None:
    # given
    class_ids = [2**53 + 1, np.iinfo(np.int64).max, 3]

    # when
    on_mps = read_rows(rows_with_class_ids(class_ids, device="mps"))
    on_cpu = read_rows(rows_with_class_ids(class_ids))

    # then
    assert on_mps.class_id.tolist() == class_ids
    assert np.array_equal(on_mps.xyxy, on_cpu.xyxy)
    assert np.array_equal(on_mps.confidence, on_cpu.confidence)


@pytest.mark.skipif(not HAS_MPS, reason="needs a local MPS device")
@pytest.mark.parametrize(
    "block_class, parameters",
    [
        (LabelVisualizationBlock, {"text": "Area", "text_position": "CENTER_OF_MASS"}),
        (BlurVisualizationBlock, {"kernel_size": 5}),
        (PixelateVisualizationBlock, {"pixel_size": 80}),
        (MaskVisualizationBlock, {}),
    ],
)
def test_mask_and_raster_blocks_on_mps_match_cpu(
    block_class: type, parameters: dict
) -> None:
    # given
    cpu_image = image_of(scene())
    mps_image = cpu_image.with_pixels(cpu_image.tensor_image.to("mps"))

    # when
    on_cpu = run_v2(
        block_class,
        image=cpu_image,
        predictions=instance_detections(cpu_image),
        **parameters,
    )
    on_mps = run_v2(
        block_class,
        image=mps_image,
        predictions=instance_detections(mps_image, device="mps"),
        **parameters,
    )

    # then
    assert on_mps.device.type == "mps"
    assert np.array_equal(pixels_of(on_mps), pixels_of(on_cpu))


def test_predictions_of_another_workflow_input_are_rejected_with_a_remedy() -> None:
    # given
    image = image_of(scene(), image_id="frame-a")
    other = image_of(scene(), image_id="frame-b")

    # when
    with pytest.raises(ValueError) as error:
        placed_in(detections(other), image, block="test")

    # then
    assert "'frame-b'" in str(error.value) and "'frame-a'" in str(error.value)
    assert "Draw on the image the predictions were made on" in str(error.value)


def test_unnamed_roots_do_not_match_each_other() -> None:
    # given
    pixels = scene()
    image = unnamed_root_image(pixels, image_id="drawn")
    predicted = unnamed_root_image(pixels, image_id="predicted")
    boxes = detections(image)
    boxes = Detections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        image_metadata=FrameRef.from_image(predicted).prediction_metadata(),
    )

    # when
    with pytest.raises(ValueError) as error:
        placed_in(boxes, image, block="test")

    # then
    assert "input None" in str(error.value)


def test_same_image_with_unnamed_root_is_drawn_as_given() -> None:
    # given
    image = unnamed_root_image(scene(), image_id="drawn")
    boxes = detections(image)

    # when
    placed = placed_in(boxes, image, block="test")

    # then
    assert placed is boxes


def test_semantic_grid_of_a_resized_image_names_the_image_to_draw_on() -> None:
    # given
    image = image_of(scene())
    resized = image.resize((24, 32), image_id="resized-for-model")

    # when
    with pytest.raises(ValueError) as error:
        run_v2(
            SemanticSegmentationVisualizationBlock,
            image=image,
            predictions=semantic(resized),
        )

    # then
    assert "'resized-for-model'" in str(error.value)
    assert "Draw a semantic segmentation on the image the model saw" in str(error.value)


@pytest.mark.parametrize("block_class", MASK_READERS)
def test_mask_quality_literals_are_checked_at_compile(block_class: type) -> None:
    # given
    accepted = [
        validate(block_class, mask_quality=value) for value in ("accurate", "coarse")
    ]

    # then
    assert [params.mask_quality for params in accepted] == ["accurate", "coarse"]
    assert validate(block_class).mask_quality == "accurate"
    for rejected in ({"mask_quality": "reference"}, {"mask_fidelity": "reference"}):
        with pytest.raises(ParamsValidationError):
            validate(block_class, **rejected)


@pytest.mark.parametrize(
    "block_class, parameters",
    [
        (MaskVisualizationBlock, {}),
        (PolygonVisualizationBlock, {}),
        (BlurVisualizationBlock, {}),
        (CropVisualizationBlock, {}),
        (LabelVisualizationBlock, {}),
        (BlurVisualizationBlock, {"boxes_only": True}),
        (CropVisualizationBlock, {"boxes_only": True}),
        (LabelVisualizationBlock, {"boxes_only": True}),
    ],
)
def test_selected_mask_quality_is_checked_at_run_on_every_path(
    block_class: type, parameters: dict
) -> None:
    # given
    image = image_of(scene())
    boxes_only = parameters.pop("boxes_only", False)
    predictions = detections(image) if boxes_only else instance_detections(image)

    # when
    with pytest.raises(ValueError) as error:
        run_with_selected_quality(
            block_class, image=image, predictions=predictions, quality="fast"
        )

    # then
    assert QUALITY_MESSAGE in str(error.value)


@pytest.mark.parametrize(
    "block_class", (MaskVisualizationBlock, PolygonVisualizationBlock)
)
def test_accurate_and_coarse_are_equal_for_dense_masks(block_class: type) -> None:
    # given
    image = image_of(scene())
    predictions = instance_detections(image)

    # when
    accurate = run_v2(
        block_class, image=image, predictions=predictions, mask_quality="accurate"
    )
    coarse = run_v2(
        block_class, image=image, predictions=predictions, mask_quality="coarse"
    )

    # then
    assert np.array_equal(pixels_of(accurate), pixels_of(coarse))


@pytest.mark.parametrize("block_class", MASK_READERS)
def test_mask_quality_description_states_cost_and_current_no_op(
    block_class: type,
) -> None:
    # given
    field = spec_of(block_class).params_model.model_fields["mask_quality"]

    # then
    assert "one full-image mask per detection, at either quality" in field.description
    assert "blocky" in field.description
    assert "v2/instance_segmentation@v1 currently returns full-resolution" in (
        field.description
    )
    assert "same result" in field.description


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize(
    "block_class", (BoundingBoxVisualizationBlock, MaskVisualizationBlock)
)
def test_copy_image_is_rejected_with_delete_and_chain_remedy(
    block_class: type, value: bool
) -> None:
    # when
    with pytest.raises(ParamsValidationError) as error:
        validate(block_class, copy_image=value)

    # then
    message = str(error.value) + str(error.value.__cause__)
    assert "delete the field" in message
    assert "$steps.<previous>.image" in message
