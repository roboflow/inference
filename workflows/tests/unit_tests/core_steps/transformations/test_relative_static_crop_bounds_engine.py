import numpy as np
import pytest
from roboflow_workflows.execution_engine.core import ExecutionEngine


@pytest.mark.parametrize(
    "block_type", ["RelativeStaticCrop", "roboflow_core/relative_statoic_crop@v1"]
)
@pytest.mark.parametrize(
    "x_center,first_box,last_box",
    [
        (0.1, (0, 30, 30, 70), (0, 24, 36, 56)),
        (0.5, (30, 30, 70, 70), (36, 24, 84, 56)),
        (0.9, (70, 30, 100, 70), (84, 24, 120, 56)),
    ],
)
def test_relative_crop_keeps_mixed_batch_order_and_aliases(
    block_type: str, x_center: float, first_box: tuple, last_box: tuple
) -> None:
    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": block_type,
                "name": "crop",
                "images": "$inputs.image",
                "x_center": x_center,
                "y_center": 0.5,
                "width": 0.4,
                "height": 0.4,
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "crop", "selector": "$steps.crop.crops"}
        ],
    }
    images = [
        np.arange(100 * 100 * 3, dtype=np.uint32).astype(np.uint8).reshape(100, 100, 3),
        np.full((1, 1, 3), 100, dtype=np.uint8),
        np.arange(80 * 120 * 3, dtype=np.uint32).astype(np.uint8).reshape(80, 120, 3),
    ]
    originals = [image.copy() for image in images]
    engine = ExecutionEngine.init(workflow_definition=definition, init_parameters={})
    try:
        result = engine.run(runtime_parameters={"image": images})
    finally:
        engine.close()

    assert len(result) == 3
    assert result[1]["crop"] is None
    for index, box in [(0, first_box), (2, last_box)]:
        x1, y1, x2, y2 = box
        crop = result[index]["crop"]
        assert crop is not None
        np.testing.assert_array_equal(crop.numpy_image, images[index][y1:y2, x1:x2])
        assert crop.parent_metadata.origin_coordinates.left_top_x == x1
        assert crop.parent_metadata.origin_coordinates.left_top_y == y1
    for image, original in zip(images, originals):
        np.testing.assert_array_equal(image, original)
