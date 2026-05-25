import numpy as np
import torch

from inference.core.workflows.core_steps.common.entities import StepExecutionMode  # noqa: F401
from inference.core.workflows.core_steps.transformations.absolute_static_crop.v1_tensor import (
    AbsoluteStaticCropBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)


def _image(h: int = 64, w: int = 64) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="p",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0, left_top_y=0, origin_width=w, origin_height=h
            ),
        ),
        numpy_image=np.zeros((h, w, 3), dtype=np.uint8),
    )


def test_absolute_static_crop_tensor_slices_tensor_image() -> None:
    image = _image(h=32, w=32)
    # Mark a pixel in the source tensor so we can check the slice picks it up.
    image._tensor_image = torch.zeros((32, 32, 3), dtype=torch.uint8)
    image._tensor_image[10, 10] = torch.tensor([7, 7, 7], dtype=torch.uint8)
    image._numpy_image = None  # force tensor path

    block = AbsoluteStaticCropBlockV1()
    result = block.run(
        images=Batch(content=[image], indices=[(0,)]),
        x_center=10, y_center=10, width=4, height=4,
    )

    crop = result[0]["crops"]
    assert crop is not None
    # Slice should be (4, 4, 3) and contain the marked pixel.
    assert crop.tensor_image.shape == (4, 4, 3)
    # Pixel (10,10) of source is inside the slice [y_min=8:y_max=12, x_min=8:x_max=12]
    # at local offset (2,2)
    assert int(crop.tensor_image[2, 2, 0].item()) == 7


def test_absolute_static_crop_tensor_returns_none_for_empty_crop() -> None:
    image = _image(h=32, w=32)
    image._tensor_image = torch.zeros((32, 32, 3), dtype=torch.uint8)
    image._numpy_image = None

    block = AbsoluteStaticCropBlockV1()
    # Coordinates entirely outside image -> empty slice
    result = block.run(
        images=Batch(content=[image], indices=[(0,)]),
        x_center=500, y_center=500, width=10, height=10,
    )
    assert result[0]["crops"] is None
