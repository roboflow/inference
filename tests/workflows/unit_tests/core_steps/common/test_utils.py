import numpy as np

from inference.core.workflows.core_steps.common.utils import (
    extract_origin_size_from_images_batch,
)


def test_extract_origin_size_from_images() -> None:
    # given
    input_images = [
        {
            "type": "url",
            "value": "https://some/image.jpg",
            "origin_coordinates": {
                "origin_image_size": {"height": 1000, "width": 1000}
            },
        },
        {
            "type": "numpy_object",
            "value": np.zeros((192, 168, 3), dtype=np.uint8),
        },
    ]
    decoded_image = [
        np.zeros((200, 200, 3), dtype=np.uint8),
        np.zeros((192, 168, 3), dtype=np.uint8),
    ]

    # when
    result = extract_origin_size_from_images_batch(
        input_images=input_images,
        decoded_images=decoded_image,
    )

    # then
    assert result == [{"height": 1000, "width": 1000}, {"height": 192, "width": 168}]
