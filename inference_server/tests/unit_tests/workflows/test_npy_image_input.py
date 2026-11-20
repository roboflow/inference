import base64
import io
import pickle

import numpy as np
import pytest


@pytest.fixture
def codec():
    from inference_server.workflows.host import ServerImageCodec

    return ServerImageCodec()


@pytest.mark.parametrize("tensor_mode", [False, True])
@pytest.mark.parametrize("shape", [(5, 7, 3), (5, 7), (5, 7, 1)])
def test_workflow_npy_preserves_pixels_and_metadata(codec, tensor_mode, shape):
    from roboflow_workflows.core_steps.common import deserializers, deserializers_tensor

    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    buffer = io.BytesIO()
    np.save(buffer, image, allow_pickle=False)
    deserializer = deserializers_tensor if tensor_mode else deserializers

    result = deserializer.deserialize_image_kind(
        parameter="image",
        image={
            "type": "NPY",
            "value": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "parent_id": "source-frame",
        },
        image_codec=codec,
    )

    expected = image
    if len(shape) == 2:
        expected = np.repeat(image[..., None], 3, axis=2)
    elif shape[2] == 1:
        expected = np.repeat(image, 3, axis=2)
    np.testing.assert_array_equal(result.numpy_image, expected)
    assert result.parent_metadata.parent_id == "source-frame"
    assert base64.b64decode(result.base64_image).startswith(b"\xff\xd8")


def test_workflow_npy_maps_invalid_input_to_workflow_error(codec):
    from roboflow_workflows.core_steps.common import deserializers
    from roboflow_workflows.errors import RuntimeInputError, WorkflowImageLoadError

    image = {
        "type": "npy",
        "value": base64.b64encode(
            pickle.dumps(np.zeros((5, 7, 3), dtype=np.uint8))
        ).decode(),
    }

    with pytest.raises(RuntimeInputError) as error:
        deserializers.deserialize_image_kind(
            parameter="image", image=image, image_codec=codec
        )

    assert isinstance(error.value.__cause__, WorkflowImageLoadError)
