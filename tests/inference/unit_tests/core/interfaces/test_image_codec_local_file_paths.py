"""ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM applies to all three gated paths.

Before Phase 10 the flag only covered `image_utils.load_image`'s declared-FILE
and inferred-path branches; `WorkflowImageData`'s two decoders and the runtime
deserializer read local paths without consulting it. Release note 1.
"""

from unittest import mock

import cv2
import numpy as np
import pytest

from inference.core.exceptions import InputImageLoadError
from inference.core.interfaces.workflows_image_codec import install_guarded_image_codec
from inference.core.utils import image_utils
from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_image_kind,
)
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.image_codec import reset_image_codec


@pytest.fixture()
def image_file(tmp_path) -> str:
    path = str(tmp_path / "source.png")
    assert cv2.imwrite(path, np.zeros((4, 6, 3), dtype=np.uint8))
    return path


@pytest.fixture(autouse=True)
def _guarded_codec():
    reset_image_codec()
    install_guarded_image_codec()
    yield
    reset_image_codec()


@mock.patch.object(image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", True)
def test_all_three_paths_read_the_file_when_the_flag_is_on(image_file: str) -> None:
    numpy_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"), image_reference=image_file
    )
    assert numpy_born.numpy_image.shape == (4, 6, 3)

    tensor_born = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"), image_reference=image_file
    )
    assert tuple(tensor_born.tensor_image.shape) == (3, 4, 6)

    deserialized = deserialize_image_kind("image", image_file)
    assert deserialized.numpy_image.shape == (4, 6, 3)


@mock.patch.object(image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", False)
def test_numpy_path_is_refused_when_the_flag_is_off(image_file: str) -> None:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"), image_reference=image_file
    )
    with pytest.raises(InputImageLoadError, match="local filesystem"):
        _ = image.numpy_image


@mock.patch.object(image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", False)
def test_tensor_path_is_refused_when_the_flag_is_off(image_file: str) -> None:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"), image_reference=image_file
    )
    with pytest.raises(InputImageLoadError, match="local filesystem"):
        _ = image.tensor_image


@mock.patch.object(image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", False)
def test_deserializer_path_is_refused_when_the_flag_is_off(image_file: str) -> None:
    # deserialize_image_kind wraps every failure in RuntimeInputError; the cause
    # is the local-filesystem refusal.
    with pytest.raises(RuntimeInputError) as error:
        deserialize_image_kind("image", image_file)
    assert "local filesystem" in str(error.value)
