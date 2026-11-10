"""The compatibility bootstrap installs the tensor mode the environment asks for."""

import os

from roboflow_workflows import environment
from roboflow_workflows.configuration import get_configuration


def test_tensor_mode_follows_the_environment() -> None:
    requested = (
        os.environ.get("ENABLE_TENSOR_DATA_REPRESENTATION", "False").lower() == "true"
    )
    tensor = get_configuration().tensor

    assert tensor.representation_enabled is requested
    assert environment.ENABLE_TENSOR_DATA_REPRESENTATION is requested
    assert (tensor.image_tensor_device is None) is (not requested)
