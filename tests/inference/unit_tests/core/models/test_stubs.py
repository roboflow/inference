from typing import Type
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.entities.requests.inference import (
    CVInferenceRequest,
    InferenceRequest,
)
from inference.core.entities.responses.inference import StubResponse
from inference.core.models.stubs import (
    ClassificationModelStub,
    InstanceSegmentationModelStub,
    KeypointsDetectionModelStub,
    ModelStub,
    ObjectDetectionModelStub,
)


@pytest.mark.parametrize(
    "stub_class, expected_task_type",
    [
        (ClassificationModelStub, "classification"),
        (ObjectDetectionModelStub, "object-detection"),
        (InstanceSegmentationModelStub, "instance-segmentation"),
        (KeypointsDetectionModelStub, "keypoint-detection"),
    ],
)
def test_model_stub(stub_class: Type[ModelStub], expected_task_type: str) -> None:
    # given
    model = stub_class(model_id="some/0", api_key="my_api_key")
    request = MagicMock()
    request.visualize_predictions = (True,)
    request.image = np.zeros((128, 128, 3), dtype=np.uint8)

    # when
    result = model.infer_from_request(request=request)

    # then
    assert type(result) == StubResponse
    assert result.is_stub is True
    assert result.model_id == "some/0"
    assert result.task_type == expected_task_type
    assert getattr(result, "visualization", None) is not None
