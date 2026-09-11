"""CR-1 regression: the block must decode the image BEFORE registering the
model with `load_core_model`, matching the pre-port (BASE) ordering. If a
future edit moves the decode back after registration, `add_model` fires
before the `ValueError` from a failed decode, and this test catches it.

One case per task; Task 11.12 and 11.14 add their own cases here.
"""

from unittest import mock

import pytest

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.clip.v1 import (
    ClipModelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def test_failed_image_preparation_leaves_registration_and_inference_uncalled() -> None:
    # given
    model_manager = mock.MagicMock()
    block = ClipModelBlockV1(
        model_manager=model_manager,
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    image = mock.MagicMock(spec=WorkflowImageData)
    image.to_inference_format.side_effect = ValueError("decode failed")

    # when
    with pytest.raises(ValueError, match="decode failed"):
        block.run(data=image, version="RN50")

    # then — HEAD parity: nothing registered, nothing inferred
    model_manager.add_model.assert_not_called()
    model_manager.run_clip_image_embedding.assert_not_called()
