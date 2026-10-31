import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.foundation.clip import v1 as clip_v1
from roboflow_workflows.core_steps.models.foundation.clip import (
    v1_tensor as clip_v1_tensor,
)
from roboflow_workflows.core_steps.models.foundation.clip_comparison import (
    v1 as clip_comparison_v1,
)
from roboflow_workflows.core_steps.models.foundation.clip_comparison import (
    v1_tensor as clip_comparison_v1_tensor,
)
from roboflow_workflows.core_steps.models.foundation.clip_comparison import (
    v2 as clip_comparison_v2,
)
from roboflow_workflows.core_steps.models.foundation.clip_comparison import (
    v2_tensor as clip_comparison_v2_tensor,
)
from roboflow_workflows.errors import RuntimeInputError
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

from inference_models.errors import ModelInputError

TEXT_TOO_LONG_ERROR = ModelInputError(
    message="Text input is too long for the model context length. "
    "Shorten the text and retry."
)
OTHER_MODEL_INPUT_ERROR = ModelInputError(message="Unsupported input type.")


def _image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((32, 32, 3), dtype=np.uint8),
    )


# (block class, run_locally kwargs factory) - texts are unique to bypass the text cache
BLOCKS = [
    (
        clip_v1.ClipModelBlockV1,
        lambda: {"data": f"text-{uuid.uuid4()}", "version": "ViT-B-16"},
    ),
    (
        clip_v1_tensor.ClipModelBlockV1,
        lambda: {"data": f"text-{uuid.uuid4()}", "version": "ViT-B-16"},
    ),
    (
        clip_comparison_v1.ClipComparisonBlockV1,
        lambda: {"images": [_image()], "texts": ["cat"]},
    ),
    (
        clip_comparison_v1_tensor.ClipComparisonBlockV1,
        lambda: {"images": [_image()], "texts": ["cat"]},
    ),
    (
        clip_comparison_v2.ClipComparisonBlockV2,
        lambda: {"images": [_image()], "classes": ["cat"], "version": "ViT-B-16"},
    ),
    (
        clip_comparison_v2_tensor.ClipComparisonBlockV2,
        lambda: {"images": [_image()], "classes": ["cat"], "version": "ViT-B-16"},
    ),
]


def _block_with_failing_model(block_class, error: Exception):
    model_manager = MagicMock()
    for method in (
        "run_clip_text_embedding",
        "run_clip_image_embedding",
        "run_clip_comparison",
        "run_tensor_native_inference",
    ):
        getattr(model_manager, method).side_effect = error
    return block_class(
        model_manager=model_manager,
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )


@pytest.mark.parametrize("block_class, kwargs_factory", BLOCKS)
def test_clip_block_when_text_exceeds_context_length(
    block_class, kwargs_factory
) -> None:
    # given
    block = _block_with_failing_model(block_class, TEXT_TOO_LONG_ERROR)

    # when
    with pytest.raises(RuntimeInputError) as error:
        block.run_locally(**kwargs_factory())

    # then
    assert error.value.inner_error is TEXT_TOO_LONG_ERROR


@pytest.mark.parametrize("block_class, kwargs_factory", BLOCKS)
def test_clip_block_when_other_model_input_error_occurs(
    block_class, kwargs_factory
) -> None:
    # given
    block = _block_with_failing_model(block_class, OTHER_MODEL_INPUT_ERROR)

    # when
    with pytest.raises(ModelInputError) as error:
        block.run_locally(**kwargs_factory())

    # then
    assert error.value is OTHER_MODEL_INPUT_ERROR
