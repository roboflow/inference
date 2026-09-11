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
from inference.core.workflows.core_steps.models.foundation.easy_ocr.v1 import (
    EasyOCRBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.ocr.v1 import OCRModelBlockV1
from inference.core.workflows.core_steps.models.foundation.yolo_world.v1 import (
    YoloWorldModelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)


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


def _failing_image_batch() -> Batch:
    image = mock.MagicMock(spec=WorkflowImageData)
    image.to_inference_format.side_effect = ValueError("decode failed")
    return Batch(content=[image], indices=[(0,)])


def test_failed_doctr_ocr_image_preparation_leaves_registration_and_inference_uncalled() -> (
    None
):
    # Task 11.12 CR-1 site: ocr/v1.py:173 decodes before load_core_model:177.
    model_manager = mock.MagicMock()
    block = OCRModelBlockV1(
        model_manager=model_manager,
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    images = _failing_image_batch()

    with pytest.raises(ValueError, match="decode failed"):
        block.run(images=images)

    model_manager.add_model.assert_not_called()
    model_manager.run_doctr_ocr.assert_not_called()


def test_failed_easy_ocr_image_preparation_leaves_registration_and_inference_uncalled() -> (
    None
):
    # Task 11.12 CR-1 site: easy_ocr/v1.py:242 decodes before load_core_model:247.
    model_manager = mock.MagicMock()
    block = EasyOCRBlockV1(
        model_manager=model_manager,
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    images = _failing_image_batch()

    with pytest.raises(ValueError, match="decode failed"):
        block.run(images=images, language="English", quantize=False)

    model_manager.add_model.assert_not_called()
    model_manager.run_easy_ocr.assert_not_called()


def test_failed_yolo_world_image_preparation_leaves_registration_and_inference_uncalled() -> (
    None
):
    # Task 11.12 CR-1 site: yolo_world/v1.py:220 decodes before load_core_model:226.
    model_manager = mock.MagicMock()
    block = YoloWorldModelBlockV1(
        model_manager=model_manager,
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    images = _failing_image_batch()

    with pytest.raises(ValueError, match="decode failed"):
        block.run(images=images, class_names=["cat"], version="s", confidence=0.3)

    model_manager.add_model.assert_not_called()
    model_manager.run_yolo_world.assert_not_called()
