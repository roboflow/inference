from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.pp_ocr.v1 import (
    BlockManifest,
    PPOCRBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_pp_ocr_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/pp_ocr@v1",
        "name": "some",
        images_field_alias: "$steps.crop.crops",
        "text_detection": "medium",
        "text_recognition": "tiny",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="roboflow_core/pp_ocr@v1",
        name="some",
        images="$steps.crop.crops",
        text_detection="medium",
        text_recognition="tiny",
    )


def test_pp_ocr_validation_when_versions_default_to_small() -> None:
    # given
    data = {
        "type": "roboflow_core/pp_ocr@v1",
        "name": "some",
        "images": "$steps.crop.crops",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.text_detection == "small"
    assert result.text_recognition == "small"


def test_pp_ocr_validation_when_both_stages_none() -> None:
    # given
    data = {
        "type": "roboflow_core/pp_ocr@v1",
        "name": "some",
        "images": "$steps.crop.crops",
        "text_detection": "none",
        "text_recognition": "none",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_pp_ocr_validation_when_invalid_version_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/pp_ocr@v1",
        "name": "some",
        "images": "$steps.crop.crops",
        "text_detection": "large",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_pp_ocr_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/pp_ocr@v1",
        "name": "some",
        "images": "invalid",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_pp_ocr_describe_outputs() -> None:
    # when
    outputs = BlockManifest.describe_outputs()

    # then
    assert {o.name for o in outputs} == {
        "result",
        "predictions",
        "parent_id",
        "root_parent_id",
        "prediction_type",
    }


def test_pp_ocr_supported_model_variants() -> None:
    # when
    variants = BlockManifest.get_supported_model_variants()

    # then
    assert variants[0] == "pp_ocr/small-small"
    assert set(variants) == {
        "pp_ocr/small-small",
        "pp_ocr/tiny-tiny",
        "pp_ocr/medium-medium",
        "pp_ocr/small-none",
        "pp_ocr/none-small",
    }


def _make_block(model_manager: MagicMock) -> PPOCRBlockV1:
    return PPOCRBlockV1(
        model_manager=model_manager,
        api_key="my-api-key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )


def _make_images() -> Batch:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )
    return Batch(content=[image], indices=[(0,)])


def test_pp_ocr_run_locally_full_mode() -> None:
    # given
    model_manager = MagicMock()
    model_manager.infer_from_request_sync.return_value = MagicMock(
        model_dump=MagicMock(
            return_value={
                "result": "HELLO",
                "image": {"width": 168, "height": 192},
                "predictions": [
                    {
                        "x": 15.0,
                        "y": 10.0,
                        "width": 30.0,
                        "height": 20.0,
                        "confidence": 0.9,
                        "class": "HELLO",
                        "class_id": 0,
                    }
                ],
            }
        )
    )
    block = _make_block(model_manager)
    images = _make_images()

    # when
    result = block.run(
        images=images, text_detection="small", text_recognition="small"
    )

    # then
    model_manager.infer_from_request_sync.assert_called_once()
    assert len(result) == 1
    single = result[0]
    assert single["result"] == "HELLO"
    assert single["prediction_type"] == "ocr"
    assert single["parent_id"] == "parent"
    assert single["root_parent_id"] == "parent"
    assert len(single["predictions"]) == 1
    assert set(single.keys()) == {
        "result",
        "predictions",
        "parent_id",
        "root_parent_id",
        "prediction_type",
    }


def test_pp_ocr_run_locally_detect_only() -> None:
    # given
    model_manager = MagicMock()
    model_manager.infer_from_request_sync.return_value = MagicMock(
        model_dump=MagicMock(
            return_value={
                "result": "",
                "image": {"width": 168, "height": 192},
                "predictions": [
                    {
                        "x": 15.0,
                        "y": 10.0,
                        "width": 30.0,
                        "height": 20.0,
                        "confidence": 0.9,
                        "class": "",
                        "class_id": 0,
                    }
                ],
            }
        )
    )
    block = _make_block(model_manager)
    images = _make_images()

    # when
    result = block.run(
        images=images, text_detection="small", text_recognition="none"
    )

    # then
    assert len(result) == 1
    single = result[0]
    assert single["result"] == ""
    assert single["prediction_type"] == "ocr"
    assert len(single["predictions"]) == 1


def test_pp_ocr_run_locally_recognize_only() -> None:
    # given
    model_manager = MagicMock()
    model_manager.infer_from_request_sync.return_value = MagicMock(
        model_dump=MagicMock(
            return_value={
                "result": "HELLO",
                "image": {"width": 168, "height": 192},
                "predictions": [],
            }
        )
    )
    block = _make_block(model_manager)
    images = _make_images()

    # when
    result = block.run(
        images=images, text_detection="none", text_recognition="small"
    )

    # then
    assert len(result) == 1
    single = result[0]
    assert single["result"] == "HELLO"
    assert single["prediction_type"] == "ocr"
    assert len(single["predictions"]) == 0
