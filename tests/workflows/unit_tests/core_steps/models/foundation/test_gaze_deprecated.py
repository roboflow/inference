"""Behavioural tests for the deprecated GazeBlockV1 stub."""

from unittest.mock import MagicMock

import pytest

from inference.core.exceptions import FeatureDeprecatedError
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.gaze.v1 import (
    BlockManifest,
    GazeBlockV1,
)


def test_gaze_block_manifest_keeps_v1_type_literal() -> None:
    fields = BlockManifest.model_fields
    type_field = fields["type"]
    assert type_field.annotation.__args__ == ("roboflow_core/gaze@v1",)


def test_gaze_block_manifest_is_marked_deprecated_in_ui_manifest() -> None:
    extra = BlockManifest.model_config["json_schema_extra"]
    assert extra.get("deprecated") is True


def test_gaze_block_run_raises_feature_deprecated_error() -> None:
    # given
    block = GazeBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # when / then
    with pytest.raises(FeatureDeprecatedError) as captured:
        block.run(images=[], do_run_face_detection=True)

    assert captured.value.feature == "roboflow_core/gaze@v1"
    assert captured.value.reason == "MediaPipe dependency removed from inference"


def test_gaze_block_run_raises_in_remote_execution_mode_as_well() -> None:
    # given
    block = GazeBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    # when / then — execution mode is irrelevant; the deprecation precedes any branch
    with pytest.raises(FeatureDeprecatedError):
        block.run(images=[], do_run_face_detection=True)
