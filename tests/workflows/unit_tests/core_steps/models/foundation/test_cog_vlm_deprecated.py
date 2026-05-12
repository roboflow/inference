"""Behavioural tests for the deprecated CogVLM workflow block."""

from unittest.mock import MagicMock

import pytest

from inference.core.exceptions import FeatureDeprecatedError
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.cog_vlm.v1 import (
    CogVLMBlockV1,
)


def test_cog_vlm_block_run_raises_feature_deprecated_error_with_removal_release() -> None:
    # given
    block = CogVLMBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # when / then
    with pytest.raises(FeatureDeprecatedError) as captured:
        block.run(images=[], prompt="anything", json_output_format=None)

    assert captured.value.feature == "roboflow_core/cog_vlm@v1"
    assert captured.value.removal_release == "0.54.0"
    assert "CVE-2024-11393" in (captured.value.reason or "")
