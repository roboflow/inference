"""Both LMM blocks call OpenAI directly on both step execution paths.

`LMM_ENABLED` gates the Roboflow LMM endpoint. Neither `roboflow_core/lmm@v1`
nor `roboflow_core/lmm_for_classification@v1` calls that endpoint: LOCAL and
REMOTE step execution both hand the images to `run_gpt_4v_llm_prompting()`,
which talks to OpenAI with the caller's key. So the flag must not change the
execution path, and the blocks declare no restriction for it.

The OpenAI helper is mocked; no network, provider API or model is touched.
"""

from typing import Any, Dict
from unittest import mock

import numpy as np
import pytest
from roboflow_workflows import environment
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.foundation.lmm import v1 as lmm_module
from roboflow_workflows.core_steps.models.foundation.lmm.v1 import LMMBlockV1, LMMConfig
from roboflow_workflows.core_steps.models.foundation.lmm_classifier import (
    v1 as lmm_classifier_module,
)
from roboflow_workflows.core_steps.models.foundation.lmm_classifier.v1 import (
    LMMForClassificationBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)

RAW_OUTPUT = [{"content": '{"top": "cat"}', "image": {"width": 4, "height": 3}}]


def _image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((3, 4, 3), dtype=np.uint8),
    )


@pytest.mark.parametrize(
    "module, block_class, extra",
    [
        (lmm_module, LMMBlockV1, {"prompt": "describe", "json_output": None}),
        (lmm_classifier_module, LMMForClassificationBlockV1, {"classes": ["cat"]}),
    ],
)
@pytest.mark.parametrize(
    "step_execution_mode", [StepExecutionMode.LOCAL, StepExecutionMode.REMOTE]
)
@pytest.mark.parametrize("lmm_enabled", [True, False])
def test_lmm_blocks_call_the_openai_helper_whatever_the_mode_and_flag(
    module: Any,
    block_class: type,
    extra: Dict[str, Any],
    step_execution_mode: StepExecutionMode,
    lmm_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    monkeypatch.setattr(environment, "LMM_ENABLED", lmm_enabled)
    helper = mock.MagicMock(return_value=RAW_OUTPUT)
    monkeypatch.setattr(module, "run_gpt_4v_llm_prompting", helper)
    roboflow_client = mock.MagicMock()
    monkeypatch.setattr(lmm_module, "InferenceHTTPClient", roboflow_client)
    lmm_config = LMMConfig()
    block = block_class(api_key=None, step_execution_mode=step_execution_mode)

    # when
    result = block.run(
        images=Batch(content=[_image()], indices=[(0,)]),
        lmm_type="gpt_4v",
        lmm_config=lmm_config,
        remote_api_key="sk-test",
        **extra,
    )

    # then - the direct OpenAI helper ran once, no Roboflow client was built
    helper.assert_called_once()
    call_kwargs = helper.call_args.kwargs
    assert len(call_kwargs["image"]) == 1
    assert call_kwargs["remote_api_key"] == "sk-test"
    assert call_kwargs["lmm_config"] is lmm_config
    roboflow_client.assert_not_called()
    assert len(result) == 1
    assert result[0]["raw_output"] == '{"top": "cat"}'
    assert result[0]["parent_id"] == "parent"
