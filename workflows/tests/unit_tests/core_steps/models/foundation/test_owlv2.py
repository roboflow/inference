from unittest.mock import Mock

import numpy as np
import pytest
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.foundation.owlv2 import v1
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


def test_local_block_sends_the_image_and_prompts_to_the_workspace_model():
    provider = Mock()
    provider.run_owlv2_text.return_value = [
        {"image": {"width": 8, "height": 8}, "predictions": []}
    ]
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"), numpy_image=pixels
    )
    block = v1.OwlV2ModelBlockV1(provider, "key", StepExecutionMode.LOCAL)

    result = block.run(Batch([image], indices=[(0,)]), "ds/1", ["bus"], 0.4)

    provider.add_model.assert_called_once_with(model_id="ds/1", api_key="key")
    provider.run_owlv2_text.assert_called_once_with(
        model_id="ds/1",
        images=[{"type": "numpy_object", "value": image.numpy_image}],
        text_prompts=["bus"],
        confidence=0.4,
        api_key="key",
    )
    assert len(result) == 1 and "predictions" in result[0]


def test_remote_block_explains_how_to_upgrade_an_older_sdk(monkeypatch):
    monkeypatch.setattr(v1, "InferenceHTTPClient", Mock(return_value=object()))
    block = v1.OwlV2ModelBlockV1(Mock(), "key", StepExecutionMode.REMOTE)

    with pytest.raises(
        RuntimeError, match="Upgrade inference-sdk or use local execution"
    ):
        block.run(Batch([], indices=[]), "ds/1", ["bus"], 0.4)
