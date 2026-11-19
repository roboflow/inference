from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.owlv2.v1 import (
    OwlV2ModelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="asserts on the supervision output of the numpy representation",
)
def test_local_run_labels_boxes_with_the_text_queries():
    model_manager = MagicMock()
    model_manager.run_owlv2_text.return_value = [
        {
            "image": {"width": 4, "height": 4},
            "predictions": [
                {
                    "x": 2,
                    "y": 2,
                    "width": 2,
                    "height": 2,
                    "confidence": 0.9,
                    "class": "bus",
                    "class_id": 1,
                    "detection_id": "a",
                }
            ],
        }
    ]
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    block = OwlV2ModelBlockV1(
        model_manager=model_manager,
        api_key="key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    (result,) = block.run(
        images=Batch(content=[image], indices=[(0,)]),
        model_id="workspace/detector",
        class_names=["bicycle", "bus"],
        confidence=0.2,
    )

    model_manager.add_model.assert_called_once_with(
        model_id="workspace/detector", api_key="key"
    )
    request = model_manager.run_owlv2_text.call_args.kwargs
    assert request["text"] == ["bicycle", "bus"]
    assert request["confidence"] == 0.2
    assert result["predictions"].data["class_name"].tolist() == ["bus"]
