import numpy as np
import pytest

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    InstanceSegmentationInferenceResponse,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v1 import (
    RoboflowInstanceSegmentationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v2 import (
    RoboflowInstanceSegmentationModelBlockV2,
)
from inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v4 import (
    RoboflowInstanceSegmentationModelBlockV4,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


class _FakeModelManager:
    def __init__(self) -> None:
        self.requests = []

    def add_model(self, **kwargs) -> None:
        pass

    def infer_from_request_sync(self, model_id, request):
        self.requests.append(request)
        return [
            InstanceSegmentationInferenceResponse(
                predictions=[],
                image=InferenceResponseImage(width=10, height=10),
            )
        ]


def _workflow_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="root"),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )


@pytest.mark.parametrize(
    "block_class, extra_kwargs",
    [
        (
            RoboflowInstanceSegmentationModelBlockV1,
            {
                "confidence": 0.4,
                "enforce_dense_masks_in_inference_models": False,
            },
        ),
        (
            RoboflowInstanceSegmentationModelBlockV2,
            {
                "confidence": 0.4,
                "enforce_dense_masks_in_inference_models": False,
            },
        ),
        (RoboflowInstanceSegmentationModelBlockV4, {"confidence": 0.4}),
    ],
)
def test_non_deferred_instance_segmentation_blocks_disable_stream_pipeline(
    block_class,
    extra_kwargs,
) -> None:
    manager = _FakeModelManager()
    block = block_class(
        model_manager=manager,
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(
        images=[_workflow_image()],
        model_id="model/1",
        class_agnostic_nms=None,
        class_filter=None,
        iou_threshold=None,
        max_detections=None,
        max_candidates=None,
        mask_decode_mode="accurate",
        tradeoff_factor=None,
        disable_active_learning=None,
        active_learning_target_dataset=None,
        **extra_kwargs,
    )

    assert manager.requests[0].disable_stream_pipeline is True
