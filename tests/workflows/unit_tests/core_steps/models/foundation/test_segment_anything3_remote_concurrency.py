"""Remote execution of the SAM3 concept-segmentation blocks.

The SAM3 endpoint takes exactly one image per request, so the blocks cannot
pack a batch into a single payload. They must still hand the whole batch to the
SDK in one call, which issues the per-image requests concurrently; issuing them
one at a time made a preview run cost one full round trip per frame.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v1 import (
    SegmentAnything3BlockV1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v1_tensor import (
    SegmentAnything3BlockV1 as TensorSegmentAnything3BlockV1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v2 import (
    SegmentAnything3BlockV2,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v2_tensor import (
    SegmentAnything3BlockV2 as TensorSegmentAnything3BlockV2,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 import (
    SegmentAnything3BlockV3,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v3_tensor import (
    SegmentAnything3BlockV3 as TensorSegmentAnything3BlockV3,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

V1_MODULE = "inference.core.workflows.core_steps.models.foundation.segment_anything3.v1"
V1_TENSOR_MODULE = (
    "inference.core.workflows.core_steps.models.foundation.segment_anything3.v1_tensor"
)
V2_MODULE = "inference.core.workflows.core_steps.models.foundation.segment_anything3.v2"
V2_TENSOR_MODULE = (
    "inference.core.workflows.core_steps.models.foundation.segment_anything3.v2_tensor"
)
V3_MODULE = "inference.core.workflows.core_steps.models.foundation.segment_anything3.v3"
V3_TENSOR_MODULE = (
    "inference.core.workflows.core_steps.models.foundation.segment_anything3.v3_tensor"
)

# Each block version takes a different run() signature; the tensor variants
# additionally require every argument explicitly.
_V2_KWARGS = {
    "confidence": 0.5,
    "per_class_confidence": None,
    "apply_nms": True,
    "nms_iou_threshold": 0.9,
}
_V3_KWARGS = {
    **_V2_KWARGS,
    "class_mapping": None,
    "output_format": "rle",
}

# (block class, module path, extra run kwargs on top of images/model_id/class_names)
BLOCK_VARIANTS = [
    (SegmentAnything3BlockV1, V1_MODULE, {"threshold": 0.5}),
    (TensorSegmentAnything3BlockV1, V1_TENSOR_MODULE, {"threshold": 0.5}),
    (SegmentAnything3BlockV2, V2_MODULE, _V2_KWARGS),
    (TensorSegmentAnything3BlockV2, V2_TENSOR_MODULE, _V2_KWARGS),
    (SegmentAnything3BlockV3, V3_MODULE, _V3_KWARGS),
    (TensorSegmentAnything3BlockV3, V3_TENSOR_MODULE, _V3_KWARGS),
]

VARIANT_IDS = ["v1", "v1_tensor", "v2", "v2_tensor", "v3", "v3_tensor"]


def _image(parent_id: str) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=parent_id),
        numpy_image=np.zeros((480, 640, 3), dtype=np.uint8),
    )


def _empty_response() -> dict:
    """A well-formed response carrying no detections.

    Covers both response shapes the blocks parse: `prompt_results` (v1/v2) and
    the RLE/polygon payload (v3).
    """
    return {
        "prompt_results": [{"prompt_index": 0, "predictions": []}],
        "predictions": [],
        "image": {"width": 640, "height": 480},
        "time": 0.01,
    }


def _run_block(block_cls, module_path: str, extra_kwargs: dict, image_count: int):
    """Run one block against a mocked SDK client and return the client mock."""
    images = [_image(f"parent-{i}") for i in range(image_count)]
    with patch(f"{module_path}.InferenceHTTPClient") as mock_client_cls:
        mock_client = MagicMock()
        # One response per image; the SDK returns a bare dict for a single image.
        mock_client.sam3_concept_segment.return_value = (
            _empty_response()
            if image_count == 1
            else [_empty_response() for _ in range(image_count)]
        )
        mock_client_cls.return_value = mock_client
        block = block_cls(
            model_manager=MagicMock(),
            api_key="test_api_key",
            step_execution_mode=StepExecutionMode.REMOTE,
        )
        result = block.run(
            images=images,
            model_id="sam3/sam3_final",
            class_names=["cat"],
            **extra_kwargs,
        )
    return mock_client, result


@pytest.mark.parametrize("block_cls,module_path,extra", BLOCK_VARIANTS, ids=VARIANT_IDS)
def test_remote_execution_sends_whole_batch_in_one_sdk_call(
    block_cls, module_path, extra
) -> None:
    """A 4-frame batch must reach the SDK as one call, not four.

    The SDK fans the batch out into concurrent per-image requests; calling it
    per image serialises the round trips instead.
    """
    mock_client, _ = _run_block(block_cls, module_path, extra, image_count=4)

    assert mock_client.sam3_concept_segment.call_count == 1
    passed = mock_client.sam3_concept_segment.call_args.kwargs["inference_input"]
    assert isinstance(passed, list)
    assert len(passed) == 4


@pytest.mark.parametrize("block_cls,module_path,extra", BLOCK_VARIANTS, ids=VARIANT_IDS)
def test_remote_execution_requests_one_image_per_request_and_concurrency(
    block_cls, module_path, extra
) -> None:
    """SAM3 accepts a single image per request, so batching must stay off.

    max_batch_size above 1 makes the SDK put a LIST into the payload's `image`
    field, which the endpoint rejects. Concurrency is the lever that is safe to
    raise.
    """
    with patch(f"{module_path}.InferenceConfiguration") as mock_config:
        _run_block(block_cls, module_path, extra, image_count=4)

    assert mock_config.call_count == 1
    kwargs = mock_config.call_args.kwargs
    assert kwargs["max_batch_size"] == 1
    assert kwargs["max_concurrent_requests"] > 1


@pytest.mark.parametrize("block_cls,module_path,extra", BLOCK_VARIANTS, ids=VARIANT_IDS)
def test_remote_execution_returns_one_result_per_image(
    block_cls, module_path, extra
) -> None:
    """Batching the transport must not disturb per-image result alignment."""
    _, result = _run_block(block_cls, module_path, extra, image_count=4)

    assert len(result) == 4


@pytest.mark.parametrize("block_cls,module_path,extra", BLOCK_VARIANTS, ids=VARIANT_IDS)
def test_remote_execution_handles_single_image_response(
    block_cls, module_path, extra
) -> None:
    """The SDK unwraps a one-element list into a bare dict; still one result."""
    mock_client, result = _run_block(block_cls, module_path, extra, image_count=1)

    assert mock_client.sam3_concept_segment.call_count == 1
    assert len(result) == 1
