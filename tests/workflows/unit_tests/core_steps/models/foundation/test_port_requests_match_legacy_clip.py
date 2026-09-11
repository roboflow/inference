"""Block -> adapter -> request differential tests for Task 11.11's CLIP and
Perception Encoder blocks.

Each test drives a block's real `run_locally` through a real
`ModelManagerModelsProvider` wrapping a `MagicMock` `ModelManager`, then
compares the pydantic request the adapter built against the exact request
construction the block used to run inline before Task 11.11 (this task's BASE
commit, e6ad76d4e, is also HEAD of this checkout - the pre-port construction
below is copied straight from `git show e6ad76d4e:<path>`, i.e. the source
this task started from).

One case per distinct request-building shape the task touches:
`ClipTextEmbeddingRequest`/`ClipImageEmbeddingRequest` (`clip/v1.py`),
`ClipCompareRequest` with the version left UNSET (`clip_comparison/v1.py`,
which never set `clip_version_id`) and with it forwarded explicitly
(`clip_comparison/v2.py`), and
`PerceptionEncoderTextEmbeddingRequest`/`PerceptionEncoderImageEmbeddingRequest`
(`perception_encoder/v1.py`).
"""

from unittest.mock import MagicMock

import numpy as np

from inference.core.entities.requests.clip import (
    ClipCompareRequest,
    ClipImageEmbeddingRequest,
    ClipTextEmbeddingRequest,
)
from inference.core.entities.requests.perception_encoder import (
    PerceptionEncoderImageEmbeddingRequest,
    PerceptionEncoderTextEmbeddingRequest,
)
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.roboflow_api import ModelEndpointType
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.clip.v1 import (
    ClipModelBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.clip_comparison.v1 import (
    ClipComparisonBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.clip_comparison.v2 import (
    ClipComparisonBlockV2,
)
from inference.core.workflows.core_steps.models.foundation.perception_encoder.v1 import (
    PerceptionEncoderModelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.models_provider import CORE_MODEL_ENDPOINT_TYPE


class _EmbeddingResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _ComparisonResponse:
    def __init__(self, similarity):
        self._similarity = similarity

    def model_dump(self, **_kwargs):
        return {"similarity": self._similarity}


def _make_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        numpy_image=np.zeros((20, 10, 3), dtype=np.uint8),
    )


def _manager() -> MagicMock:
    return MagicMock()


def _captured_request(manager: MagicMock):
    assert manager.infer_from_request_sync.call_count == 1
    call = manager.infer_from_request_sync.call_args
    return call.kwargs["request"] if "request" in call.kwargs else call.args[1]


def test_clip_text_embedding_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show e6ad76d4e:.../clip/v1.py`:
    # `ClipTextEmbeddingRequest(clip_version_id=version, text=[data], api_key=...)`.
    manager = _manager()
    manager.infer_from_request_sync.return_value = _EmbeddingResponse([[0.1, 0.2]])
    block = ClipModelBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run_locally(data="a parity test cat", version="ViT-B-16")

    manager.add_model.assert_called_once_with(
        model_id="clip/ViT-B-16", api_key="k", endpoint_type=CORE_MODEL_ENDPOINT_TYPE
    )
    request = _captured_request(manager)
    expected = ClipTextEmbeddingRequest(
        clip_version_id="ViT-B-16", text=["a parity test cat"], api_key="k"
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert result == {"embedding": [0.1, 0.2]}


def test_clip_image_embedding_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show e6ad76d4e:.../clip/v1.py`:
    # `ClipImageEmbeddingRequest(clip_version_id=version,
    # image=[data.to_inference_format(numpy_preferred=True)], api_key=...)`.
    manager = _manager()
    manager.infer_from_request_sync.return_value = _EmbeddingResponse([[0.3]])
    image = _make_image()
    block = ClipModelBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run_locally(data=image, version="RN50")

    manager.add_model.assert_called_once_with(
        model_id="clip/RN50", api_key="k", endpoint_type=CORE_MODEL_ENDPOINT_TYPE
    )
    request = _captured_request(manager)
    expected = ClipImageEmbeddingRequest(
        clip_version_id="RN50",
        image=[image.to_inference_format(numpy_preferred=True)],
        api_key="k",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert result == {"embedding": [0.3]}


def test_clip_comparison_v1_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show e6ad76d4e:.../clip_comparison/v1.py`:
    # `ClipCompareRequest(subject=..., subject_type="image", prompt=texts,
    # prompt_type="text", api_key=...)` - no `clip_version_id`, so the
    # pydantic default (`env.CLIP_VERSION_ID`) applies.
    manager = _manager()
    manager.infer_from_request_sync.return_value = _ComparisonResponse([0.5, 0.6])
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    block = ClipComparisonBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(images=images, texts=["cat", "dog"])

    request = _captured_request(manager)
    expected = ClipCompareRequest(
        api_key="k",
        subject=image.to_inference_format(numpy_preferred=True),
        subject_type="image",
        prompt=["cat", "dog"],
        prompt_type="text",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    manager.add_model.assert_called_once_with(
        f"clip/{expected.clip_version_id}",
        "k",
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )


def test_clip_comparison_v2_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show e6ad76d4e:.../clip_comparison/v2.py`:
    # `ClipCompareRequest(clip_version_id=version, subject=..., ...)`.
    manager = _manager()
    manager.infer_from_request_sync.return_value = _ComparisonResponse([0.2, 0.9])
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    block = ClipComparisonBlockV2(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(images=images, classes=["a", "b"], version="RN50")

    request = _captured_request(manager)
    expected = ClipCompareRequest(
        clip_version_id="RN50",
        subject=image.to_inference_format(numpy_preferred=True),
        subject_type="image",
        prompt=["a", "b"],
        prompt_type="text",
        api_key="k",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    manager.add_model.assert_called_once_with(
        "clip/RN50", "k", endpoint_type=ModelEndpointType.CORE_MODEL
    )


def test_perception_encoder_text_embedding_request_matches_the_pre_port_construction() -> (
    None
):
    # Copied verbatim from `git show e6ad76d4e:.../perception_encoder/v1.py`.
    manager = _manager()
    manager.infer_from_request_sync.return_value = _EmbeddingResponse([[0.4]])
    block = PerceptionEncoderModelBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run_locally(data="a parity test string", version="PE-Core-B16-224")

    manager.add_model.assert_called_once_with(
        model_id="perception_encoder/PE-Core-B16-224",
        api_key="k",
        endpoint_type=CORE_MODEL_ENDPOINT_TYPE,
    )
    request = _captured_request(manager)
    expected = PerceptionEncoderTextEmbeddingRequest(
        perception_encoder_version_id="PE-Core-B16-224",
        text=["a parity test string"],
        api_key="k",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert result == {"embedding": [0.4]}


def test_perception_encoder_image_embedding_request_matches_the_pre_port_construction() -> (
    None
):
    # Copied verbatim from `git show e6ad76d4e:.../perception_encoder/v1.py`.
    manager = _manager()
    manager.infer_from_request_sync.return_value = _EmbeddingResponse([[0.5]])
    image = _make_image()
    block = PerceptionEncoderModelBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run_locally(data=image, version="PE-Core-L14-336")

    manager.add_model.assert_called_once_with(
        model_id="perception_encoder/PE-Core-L14-336",
        api_key="k",
        endpoint_type=CORE_MODEL_ENDPOINT_TYPE,
    )
    request = _captured_request(manager)
    expected = PerceptionEncoderImageEmbeddingRequest(
        perception_encoder_version_id="PE-Core-L14-336",
        image=[image.to_inference_format(numpy_preferred=True)],
        api_key="k",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert result == {"embedding": [0.5]}
