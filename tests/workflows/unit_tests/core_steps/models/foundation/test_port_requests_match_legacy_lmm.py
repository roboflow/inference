"""Block -> adapter -> request differential tests for Task 11.10's LMM family,
depth estimation and Moondream2 blocks.

Each test drives a block's real `run_locally` through a real
`ModelManagerModelsProvider` wrapping a `MagicMock` `ModelManager`, then
compares the pydantic request the adapter built against the exact request
construction the block used to run inline before Task 11.10 (copied from
`git show 32b749865:.../<family>/v1.py` - the commit this task was actually
built on top of; it landed after this task's nominal BASE, 723646aed, but
touched none of the files these tests read, so the pre-port construction is
unchanged there).

Every case asserts both `model_dump()` equality AND `model_fields_set`
equality: an omitted (UNSET) field must stay omitted, not be re-supplied as
its own default - `model_dump()` alone can't tell "never set" from
"explicitly set to the same value as the default" apart, but the pydantic
`model_fields_set` bookkeeping can.

One case per distinct request-building shape the task touches:
`LMMInferenceRequest` with neither optional field set (`cosmos3`), with only
`max_new_tokens` forwarded (`glm_ocr`), with both `enable_thinking` and
`max_new_tokens` forwarded (`qwen3_5vl` v1), and with `enable_thinking` set
but `max_new_tokens` left `None` - and therefore omitted, per the old
`if max_new_tokens is not None` guard also present in `qwen_vlm/v3.py`'s
`_run_native_locally` (`qwen3_5vl` v1 again); plus `DepthEstimationRequest`
(`depth_estimation`) and `Moondream2InferenceRequest` (`moondream2`).
"""

from unittest.mock import MagicMock

import numpy as np

from inference.core.entities.requests.inference import (
    DepthEstimationRequest,
    LMMInferenceRequest,
)
from inference.core.entities.requests.moondream2 import Moondream2InferenceRequest
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.cosmos3.v1 import (
    Cosmos3EdgeBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.depth_estimation.v1 import (
    DepthEstimationBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.glm_ocr.v1 import (
    GLMOCRBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.moondream2.v1 import (
    Moondream2BlockV1,
)
from inference.core.workflows.core_steps.models.foundation.qwen3_5vl.v1 import (
    Qwen35VLBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


def _make_images() -> Batch:
    # Every LMM/depth/moondream2 block calls `to_inference_format(numpy_preferred=False)`
    # (unlike the roboflow model family, which prefers numpy). Building the
    # "expected" request from the SAME Batch/WorkflowImageData instance the
    # block runs against keeps `value` the identical object both times
    # (WorkflowImageData caches the conversion), so dict/list equality on the
    # dumped requests hits Python's identity fast path instead of an
    # element-wise ndarray `==` (which raises on a >1-element array).
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        numpy_image=np.zeros((20, 10, 3), dtype=np.uint8),
    )
    return Batch(content=[image], indices=[(0,)])


def _manager() -> MagicMock:
    return MagicMock()


def _captured_request(manager: MagicMock):
    assert manager.infer_from_request_sync.call_count == 1
    call = manager.infer_from_request_sync.call_args
    return call.kwargs["request"] if "request" in call.kwargs else call.args[1]


def test_lmm_request_matches_the_pre_port_construction_without_optional_fields() -> (
    None
):
    # Copied verbatim from `git show 32b749865:.../cosmos3/v1.py`: neither
    # `enable_thinking` nor `max_new_tokens` is ever set by this block.
    manager = _manager()
    images = _make_images()
    block = Cosmos3EdgeBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(
        images=images,
        model_version="m/1",
        prompt="custom prompt",
        system_prompt=None,
    )

    manager.add_model.assert_called_once_with(model_id="m/1", api_key="k")
    request = _captured_request(manager)

    inference_images = [i.to_inference_format(numpy_preferred=False) for i in images]
    expected = LMMInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images[0],
        source="workflow-execution",
        prompt="custom prompt",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_lmm_request_matches_the_pre_port_construction_with_max_new_tokens_only() -> (
    None
):
    # Copied verbatim from `git show 32b749865:.../glm_ocr/v1.py`: forwards
    # `max_new_tokens` only when it is not None; never sets `enable_thinking`.
    manager = _manager()
    images = _make_images()
    block = GLMOCRBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(
        images=images,
        model_version="m/1",
        prompt="p",
        max_new_tokens=64,
    )

    manager.add_model.assert_called_once_with(model_id="m/1", api_key="k")
    request = _captured_request(manager)

    inference_images = [i.to_inference_format(numpy_preferred=False) for i in images]
    expected = LMMInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images[0],
        source="workflow-execution",
        prompt="p",
        max_new_tokens=64,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_lmm_request_matches_the_pre_port_construction_with_thinking_and_tokens() -> (
    None
):
    # Copied verbatim from `git show 32b749865:.../qwen3_5vl/v1.py`: always
    # sets `enable_thinking`, forwards `max_new_tokens` only when not None.
    manager = _manager()
    images = _make_images()
    block = Qwen35VLBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(
        images=images,
        model_version="m/1",
        prompt="Hi",
        system_prompt="Sys",
        enable_thinking=True,
        max_new_tokens=100,
    )

    manager.add_model.assert_called_once_with(model_id="m/1", api_key="k")
    request = _captured_request(manager)

    inference_images = [i.to_inference_format(numpy_preferred=False) for i in images]
    expected = LMMInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images[0],
        source="workflow-execution",
        prompt="Hi<system_prompt>Sys",
        enable_thinking=True,
        max_new_tokens=100,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_lmm_request_matches_the_pre_port_construction_with_thinking_only() -> None:
    # Copied verbatim from `git show 32b749865:.../qwen3_5vl/v1.py` (the same
    # `if max_new_tokens is not None: request_kwargs["max_new_tokens"] = ...`
    # guard is in `qwen_vlm/v3.py`'s `_run_native_locally`): `enable_thinking`
    # is always forwarded, but with `max_new_tokens=None` the guard never
    # fires, so the OLD request never had `max_new_tokens` in
    # `model_fields_set` either - only `model_dump()` equality would miss a
    # regression here, since an omitted field and one explicitly set to its
    # own default (`None`) dump identically.
    manager = _manager()
    images = _make_images()
    block = Qwen35VLBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(
        images=images,
        model_version="m/1",
        prompt="Hi",
        system_prompt="Sys",
        enable_thinking=True,
        max_new_tokens=None,
    )

    manager.add_model.assert_called_once_with(model_id="m/1", api_key="k")
    request = _captured_request(manager)

    inference_images = [i.to_inference_format(numpy_preferred=False) for i in images]
    expected = LMMInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images[0],
        source="workflow-execution",
        prompt="Hi<system_prompt>Sys",
        enable_thinking=True,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set
    assert "max_new_tokens" not in request.model_fields_set


def test_depth_estimation_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 32b749865:.../depth_estimation/v1.py`:
    # `DepthEstimationRequest(image=image)` - no `model_id` on the request
    # itself (it only goes to `infer_from_request_sync`'s own `model_id` kwarg).
    manager = _manager()
    images = _make_images()
    block = DepthEstimationBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(images=images, model_version="m/1")

    manager.add_model.assert_called_once_with(model_id="m/1", api_key="k")
    request = _captured_request(manager)
    assert manager.infer_from_request_sync.call_args.kwargs["model_id"] == "m/1"

    inference_images = [i.to_inference_format(numpy_preferred=False) for i in images]
    expected = DepthEstimationRequest(image=inference_images[0])
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_moondream2_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 32b749865:.../moondream2/v1.py`.
    manager = _manager()
    images = _make_images()
    block = Moondream2BlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._post_process_result = lambda **_: None

    block.run_locally(images=images, model_version="m/1", prompt="p")

    manager.add_model.assert_called_once_with(model_id="m/1", api_key="k")
    request = _captured_request(manager)

    inference_images = [i.to_inference_format(numpy_preferred=False) for i in images]
    expected = Moondream2InferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images[0],
        text=[],
        prompt="p",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set
