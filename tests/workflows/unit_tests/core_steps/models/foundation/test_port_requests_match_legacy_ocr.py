"""Block -> adapter -> request differential tests for Task 11.12's OCR family
(DocTR, EasyOCR, PP-OCR) and YOLO-World blocks.

Each test drives a block's real ``run_locally`` through a real
``ModelManagerModelsProvider`` wrapping a ``MagicMock`` ``ModelManager``, then
compares the pydantic request the adapter built against the exact request
construction the block used to run inline before Task 11.12 (this task's BASE
commit, 3dc9e17cb, is also HEAD of this checkout - the pre-port construction
below is copied straight from ``git show 3dc9e17cb:<path>``, i.e. the source
this task started from).

One case per distinct request-building shape the task touches:
``DoctrOCRInferenceRequest`` (``ocr/v1.py``), ``EasyOCRInferenceRequest``
(``easy_ocr/v1.py``), ``PPOCRInferenceRequest`` (``pp_ocr/v1.py`` - registers
via the validator-derived id, same as the adapter-level tests in
``test_workflows_models_provider.py``) and ``YOLOWorldInferenceRequest``
(``yolo_world/v1.py``).

Fix round 1 adds a parity case for the tensor-native sibling
(``pp_ocr/v1_tensor.py``), which shares ``PPOCRInferenceRequest`` construction
with ``pp_ocr/v1.py`` but returns a native ``Detections`` instead of the numpy
``post_process_ocr_result`` shape.

Every case asserts both ``model_dump()`` equality AND ``model_fields_set``
equality: an omitted field must stay omitted, not be re-supplied as its own
default.
"""

from unittest.mock import MagicMock

import numpy as np

from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.requests.pp_ocr import PPOCRInferenceRequest
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.roboflow_api import ModelEndpointType
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.easy_ocr.v1 import (
    EasyOCRBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.ocr.v1 import OCRModelBlockV1
from inference.core.workflows.core_steps.models.foundation.pp_ocr.v1 import PPOCRBlockV1
from inference.core.workflows.core_steps.models.foundation.pp_ocr.v1_tensor import (
    PPOCRBlockV1 as PPOCRTensorBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.yolo_world.v1 import (
    YoloWorldModelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.models_provider import CORE_MODEL_ENDPOINT_TYPE
from inference_models.models.base.object_detection import Detections


class _DictResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, **_kwargs):
        return self._payload


def _make_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        numpy_image=np.zeros((20, 10, 3), dtype=np.uint8),
    )


def _ocr_response() -> _DictResponse:
    return _DictResponse(
        {"result": "HELLO", "image": {"width": 10, "height": 20}, "predictions": []}
    )


def _manager() -> MagicMock:
    return MagicMock()


def _captured_request(manager: MagicMock):
    assert manager.infer_from_request_sync.call_count == 1
    call = manager.infer_from_request_sync.call_args
    return call.kwargs["request"] if "request" in call.kwargs else call.args[1]


def test_doctr_ocr_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 3dc9e17cb:.../ocr/v1.py`:
    # `DoctrOCRInferenceRequest(image=..., api_key=..., generate_bounding_boxes=True)`
    # with no `doctr_version_id` set - the field's own pydantic default
    # ("default", requests/doctr.py:20) applies, and the pre-port
    # `load_core_model` read exactly that default off the validated request.
    manager = _manager()
    manager.infer_from_request_sync.return_value = _ocr_response()
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    block = OCRModelBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(images=images)

    manager.add_model.assert_called_once_with(
        model_id="doctr/default", api_key="k", endpoint_type=CORE_MODEL_ENDPOINT_TYPE
    )
    request = _captured_request(manager)
    expected = DoctrOCRInferenceRequest(
        image=image.to_inference_format(numpy_preferred=True),
        api_key="k",
        generate_bounding_boxes=True,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_easy_ocr_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 3dc9e17cb:.../easy_ocr/v1.py`:
    # `EasyOCRInferenceRequest(easy_ocr_version_id=version, image=..., api_key=...,
    # language_codes=language_codes, quantize=quantize)`.
    manager = _manager()
    manager.infer_from_request_sync.return_value = _ocr_response()
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    block = EasyOCRBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(
        images=images,
        language_codes=["en"],
        version="english_g2",
        quantize=False,
    )

    manager.add_model.assert_called_once_with(
        model_id="easy_ocr/english_g2",
        api_key="k",
        endpoint_type=CORE_MODEL_ENDPOINT_TYPE,
    )
    request = _captured_request(manager)
    expected = EasyOCRInferenceRequest(
        easy_ocr_version_id="english_g2",
        image=image.to_inference_format(numpy_preferred=True),
        api_key="k",
        language_codes=["en"],
        quantize=False,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_pp_ocr_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 3dc9e17cb:.../pp_ocr/v1.py`:
    # `PPOCRInferenceRequest(text_detection=..., text_recognition=..., image=...,
    # api_key=...)`. Registration is the round-2-defect-4 exception (round-2 in
    # the Phase 11 plan predates this task): the validator derives
    # `pp_ocr_version_id`, which only exists post-validation, so the adapter
    # registers (build -> register -> infer) instead of the block calling
    # `load_core_model`.
    manager = _manager()
    manager.infer_from_request_sync.return_value = _ocr_response()
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    block = PPOCRBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(images=images, text_detection="small", text_recognition="small")

    request = _captured_request(manager)
    expected = PPOCRInferenceRequest(
        text_detection="small",
        text_recognition="small",
        image=image.to_inference_format(numpy_preferred=True),
        api_key="k",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set
    manager.add_model.assert_called_once_with(
        "pp_ocr/small-small", "k", endpoint_type=ModelEndpointType.CORE_MODEL
    )


def test_pp_ocr_tensor_request_matches_the_pre_port_construction() -> None:
    """Fix round 1 (minor): the tensor-native sibling shares `PPOCRInferenceRequest`
    construction with `pp_ocr/v1.py`. Copied verbatim from
    `git show fbe20ca6e:.../pp_ocr/v1_tensor.py` (identical to `pp_ocr/v1.py`'s
    pre-port body). Also asserts the native-output path is taken: the block
    returns a `Detections` object directly (not the numpy `post_process_ocr_result`
    dict-of-`sv.Detections` shape `pp_ocr/v1.py` uses)."""
    manager = _manager()
    manager.infer_from_request_sync.return_value = _DictResponse(
        {
            "result": "HELLO",
            "image": {"width": 10, "height": 20},
            "predictions": [
                {
                    "x": 5.0,
                    "y": 5.0,
                    "width": 4.0,
                    "height": 4.0,
                    "confidence": 0.9,
                    "class": "HELLO",
                    "class_id": 0,
                }
            ],
        }
    )
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    block = PPOCRTensorBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run_locally(
        images=images, text_detection="small", text_recognition="small"
    )

    request = _captured_request(manager)
    expected = PPOCRInferenceRequest(
        text_detection="small",
        text_recognition="small",
        image=image.to_inference_format(numpy_preferred=True),
        api_key="k",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set
    manager.add_model.assert_called_once_with(
        "pp_ocr/small-small", "k", endpoint_type=ModelEndpointType.CORE_MODEL
    )

    # Native-output path: one dict per image, `predictions` is a native
    # `Detections`, not `sv.Detections` (the numpy `pp_ocr/v1.py` shape).
    assert len(result) == 1
    assert result[0]["result"] == "HELLO"
    assert isinstance(result[0]["predictions"], Detections)
    assert len(result[0]["predictions"]) == 1


def test_yolo_world_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 3dc9e17cb:.../yolo_world/v1.py`:
    # `YOLOWorldInferenceRequest(image=..., yolo_world_version_id=version,
    # confidence=confidence, text=class_names, api_key=...)`.
    manager = _manager()
    manager.infer_from_request_sync.return_value = _DictResponse(
        {"image": {"width": 10, "height": 20}, "predictions": []}
    )
    image = _make_image()
    images = Batch(content=[image], indices=[(0,)])
    block = YoloWorldModelBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(images=images, class_names=["cat"], version="l", confidence=0.3)

    manager.add_model.assert_called_once_with(
        model_id="yolo_world/l", api_key="k", endpoint_type=CORE_MODEL_ENDPOINT_TYPE
    )
    request = _captured_request(manager)
    expected = YOLOWorldInferenceRequest(
        image=image.to_inference_format(numpy_preferred=True),
        yolo_world_version_id="l",
        confidence=0.3,
        text=["cat"],
        api_key="k",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set
