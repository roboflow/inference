"""Block -> adapter -> request differential tests for Task 11.8's six block
families.

Each test drives a block's real `run_locally` through a real
`ModelManagerModelsProvider` wrapping a `MagicMock` `ModelManager`, then
compares the pydantic request the adapter built against the exact request
construction the block used to run inline before Task 11.8 (copied from
`git show 93b644ca7:.../<family>/v1.py`, the commit immediately before that
task). `_post_process_result` is stubbed out because these tests only care
about the request that reaches `infer_from_request_sync`, not about parsing
a mocked response into detections/predictions.
"""

from unittest.mock import MagicMock

import numpy as np

from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
    SemanticSegmentationInferenceRequest,
)
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.keypoint_detection.v1 import (
    RoboflowKeypointDetectionModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.multi_class_classification.v1 import (
    RoboflowClassificationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v1 import (
    RoboflowMultiLabelClassificationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v2 import (
    RoboflowMultiLabelClassificationModelBlockV2,
)
from inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v3 import (
    RoboflowMultiLabelClassificationModelBlockV3,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v1 import (
    RoboflowObjectDetectionModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v3 import (
    RoboflowObjectDetectionModelBlockV3,
)
from inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v1 import (
    RoboflowSemanticSegmentationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v2 import (
    RoboflowSemanticSegmentationModelBlockV2,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


def _make_images() -> Batch:
    # numpy_preferred=True (what every one of these blocks calls) turns this
    # into {"type": "numpy_object", "value": <this exact ndarray>}. Building
    # the "expected" request from the SAME Batch/WorkflowImageData instance
    # the block runs against keeps `value` the identical object both times
    # (WorkflowImageData caches `numpy_image`), so dict/list equality on the
    # dumped requests hits Python's identity fast path instead of trying an
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


def _assert_registers_before_inferring(
    manager: MagicMock, model_id: str, api_key: str
) -> None:
    names = [call[0] for call in manager.method_calls]
    assert "add_model" in names and "infer_from_request_sync" in names
    assert names.index("add_model") < names.index("infer_from_request_sync"), names
    manager.add_model.assert_called_once_with(model_id=model_id, api_key=api_key)


def test_object_detection_v1_request_matches_the_pre_port_construction() -> None:
    manager = _manager()
    images = _make_images()
    block = RoboflowObjectDetectionModelBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._post_process_result = lambda **_: None

    block.run_locally(
        images=images,
        model_id="m/1",
        class_agnostic_nms=True,
        class_filter=["cat"],
        confidence=0.6,
        iou_threshold=0.4,
        max_detections=10,
        max_candidates=100,
        disable_active_learning=True,
        active_learning_target_dataset="ds",
    )

    _assert_registers_before_inferring(manager, model_id="m/1", api_key="k")
    request = _captured_request(manager)

    # Copied verbatim from `git show 93b644ca7:.../object_detection/v1.py`.
    inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
    expected = ObjectDetectionInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images,
        disable_active_learning=True,
        active_learning_target_dataset="ds",
        class_agnostic_nms=True,
        class_filter=["cat"],
        confidence=0.6,
        iou_threshold=0.4,
        max_detections=10,
        max_candidates=100,
        source="workflow-execution",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_object_detection_v3_request_matches_the_pre_port_construction() -> None:
    manager = _manager()
    images = _make_images()
    block = RoboflowObjectDetectionModelBlockV3(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._post_process_result = lambda **_: None

    block.run_locally(
        images=images,
        model_id="m/1",
        class_agnostic_nms=True,
        class_filter=["cat"],
        confidence=0.6,
        iou_threshold=0.4,
        max_detections=10,
        max_candidates=100,
        disable_active_learning=True,
        active_learning_target_dataset="ds",
    )

    _assert_registers_before_inferring(manager, model_id="m/1", api_key="k")
    request = _captured_request(manager)

    # Copied verbatim from `git show 93b644ca7:.../object_detection/v3.py`
    # (identical request construction to v1).
    inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
    expected = ObjectDetectionInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images,
        disable_active_learning=True,
        active_learning_target_dataset="ds",
        class_agnostic_nms=True,
        class_filter=["cat"],
        confidence=0.6,
        iou_threshold=0.4,
        max_detections=10,
        max_candidates=100,
        source="workflow-execution",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_multi_class_classification_v1_request_matches_the_pre_port_construction() -> (
    None
):
    manager = _manager()
    images = _make_images()
    block = RoboflowClassificationModelBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._post_process_result = lambda **_: None

    block.run_locally(
        images=images,
        model_id="m/1",
        confidence=0.7,
        disable_active_learning=False,
        active_learning_target_dataset=None,
    )

    _assert_registers_before_inferring(manager, model_id="m/1", api_key="k")
    request = _captured_request(manager)

    # Copied verbatim from `git show 93b644ca7:.../multi_class_classification/v1.py`.
    inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
    expected = ClassificationInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images,
        confidence=0.7,
        disable_active_learning=False,
        source="workflow-execution",
        active_learning_target_dataset=None,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_multi_label_classification_v1_request_matches_the_pre_port_construction() -> (
    None
):
    manager = _manager()
    images = _make_images()
    block = RoboflowMultiLabelClassificationModelBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._post_process_result = lambda **_: None

    block.run_locally(
        images=images,
        model_id="m/1",
        confidence=0.7,
        disable_active_learning=False,
        active_learning_target_dataset=None,
    )

    _assert_registers_before_inferring(manager, model_id="m/1", api_key="k")
    request = _captured_request(manager)

    # Copied verbatim from `git show 93b644ca7:.../multi_label_classification/v1.py`.
    inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
    expected = ClassificationInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images,
        confidence=0.7,
        disable_active_learning=False,
        source="workflow-execution",
        active_learning_target_dataset=None,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_multi_label_classification_v2_forwards_the_separate_confidence_keyword() -> (
    None
):
    # Round-2 review finding: v2 passes `confidence` to `infer_from_request_sync`
    # as a SEPARATE keyword in addition to the request field - the adapter
    # carries this as `inference_kwargs`. Assert both the request and that
    # extra keyword match the pre-11.8 call.
    manager = _manager()
    images = _make_images()
    block = RoboflowMultiLabelClassificationModelBlockV2(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._post_process_result = lambda **_: None

    block.run_locally(
        images=images,
        model_id="m/1",
        confidence=0.7,
        disable_active_learning=False,
        active_learning_target_dataset=None,
    )

    _assert_registers_before_inferring(manager, model_id="m/1", api_key="k")
    request = _captured_request(manager)

    # Copied verbatim from `git show 93b644ca7:.../multi_label_classification/v2.py`.
    inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
    expected = ClassificationInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images,
        confidence=0.7,
        disable_active_learning=False,
        source="workflow-execution",
        active_learning_target_dataset=None,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set

    # Old code: `infer_from_request_sync(model_id=model_id, request=request,
    # confidence=confidence)` - the extra `confidence` keyword, unchanged.
    call = manager.infer_from_request_sync.call_args
    assert call.kwargs["confidence"] == 0.7


def test_multi_label_classification_v3_forwards_the_separate_confidence_keyword() -> (
    None
):
    # Same shape as v2 (round-2 review finding); v3 additionally resolves
    # `confidence` from `confidence_mode`/`custom_confidence` in `run()`, but
    # `run_locally` (called directly here) takes the resolved `confidence`.
    manager = _manager()
    images = _make_images()
    block = RoboflowMultiLabelClassificationModelBlockV3(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._post_process_result = lambda **_: None

    block.run_locally(
        images=images,
        model_id="m/1",
        confidence=0.7,
        disable_active_learning=False,
        active_learning_target_dataset=None,
    )

    _assert_registers_before_inferring(manager, model_id="m/1", api_key="k")
    request = _captured_request(manager)

    # Copied verbatim from `git show 93b644ca7:.../multi_label_classification/v3.py`.
    inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
    expected = ClassificationInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images,
        confidence=0.7,
        disable_active_learning=False,
        source="workflow-execution",
        active_learning_target_dataset=None,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set

    # Old code: `infer_from_request_sync(model_id=model_id, request=request,
    # confidence=confidence)` - the extra `confidence` keyword, unchanged.
    call = manager.infer_from_request_sync.call_args
    assert call.kwargs["confidence"] == 0.7


def test_keypoint_detection_v1_request_matches_the_pre_port_construction() -> None:
    manager = _manager()
    images = _make_images()
    block = RoboflowKeypointDetectionModelBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._post_process_result = lambda **_: None

    block.run_locally(
        images=images,
        model_id="m/1",
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.4,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        keypoint_confidence=0.5,
        disable_active_learning=False,
        active_learning_target_dataset=None,
    )

    _assert_registers_before_inferring(manager, model_id="m/1", api_key="k")
    request = _captured_request(manager)

    # Copied verbatim from `git show 93b644ca7:.../keypoint_detection/v1.py`.
    inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
    expected = KeypointsDetectionInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images,
        disable_active_learning=False,
        active_learning_target_dataset=None,
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.4,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        keypoint_confidence=0.5,
        source="workflow-execution",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_semantic_segmentation_v1_request_matches_the_pre_port_construction() -> None:
    manager = _manager()
    images = _make_images()
    block = RoboflowSemanticSegmentationModelBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._post_process_result = lambda **_: None

    block.run_locally(images=images, model_id="m/1")

    _assert_registers_before_inferring(manager, model_id="m/1", api_key="k")
    request = _captured_request(manager)

    # Copied verbatim from `git show 93b644ca7:.../semantic_segmentation/v1.py`
    # - v1 never sets `confidence`.
    inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
    expected = SemanticSegmentationInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images,
        response_mask_format="numpy",
        source="workflow-execution",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_semantic_segmentation_v2_request_matches_the_pre_port_construction() -> None:
    # v2's argument set differs from v1 (it resolves and forwards `confidence`),
    # so it gets its own case per the fix-round instructions.
    manager = _manager()
    images = _make_images()
    block = RoboflowSemanticSegmentationModelBlockV2(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._post_process_result = lambda **_: None

    block.run_locally(images=images, model_id="m/1", confidence=0.4)

    _assert_registers_before_inferring(manager, model_id="m/1", api_key="k")
    request = _captured_request(manager)

    # Copied verbatim from `git show 93b644ca7:.../semantic_segmentation/v2.py`.
    inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
    expected = SemanticSegmentationInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=inference_images,
        confidence=0.4,
        response_mask_format="numpy",
        source="workflow-execution",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set
