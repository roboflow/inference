import ast
import inspect
from unittest.mock import MagicMock

import pytest

import inference.core.interfaces.workflows_models_provider as adapter_module
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
)
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.workflows.prototypes.models_provider import ModelsProvider

# A payload `InferenceRequestImage` accepts. A bare string does not validate.
IMAGE = {"type": "base64", "value": "aGVsbG8="}
IMAGES = [IMAGE]


def empty_detection_response() -> ObjectDetectionInferenceResponse:
    return ObjectDetectionInferenceResponse(
        image=InferenceResponseImage(width=10, height=20), predictions=[]
    )


def manager_returning(value) -> MagicMock:
    manager = MagicMock()
    manager.infer_from_request_sync.return_value = value
    manager.__contains__.return_value = True  # MagicMock's default is False
    return manager


def captured_request(manager):
    assert manager.infer_from_request_sync.call_count == 1
    call = manager.infer_from_request_sync.call_args
    return call.kwargs["request"] if "request" in call.kwargs else call.args[1]


def test_adapter_implements_every_port_member() -> None:
    for name, member in vars(ModelsProvider).items():
        if not callable(member):
            continue
        if name.startswith("_") and name != "__contains__":
            continue
        assert hasattr(ModelManagerModelsProvider, name), name
    assert isinstance(
        vars(ModelManagerModelsProvider)["content_addressed_artifact_cache"], property
    )


def test_adapter_forwards_each_member_with_its_own_valid_arguments() -> None:
    manager = MagicMock()
    manager.__contains__.return_value = True
    provider = ModelManagerModelsProvider(manager)

    # `add_model` is forwarded exactly as the block called it, so a class-level
    # test patch on `ModelManager.add_model` sees the call it sees today.
    provider.add_model(model_id="m/1", api_key="key")
    manager.add_model.assert_called_once_with(model_id="m/1", api_key="key")
    manager.add_model.reset_mock()
    provider.add_model("m/2", "key", model_id_alias="alias", endpoint_type="core_model")
    manager.add_model.assert_called_once_with(
        model_id="m/2",
        api_key="key",
        model_id_alias="alias",
        endpoint_type="core_model",
    )

    sentinel = object()
    provider.infer_from_request_sync(model_id="m/1", request=sentinel)
    manager.infer_from_request_sync.assert_called_once_with(
        model_id="m/1", request=sentinel
    )

    provider.run_tensor_native_inference("m/1", images=[1])
    manager.run_tensor_native_inference.assert_called_once_with(
        model_id="m/1", images=[1]
    )

    for name in (
        "get_class_names",
        "get_keypoints_classes",
        "model_supports_stream_pipeline",
        "get_model_pipeline_depth",
        "flush_model_stream_pipeline",
        "shutdown_model_stream_pipeline",
    ):
        getattr(provider, name)("m/1")
        getattr(manager, name).assert_called_once_with("m/1")

    # Phase 9's loader (Task 9.9), forwarded with the block's keyword shape.
    assert (
        provider.load_action_recognition_model(model_id="ar/1", api_key="k")
        is manager.load_action_recognition_model.return_value
    )
    manager.load_action_recognition_model.assert_called_once_with(
        model_id="ar/1", api_key="k"
    )

    assert "m/1" in provider
    manager.__contains__.assert_called_once_with("m/1")
    assert (
        provider.content_addressed_artifact_cache
        is manager.content_addressed_artifact_cache
    )
    assert provider._model_manager is manager


def test_adapter_signatures_match_the_port() -> None:
    for name, member in vars(ModelsProvider).items():
        if not callable(member):
            continue
        if name.startswith("_") and name != "__contains__":
            continue
        port_parameters = inspect.signature(member).parameters
        adapter_parameters = inspect.signature(
            getattr(ModelManagerModelsProvider, name)
        ).parameters
        for key, port_parameter in port_parameters.items():
            if key in ("self", "kwargs"):
                continue
            assert key in adapter_parameters, f"{name}.{key}"
            assert (
                port_parameter.default == adapter_parameters[key].default
            ), f"{name}.{key}"


def test_adapter_registers_only_for_the_validator_derived_core_models() -> None:
    """Registration stays in the blocks (instance_segmentation/v3.py reads the
    loaded model's pipeline depth between registering and inferring). The two
    exceptions are the core models whose id only exists on the VALIDATED
    request - CLIP comparison (pydantic default version) and PP-OCR (validator-
    derived id) - and there the order is build -> register -> infer."""
    tree = ast.parse(inspect.getsource(adapter_module))
    registering = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("run_"):
            continue
        events = []
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            if isinstance(inner.func, ast.Attribute) and inner.func.attr in (
                "add_model",
                "_infer",
            ):
                events.append((inner.func.attr, inner.lineno))
            elif isinstance(inner.func, ast.Name) and inner.func.id.endswith("Request"):
                events.append(("build", inner.lineno))
        if any(kind == "add_model" for kind, _ in events):
            registering[node.name] = [
                kind for kind, _ in sorted(events, key=lambda event: event[1])
            ]
    assert set(registering) <= {"run_clip_comparison", "run_pp_ocr"}, registering
    for name, order in registering.items():
        assert (
            order.index("build") < order.index("add_model") < order.index("_infer")
        ), (
            name,
            order,
        )


def test_action_recognition_block_loads_through_the_adapter() -> None:
    """Round-3 defect 4: the injected wrapper must carry Phase 9's loader all
    the way to the block. Executes Phase 9's block path through the adapter."""
    from inference.core.workflows.core_steps.common.entities import StepExecutionMode
    from inference.core.workflows.core_steps.models.roboflow.action_recognition.v1 import (
        ActionRecognitionModelBlockV1,
    )

    if "model_manager" not in ActionRecognitionModelBlockV1.get_init_parameters():
        pytest.skip("Phase 9 Task 9.9 (block takes model_manager) has not landed")
    manager = MagicMock()
    block = ActionRecognitionModelBlockV1(
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
        model_manager=ModelManagerModelsProvider(manager),
    )
    assert (
        block._get_model("cosmos-3-edge")
        is manager.load_action_recognition_model.return_value
    )
    manager.load_action_recognition_model.assert_called_once_with(
        model_id="cosmos-3-edge", api_key="k"
    )


from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
    SemanticSegmentationInferenceRequest,
)


def test_run_object_detection_builds_the_request_the_block_used_to_build() -> None:
    response = empty_detection_response()
    manager = manager_returning(response)
    provider = ModelManagerModelsProvider(manager)

    result = provider.run_object_detection(
        model_id="m/1",
        images=IMAGES,
        api_key="k",
        class_agnostic_nms=True,
        class_filter=["cat"],
        confidence=0.6,
        iou_threshold=0.4,
        max_detections=10,
        max_candidates=100,
        disable_active_learning=True,
        active_learning_target_dataset="ds",
    )

    manager.add_model.assert_not_called()  # registration stays in the block
    request = captured_request(manager)
    expected = ObjectDetectionInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=IMAGES,
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
    assert isinstance(request, ObjectDetectionInferenceRequest)
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert result == [response.model_dump(by_alias=True, exclude_none=True)]


def test_run_object_detection_accepts_a_symbolic_confidence() -> None:
    manager = manager_returning(empty_detection_response())
    ModelManagerModelsProvider(manager).run_object_detection(
        model_id="m/1", images=IMAGES, api_key="k", confidence="best"
    )
    assert captured_request(manager).confidence == "best"


def test_run_classification_builds_the_request_the_block_used_to_build() -> None:
    from inference.core.entities.responses.inference import (
        ClassificationInferenceResponse,
    )

    response = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=10, height=20),
        predictions=[],
        top="cat",
        confidence=0.9,
    )
    manager = manager_returning(response)
    ModelManagerModelsProvider(manager).run_classification(
        model_id="m/1",
        images=IMAGES,
        api_key="k",
        confidence=0.7,
        disable_active_learning=False,
        active_learning_target_dataset=None,
    )
    request = captured_request(manager)
    expected = ClassificationInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=IMAGES,
        confidence=0.7,
        disable_active_learning=False,
        source="workflow-execution",
        active_learning_target_dataset=None,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})


def test_run_keypoints_detection_builds_the_request_the_block_used_to_build() -> None:
    manager = manager_returning(empty_detection_response())
    ModelManagerModelsProvider(manager).run_keypoints_detection(
        model_id="m/1",
        images=IMAGES,
        api_key="k",
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
    request = captured_request(manager)
    expected = KeypointsDetectionInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=IMAGES,
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


def test_run_semantic_segmentation_keeps_the_numpy_fast_path() -> None:
    manager = manager_returning(empty_detection_response())
    ModelManagerModelsProvider(manager).run_semantic_segmentation(
        model_id="m/1", images=IMAGES, api_key="k", response_mask_format="numpy"
    )
    request = captured_request(manager)
    assert isinstance(request, SemanticSegmentationInferenceRequest)
    assert request.response_mask_format == "numpy"
    assert request.source == "workflow-execution"


def test_run_semantic_segmentation_omits_confidence_when_never_passed() -> None:
    manager = manager_returning(empty_detection_response())
    ModelManagerModelsProvider(manager).run_semantic_segmentation(
        model_id="m/1", images=IMAGES, api_key="k"
    )
    request = captured_request(manager)
    # v1 never sets it; the pydantic default must survive, not become None.
    assert (
        request.confidence
        == SemanticSegmentationInferenceRequest(
            api_key="k", model_id="m/1", image=IMAGES
        ).confidence
    )


def test_run_semantic_segmentation_still_rejects_an_explicit_none_confidence() -> None:
    """Round-2 defect 5: v2 forwards its manifest value, which can be None, and
    that raises today. An UNSET sentinel keeps 'not passed' and 'passed None'
    distinct."""
    import pytest
    from pydantic import ValidationError

    manager = manager_returning(empty_detection_response())
    with pytest.raises(ValidationError):
        ModelManagerModelsProvider(manager).run_semantic_segmentation(
            model_id="m/1", images=IMAGES, api_key="k", confidence=None
        )
    manager.infer_from_request_sync.assert_not_called()


def test_run_classification_forwards_extra_inference_kwargs() -> None:
    """multi_label v2/v3 pass `confidence` to the model call as well as into the
    request; the port carries it as `inference_kwargs`."""
    from inference.core.entities.responses.inference import (
        ClassificationInferenceResponse,
    )

    manager = manager_returning(
        ClassificationInferenceResponse(
            image=InferenceResponseImage(width=10, height=20),
            predictions=[],
            top="cat",
            confidence=0.9,
        )
    )
    ModelManagerModelsProvider(manager).run_classification(
        model_id="m/1",
        images=IMAGES,
        api_key="k",
        confidence=0.7,
        inference_kwargs={"confidence": 0.7},
    )
    assert manager.infer_from_request_sync.call_args.kwargs["confidence"] == 0.7


def test_run_methods_normalise_a_single_response_to_a_list() -> None:
    response = empty_detection_response()
    manager = manager_returning(response)
    assert ModelManagerModelsProvider(manager).run_object_detection(
        model_id="m/1", images=IMAGES, confidence=0.4
    ) == [response.model_dump(by_alias=True, exclude_none=True)]


from inference.core.entities.requests.inference import (
    InstanceSegmentationInferenceRequest,
)
from inference.core.workflows.prototypes.models_provider import InferenceResultsDC


def test_run_instance_segmentation_omits_unset_optional_fields() -> None:
    manager = manager_returning([empty_detection_response()])
    ModelManagerModelsProvider(manager).run_instance_segmentation(
        model_id="m/1",
        images=IMAGES,
        api_key="k",
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.4,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        mask_decode_mode="accurate",
        tradeoff_factor=0.0,
        enforce_dense_masks_in_inference_models=True,
        disable_active_learning=False,
        active_learning_target_dataset=None,
    )
    manager.add_model.assert_not_called()
    request = captured_request(manager)
    assert isinstance(request, InstanceSegmentationInferenceRequest)
    assert request.enforce_dense_masks_in_inference_models is True
    assert request.stream_pipeline_context_id is None
    default = InstanceSegmentationInferenceRequest(
        api_key="k", model_id="m/1", image=IMAGES
    )
    assert request.response_mask_format == default.response_mask_format


def test_run_instance_segmentation_can_return_raw_responses() -> None:
    raw = [object(), object()]
    manager = manager_returning(raw)
    result = ModelManagerModelsProvider(manager).run_instance_segmentation(
        model_id="m/1",
        images=IMAGES,
        api_key="k",
        confidence=0.4,
        stream_pipeline_context_id="ctx-1",
        return_raw_responses=True,
    )
    assert isinstance(result, InferenceResultsDC)
    assert result.raw_responses == raw
    assert result.predictions == []
    assert captured_request(manager).stream_pipeline_context_id == "ctx-1"


from inference.core.entities.requests.inference import (
    DepthEstimationRequest,
    LMMInferenceRequest,
)
from inference.core.entities.requests.moondream2 import Moondream2InferenceRequest


class _LMMResponse:
    """Minimal stand-in with the two members the adapter and blocks touch."""

    def __init__(self, response):
        self.response = response

    def model_dump(self, **_kwargs):
        return {"response": self.response}


def test_run_lmm_omits_thinking_and_token_budget_when_unset() -> None:
    manager = manager_returning(_LMMResponse("hi"))
    provider = ModelManagerModelsProvider(manager)
    assert provider.run_lmm(model_id="m/1", image=IMAGE, prompt="p", api_key="k") == {
        "response": "hi"
    }
    request = captured_request(manager)
    expected = LMMInferenceRequest(
        api_key="k",
        model_id="m/1",
        image=IMAGE,
        source="workflow-execution",
        prompt="p",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})


def test_run_lmm_passes_thinking_and_token_budget_when_set() -> None:
    manager = manager_returning(_LMMResponse("hi"))
    ModelManagerModelsProvider(manager).run_lmm(
        model_id="m/1",
        image=IMAGE,
        prompt="p",
        api_key="k",
        enable_thinking=True,
        max_new_tokens=64,
    )
    request = captured_request(manager)
    assert request.enable_thinking is True and request.max_new_tokens == 64


def test_run_moondream2_builds_the_request_the_block_used_to_build() -> None:
    manager = manager_returning(_LMMResponse("cat"))
    ModelManagerModelsProvider(manager).run_moondream2(
        model_id="md/2", image=IMAGE, prompt="p", text=[], api_key="k"
    )
    request = captured_request(manager)
    expected = Moondream2InferenceRequest(
        api_key="k", model_id="md/2", image=IMAGE, text=[], prompt="p"
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})


def test_run_depth_estimation_returns_the_raw_response_field() -> None:
    manager = manager_returning(_LMMResponse("depth-map"))
    assert (
        ModelManagerModelsProvider(manager).run_depth_estimation(
            model_id="d/1", image=IMAGE
        )
        == "depth-map"
    )
    assert isinstance(captured_request(manager), DepthEstimationRequest)
