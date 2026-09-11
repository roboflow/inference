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


from inference.core.entities.requests.clip import (
    ClipCompareRequest,
    ClipImageEmbeddingRequest,
    ClipTextEmbeddingRequest,
)


class _EmbeddingResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _ComparisonResponse:
    def __init__(self, similarity):
        self._similarity = similarity

    def model_dump(self, **_kwargs):
        return {"similarity": self._similarity}


def test_run_clip_text_embedding_builds_the_request_and_returns_embeddings() -> None:
    manager = manager_returning(_EmbeddingResponse([[0.1, 0.2]]))
    assert ModelManagerModelsProvider(manager).run_clip_text_embedding(
        model_id="clip/ViT-B-16", version_id="ViT-B-16", text=["a cat"], api_key="k"
    ) == [[0.1, 0.2]]
    manager.add_model.assert_not_called()  # the block registers, not the adapter
    request = captured_request(manager)
    expected = ClipTextEmbeddingRequest(
        clip_version_id="ViT-B-16", text=["a cat"], api_key="k"
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})


def test_run_clip_image_embedding_builds_the_request_and_returns_embeddings() -> None:
    manager = manager_returning(_EmbeddingResponse([[0.3]]))
    assert ModelManagerModelsProvider(manager).run_clip_image_embedding(
        model_id="clip/ViT-B-16", version_id="ViT-B-16", images=IMAGES, api_key="k"
    ) == [[0.3]]
    request = captured_request(manager)
    expected = ClipImageEmbeddingRequest(
        clip_version_id="ViT-B-16", image=IMAGES, api_key="k"
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})


def test_run_clip_comparison_leaves_the_version_default_when_unset() -> None:
    from inference.core.roboflow_api import ModelEndpointType

    manager = manager_returning(_ComparisonResponse([0.5]))
    assert ModelManagerModelsProvider(manager).run_clip_comparison(
        subject=IMAGE,
        subject_type="image",
        prompt=["cat"],
        prompt_type="text",
        api_key="k",
    ) == {"similarity": [0.5]}
    request = captured_request(manager)
    expected = ClipCompareRequest(
        api_key="k",
        subject=IMAGE,
        subject_type="image",
        prompt=["cat"],
        prompt_type="text",
    )
    assert request.clip_version_id == expected.clip_version_id
    # This method - and only this method, among the CLIP/PE family - registers.
    manager.add_model.assert_called_once_with(
        f"clip/{expected.clip_version_id}",
        "k",
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )


def test_run_perception_encoder_embeddings_build_their_requests() -> None:
    from inference.core.entities.requests.perception_encoder import (
        PerceptionEncoderImageEmbeddingRequest,
        PerceptionEncoderTextEmbeddingRequest,
    )

    manager = manager_returning(_EmbeddingResponse([[1.0]]))
    provider = ModelManagerModelsProvider(manager)
    assert provider.run_perception_encoder_text_embedding(
        model_id="perception_encoder/v", version_id="v", text=["a"], api_key="k"
    ) == [[1.0]]
    request = captured_request(manager)
    expected = PerceptionEncoderTextEmbeddingRequest(
        perception_encoder_version_id="v", text=["a"], api_key="k"
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})

    manager = manager_returning(_EmbeddingResponse([[2.0]]))
    provider = ModelManagerModelsProvider(manager)
    assert provider.run_perception_encoder_image_embedding(
        model_id="perception_encoder/v", version_id="v", images=IMAGES, api_key="k"
    ) == [[2.0]]
    request = captured_request(manager)
    expected = PerceptionEncoderImageEmbeddingRequest(
        perception_encoder_version_id="v", image=IMAGES, api_key="k"
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})


from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.requests.pp_ocr import PPOCRInferenceRequest
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest


class _DictResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, **_kwargs):
        return self._payload


def test_run_doctr_ocr_builds_the_request_the_block_used_to_build() -> None:
    manager = manager_returning(_DictResponse({"result": "HELLO"}))
    assert ModelManagerModelsProvider(manager).run_doctr_ocr(
        model_id="doctr/x", image=IMAGE, api_key="k", generate_bounding_boxes=True
    ) == {"result": "HELLO"}
    manager.add_model.assert_not_called()
    request = captured_request(manager)
    expected = DoctrOCRInferenceRequest(
        image=IMAGE, api_key="k", generate_bounding_boxes=True
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})


def test_run_easy_ocr_builds_the_request_the_block_used_to_build() -> None:
    manager = manager_returning(_DictResponse({"result": "HI"}))
    ModelManagerModelsProvider(manager).run_easy_ocr(
        model_id="easy_ocr/v",
        version_id="v",
        image=IMAGE,
        api_key="k",
        language_codes=["en"],
        quantize=False,
    )
    request = captured_request(manager)
    expected = EasyOCRInferenceRequest(
        easy_ocr_version_id="v",
        image=IMAGE,
        api_key="k",
        language_codes=["en"],
        quantize=False,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})


def test_run_pp_ocr_builds_the_request_and_registers_the_derived_model_id() -> None:
    from inference.core.roboflow_api import ModelEndpointType

    manager = manager_returning(_DictResponse({"result": "HI"}))
    ModelManagerModelsProvider(manager).run_pp_ocr(
        image=IMAGE, api_key="k", text_detection="small", text_recognition="small"
    )
    request = captured_request(manager)
    expected = PPOCRInferenceRequest(
        text_detection="small", text_recognition="small", image=IMAGE, api_key="k"
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    manager.add_model.assert_called_once_with(
        "pp_ocr/small-small", "k", endpoint_type=ModelEndpointType.CORE_MODEL
    )


def test_run_pp_ocr_covers_detection_only_recognition_only_and_normalisation() -> None:
    """The validator (requests/pp_ocr.py:33-72) lower-cases the stages, maps a
    disabled stage to "none" and derives the id from the normalised pair."""
    for (detection, recognition), expected_id in [
        (("small", None), "pp_ocr/small-none"),
        ((None, "small"), "pp_ocr/none-small"),
        (("SMALL", "Medium"), "pp_ocr/small-medium"),
    ]:
        manager = manager_returning(_DictResponse({"result": "HI"}))
        ModelManagerModelsProvider(manager).run_pp_ocr(
            image=IMAGE,
            api_key="k",
            text_detection=detection,
            text_recognition=recognition,
        )
        assert manager.add_model.call_args.args[0] == expected_id, (
            detection,
            recognition,
        )


def test_run_pp_ocr_rejects_an_invalid_stage_before_registering() -> None:
    import pytest
    from pydantic import ValidationError

    manager = manager_returning(_DictResponse({}))
    with pytest.raises(ValidationError):
        ModelManagerModelsProvider(manager).run_pp_ocr(
            image=IMAGE,
            api_key="k",
            text_detection="enormous",
            text_recognition="small",
        )
    manager.add_model.assert_not_called()
    manager.infer_from_request_sync.assert_not_called()


def test_run_pp_ocr_rejects_both_stages_disabled() -> None:
    import pytest
    from pydantic import ValidationError

    manager = manager_returning(_DictResponse({}))
    with pytest.raises(ValidationError):
        ModelManagerModelsProvider(manager).run_pp_ocr(
            image=IMAGE, api_key="k", text_detection=None, text_recognition=None
        )
    manager.add_model.assert_not_called()


def test_run_pp_ocr_defaults_to_small_small_when_both_stages_omitted() -> None:
    """Fix round 1: every other `run_pp_ocr` test passes both stage keywords
    explicitly, so a regression that swapped the adapter's `UNSET` defaults for
    a plain `None` would go undetected - a plain `None` default is NOT filtered
    by `_passed()` (it only drops `_Unset` instances), so it would be forwarded
    as an explicit `text_detection=None, text_recognition=None`, which the
    validator rejects outright (see `test_run_pp_ocr_rejects_both_stages_disabled`
    above), rather than resolving to "small"/"small" as it does today."""
    from inference.core.roboflow_api import ModelEndpointType

    manager = manager_returning(_DictResponse({"result": "HI"}))
    ModelManagerModelsProvider(manager).run_pp_ocr(image=IMAGE, api_key="k")
    request = captured_request(manager)
    # Neither stage keyword is passed to the constructor either - mirroring
    # what `_passed()` forwards when both arguments are UNSET.
    expected = PPOCRInferenceRequest(image=IMAGE, api_key="k")
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set
    assert request.text_detection == "small"
    assert request.text_recognition == "small"
    manager.add_model.assert_called_once_with(
        "pp_ocr/small-small", "k", endpoint_type=ModelEndpointType.CORE_MODEL
    )


def test_run_pp_ocr_omits_one_stage_while_disabling_the_other() -> None:
    """Fix round 1: one stage omitted (UNSET, so the "small" default applies),
    the other explicitly disabled (`None`) - the two omitted-stage cases the
    existing coverage never exercised together."""
    from inference.core.roboflow_api import ModelEndpointType

    manager = manager_returning(_DictResponse({"result": "HI"}))
    ModelManagerModelsProvider(manager).run_pp_ocr(
        image=IMAGE, api_key="k", text_recognition=None
    )
    request = captured_request(manager)
    # `text_detection` is not passed to the constructor (UNSET was dropped by
    # `_passed()`); `text_recognition=None` is forwarded explicitly.
    expected = PPOCRInferenceRequest(image=IMAGE, api_key="k", text_recognition=None)
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set
    assert request.text_detection == "small"
    assert request.text_recognition == "none"
    manager.add_model.assert_called_once_with(
        "pp_ocr/small-none", "k", endpoint_type=ModelEndpointType.CORE_MODEL
    )


def test_run_yolo_world_builds_the_request_the_block_used_to_build() -> None:
    manager = manager_returning(_DictResponse({"predictions": []}))
    ModelManagerModelsProvider(manager).run_yolo_world(
        model_id="yolo_world/v",
        version_id="v",
        image=IMAGE,
        text=["cat"],
        api_key="k",
        confidence=0.3,
    )
    request = captured_request(manager)
    expected = YOLOWorldInferenceRequest(
        image=IMAGE,
        yolo_world_version_id="v",
        confidence=0.3,
        text=["cat"],
        api_key="k",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})


from inference.core.entities.requests.sam2 import (
    Box,
    Point,
    Sam2Prompt,
    Sam2PromptSet,
    Sam2SegmentationRequest,
)
from inference.core.entities.requests.sam3 import Sam3Prompt, Sam3SegmentationRequest
from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest


def test_run_sam2_segmentation_revives_box_and_point_prompts_from_a_version_id() -> (
    None
):
    response = object()
    manager = manager_returning(response)
    result = ModelManagerModelsProvider(manager).run_sam2_segmentation(
        model_id="sam2/hiera_large",
        image=IMAGE,
        prompts=[
            {"box": {"x": 5.0, "y": 5.0, "width": 4.0, "height": 4.0}},
            {"points": [{"x": 1.0, "y": 2.0, "positive": True}]},
        ],
        api_key="k",
        version_id="hiera_large",
        multimask_output=False,
        threshold=0.5,
    )
    assert result == [response]
    manager.add_model.assert_not_called()
    request = captured_request(manager)
    expected = Sam2SegmentationRequest(
        image=IMAGE,
        sam2_version_id="hiera_large",
        api_key="k",
        source="workflow-execution",
        prompts=Sam2PromptSet(
            prompts=[
                Sam2Prompt(box=Box(x=5.0, y=5.0, width=4.0, height=4.0)),
                Sam2Prompt(points=[Point(x=1.0, y=2.0, positive=True)]),
            ]
        ),
        threshold=0.5,
        multimask_output=False,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})


def test_run_sam2_segmentation_accepts_an_explicit_request_model_id() -> None:
    manager = manager_returning(object())
    ModelManagerModelsProvider(manager).run_sam2_segmentation(
        model_id="sam3-interactive",
        image=IMAGE,
        prompts=[{"box": {"x": 1.0, "y": 1.0, "width": 2.0, "height": 2.0}}],
        api_key="k",
        request_model_id="sam3-interactive",
        multimask_output=True,
    )
    assert captured_request(manager).model_id == "sam3-interactive"


def test_run_sam3_segmentation_revives_text_prompts() -> None:
    manager = manager_returning(object())
    ModelManagerModelsProvider(manager).run_sam3_segmentation(
        model_id="sam3/sam3_final",
        image=IMAGE,
        api_key="k",
        prompts=[{"type": "text", "text": "cat", "output_prob_thresh": 0.4}],
        output_prob_thresh=0.4,
        nms_iou_threshold=0.5,
    )
    request = captured_request(manager)
    expected = Sam3SegmentationRequest(
        api_key="k",
        model_id="sam3/sam3_final",
        image=IMAGE,
        prompts=[Sam3Prompt(type="text", text="cat", output_prob_thresh=0.4)],
        output_prob_thresh=0.4,
        nms_iou_threshold=0.5,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})


def test_run_sam3_segmentation_omits_unset_optionals() -> None:
    manager = manager_returning(object())
    ModelManagerModelsProvider(manager).run_sam3_segmentation(
        model_id="sam3/sam3_final",
        image=IMAGE,
        api_key="k",
        prompts=[{"type": "text", "text": "cat"}],
    )
    request = captured_request(manager)
    default = Sam3SegmentationRequest(
        api_key="k",
        model_id="sam3/sam3_final",
        image=IMAGE,
        prompts=[Sam3Prompt(type="text", text="cat")],
    )
    assert request.output_prob_thresh == default.output_prob_thresh
    assert request.nms_iou_threshold == default.nms_iou_threshold
    assert request.format == default.format


def test_run_sam3_3d_objects_builds_the_request_and_returns_the_response() -> None:
    response = object()
    manager = manager_returning(response)
    assert (
        ModelManagerModelsProvider(manager).run_sam3_3d_objects(
            model_id="sam3-3d-objects", image=IMAGE, mask_input=[[0, 1]], api_key="k"
        )
        is response
    )
    manager.add_model.assert_not_called()
    request = captured_request(manager)
    expected = Sam3_3D_Objects_InferenceRequest(
        image=IMAGE, mask_input=[[0, 1]], api_key="k", model_id="sam3-3d-objects"
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})


def test_sam2_prompt_dicts_encode_the_same_prompt_set_as_the_inline_construction() -> (
    None
):
    produced = ModelManagerModelsProvider(MagicMock())._sam2_prompt_set(
        [
            {"box": {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}},
            {"points": [{"x": 5.0, "y": 6.0, "positive": False}]},
        ]
    )
    expected = Sam2PromptSet(
        prompts=[
            Sam2Prompt(box=Box(x=1.0, y=2.0, width=3.0, height=4.0)),
            Sam2Prompt(points=[Point(x=5.0, y=6.0, positive=False)]),
        ]
    )
    assert produced.model_dump() == expected.model_dump()


# --- round-5 defect 1: an explicit None must have the SAME outcome through the
# adapter as through the block's inline construction - the same stored value,
# or the same ValidationError (type, message, error table). One row per caller
# shape whose argument the none-matrix flagged (Global Constraints, forwarding
# rule); the request classes are imported above (Tasks 11.8-11.14).
from pydantic import ValidationError

BOX_PROMPT = {"box": {"x": 30.0, "y": 30.0, "width": 40.0, "height": 40.0}}
BOX_PROMPT_SET = Sam2PromptSet(
    prompts=[Sam2Prompt(box=Box(x=30.0, y=30.0, width=40.0, height=40.0))]
)
ISEG_COMMON = dict(
    class_agnostic_nms=False,
    class_filter=None,
    confidence=0.4,
    iou_threshold=0.3,
    max_detections=300,
    max_candidates=3000,
    mask_decode_mode="accurate",
    tradeoff_factor=0.0,
    disable_active_learning=False,
    active_learning_target_dataset=None,
)


def _outcome(build):
    """What the caller observes: the request dump, or the error as the HTTP
    layer exposes it (error_handlers.py:148-152: type name and str(error))."""
    try:
        return ("ok", build().model_dump(exclude={"id"}))
    except ValidationError as error:
        return (
            "error",
            type(error).__name__,
            str(error),
            [(e["loc"], e["type"]) for e in error.errors()],
        )


# (id, adapter method, adapter kwargs, request class, the kwargs the block passed inline)
NONE_CASES = [
    (
        "segment_anything2/v1.py: multimask_output=None",
        "run_sam2_segmentation",
        dict(
            model_id="sam2/hiera_large",
            image=IMAGE,
            prompts=[BOX_PROMPT],
            api_key="k",
            version_id="hiera_large",
            threshold=0.0,
            multimask_output=None,
        ),
        Sam2SegmentationRequest,
        dict(
            image=IMAGE,
            sam2_version_id="hiera_large",
            api_key="k",
            source="workflow-execution",
            prompts=BOX_PROMPT_SET,
            threshold=0.0,
            multimask_output=None,
        ),
    ),
    (
        "segment_anything3_interactive/v1.py: multimask_output=None",
        "run_sam2_segmentation",
        dict(
            model_id="sam3-interactive",
            image=IMAGE,
            prompts=[BOX_PROMPT],
            api_key="k",
            request_model_id="sam3-interactive",
            multimask_output=None,
        ),
        Sam2SegmentationRequest,
        dict(
            image=IMAGE,
            model_id="sam3-interactive",
            api_key="k",
            source="workflow-execution",
            prompts=BOX_PROMPT_SET,
            multimask_output=None,
        ),
    ),
    (
        "segment_anything2/v1.py: version=None",
        "run_sam2_segmentation",
        dict(
            model_id="sam2/None",
            image=IMAGE,
            prompts=[BOX_PROMPT],
            api_key="k",
            version_id=None,
            threshold=0.0,
            multimask_output=True,
        ),
        Sam2SegmentationRequest,
        dict(
            image=IMAGE,
            sam2_version_id=None,
            api_key="k",
            source="workflow-execution",
            prompts=BOX_PROMPT_SET,
            threshold=0.0,
            multimask_output=True,
        ),
    ),
    (
        "segment_anything3/v1.py: threshold=None",
        "run_sam3_segmentation",
        dict(
            model_id="sam3/sam3_final",
            image=IMAGE,
            prompts=[{"type": "text", "text": "cat"}],
            api_key="k",
            output_prob_thresh=None,
        ),
        Sam3SegmentationRequest,
        dict(
            image=IMAGE,
            model_id="sam3/sam3_final",
            api_key="k",
            prompts=[Sam3Prompt(type="text", text="cat")],
            output_prob_thresh=None,
        ),
    ),
    (
        "segment_anything3/v2.py: apply_nms=False (nms_iou_threshold=None)",
        "run_sam3_segmentation",
        dict(
            model_id="sam3/sam3_final",
            image=IMAGE,
            prompts=[{"type": "text", "text": "cat"}],
            api_key="k",
            output_prob_thresh=0.5,
            nms_iou_threshold=None,
        ),
        Sam3SegmentationRequest,
        dict(
            image=IMAGE,
            model_id="sam3/sam3_final",
            api_key="k",
            prompts=[Sam3Prompt(type="text", text="cat")],
            output_prob_thresh=0.5,
            nms_iou_threshold=None,
        ),
    ),
    (
        "segment_anything3/v3.py: format=None",
        "run_sam3_segmentation",
        dict(
            model_id="sam3/sam3_final",
            image=IMAGE,
            prompts=[{"type": "text", "text": "cat"}],
            api_key="k",
            output_prob_thresh=0.5,
            nms_iou_threshold=0.9,
            format=None,
        ),
        Sam3SegmentationRequest,
        dict(
            image=IMAGE,
            model_id="sam3/sam3_final",
            api_key="k",
            prompts=[Sam3Prompt(type="text", text="cat")],
            output_prob_thresh=0.5,
            nms_iou_threshold=0.9,
            format=None,
        ),
    ),
    (
        "instance_segmentation/v1.py: enforce_dense_masks_in_inference_models=None",
        "run_instance_segmentation",
        dict(
            model_id="m/1",
            images=IMAGES,
            api_key="k",
            enforce_dense_masks_in_inference_models=None,
            **ISEG_COMMON,
        ),
        InstanceSegmentationInferenceRequest,
        dict(
            api_key="k",
            model_id="m/1",
            image=IMAGES,
            source="workflow-execution",
            enforce_dense_masks_in_inference_models=None,
            **ISEG_COMMON,
        ),
    ),
    (
        "qwen3_5vl/v1.py (any LMM block passing it): enable_thinking=None",
        "run_lmm",
        dict(
            model_id="m/1", image=IMAGE, prompt="p", api_key="k", enable_thinking=None
        ),
        LMMInferenceRequest,
        dict(
            api_key="k",
            model_id="m/1",
            image=IMAGE,
            source="workflow-execution",
            prompt="p",
            enable_thinking=None,
        ),
    ),
    (
        "clip_comparison/v2.py: version=None",
        "run_clip_comparison",
        dict(
            subject=IMAGE,
            subject_type="image",
            prompt=["cat"],
            prompt_type="text",
            api_key="k",
            version_id=None,
        ),
        ClipCompareRequest,
        dict(
            clip_version_id=None,
            subject=IMAGE,
            subject_type="image",
            prompt=["cat"],
            prompt_type="text",
            api_key="k",
        ),
    ),
    (
        "semantic_segmentation/v2.py: confidence=None",
        "run_semantic_segmentation",
        dict(
            model_id="m/1",
            images=IMAGES,
            api_key="k",
            confidence=None,
            response_mask_format="numpy",
        ),
        SemanticSegmentationInferenceRequest,
        dict(
            api_key="k",
            model_id="m/1",
            image=IMAGES,
            confidence=None,
            response_mask_format="numpy",
            source="workflow-execution",
        ),
    ),
]


@pytest.mark.parametrize("case", NONE_CASES, ids=[case[0] for case in NONE_CASES])
def test_an_explicit_none_has_the_same_outcome_through_the_adapter_as_inline(
    case,
) -> None:
    _, method, adapter_kwargs, request_class, inline_kwargs = case
    manager = MagicMock()
    manager.__contains__.return_value = True
    manager.infer_from_request_sync.return_value = MagicMock(embeddings=[[0.0]])
    provider = ModelManagerModelsProvider(manager)

    def through_adapter():
        getattr(provider, method)(**adapter_kwargs)
        return captured_request(manager)

    inline = _outcome(lambda: request_class(**inline_kwargs))
    assert _outcome(through_adapter) == inline
    if inline[0] == "error":
        manager.infer_from_request_sync.assert_not_called()


def test_clip_comparison_registers_the_id_the_block_derived_for_a_none_version() -> (
    None
):
    """`load_core_model` read `inference_request.clip_version_id` (None) and
    registered `clip/None`; the adapter must not silently upgrade that to the
    default version."""
    manager = MagicMock()
    manager.infer_from_request_sync.return_value = MagicMock()
    ModelManagerModelsProvider(manager).run_clip_comparison(
        subject=IMAGE,
        subject_type="image",
        prompt=["cat"],
        prompt_type="text",
        api_key="k",
        version_id=None,
    )
    assert manager.add_model.call_args.args[0] == "clip/None"


def test_sam2_multimask_none_is_rejected_before_inference_with_the_inline_message() -> (
    None
):
    """The reviewer's reproduction: both consuming manifests allow None, both
    forward it, and the request field is a non-optional bool."""
    manager = MagicMock()
    with pytest.raises(ValidationError) as through_adapter:
        ModelManagerModelsProvider(manager).run_sam2_segmentation(
            model_id="sam2/hiera_large",
            image=IMAGE,
            prompts=[BOX_PROMPT],
            api_key="k",
            version_id="hiera_large",
            threshold=0.0,
            multimask_output=None,
        )
    with pytest.raises(ValidationError) as inline:
        Sam2SegmentationRequest(
            image=IMAGE,
            sam2_version_id="hiera_large",
            api_key="k",
            source="workflow-execution",
            prompts=BOX_PROMPT_SET,
            threshold=0.0,
            multimask_output=None,
        )
    assert type(through_adapter.value) is type(inline.value)
    assert str(through_adapter.value) == str(inline.value)
    assert "multimask_output" in str(inline.value) and "bool_type" in str(inline.value)
    manager.infer_from_request_sync.assert_not_called()
