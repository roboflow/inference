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
