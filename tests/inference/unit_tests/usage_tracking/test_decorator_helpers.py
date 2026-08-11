from types import SimpleNamespace
from unittest import mock

from inference.core.entities.requests.sam2 import Sam2InferenceRequest
from inference.core.env import SAM2_VERSION_ID, SAM3_EXEC_MODE
from inference.usage_tracking import decorator_helpers
from inference.usage_tracking.decorator_helpers import (
    get_model_type_from_kwargs,
    get_request_resource_details_from_kwargs,
)
from inference.usage_tracking.model_types import (
    clear_recorded_model_types,
    get_recorded_model_type,
    record_model_type,
)


def test_get_request_resource_details_ignores_default_countinference():
    result = get_request_resource_details_from_kwargs({"countinference": None})

    assert "billable" not in result


def test_get_request_resource_details_respects_explicit_countinference():
    result = get_request_resource_details_from_kwargs({"countinference": False})

    assert result["billable"] is False


def test_get_request_resource_details_tags_sam3_execution_mode(monkeypatch):
    monkeypatch.setattr(decorator_helpers, "SAM3_EXEC_MODE", "remote", raising=False)

    result = get_request_resource_details_from_kwargs(
        {"inference_request": SimpleNamespace(model_id="sam3/sam3_interactive")}
    )

    assert result["execution_mode"] == "remote"


def test_get_request_resource_details_skips_execution_mode_for_non_sam3_model():
    result = get_request_resource_details_from_kwargs(
        {"inference_request": SimpleNamespace(model_id="some-project/1")}
    )

    assert "execution_mode" not in result


def test_sam2_inference_request_model_id_defaults_to_sam2_version():
    request = Sam2InferenceRequest(api_key="key")

    assert request.model_id == f"sam2/{SAM2_VERSION_ID}"


def test_extract_usage_params_for_sam3_request(usage_collector_with_mocked_threads):
    def handler(
        inference_request,
        request,
        api_key=None,
        countinference=None,
        service_secret=None,
        request_source=None,
        request_source_info=None,
    ): ...

    usage_params = (
        usage_collector_with_mocked_threads._extract_usage_params_from_func_kwargs(
            usage_fps=0,
            usage_api_key="",
            usage_workflow_id="",
            usage_workflow_preview=False,
            usage_inference_test_run=False,
            usage_billable=True,
            execution_duration=0.1,
            func=handler,
            category="request",
            error_details=None,
            args=(
                SimpleNamespace(
                    api_key=None,
                    model_id="sam3/sam3_interactive",
                    source="app",
                    source_info="async-serverless-gpu",
                ),
                SimpleNamespace(),
            ),
            kwargs={
                "api_key": "query-api-key",
                "service_secret": "internal-secret",
                "request_source_info": "async-serverless-gpu",
            },
        )
    )

    assert usage_params["api_key"] == "query-api-key"
    assert usage_params["resource_id"] == "sam3/sam3_interactive"
    assert usage_params["roboflow_service_name"] == "async-serverless-gpu"
    assert usage_params["roboflow_internal_secret"] == "internal-secret"
    assert usage_params["resource_details"]["billable"] is True
    assert usage_params["resource_details"]["source_info"] == "async-serverless-gpu"
    assert usage_params["resource_details"]["execution_mode"] == SAM3_EXEC_MODE


def test_extract_usage_params_for_sam3_model_uses_current_request_identity(
    usage_collector_with_mocked_threads,
):
    class CachedModel:
        api_key = "first-loader-api-key"
        task_type = "unsupervised-segmentation"
        model_type = "sam3"

        def infer_from_request(self, request): ...

    usage_params = (
        usage_collector_with_mocked_threads._extract_usage_params_from_func_kwargs(
            usage_fps=0,
            usage_api_key="",
            usage_workflow_id="",
            usage_workflow_preview=False,
            usage_inference_test_run=False,
            usage_billable=True,
            execution_duration=0.1,
            func=CachedModel.infer_from_request,
            category="model",
            error_details=None,
            args=(
                CachedModel(),
                SimpleNamespace(
                    api_key="current-caller-api-key",
                    model_id="sam3/sam3_interactive",
                ),
            ),
            kwargs={},
        )
    )

    assert usage_params["api_key"] == "current-caller-api-key"
    assert usage_params["resource_id"] == "sam3/sam3_interactive"
    assert usage_params["resource_details"]["task_type"] == (
        "unsupervised-segmentation"
    )
    assert usage_params["resource_details"]["model_type"] == "sam3"


def test_extract_usage_params_for_model_includes_megapixel_buckets(
    usage_collector_with_mocked_threads,
):
    class FixedInputModel:
        api_key = "test_key"
        dataset_id = "st-inst-seg"
        version_id = "9"
        task_type = "instance-segmentation"
        model_type = "rfdetr-seg-nano"
        img_size_h = 640
        img_size_w = 640

        def infer(self, image, **kwargs): ...

    usage_params = (
        usage_collector_with_mocked_threads._extract_usage_params_from_func_kwargs(
            usage_fps=0,
            usage_api_key="",
            usage_workflow_id="",
            usage_workflow_preview=False,
            usage_inference_test_run=False,
            usage_billable=True,
            execution_duration=0.25,
            func=FixedInputModel.infer,
            category="model",
            error_details=None,
            args=(FixedInputModel(), [object(), object(), object()]),
            kwargs={},
        )
    )

    assert usage_params["resource_id"] == "st-inst-seg/9"
    assert usage_params["frames"] == 3
    assert usage_params["resource_details"]["model_type"] == "rfdetr-seg-nano"
    assert usage_params["megapixel_buckets"] == {
        "0.25-0.5": {
            "processed_frames": 3,
            "execution_duration": 0.25,
        }
    }


def test_extract_usage_params_for_sam_uses_encoder_image_size(
    usage_collector_with_mocked_threads,
):
    class SamLikeModel:
        api_key = "test_key"
        dataset_id = "sam2"
        version_id = "hiera_tiny"
        task_type = "unsupervised-segmentation"
        model_type = "sam2"
        image_size = 1024

        def infer_from_request(self, request): ...

    usage_params = (
        usage_collector_with_mocked_threads._extract_usage_params_from_func_kwargs(
            usage_fps=0,
            usage_api_key="",
            usage_workflow_id="",
            usage_workflow_preview=False,
            usage_inference_test_run=False,
            usage_billable=True,
            execution_duration=0.5,
            func=SamLikeModel.infer_from_request,
            category="model",
            error_details=None,
            args=(
                SamLikeModel(),
                SimpleNamespace(image=object(), api_key="test_key"),
            ),
            kwargs={},
        )
    )

    assert usage_params["resource_id"] == "sam2/hiera_tiny"
    assert usage_params["frames"] == 1
    assert usage_params["resource_details"]["model_type"] == "sam2"
    # 1024x1024 = ~1.05 MP -> 1-2 bucket
    assert usage_params["megapixel_buckets"] == {
        "1-2": {
            "processed_frames": 1,
            "execution_duration": 0.5,
        }
    }


def test_get_model_type_reads_recorded_map_without_calling_registry():
    class UnlabelledModel:
        dataset_id = "st-inst-seg"
        version_id = "9"

        def infer(self, image, **kwargs): ...

    func_kwargs = {"self": UnlabelledModel()}

    with mock.patch(
        "inference.core.registries.roboflow.get_model_type"
    ) as registry_get_model_type:
        assert get_model_type_from_kwargs(func_kwargs) is None

        record_model_type(model_id="st-inst-seg/9", model_type="rfdetr-seg-nano")
        try:
            assert get_model_type_from_kwargs(func_kwargs) == "rfdetr-seg-nano"
        finally:
            clear_recorded_model_types()

    # Resolving a model type must never reach the registry: that call can issue
    # an authenticated HTTP request from the inference hot path.
    assert not registry_get_model_type.called


def test_registry_records_model_type_for_usage_tracking():
    from inference.core.registries.roboflow import get_model_type

    try:
        _, model_type = get_model_type(model_id="sam2/hiera_tiny")

        assert model_type == "sam2"
        assert get_recorded_model_type("sam2/hiera_tiny") == "sam2"
    finally:
        clear_recorded_model_types()


def test_explicit_model_usage_api_key_takes_precedence_over_request(
    usage_collector_with_mocked_threads,
):
    class CachedModel:
        api_key = "first-loader-api-key"

        def infer_from_request(self, request): ...

    usage_params = (
        usage_collector_with_mocked_threads._extract_usage_params_from_func_kwargs(
            usage_fps=0,
            usage_api_key="explicit-usage-api-key",
            usage_workflow_id="",
            usage_workflow_preview=False,
            usage_inference_test_run=False,
            usage_billable=True,
            execution_duration=0.1,
            func=CachedModel.infer_from_request,
            category="model",
            error_details=None,
            args=(
                CachedModel(),
                SimpleNamespace(
                    api_key="current-caller-api-key",
                    model_id="sam3/sam3_interactive",
                ),
            ),
            kwargs={},
        )
    )

    assert usage_params["api_key"] == "explicit-usage-api-key"
