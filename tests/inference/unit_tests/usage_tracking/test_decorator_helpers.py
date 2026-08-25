from types import SimpleNamespace
from unittest import mock

from inference.core.entities.requests.sam2 import Sam2InferenceRequest
from inference.core.env import SAM2_VERSION_ID, SAM3_EXEC_MODE
from inference.usage_tracking import decorator_helpers
from inference.usage_tracking.decorator_helpers import (
    get_model_descriptor_from_kwargs,
    get_model_frames_and_input_hw,
    get_model_id_from_kwargs,
    get_model_resource_details_from_kwargs,
    get_request_resource_details_from_kwargs,
)
from inference.usage_tracking.megapixel_buckets import (
    clear_measured_model_input,
    record_measured_model_hw,
)
from inference.usage_tracking.model_types import (
    ModelDescriptor,
    bind_usage_model_descriptor,
    clear_recorded_model_descriptors,
    get_recorded_model_descriptor,
    record_model_descriptor,
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
        model_architecture = "sam3"
        model_variant = "sam3_interactive"

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
    assert usage_params["resource_details"]["model_architecture"] == "sam3"
    assert usage_params["resource_details"]["model_variant"] == "sam3_interactive"


def test_extract_usage_params_for_model_includes_megapixel_buckets(
    usage_collector_with_mocked_threads,
):
    class FixedInputModel:
        api_key = "test_key"
        dataset_id = "st-inst-seg"
        version_id = "9"
        task_type = "instance-segmentation"
        model_architecture = "rfdetr"
        model_variant = "rfdetr-seg-nano"
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
    assert usage_params["resource_details"]["model_architecture"] == "rfdetr"
    assert usage_params["resource_details"]["model_variant"] == "rfdetr-seg-nano"
    assert usage_params["resource_details"]["model_input_height"] == 640
    assert usage_params["resource_details"]["model_input_width"] == 640
    assert usage_params["megapixel_buckets"] == {
        "0.25-0.5": {
            "processed_frames": 3,
            "execution_duration": 0.25,
        }
    }


def test_model_resource_details_report_non_square_configured_input():
    class LetterboxedModel:
        task_type = "object-detection"
        preproc = {"resize": {"height": 480, "width": 640}}

    resource_details = get_model_resource_details_from_kwargs(
        {"self": LetterboxedModel()}
    )

    assert resource_details["model_input_height"] == 480
    assert resource_details["model_input_width"] == 640


def test_model_resource_details_omit_input_size_for_dynamic_input_model():
    # A model with no configured canvas sizes itself from the upload, so the
    # size varies per call and must not be reported as a row-level scalar.
    class DynamicInputModel:
        task_type = "lmm"

    resource_details = get_model_resource_details_from_kwargs(
        {"self": DynamicInputModel()}
    )

    assert "model_input_height" not in resource_details
    assert "model_input_width" not in resource_details


def test_model_resource_details_input_size_does_not_consume_measured_input():
    # get_model_resource_details_from_kwargs runs before the megapixel buckets
    # are built; consuming the measured-input ContextVar here would blank them.
    class DynamicInputModel:
        task_type = "object-detection"

        def infer(self, image, **kwargs): ...

    model = DynamicInputModel()
    record_measured_model_hw(height=1080, width=1920)
    try:
        get_model_resource_details_from_kwargs({"self": model})

        assert get_model_frames_and_input_hw({"self": model, "image": object()}) == (
            1,
            (1080, 1920),
        )
    finally:
        clear_measured_model_input()


def test_extract_usage_params_for_sam_uses_encoder_image_size(
    usage_collector_with_mocked_threads,
):
    class SamLikeModel:
        api_key = "test_key"
        dataset_id = "sam2"
        version_id = "hiera_tiny"
        task_type = "unsupervised-segmentation"
        model_architecture = "sam2"
        model_variant = "hiera_tiny"
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
    assert usage_params["resource_details"]["model_architecture"] == "sam2"
    assert usage_params["resource_details"]["model_variant"] == "hiera_tiny"
    # 1024x1024 = ~1.05 MP -> 1-2 bucket
    assert usage_params["megapixel_buckets"] == {
        "1-2": {
            "processed_frames": 1,
            "execution_duration": 0.5,
        }
    }


def test_bind_usage_model_descriptor_copies_recorded_variant():
    model = SimpleNamespace()
    record_model_descriptor(
        "paligemma-3b-mix-224",
        architecture="paligemma",
        variant="paligemma-3b-mix-224",
    )
    try:
        bind_usage_model_descriptor(model, "paligemma-3b-mix-224")

        assert getattr(model, "model_id", None) is None
        assert model.model_architecture == "paligemma"
        assert model.model_variant == "paligemma-3b-mix-224"
    finally:
        clear_recorded_model_descriptors()


def test_bind_usage_model_descriptor_leaves_variant_empty_when_unknown():
    model = SimpleNamespace()
    record_model_descriptor("clip", architecture="clip")
    try:
        bind_usage_model_descriptor(model, "clip")

        assert model.model_architecture == "clip"
        assert model.model_variant is None
        assert get_model_descriptor_from_kwargs({"self": model}) == ModelDescriptor(
            "clip", None
        )
    finally:
        clear_recorded_model_descriptors()


def test_bind_does_not_change_alias_resource_id():
    # HTTP add_model(de_aliased, ..., model_id_alias=alias) used to write the
    # de-aliased id onto adapter instances. resource_id must stay the alias.
    model = SimpleNamespace()
    record_model_descriptor("coco/25", architecture="yolov11", variant="yolov11-n")
    record_model_descriptor("yolov11n-640", architecture="yolov11", variant="yolov11-n")
    try:
        bind_usage_model_descriptor(model, "coco/25", "yolov11n-640", "yolov11n-640")

        assert getattr(model, "model_id", None) is None
        assert model.model_architecture == "yolov11"
        assert model.model_variant == "yolov11-n"
        assert (
            get_model_id_from_kwargs({"self": model, "model_id": "yolov11n-640"})
            == "yolov11n-640"
        )
        assert get_model_descriptor_from_kwargs({"self": model}) == ModelDescriptor(
            "yolov11", "yolov11-n"
        )
    finally:
        clear_recorded_model_descriptors()


def test_get_model_descriptor_from_kwargs_uses_instance_after_map_cleared():
    model = SimpleNamespace(
        model_id="paligemma-3b-mix-224",
        model_architecture="paligemma",
        model_variant="paligemma-3b-mix-224",
    )
    clear_recorded_model_descriptors()

    assert get_model_descriptor_from_kwargs({"self": model}) == ModelDescriptor(
        "paligemma", "paligemma-3b-mix-224"
    )


def test_get_model_id_skips_null_kwargs_and_uses_instance():
    model = SimpleNamespace(model_id="qwen25-vl-7b")

    assert get_model_id_from_kwargs({"self": model, "model_id": None}) == "qwen25-vl-7b"


def test_caller_model_id_beats_normalized_instance_id():
    # vLLM Qwen stores the de-aliased id; TrOCR rewrites trocr/ → microsoft/.
    qwen = SimpleNamespace(model_id="qwen-pretrains/2")
    trocr = SimpleNamespace(model_id="microsoft/trocr-base-printed")

    assert (
        get_model_id_from_kwargs({"self": qwen, "model_id": "qwen3vl-2b-instruct"})
        == "qwen3vl-2b-instruct"
    )
    assert (
        get_model_id_from_kwargs(
            {"self": trocr, "model_id": "trocr/trocr-base-printed"}
        )
        == "trocr/trocr-base-printed"
    )


def test_get_model_binds_recorded_variant_on_instance():
    from inference.core.models.base import Model
    from inference.models import utils as model_utils

    class DummyAdapter(Model):
        def __init__(self, model_id, api_key=None, **kwargs):
            super().__init__()
            self.task_type = "lmm"

        def preprocess(self, image, **kwargs):
            return image, None

        def predict(self, img_in, **kwargs):
            return (img_in,)

        def postprocess(self, predictions, preprocess_return_metadata, **kwargs):
            return predictions

    record_model_descriptor(
        "paligemma-3b-mix-224",
        architecture="paligemma",
        variant="paligemma-3b-mix-224",
    )
    previous = model_utils.ROBOFLOW_MODEL_TYPES.get(("lmm", "paligemma"))
    try:
        with mock.patch.object(
            model_utils, "get_model_type", return_value=("lmm", "paligemma")
        ):
            model_utils.ROBOFLOW_MODEL_TYPES[("lmm", "paligemma")] = DummyAdapter
            model = model_utils.get_model("paligemma-3b-mix-224")

        assert getattr(model, "model_id", None) is None
        assert get_model_descriptor_from_kwargs({"self": model}) == ModelDescriptor(
            "paligemma", "paligemma-3b-mix-224"
        )
    finally:
        if previous is None:
            model_utils.ROBOFLOW_MODEL_TYPES.pop(("lmm", "paligemma"), None)
        else:
            model_utils.ROBOFLOW_MODEL_TYPES[("lmm", "paligemma")] = previous
        clear_recorded_model_descriptors()


def test_get_model_descriptor_reads_recorded_map_without_calling_registry():
    class UnlabelledModel:
        dataset_id = "st-inst-seg"
        version_id = "9"

        def infer(self, image, **kwargs): ...

    func_kwargs = {"self": UnlabelledModel()}

    with mock.patch(
        "inference.core.registries.roboflow.get_model_type"
    ) as registry_get_model_type:
        assert get_model_descriptor_from_kwargs(func_kwargs) is None

        record_model_descriptor(
            model_id="st-inst-seg/9",
            architecture="rfdetr",
            variant="rfdetr-seg-nano",
        )
        try:
            assert get_model_descriptor_from_kwargs(func_kwargs) == ModelDescriptor(
                "rfdetr", "rfdetr-seg-nano"
            )
        finally:
            clear_recorded_model_descriptors()

    # Resolving a model descriptor must never reach the registry: that call can
    # issue an authenticated HTTP request from the inference hot path.
    assert not registry_get_model_type.called


def test_registry_records_model_descriptor_for_usage_tracking():
    from inference.core.registries.roboflow import get_model_type

    try:
        _, model_type = get_model_type(model_id="sam2/hiera_tiny")

        assert model_type == "sam2"
        assert get_recorded_model_descriptor("sam2/hiera_tiny") == ModelDescriptor(
            "sam2", "hiera_tiny"
        )

        model = SimpleNamespace()
        bind_usage_model_descriptor(model, "sam2/hiera_tiny")
        assert model.model_architecture == "sam2"
        assert model.model_variant == "hiera_tiny"
    finally:
        clear_recorded_model_descriptors()


def test_registry_records_model_variant_for_usage_tracking_not_architecture():
    from inference.core.registries import roboflow
    from inference.core.registries.roboflow import get_model_type

    try:
        with mock.patch.object(
            roboflow, "USE_INFERENCE_MODELS", True
        ), mock.patch.object(
            roboflow,
            "get_model_metadata_from_inference_models_registry",
            return_value={
                "modelType": "paligemma",
                "taskType": "lmm",
                "modelVariant": "paligemma-3b-mix-224",
            },
        ), mock.patch.object(
            roboflow,
            "get_model_metadata_from_cache",
            return_value=None,
        ), mock.patch.object(
            roboflow,
            "_get_cached_model_metadata",
            return_value=None,
        ), mock.patch.object(
            roboflow,
            "save_model_metadata_in_cache",
        ):
            task_type, model_type = get_model_type(model_id="paligemma-3b-mix-224")

        assert task_type == "lmm"
        assert model_type == "paligemma"
        assert get_recorded_model_descriptor("paligemma-3b-mix-224") == ModelDescriptor(
            "paligemma", "paligemma-3b-mix-224"
        )
    finally:
        clear_recorded_model_descriptors()


def test_recorded_model_identities_evict_oldest_when_full(monkeypatch):
    import inference.usage_tracking.model_types as model_types

    monkeypatch.setattr(model_types, "_MAX_TRACKED_MODELS", 2)
    clear_recorded_model_descriptors()
    try:
        record_model_descriptor(model_id="a/1", architecture="yolov8", variant="yolov8-n")
        record_model_descriptor(model_id="b/1", architecture="yolov8", variant="yolov8-s")
        record_model_descriptor(model_id="c/1", architecture="yolov8", variant="yolov8-m")

        assert get_recorded_model_descriptor("a/1") is None
        assert get_recorded_model_descriptor("b/1") == ModelDescriptor("yolov8", "yolov8-s")
        assert get_recorded_model_descriptor("c/1") == ModelDescriptor("yolov8", "yolov8-m")

        # Refreshing an existing key keeps it from being the next eviction victim.
        record_model_descriptor(model_id="b/1", architecture="yolov8", variant="yolov8-s")
        record_model_descriptor(model_id="d/1", architecture="yolov8", variant="yolov8-l")

        assert get_recorded_model_descriptor("c/1") is None
        assert get_recorded_model_descriptor("b/1") == ModelDescriptor("yolov8", "yolov8-s")
        assert get_recorded_model_descriptor("d/1") == ModelDescriptor("yolov8", "yolov8-l")
    finally:
        clear_recorded_model_descriptors()


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
