from types import SimpleNamespace
from unittest import mock

import pytest

from inference.core.entities.requests.sam2 import Sam2InferenceRequest
from inference.core.env import SAM2_VERSION_ID, SAM3_EXEC_MODE
from inference.usage_tracking import decorator_helpers
from inference.usage_tracking.decorator_helpers import (
    call_carries_authenticated_non_billable_intent,
    get_model_api_key_from_kwargs,
    get_model_frames_and_input_hw,
    get_model_id_from_kwargs,
    get_model_resource_details_from_kwargs,
    get_model_type_from_kwargs,
    get_request_resource_details_from_kwargs,
    get_source_info_from_kwargs,
    non_billable_intent_is_authenticated,
    read_source_tags_bound_to_call,
)
from inference.usage_tracking.model_types import (
    bind_usage_model_identity,
    clear_recorded_model_types,
    get_recorded_model_type,
    record_model_type,
)


@pytest.mark.parametrize(
    "countinference, expected",
    [
        (False, True),
        ("false", True),
        ("FALSE", True),
        ("0", True),
        (True, False),
        ("true", False),
        (None, False),
        ("nonsense", False),
    ],
)
def test_non_billable_intent_coerces_string_flags(
    configured_service_secret,
    countinference,
    expected,
):
    result = non_billable_intent_is_authenticated(
        countinference, configured_service_secret
    )

    assert result is expected


def test_non_billable_intent_requires_a_valid_service_secret():
    assert non_billable_intent_is_authenticated(False, None) is False


def test_bound_call_reads_the_billing_arguments_it_was_given(
    configured_service_secret,
):
    def handler(inference_request, countinference=None, service_secret=None): ...

    assert (
        call_carries_authenticated_non_billable_intent(
            handler,
            (SimpleNamespace(),),
            {
                "countinference": False,
                "service_secret": configured_service_secret,
            },
        )
        is True
    )


def test_bound_call_ignores_an_unauthenticated_opt_out(configured_service_secret):
    def handler(inference_request, countinference=None, service_secret=None): ...

    assert (
        call_carries_authenticated_non_billable_intent(
            handler,
            (SimpleNamespace(),),
            {"countinference": False, "service_secret": "not-the-secret"},
        )
        is False
    )


def test_bound_call_skips_binding_a_signature_without_billing_arguments():
    # Every usage-collected call pays for this check, including the model hot
    # path, so binding a signature that cannot carry the intent is pure
    # overhead.
    def infer(self, image, **kwargs): ...

    with mock.patch.object(
        decorator_helpers, "collect_func_params"
    ) as collect_func_params_mock:
        result = call_carries_authenticated_non_billable_intent(
            infer, (None, "img"), {"countinference": False}
        )

    assert result is False
    collect_func_params_mock.assert_not_called()


def test_bound_call_swallows_binding_failures():
    def handler(inference_request, countinference=None, service_secret=None): ...

    with mock.patch.object(
        decorator_helpers,
        "collect_func_params",
        side_effect=ValueError("boom"),
    ):
        result = call_carries_authenticated_non_billable_intent(handler, (), {})

    assert result is False


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


def test_bind_usage_model_identity_copies_recorded_variant():
    model = SimpleNamespace()
    record_model_type("paligemma-3b-mix-224", "paligemma-3b-mix-224")
    try:
        bind_usage_model_identity(model, "paligemma-3b-mix-224")

        assert getattr(model, "model_id", None) is None
        assert model.model_type == "paligemma-3b-mix-224"
    finally:
        clear_recorded_model_types()


def test_bind_does_not_change_alias_resource_id():
    # HTTP add_model(de_aliased, ..., model_id_alias=alias) used to write the
    # de-aliased id onto adapter instances. resource_id must stay the alias.
    model = SimpleNamespace()
    record_model_type("coco/25", "yolov11-n")
    record_model_type("yolov11n-640", "yolov11-n")
    try:
        bind_usage_model_identity(model, "coco/25", "yolov11n-640", "yolov11n-640")

        assert getattr(model, "model_id", None) is None
        assert model.model_type == "yolov11-n"
        assert (
            get_model_id_from_kwargs({"self": model, "model_id": "yolov11n-640"})
            == "yolov11n-640"
        )
        assert get_model_type_from_kwargs({"self": model}) == "yolov11-n"
    finally:
        clear_recorded_model_types()


def test_get_model_type_from_kwargs_uses_instance_after_map_cleared():
    model = SimpleNamespace(
        model_id="paligemma-3b-mix-224",
        model_type="paligemma-3b-mix-224",
    )
    clear_recorded_model_types()

    assert get_model_type_from_kwargs({"self": model}) == "paligemma-3b-mix-224"


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

    record_model_type("paligemma-3b-mix-224", "paligemma-3b-mix-224")
    previous = model_utils.ROBOFLOW_MODEL_TYPES.get(("lmm", "paligemma"))
    try:
        with mock.patch.object(
            model_utils, "get_model_type", return_value=("lmm", "paligemma")
        ):
            model_utils.ROBOFLOW_MODEL_TYPES[("lmm", "paligemma")] = DummyAdapter
            model = model_utils.get_model("paligemma-3b-mix-224")

        assert getattr(model, "model_id", None) is None
        assert model.model_type == "paligemma-3b-mix-224"
        assert get_model_type_from_kwargs({"self": model}) == "paligemma-3b-mix-224"
    finally:
        if previous is None:
            model_utils.ROBOFLOW_MODEL_TYPES.pop(("lmm", "paligemma"), None)
        else:
            model_utils.ROBOFLOW_MODEL_TYPES[("lmm", "paligemma")] = previous
        clear_recorded_model_types()


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
        assert get_recorded_model_type("paligemma-3b-mix-224") == (
            "paligemma-3b-mix-224"
        )
    finally:
        clear_recorded_model_types()


def test_recorded_model_types_evict_oldest_when_full(monkeypatch):
    import inference.usage_tracking.model_types as model_types

    monkeypatch.setattr(model_types, "_MAX_TRACKED_MODELS", 2)
    clear_recorded_model_types()
    try:
        record_model_type(model_id="a/1", model_type="yolov8n")
        record_model_type(model_id="b/1", model_type="yolov8s")
        record_model_type(model_id="c/1", model_type="yolov8m")

        assert get_recorded_model_type("a/1") is None
        assert get_recorded_model_type("b/1") == "yolov8s"
        assert get_recorded_model_type("c/1") == "yolov8m"

        # Refreshing an existing key keeps it from being the next eviction victim.
        record_model_type(model_id="b/1", model_type="yolov8s")
        record_model_type(model_id="d/1", model_type="yolov8l")

        assert get_recorded_model_type("c/1") is None
        assert get_recorded_model_type("b/1") == "yolov8s"
        assert get_recorded_model_type("d/1") == "yolov8l"
    finally:
        clear_recorded_model_types()


def test_model_frames_count_images_kwarg_used_by_video_blocks():
    frames, _ = get_model_frames_and_input_hw(
        {"images": [object(), object(), object()]}
    )

    assert frames == 3


def test_model_frames_count_workflow_batch_used_by_video_blocks():
    class FakeWorkflowBatch:
        def __init__(self, items):
            self._content = items

        def __len__(self):
            return len(self._content)

        def __iter__(self):
            return iter(self._content)

        def iter_with_indices(self):
            return enumerate(self._content)

    frames, _ = get_model_frames_and_input_hw(
        {"images": FakeWorkflowBatch([object(), object(), object()])}
    )

    assert frames == 3


def test_model_api_key_from_workflow_block_private_attr():
    api_key = get_model_api_key_from_kwargs(
        {"self": SimpleNamespace(_api_key="workflow-block-key")}
    )

    assert api_key == "workflow-block-key"


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


@pytest.mark.parametrize(
    "handler_kwargs",
    [
        {"request_source": "app", "request_source_info": "smart-polygon"},
        {"source": "app", "source_info": "smart-polygon"},
        {
            "request": SimpleNamespace(
                query_params={"source": "app", "source_info": "smart-polygon"}
            )
        },
    ],
    ids=["aliased", "plain", "query_string_only"],
)
def test_source_tags_are_read_however_the_handler_declares_them(handler_kwargs):
    def handler(
        inference_request,
        request=None,
        countinference=None,
        source=None,
        source_info=None,
        request_source=None,
        request_source_info=None,
    ): ...

    tags = read_source_tags_bound_to_call(
        handler, (), {"inference_request": SimpleNamespace(), **handler_kwargs}
    )

    assert tags == {"source": "app", "source_info": "smart-polygon"}


def test_source_tags_ignore_the_external_placeholder():
    # The legacy route defaults both parameters to "external", which describes
    # no caller and must not land in a synthetic bucket.
    def handler(
        inference_request, countinference=None, source=None, source_info=None
    ): ...

    tags = read_source_tags_bound_to_call(
        handler,
        (),
        {
            "inference_request": SimpleNamespace(source=None, source_info=None),
            "source": "external",
            "source_info": "external",
        },
    )

    assert tags == {}


def test_source_tags_come_from_the_bound_request_payload():
    # /infer/* routes declare no source parameters; a caller can still set the
    # field in the request body.
    def handler(inference_request, countinference=None): ...

    tags = read_source_tags_bound_to_call(
        handler,
        (),
        {"inference_request": SimpleNamespace(source="app", source_info=None)},
    )

    assert tags == {"source": "app"}


def test_source_tags_skip_binding_a_signature_without_billing_arguments():
    def infer(self, image, **kwargs): ...

    with mock.patch.object(
        decorator_helpers, "collect_func_params"
    ) as collect_func_params_mock:
        tags = read_source_tags_bound_to_call(infer, (None, "img"), {"source": "app"})

    assert tags == {}
    collect_func_params_mock.assert_not_called()


def test_source_tags_swallow_binding_failures():
    def handler(inference_request, countinference=None, source=None): ...

    with mock.patch.object(
        decorator_helpers,
        "collect_func_params",
        side_effect=ValueError("boom"),
    ):
        assert read_source_tags_bound_to_call(handler, (), {"source": "app"}) == {}


def test_model_row_records_a_source_the_caller_actually_supplied():
    result = get_model_resource_details_from_kwargs({"source": "app"})

    assert result["source"] == "app"


@pytest.mark.parametrize("placeholder", ["external", None])
def test_model_row_omits_the_placeholder_the_legacy_route_supplies(placeholder):
    """The legacy route defaults ``source`` to "external", and ``/infer/*`` to None.

    ``infer_from_request`` unpacks ``request.dict()`` into ``infer(**kwargs)``, so
    the field always reaches this function whether or not the caller set it, and
    the row used to carry it either way. Neither value names a caller, so the key
    is omitted instead - a deliberate narrowing of what the dimension holds.
    """
    result = get_model_resource_details_from_kwargs({"source": placeholder})

    assert "source" not in result


def test_model_row_takes_source_from_the_request_it_was_handed():
    result = get_model_resource_details_from_kwargs(
        {"inference_request": SimpleNamespace(source="app")}
    )

    assert result["source"] == "app"


def test_model_row_takes_source_from_the_request_scope():
    # A nested model decorator is handed the typed request and nothing else; a
    # tag that arrived on the handler's query string reaches it only as context.
    token = decorator_helpers.usage_source_tags.set({"source": "app"})
    try:
        result = get_model_resource_details_from_kwargs({})
    finally:
        decorator_helpers.usage_source_tags.reset(token)

    assert result["source"] == "app"


def test_request_row_records_source():
    result = get_request_resource_details_from_kwargs(
        {"inference_request": SimpleNamespace(source="app")}
    )

    assert result["source"] == "app"


def test_request_row_omits_source_for_the_external_placeholder():
    # Every legacy-route call would otherwise land in a synthetic bucket.
    result = get_request_resource_details_from_kwargs(
        {"source": "external", "countinference": True}
    )

    assert "source" not in result


def test_source_info_falls_back_to_the_request_scope():
    token = decorator_helpers.usage_source_tags.set({"source_info": "smart-polygon"})
    try:
        result = get_source_info_from_kwargs({})
    finally:
        decorator_helpers.usage_source_tags.reset(token)

    assert result == "smart-polygon"


def test_source_info_still_ignores_the_external_placeholder():
    assert get_source_info_from_kwargs({"source_info": "external"}) is None
