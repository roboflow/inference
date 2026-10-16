import pytest

from inference_sdk.config import outbound_service_secret, usage_source_tags
from inference_sdk.http.entities import (
    CLASSIFICATION_TASK,
    DEFAULT_IMAGE_EXTENSIONS,
    INSTANCE_SEGMENTATION_TASK,
    OBJECT_DETECTION_TASK,
    HTTPClientMode,
    InferenceConfiguration,
    get_non_empty_attributes,
)

REFERENCE_IMAGE_CONFIGURATION = InferenceConfiguration(
    confidence_threshold=0.5,
    format="json",
    mask_decode_mode="fast",
    tradeoff_factor=0.0,
    max_candidates=10,
    max_detections=20,
    stroke_width=1,
    count_inference=True,
    service_secret="xxx",
    disable_preproc_auto_orientation=True,
    disable_preproc_contrast=False,
    disable_preproc_grayscale=True,
    disable_preproc_static_crop=False,
    class_agnostic_nms=True,
    class_filter=["a", "b"],
    fix_batch_size=False,
    visualize_predictions=False,
    visualize_labels=True,
    iou_threshold=0.7,
    disable_active_learning=True,
    source="config-test",
    source_info="config-test-source-info",
)


def test_get_non_empty_attributes() -> None:
    # given
    reference_object = InferenceConfiguration(
        confidence_threshold=0.5,
        fix_batch_size=True,
    )

    # when
    result = get_non_empty_attributes(
        source_object=reference_object,
        specification=[
            ("confidence_threshold", "A"),
            ("image_extensions_for_directory_scan", "B"),
            ("visualize_labels", "C"),
        ],
    )

    # then
    assert result == {
        "A": 0.5,
        "B": DEFAULT_IMAGE_EXTENSIONS,
    }


def test_source_attributes() -> None:
    # given
    reference_object = InferenceConfiguration(
        source="source-test",
        source_info="source-info-test",
    )

    # when
    result = get_non_empty_attributes(
        source_object=reference_object,
        specification=[
            ("source", "A"),
            ("source_info", "B"),
        ],
    )

    # then
    assert result == {
        "A": "source-test",
        "B": "source-info-test",
    }


@pytest.mark.parametrize(
    "serializer",
    [
        "to_object_detection_parameters",
        "to_classification_parameters",
        "to_keypoints_detection_parameters",
        "to_instance_segmentation_parameters",
        "to_legacy_call_parameters",
    ],
)
def test_source_context_overrides_block_defaults_at_send_time(serializer) -> None:
    """Use request attribution even when a client was configured earlier.

    Args:
        serializer (str): Configuration serializer for the outgoing model call.
    """
    configuration = InferenceConfiguration(
        source="workflow-execution", source_info="block-default"
    )
    token = usage_source_tags.set({"source": "app", "source_info": "workflowEvals"})
    try:
        parameters = getattr(configuration, serializer)()
    finally:
        usage_source_tags.reset(token)

    assert parameters["source"] == "app"
    assert parameters["source_info"] == "workflowEvals"
    if serializer == "to_legacy_call_parameters":
        assert parameters["source_info_origin"] == "context"
    else:
        assert "source_info_origin" not in parameters

    direct_parameters = getattr(configuration, serializer)()
    assert direct_parameters["source_info"] == "block-default"
    assert "source_info_origin" not in direct_parameters


def test_source_context_does_not_change_billing_suppression() -> None:
    """Forward attribution alongside an authenticated billing opt-out."""
    source_token = usage_source_tags.set({"source_info": "labelAssist"})
    secret_token = outbound_service_secret.set("ctx-secret")
    try:
        parameters = InferenceConfiguration(
            count_inference=True
        ).to_legacy_call_parameters()
    finally:
        outbound_service_secret.reset(secret_token)
        usage_source_tags.reset(source_token)

    assert parameters["source_info"] == "labelAssist"
    assert parameters["service_secret"] == "ctx-secret"
    assert parameters["countinference"] is False


def test_to_api_call_parameters_for_api_v0() -> None:
    # when
    result = REFERENCE_IMAGE_CONFIGURATION.to_api_call_parameters(
        client_mode=HTTPClientMode.V0, task_type="does-not-matter"
    )

    # then
    assert result == {
        "confidence": 0.5,
        "format": "json",
        "labels": True,
        "mask_decode_mode": "fast",
        "allow_reduced_mask_resolution": False,
        "tradeoff_factor": 0.0,
        "max_detections": 20,
        "overlap": 0.7,
        "stroke": 1,
        "countinference": True,
        "service_secret": "xxx",
        "disable_preproc_auto_orient": True,
        "disable_preproc_contrast": False,
        "disable_preproc_grayscale": True,
        "disable_preproc_static_crop": False,
        "disable_active_learning": True,
        "source": "config-test",
        "source_info": "config-test-source-info",
    }


def test_to_api_call_parameters_for_api_v1_classification() -> None:
    # when
    result = REFERENCE_IMAGE_CONFIGURATION.to_api_call_parameters(
        client_mode=HTTPClientMode.V1,
        task_type=CLASSIFICATION_TASK,
    )

    # then
    assert result == {
        "confidence": 0.5,
        "visualize_predictions": False,
        "visualization_stroke_width": 1,
        "disable_preproc_auto_orient": True,
        "disable_preproc_contrast": False,
        "disable_preproc_grayscale": True,
        "disable_preproc_static_crop": False,
        "disable_active_learning": True,
        "source": "config-test",
        "source_info": "config-test-source-info",
    }


def test_to_api_call_parameters_for_api_v1_object_detection() -> None:
    # when
    result = REFERENCE_IMAGE_CONFIGURATION.to_api_call_parameters(
        client_mode=HTTPClientMode.V1, task_type=OBJECT_DETECTION_TASK
    )

    # then
    assert result == {
        "disable_preproc_auto_orient": True,
        "disable_preproc_contrast": False,
        "disable_preproc_grayscale": True,
        "disable_preproc_static_crop": False,
        "class_agnostic_nms": True,
        "class_filter": ["a", "b"],
        "confidence": 0.5,
        "fix_batch_size": False,
        "iou_threshold": 0.7,
        "max_detections": 20,
        "max_candidates": 10,
        "visualization_labels": True,
        "visualize_predictions": False,
        "visualization_stroke_width": 1,
        "disable_active_learning": True,
        "source": "config-test",
        "source_info": "config-test-source-info",
    }


def test_to_api_v1_query_parameters_when_both_set() -> None:
    # when
    query = REFERENCE_IMAGE_CONFIGURATION.to_api_v1_query_parameters()

    # then
    assert query == {"service_secret": "xxx", "countinference": True}


def test_to_api_v1_query_parameters_when_credentials_unset() -> None:
    # given
    configuration = InferenceConfiguration(confidence_threshold=0.5)

    # when
    query = configuration.to_api_v1_query_parameters()

    # then — no creds set means None so request builders skip the query string
    assert query is None


def test_to_api_v1_query_parameters_when_only_service_secret_set() -> None:
    # given
    configuration = InferenceConfiguration(service_secret="abc")

    # when
    query = configuration.to_api_v1_query_parameters()

    # then
    assert query == {"service_secret": "abc"}


def test_to_api_v1_query_parameters_when_only_count_inference_set() -> None:
    # given
    configuration = InferenceConfiguration(count_inference=True)

    # when
    query = configuration.to_api_v1_query_parameters()

    # then
    assert query == {"countinference": True}


def test_to_billing_query_parameters_uses_outbound_context_when_present() -> None:
    # given - no explicit billing configuration on the object itself
    configuration = InferenceConfiguration(confidence_threshold=0.5)
    token = outbound_service_secret.set("ctx-secret")

    try:
        # when
        query = configuration.to_billing_query_parameters()
    finally:
        outbound_service_secret.reset(token)

    # then
    assert query == {"service_secret": "ctx-secret", "countinference": False}


def test_to_billing_query_parameters_context_overrides_explicit_configuration() -> None:
    # given - explicit configuration says billable, and even names its own secret
    configuration = InferenceConfiguration(
        count_inference=True, service_secret="explicit-secret"
    )
    token = outbound_service_secret.set("ctx-secret")

    try:
        # when - the outbound context wins: it must not be possible to
        # upgrade an authenticated suppression back to billable
        query = configuration.to_billing_query_parameters()
    finally:
        outbound_service_secret.reset(token)

    assert query == {"service_secret": "ctx-secret", "countinference": False}


def test_to_billing_query_parameters_preserves_explicit_configuration_without_context() -> (
    None
):
    # given - no outbound context is active
    assert outbound_service_secret.get() is None
    configuration = InferenceConfiguration(
        count_inference=True, service_secret="explicit-secret"
    )

    # when
    query = configuration.to_billing_query_parameters()

    # then
    assert query == {"service_secret": "explicit-secret", "countinference": True}


def test_to_legacy_call_parameters_uses_outbound_context_when_present() -> None:
    # given - a v0 caller configured nothing about billing
    configuration = InferenceConfiguration(confidence_threshold=0.5)
    token = outbound_service_secret.set("ctx-secret")

    try:
        # when
        parameters = configuration.to_legacy_call_parameters()
    finally:
        outbound_service_secret.reset(token)

    # then
    assert parameters["countinference"] is False
    assert parameters["service_secret"] == "ctx-secret"


def test_to_legacy_call_parameters_context_overrides_explicit_count_inference_true() -> (
    None
):
    # given
    configuration = InferenceConfiguration(
        count_inference=True, service_secret="explicit-secret"
    )
    token = outbound_service_secret.set("ctx-secret")

    try:
        # when
        parameters = configuration.to_legacy_call_parameters()
    finally:
        outbound_service_secret.reset(token)

    # then
    assert parameters["countinference"] is False
    assert parameters["service_secret"] == "ctx-secret"


@pytest.mark.parametrize("client_mode", [HTTPClientMode.V0, HTTPClientMode.V1])
@pytest.mark.parametrize("requested", [None, False, True])
def test_include_anomaly_map_reaches_both_api_versions(client_mode, requested):
    # given - the anomaly heatmap is requested the same way on the legacy route and v1
    configuration = InferenceConfiguration(include_anomaly_map=requested)

    # when
    parameters = configuration.to_api_call_parameters(
        client_mode=client_mode, task_type=CLASSIFICATION_TASK
    )

    # then
    if requested is None:
        assert "include_anomaly_map" not in parameters
    else:
        assert parameters["include_anomaly_map"] is requested


@pytest.mark.parametrize("client_mode", [HTTPClientMode.V0, HTTPClientMode.V1])
@pytest.mark.parametrize("threshold", [None, 0.0, 1.25])
def test_anomaly_threshold_reaches_both_api_versions(client_mode, threshold):
    # given - the caller threshold is sent as a body field on v1 and a query
    # parameter on the legacy route; 0.0 is a valid threshold, not "unset"
    configuration = InferenceConfiguration(anomaly_threshold=threshold)

    # when
    parameters = configuration.to_api_call_parameters(
        client_mode=client_mode, task_type=CLASSIFICATION_TASK
    )

    # then
    if threshold is None:
        assert "anomaly_threshold" not in parameters
    else:
        assert parameters["anomaly_threshold"] == threshold


@pytest.mark.parametrize(
    "task_type", [OBJECT_DETECTION_TASK, INSTANCE_SEGMENTATION_TASK]
)
def test_anomaly_threshold_is_not_sent_to_other_v1_task_routes(task_type):
    # given
    configuration = InferenceConfiguration(anomaly_threshold=1.25)

    # when
    parameters = configuration.to_api_call_parameters(
        client_mode=HTTPClientMode.V1, task_type=task_type
    )

    # then
    assert "anomaly_threshold" not in parameters


@pytest.mark.parametrize("client_mode", [HTTPClientMode.V0, HTTPClientMode.V1])
@pytest.mark.parametrize("opt_in", [None, False, True])
def test_mask_resolution_opt_in_is_serialized_only_when_selected(client_mode, opt_in):
    configuration = InferenceConfiguration(allow_reduced_mask_resolution=opt_in)

    parameters = configuration.to_api_call_parameters(
        client_mode=client_mode, task_type=INSTANCE_SEGMENTATION_TASK
    )

    if opt_in is None:
        assert "allow_reduced_mask_resolution" not in parameters
    else:
        assert parameters["allow_reduced_mask_resolution"] is opt_in


def test_mask_resolution_opt_in_does_not_leak_to_object_detection():
    configuration = InferenceConfiguration(allow_reduced_mask_resolution=True)

    parameters = configuration.to_api_call_parameters(
        client_mode=HTTPClientMode.V1, task_type=OBJECT_DETECTION_TASK
    )

    assert "allow_reduced_mask_resolution" not in parameters


@pytest.mark.parametrize("client_mode", [HTTPClientMode.V0, HTTPClientMode.V1])
def test_mask_resolution_opt_in_defaults_to_false(client_mode):
    configuration = InferenceConfiguration()
    assert configuration.allow_reduced_mask_resolution is False
    assert (
        configuration.to_api_call_parameters(
            client_mode=client_mode, task_type=INSTANCE_SEGMENTATION_TASK
        )["allow_reduced_mask_resolution"]
        is False
    )
