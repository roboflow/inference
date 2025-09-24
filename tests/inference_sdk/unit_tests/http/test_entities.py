from inference_sdk.http.entities import (
    CLASSIFICATION_TASK,
    DEFAULT_IMAGE_EXTENSIONS,
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
