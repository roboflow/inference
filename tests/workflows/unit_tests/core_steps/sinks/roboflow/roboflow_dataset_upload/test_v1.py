import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import supervision as sv
from fastapi import BackgroundTasks

from inference.core.cache import MemoryCache
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload import v1
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v1 import (
    BatchCreationFrequency,
    RoboflowDatasetUploadBlockV1,
    encode_prediction,
    execute_registration,
    generate_batch_name,
    get_workspace_name,
    is_prediction_registration_forbidden,
    register_datapoint,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


def test_encode_prediction_when_classification_prediction_provided() -> None:
    # given
    prediction = {
        "top": "car",
        "predictions": [
            {"class": "car", "confidence": 0.7},
            {"class": "truck", "confidence": 0.3},
        ],
    }

    # when
    result = encode_prediction(prediction=prediction)

    # then
    assert result == ("car", "txt"), "Expected top class with txt format returned"


def test_encode_prediction_when_sv_detections_provided() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64),
        class_id=np.array([1, 2]),
        confidence=np.array([0.1, 0.9], dtype=np.float64),
        data={
            "class_name": np.array(["cat", "dog"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
            "parent_dimensions": np.array(
                [
                    [192, 168],
                    [192, 168],
                ]
            ),
            "image_dimensions": np.array(
                [
                    [192, 168],
                    [192, 168],
                ]
            ),
        },
    )

    # when
    result = encode_prediction(prediction=detections)

    # then
    assert result[1] == "json", "Expected JSON format of encoding"
    assert json.loads(result[0]) == {
        "image": {
            "width": 168,
            "height": 192,
        },
        "predictions": [
            {
                "width": 1.0,
                "height": 1.0,
                "x": 1.5,
                "y": 1.5,
                "confidence": 0.1,
                "class_id": 1,
                "class": "cat",
                "detection_id": "first",
                "parent_id": "image",
            },
            {
                "width": 1.0,
                "height": 1.0,
                "x": 3.5,
                "y": 3.5,
                "confidence": 0.9,
                "class_id": 2,
                "class": "dog",
                "detection_id": "second",
                "parent_id": "image",
            },
        ],
    }, "Expected prediction to be serialised properly"


def test_is_prediction_registration_forbidden_when_prediction_is_empty() -> None:
    # when
    result = is_prediction_registration_forbidden(prediction=None)

    # then
    assert result is True


def test_is_prediction_registration_forbidden_when_sv_detection_is_empty() -> None:
    # when
    result = is_prediction_registration_forbidden(prediction=sv.Detections.empty())

    # then
    assert result is True


def test_is_prediction_registration_forbidden_when_non_classification_dict_provided() -> (
    None
):
    # when
    result = is_prediction_registration_forbidden(prediction={"some": "prediction"})

    # then
    assert result is True


def test_is_prediction_registration_forbidden_when_classification_dict_provided() -> (
    None
):
    # given
    prediction = {
        "top": "car",
        "predictions": [
            {"class": "car", "confidence": 0.7},
            {"class": "truck", "confidence": 0.3},
        ],
    }

    # when
    result = is_prediction_registration_forbidden(prediction=prediction)

    # then
    assert result is False


def test_is_prediction_registration_forbidden_when_non_empty_sv_detection_provided() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.1], dtype=np.float64),
        data={
            "class_name": np.array(["cat"]),
        },
    )

    # when
    result = is_prediction_registration_forbidden(prediction=detections)

    # then
    assert result is False


@mock.patch.object(v1, "register_image_at_roboflow")
def test_register_datapoint_when_duplicate_found(
    register_image_at_roboflow_mock: MagicMock,
) -> None:
    # given
    register_image_at_roboflow_mock.return_value = {"duplicate": True}
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.1], dtype=np.float64),
        data={
            "class_name": np.array(["cat"]),
        },
    )

    # when
    result = register_datapoint(
        target_project="my_project",
        encoded_image=b"image",
        local_image_id="local_id",
        prediction=detections,
        api_key="my_api_key",
        batch_name="my_batch",
        tags=[],
    )

    # then
    assert result == "Duplicated image", "Duplicate status is expected to be reported"
    register_image_at_roboflow_mock.assert_called_once()
    assert (
        register_image_at_roboflow_mock.call_args[1]["inference_id"] is None
    ), "No inference id found in sv detection"


@mock.patch.object(v1, "annotate_image_at_roboflow")
@mock.patch.object(v1, "register_image_at_roboflow")
def test_register_datapoint_when_prediction_is_not_delivered(
    register_image_at_roboflow_mock: MagicMock,
    annotate_image_at_roboflow: MagicMock,
) -> None:
    # given
    register_image_at_roboflow_mock.return_value = {"duplicate": True}

    # when
    result = register_datapoint(
        target_project="my_project",
        encoded_image=b"image",
        local_image_id="local_id",
        prediction=None,
        api_key="my_api_key",
        batch_name="my_batch",
        tags=[],
    )

    # then
    assert result == "Duplicated image", "Duplicate status is expected to be reported"
    register_image_at_roboflow_mock.assert_called_once()
    assert (
        register_image_at_roboflow_mock.call_args[1]["inference_id"] is None
    ), "No inference id found in sv detection"
    annotate_image_at_roboflow.assert_not_called()


@mock.patch.object(v1, "register_image_at_roboflow")
def test_register_datapoint_when_prediction_registration_should_be_forbidden(
    register_image_at_roboflow_mock: MagicMock,
) -> None:
    # given
    register_image_at_roboflow_mock.return_value = {"id": "backend_id"}
    detections = None

    # when
    result = register_datapoint(
        target_project="my_project",
        encoded_image=b"image",
        local_image_id="local_id",
        prediction=detections,
        api_key="my_api_key",
        batch_name="my_batch",
        tags=[],
    )

    # then
    assert (
        result == "Successfully registered image"
    ), "Status reporting success on image registration is expected"
    register_image_at_roboflow_mock.assert_called_once()
    assert (
        register_image_at_roboflow_mock.call_args[1]["inference_id"] is None
    ), "No inference id found in sv detection"


@mock.patch.object(v1, "annotate_image_at_roboflow")
@mock.patch.object(v1, "register_image_at_roboflow")
def test_register_datapoint_when_prediction_registration_should_be_successful(
    register_image_at_roboflow_mock: MagicMock,
    annotate_image_at_roboflow_mock: MagicMock,
) -> None:
    # given
    register_image_at_roboflow_mock.return_value = {"id": "backend_id"}
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64),
        class_id=np.array([1, 2]),
        confidence=np.array([0.1, 0.9], dtype=np.float64),
        data={
            "class_name": np.array(["cat", "dog"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
            "parent_dimensions": np.array(
                [
                    [192, 168],
                    [192, 168],
                ]
            ),
            "image_dimensions": np.array(
                [
                    [192, 168],
                    [192, 168],
                ]
            ),
            "inference_id": np.array(["a", "a"]),
        },
    )
    expected_registered_prediction = json.dumps(
        {
            "image": {
                "width": 168,
                "height": 192,
            },
            "predictions": [
                {
                    "width": 1.0,
                    "height": 1.0,
                    "x": 1.5,
                    "y": 1.5,
                    "confidence": 0.1,
                    "class_id": 1,
                    "class": "cat",
                    "detection_id": "first",
                    "parent_id": "image",
                },
                {
                    "width": 1.0,
                    "height": 1.0,
                    "x": 3.5,
                    "y": 3.5,
                    "confidence": 0.9,
                    "class_id": 2,
                    "class": "dog",
                    "detection_id": "second",
                    "parent_id": "image",
                },
            ],
        }
    )

    # when
    result = register_datapoint(
        target_project="my_project",
        encoded_image=b"image",
        local_image_id="local_id",
        prediction=detections,
        api_key="my_api_key",
        batch_name="my_batch",
        tags=[],
    )

    # then
    assert (
        result == "Successfully registered image and annotation"
    ), "Success status report expected"
    register_image_at_roboflow_mock.assert_called_once()
    assert (
        register_image_at_roboflow_mock.call_args[1]["inference_id"] == "a"
    ), "Expected inference ID to be denoted"
    annotate_image_at_roboflow_mock.assert_called_once_with(
        api_key="my_api_key",
        dataset_id="my_project",
        local_image_id="local_id",
        roboflow_image_id="backend_id",
        annotation_content=expected_registered_prediction,
        annotation_file_type="json",
        is_prediction=True,
    )


@mock.patch.object(v1, "annotate_image_at_roboflow")
@mock.patch.object(v1, "register_image_at_roboflow")
def test_register_datapoint_when_prediction_registration_should_be_successful_but_without_inference_id(
    register_image_at_roboflow_mock: MagicMock,
    annotate_image_at_roboflow_mock: MagicMock,
) -> None:
    # given
    register_image_at_roboflow_mock.return_value = {"id": "backend_id"}
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64),
        class_id=np.array([1, 2]),
        confidence=np.array([0.1, 0.9], dtype=np.float64),
        data={
            "class_name": np.array(["cat", "dog"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
            "parent_dimensions": np.array(
                [
                    [192, 168],
                    [192, 168],
                ]
            ),
            "image_dimensions": np.array(
                [
                    [192, 168],
                    [192, 168],
                ]
            ),
        },
    )
    expected_registered_prediction = json.dumps(
        {
            "image": {
                "width": 168,
                "height": 192,
            },
            "predictions": [
                {
                    "width": 1.0,
                    "height": 1.0,
                    "x": 1.5,
                    "y": 1.5,
                    "confidence": 0.1,
                    "class_id": 1,
                    "class": "cat",
                    "detection_id": "first",
                    "parent_id": "image",
                },
                {
                    "width": 1.0,
                    "height": 1.0,
                    "x": 3.5,
                    "y": 3.5,
                    "confidence": 0.9,
                    "class_id": 2,
                    "class": "dog",
                    "detection_id": "second",
                    "parent_id": "image",
                },
            ],
        }
    )

    # when
    result = register_datapoint(
        target_project="my_project",
        encoded_image=b"image",
        local_image_id="local_id",
        prediction=detections,
        api_key="my_api_key",
        batch_name="my_batch",
        tags=[],
    )

    # then
    assert (
        result == "Successfully registered image and annotation"
    ), "Success status report expected"
    register_image_at_roboflow_mock.assert_called_once()
    assert (
        register_image_at_roboflow_mock.call_args[1]["inference_id"] is None
    ), "Expected inference ID not to be denoted"
    annotate_image_at_roboflow_mock.assert_called_once_with(
        api_key="my_api_key",
        dataset_id="my_project",
        local_image_id="local_id",
        roboflow_image_id="backend_id",
        annotation_content=expected_registered_prediction,
        annotation_file_type="json",
        is_prediction=True,
    )


@mock.patch.object(v1, "register_image_at_roboflow")
def test_register_datapoint_when_prediction_is_empty(
    register_image_at_roboflow_mock: MagicMock,
) -> None:
    # given
    register_image_at_roboflow_mock.return_value = {"id": "backend_id"}
    detections = sv.Detections.empty()
    expected_registered_prediction = json.dumps(
        {
            "image": {
                "width": None,
                "height": None,
            },
            "predictions": [],
        }
    )

    # when
    result = register_datapoint(
        target_project="my_project",
        encoded_image=b"image",
        local_image_id="local_id",
        prediction=detections,
        api_key="my_api_key",
        batch_name="my_batch",
        tags=[],
    )

    # then
    assert result == "Successfully registered image", "Success status report expected"
    register_image_at_roboflow_mock.assert_called_once()
    assert (
        register_image_at_roboflow_mock.call_args[1]["inference_id"] is None
    ), "Expected inference ID not to be denoted"


@mock.patch.object(v1, "annotate_image_at_roboflow")
@mock.patch.object(v1, "register_image_at_roboflow")
def test_register_datapoint_when_classification_prediction_registration_should_be_successful_with_inference_id(
    register_image_at_roboflow_mock: MagicMock,
    annotate_image_at_roboflow_mock: MagicMock,
) -> None:
    # given
    register_image_at_roboflow_mock.return_value = {"id": "backend_id"}
    prediction = {
        "top": "some",
        "confidence": 0.3,
        "parent_id": "parent",
        "predictions": [{"class": "some", "class_id": 1, "confidence": 0.3}],
        "inference_id": "a",
    }
    expected_registered_prediction = "some"

    # when
    result = register_datapoint(
        target_project="my_project",
        encoded_image=b"image",
        local_image_id="local_id",
        prediction=prediction,
        api_key="my_api_key",
        batch_name="my_batch",
        tags=[],
    )

    # then
    assert (
        result == "Successfully registered image and annotation"
    ), "Success status report expected"
    register_image_at_roboflow_mock.assert_called_once()
    assert (
        register_image_at_roboflow_mock.call_args[1]["inference_id"] == "a"
    ), "Expected inference ID to be denoted"
    annotate_image_at_roboflow_mock.assert_called_once_with(
        api_key="my_api_key",
        dataset_id="my_project",
        local_image_id="local_id",
        roboflow_image_id="backend_id",
        annotation_content=expected_registered_prediction,
        annotation_file_type="txt",
        is_prediction=True,
    )


@mock.patch.object(v1, "annotate_image_at_roboflow")
@mock.patch.object(v1, "register_image_at_roboflow")
def test_register_datapoint_when_classification_prediction_registration_should_be_successful_without_inference_id(
    register_image_at_roboflow_mock: MagicMock,
    annotate_image_at_roboflow_mock: MagicMock,
) -> None:
    # given
    register_image_at_roboflow_mock.return_value = {"id": "backend_id"}
    prediction = {
        "top": "some",
        "confidence": 0.3,
        "parent_id": "parent",
        "predictions": [{"class": "some", "class_id": 1, "confidence": 0.3}],
    }
    expected_registered_prediction = "some"

    # when
    result = register_datapoint(
        target_project="my_project",
        encoded_image=b"image",
        local_image_id="local_id",
        prediction=prediction,
        api_key="my_api_key",
        batch_name="my_batch",
        tags=[],
    )

    # then
    assert (
        result == "Successfully registered image and annotation"
    ), "Success status report expected"
    register_image_at_roboflow_mock.assert_called_once()
    assert (
        register_image_at_roboflow_mock.call_args[1]["inference_id"] is None
    ), "Expected inference ID not to be denoted"
    annotate_image_at_roboflow_mock.assert_called_once_with(
        api_key="my_api_key",
        dataset_id="my_project",
        local_image_id="local_id",
        roboflow_image_id="backend_id",
        annotation_content=expected_registered_prediction,
        annotation_file_type="txt",
        is_prediction=True,
    )


@pytest.mark.parametrize(
    "labeling_batch_prefix, creation_frequency, expected_result",
    [
        ("my_batch", "never", "my_batch"),
        ("my_batch", "daily", "my_batch_2024_05_28"),
        ("my_batch", "weekly", "my_batch_2024_05_27"),
        ("my_batch", "monthly", "my_batch_2024_05_01"),
    ],
)
@mock.patch.object(v1, "datetime")
def test_generate_batch_name(
    datetime_mock: MagicMock,
    labeling_batch_prefix: str,
    creation_frequency: BatchCreationFrequency,
    expected_result: str,
) -> None:
    # given
    datetime_mock.today.return_value = datetime(year=2024, month=5, day=28)

    # when
    result = generate_batch_name(
        labeling_batch_prefix=labeling_batch_prefix,
        new_labeling_batch_frequency=creation_frequency,
    )

    # then
    assert result == expected_result


def test_get_workspace_name_when_cache_contains_workspace_name() -> None:
    # given
    api_key = "my_api_key"
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    expected_cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cache = MemoryCache()
    cache.set(key=expected_cache_key, value="my_workspace")

    # when
    result = get_workspace_name(api_key=api_key, cache=cache)

    # then
    assert (
        result == "my_workspace"
    ), "Expected return value from the cache to be returned"


@mock.patch.object(v1, "get_roboflow_workspace")
def test_get_workspace_name_when_cache_does_not_contain_workspace_name(
    get_roboflow_workspace_mock: MagicMock,
) -> None:
    # given
    api_key = "my_api_key"
    cache = MemoryCache()
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    expected_cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    get_roboflow_workspace_mock.return_value = "workspace_from_api"

    # when
    result = get_workspace_name(api_key=api_key, cache=cache)

    # then
    assert (
        result == "workspace_from_api"
    ), "Expected return value from the API to be returned"
    assert (
        cache.get(expected_cache_key) == "workspace_from_api"
    ), "Expected retrieved workspace to be saved in cache"


@mock.patch.object(v1, "use_credit_of_matching_strategy")
def test_execute_registration_when_quota_limit_exceeded(
    use_credit_of_matching_strategy_mock: MagicMock,
) -> None:
    # given
    api_key = "my_api_key"
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    expected_cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cache = MemoryCache()
    cache.set(key=expected_cache_key, value="my_workspace")
    use_credit_of_matching_strategy_mock.return_value = None
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((512, 256, 3), dtype=np.uint8),
    )
    prediction = {
        "top": "car",
        "predictions": [
            {"class": "car", "confidence": 0.7},
            {"class": "truck", "confidence": 0.3},
        ],
    }

    # when
    result = execute_registration(
        image=image,
        prediction=prediction,
        target_project="my_project",
        usage_quota_name="my_quota",
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(100, 100),
        compression_level=75,
        registration_tags=["some"],
        labeling_batch_prefix="my_batch",
        new_labeling_batch_frequency="never",
        cache=cache,
        api_key=api_key,
    )

    # then
    assert result == (
        False,
        "Registration skipped due to usage quota exceeded",
    ), "Expected quota hut to be marked"


@mock.patch.object(v1, "return_strategy_credit")
@mock.patch.object(v1, "register_datapoint")
@mock.patch.object(v1, "use_credit_of_matching_strategy")
def test_execute_registration_when_error_in_registration_happened(
    use_credit_of_matching_strategy_mock: MagicMock,
    register_datapoint_mock: MagicMock,
    return_strategy_credit_mock: MagicMock,
) -> None:
    # given
    api_key = "my_api_key"
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    expected_cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cache = MemoryCache()
    cache.set(key=expected_cache_key, value="my_workspace")
    use_credit_of_matching_strategy_mock.return_value = "my_strategy"
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((512, 256, 3), dtype=np.uint8),
    )
    prediction = {
        "top": "car",
        "predictions": [
            {"class": "car", "confidence": 0.7},
            {"class": "truck", "confidence": 0.3},
        ],
    }
    register_datapoint_mock.side_effect = Exception()

    # when
    result = execute_registration(
        image=image,
        prediction=prediction,
        target_project="my_project",
        usage_quota_name="my_quota",
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(100, 100),
        compression_level=75,
        registration_tags=["some"],
        labeling_batch_prefix="my_batch",
        new_labeling_batch_frequency="never",
        cache=cache,
        api_key=api_key,
    )

    # then
    assert result[0] is True, "Expected error to be marked"
    return_strategy_credit_mock.assert_called_once_with(
        cache=cache,
        workspace="my_workspace",
        project="my_project",
        strategy_name="my_strategy",
    )


@mock.patch.object(v1, "return_strategy_credit")
@mock.patch.object(v1, "register_datapoint")
@mock.patch.object(v1, "use_credit_of_matching_strategy")
def test_execute_registration_when_registration_should_be_successful(
    use_credit_of_matching_strategy_mock: MagicMock,
    register_datapoint_mock: MagicMock,
    return_strategy_credit_mock: MagicMock,
) -> None:
    # given
    api_key = "my_api_key"
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    expected_cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cache = MemoryCache()
    cache.set(key=expected_cache_key, value="my_workspace")
    use_credit_of_matching_strategy_mock.return_value = "my_strategy"
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((128, 256, 3), dtype=np.uint8),
    )
    detections = sv.Detections(
        xyxy=np.array([[2, 2, 4, 4], [4, 4, 8, 8]], dtype=np.float64),
        class_id=np.array([1, 2]),
        confidence=np.array([0.1, 0.9], dtype=np.float64),
        data={
            "class_name": np.array(["cat", "dog"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
            "parent_dimensions": np.array(
                [
                    [128, 128],
                    [128, 128],
                ]
            ),
            "image_dimensions": np.array(
                [
                    [128, 128],
                    [128, 128],
                ]
            ),
        },
    )
    register_datapoint_mock.return_value = "STATUS OK"

    # when
    result = execute_registration(
        image=image,
        prediction=detections,
        target_project="my_project",
        usage_quota_name="my_quota",
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 64),
        compression_level=75,
        registration_tags=["some"],
        labeling_batch_prefix="my_batch",
        new_labeling_batch_frequency="never",
        cache=cache,
        api_key=api_key,
    )

    # then
    assert result == (False, "STATUS OK"), "Expected correct status to be marked"
    register_datapoint_mock.assert_called_once()
    assert (
        register_datapoint_mock.call_args[1]["prediction"].xyxy
        == np.array([[1, 1, 2, 2], [2, 2, 4, 4]])
    ).all(), "Expected scaling of prediction to happen in both axis, as this scaling keeps aspect ratio"
    return_strategy_credit_mock.assert_not_called()


def test_run_sink_when_api_key_is_not_specified() -> None:
    # given
    data_collector_block = RoboflowDatasetUploadBlockV1(
        cache=MemoryCache(),
        api_key=None,
        background_tasks=None,
        thread_pool_executor=None,
    )

    # when
    with pytest.raises(ValueError):
        _ = data_collector_block.run(
            images=Batch(content=[], indices=[]),
            predictions=Batch(content=[], indices=[]),
            target_project="my_project",
            usage_quota_name="my_quota",
            persist_predictions=True,
            minutely_usage_limit=10,
            hourly_usage_limit=100,
            daily_usage_limit=1000,
            max_image_size=(128, 128),
            compression_level=75,
            registration_tags=["some"],
            disable_sink=False,
            fire_and_forget=True,
            labeling_batch_prefix="my_batch",
            labeling_batches_recreation_frequency="never",
        )


def test_run_sink_when_sink_is_disabled_by_configuration() -> None:
    # given
    data_collector_block = RoboflowDatasetUploadBlockV1(
        cache=MemoryCache(),
        api_key="my_api_key",
        background_tasks=None,
        thread_pool_executor=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((512, 256, 3), dtype=np.uint8),
    )
    prediction = {
        "top": "car",
        "predictions": [
            {"class": "car", "confidence": 0.7},
            {"class": "truck", "confidence": 0.3},
        ],
    }
    indices = [(0,), (1,), (2,)]

    # when
    result = data_collector_block.run(
        images=Batch(content=[image, image, image], indices=indices),
        predictions=Batch(
            content=[prediction, prediction, prediction], indices=indices
        ),
        target_project="my_project",
        usage_quota_name="my_quota",
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 128),
        compression_level=75,
        registration_tags=["some"],
        disable_sink=True,
        fire_and_forget=True,
        labeling_batch_prefix="my_batch",
        labeling_batches_recreation_frequency="never",
    )

    # then
    assert (
        result
        == [
            {
                "error_status": False,
                "message": "Sink was disabled by parameter `disable_sink`",
            }
        ]
        * 3
    ), "Expected disable sink status to be returned"


@mock.patch.object(v1, "execute_registration", MagicMock())
def test_run_sink_when_registration_should_happen_in_background_tasks() -> None:
    # given
    background_tasks = BackgroundTasks()
    data_collector_block = RoboflowDatasetUploadBlockV1(
        cache=MemoryCache(),
        api_key="my_api_key",
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((512, 256, 3), dtype=np.uint8),
    )
    prediction = {
        "top": "car",
        "predictions": [
            {"class": "car", "confidence": 0.7},
            {"class": "truck", "confidence": 0.3},
        ],
    }
    indices = [(0,), (1,), (2,)]

    # when
    result = data_collector_block.run(
        images=Batch(content=[image, image, image], indices=indices),
        predictions=Batch(
            content=[prediction, prediction, prediction], indices=indices
        ),
        target_project="my_project",
        usage_quota_name="my_quota",
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 128),
        compression_level=75,
        registration_tags=["some"],
        disable_sink=False,
        fire_and_forget=True,
        labeling_batch_prefix="my_batch",
        labeling_batches_recreation_frequency="never",
    )

    # then
    assert (
        result
        == [
            {
                "error_status": False,
                "message": "Element registration happens in the background task",
            }
        ]
        * 3
    ), "Expected async execution status to be presented"
    assert len(background_tasks.tasks) == 3, "Async tasks to be added"


@mock.patch.object(v1, "execute_registration", MagicMock())
def test_run_sink_when_registration_should_happen_in_thread_pool() -> None:
    # given
    with ThreadPoolExecutor() as thread_pool_executor:
        data_collector_block = RoboflowDatasetUploadBlockV1(
            cache=MemoryCache(),
            api_key="my_api_key",
            background_tasks=None,
            thread_pool_executor=thread_pool_executor,
        )
        image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="parent"),
            numpy_image=np.zeros((512, 256, 3), dtype=np.uint8),
        )
        prediction = {
            "top": "car",
            "predictions": [
                {"class": "car", "confidence": 0.7},
                {"class": "truck", "confidence": 0.3},
            ],
        }
        indices = [(0,), (1,), (2,)]

        # when
        result = data_collector_block.run(
            images=Batch(content=[image, image, image], indices=indices),
            predictions=Batch(
                content=[prediction, prediction, prediction], indices=indices
            ),
            target_project="my_project",
            usage_quota_name="my_quota",
            persist_predictions=True,
            minutely_usage_limit=10,
            hourly_usage_limit=100,
            daily_usage_limit=1000,
            max_image_size=(128, 128),
            compression_level=75,
            registration_tags=["some"],
            disable_sink=False,
            fire_and_forget=True,
            labeling_batch_prefix="my_batch",
            labeling_batches_recreation_frequency="never",
        )

        # then
        assert (
            result
            == [
                {
                    "error_status": False,
                    "message": "Element registration happens in the background task",
                }
            ]
            * 3
        ), "Expected async execution status to be presented"


@mock.patch.object(v1, "execute_registration")
def test_run_sink_when_registration_should_happen_in_foreground_despite_providing_background_tasks(
    execute_registration_mock: MagicMock,
) -> None:
    # given
    background_tasks = BackgroundTasks()
    cache = MemoryCache()
    data_collector_block = RoboflowDatasetUploadBlockV1(
        cache=cache,
        api_key="my_api_key",
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((512, 256, 3), dtype=np.uint8),
    )
    prediction = {
        "top": "car",
        "predictions": [
            {"class": "car", "confidence": 0.7},
            {"class": "truck", "confidence": 0.3},
        ],
    }
    execute_registration_mock.return_value = False, "OK"
    indices = [(0,), (1,), (2,)]

    # when
    result = data_collector_block.run(
        images=Batch(content=[image, image, image], indices=indices),
        predictions=Batch(
            content=[prediction, prediction, prediction], indices=indices
        ),
        target_project="my_project",
        usage_quota_name="my_quota",
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 128),
        compression_level=75,
        registration_tags=["some"],
        disable_sink=False,
        fire_and_forget=False,
        labeling_batch_prefix="my_batch",
        labeling_batches_recreation_frequency="never",
    )

    # then
    assert (
        result
        == [
            {
                "error_status": False,
                "message": "OK",
            }
        ]
        * 3
    ), "Expected sync execution status to be presented"
    execute_registration_mock.assert_has_calls(
        [
            call(
                image=image,
                prediction=prediction,
                target_project="my_project",
                usage_quota_name="my_quota",
                persist_predictions=True,
                minutely_usage_limit=10,
                hourly_usage_limit=100,
                daily_usage_limit=1000,
                max_image_size=(128, 128),
                compression_level=75,
                registration_tags=["some"],
                labeling_batch_prefix="my_batch",
                new_labeling_batch_frequency="never",
                cache=cache,
                api_key="my_api_key",
            )
        ]
        * 3
    )
    assert len(background_tasks.tasks) == 0, "Async tasks not to be added"


@mock.patch.object(v1, "execute_registration")
def test_run_sink_when_predictions_not_provided(
    execute_registration_mock: MagicMock,
) -> None:
    # given
    background_tasks = BackgroundTasks()
    cache = MemoryCache()
    data_collector_block = RoboflowDatasetUploadBlockV1(
        cache=cache,
        api_key="my_api_key",
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((512, 256, 3), dtype=np.uint8),
    )
    execute_registration_mock.return_value = False, "OK"
    indices = [(0,), (1,), (2,)]

    # when
    result = data_collector_block.run(
        images=Batch(content=[image, image, image], indices=indices),
        predictions=None,
        target_project="my_project",
        usage_quota_name="my_quota",
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 128),
        compression_level=75,
        registration_tags=["some"],
        disable_sink=False,
        fire_and_forget=False,
        labeling_batch_prefix="my_batch",
        labeling_batches_recreation_frequency="never",
    )

    # then
    assert (
        result
        == [
            {
                "error_status": False,
                "message": "OK",
            }
        ]
        * 3
    ), "Expected sync execution status to be presented"
    execute_registration_mock.assert_has_calls(
        [
            call(
                image=image,
                prediction=None,
                target_project="my_project",
                usage_quota_name="my_quota",
                persist_predictions=True,
                minutely_usage_limit=10,
                hourly_usage_limit=100,
                daily_usage_limit=1000,
                max_image_size=(128, 128),
                compression_level=75,
                registration_tags=["some"],
                labeling_batch_prefix="my_batch",
                new_labeling_batch_frequency="never",
                cache=cache,
                api_key="my_api_key",
            )
        ]
        * 3
    )
    assert len(background_tasks.tasks) == 0, "Async tasks not to be added"


@mock.patch.object(v1, "return_strategy_credit")
@mock.patch.object(v1, "register_datapoint")
@mock.patch.object(v1, "use_credit_of_matching_strategy")
def test_execute_registration_with_custom_image_name(
    use_credit_of_matching_strategy_mock: MagicMock,
    register_datapoint_mock: MagicMock,
    return_strategy_credit_mock: MagicMock,
) -> None:
    # given
    api_key = "my_api_key"
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    expected_cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cache = MemoryCache()
    cache.set(key=expected_cache_key, value="my_workspace")
    use_credit_of_matching_strategy_mock.return_value = "my_strategy"
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((128, 256, 3), dtype=np.uint8),
    )
    register_datapoint_mock.return_value = "STATUS OK"

    # when
    result = execute_registration(
        image=image,
        prediction=None,
        target_project="my_project",
        usage_quota_name="my_quota",
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 64),
        compression_level=75,
        registration_tags=["some"],
        labeling_batch_prefix="my_batch",
        new_labeling_batch_frequency="never",
        cache=cache,
        api_key=api_key,
        image_name="custom_serial_number_123",
    )

    # then
    assert result == (False, "STATUS OK"), "Expected correct status to be marked"
    register_datapoint_mock.assert_called_once()
    # Verify custom image_name was used as local_image_id
    assert (
        register_datapoint_mock.call_args[1]["local_image_id"]
        == "custom_serial_number_123"
    ), "Expected custom image_name to be used as local_image_id"
    return_strategy_credit_mock.assert_not_called()


@mock.patch.object(v1, "return_strategy_credit")
@mock.patch.object(v1, "register_datapoint")
@mock.patch.object(v1, "use_credit_of_matching_strategy")
def test_execute_registration_without_image_name_uses_uuid(
    use_credit_of_matching_strategy_mock: MagicMock,
    register_datapoint_mock: MagicMock,
    return_strategy_credit_mock: MagicMock,
) -> None:
    # given
    api_key = "my_api_key"
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    expected_cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cache = MemoryCache()
    cache.set(key=expected_cache_key, value="my_workspace")
    use_credit_of_matching_strategy_mock.return_value = "my_strategy"
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((128, 256, 3), dtype=np.uint8),
    )
    register_datapoint_mock.return_value = "STATUS OK"

    # when
    result = execute_registration(
        image=image,
        prediction=None,
        target_project="my_project",
        usage_quota_name="my_quota",
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 64),
        compression_level=75,
        registration_tags=["some"],
        labeling_batch_prefix="my_batch",
        new_labeling_batch_frequency="never",
        cache=cache,
        api_key=api_key,
        # No image_name provided, should fall back to UUID
    )

    # then
    assert result == (False, "STATUS OK"), "Expected correct status to be marked"
    register_datapoint_mock.assert_called_once()
    # Verify UUID format is used when no image_name provided
    local_image_id = register_datapoint_mock.call_args[1]["local_image_id"]
    # UUID4 has format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx (36 chars with hyphens)
    assert (
        len(local_image_id) == 36 and local_image_id.count("-") == 4
    ), "Expected UUID format to be used when image_name not provided"
    return_strategy_credit_mock.assert_not_called()
