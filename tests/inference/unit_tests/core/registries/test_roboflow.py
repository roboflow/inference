import json
import os.path
from typing import Tuple
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.cache import model_artifacts
from inference.core.devices.utils import GLOBAL_DEVICE_ID
from inference.core.entities.types import ModelType, TaskType
from inference.core.exceptions import (
    MissingApiKeyError,
    ModelDeploymentNotSupportedError,
    ModelNotRecognisedError,
    RoboflowAPINotAuthorizedError,
)
from inference.core.registries import roboflow
from inference.core.registries.roboflow import (
    FINE_TUNED_SAM3_DEPLOYMENT_ERROR,
    RoboflowModelRegistry,
    _in_process_metadata_cache,
    get_model_metadata_from_cache,
    get_model_type,
    model_metadata_content_is_invalid,
    save_model_metadata_in_cache,
)
from inference.core.roboflow_api import ModelEndpointType


@pytest.fixture(autouse=True)
def clear_in_process_metadata_cache():
    _in_process_metadata_cache.cache.clear()
    roboflow._check_if_api_key_has_access_to_model.cache_clear()
    yield
    _in_process_metadata_cache.cache.clear()
    roboflow._check_if_api_key_has_access_to_model.cache_clear()


@pytest.mark.parametrize("is_lambda", [False, True])
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_file_does_not_exist(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
    is_lambda: bool,
) -> None:
    # given
    construct_model_type_cache_path_mock.return_value = os.path.join(
        empty_local_dir, "model_type.json"
    )

    # when
    with mock.patch.object(roboflow, "LAMBDA", is_lambda):
        result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@pytest.mark.parametrize("is_lambda", [False, True])
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_file_is_not_json(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
    is_lambda: bool,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write("FOR SURE NOT JSON :)")

    # when
    with mock.patch.object(roboflow, "LAMBDA", is_lambda):
        result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@pytest.mark.parametrize("is_lambda", [False, True])
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_file_is_empty(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
    is_lambda: bool,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write("")

    # when
    with mock.patch.object(roboflow, "LAMBDA", is_lambda):
        result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@pytest.mark.parametrize("is_lambda", [False, True])
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_is_invalid(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
    is_lambda: bool,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write(json.dumps({"some": "key"}))

    # when
    with mock.patch.object(roboflow, "LAMBDA", is_lambda):
        result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result is None


@pytest.mark.parametrize("is_lambda", [False, True])
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_metadata_from_cache_when_metadata_is_valid(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
    is_lambda: bool,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "project_task_type": "object-detection",
                    "model_type": "yolov8n",
                }
            )
        )

    # when
    with mock.patch.object(roboflow, "LAMBDA", is_lambda):
        result = get_model_metadata_from_cache(dataset_id="some", version_id="1")

    # then
    assert result == ("object-detection", "yolov8n")


def test_model_metadata_content_is_invalid_when_content_is_empty() -> None:
    # when
    result = model_metadata_content_is_invalid(content=None)

    # then
    assert result is True


def test_model_metadata_content_is_invalid_when_content_is_not_dict() -> None:
    # when
    result = model_metadata_content_is_invalid(content=[1, 2, 3])

    # then
    assert result is True


def test_model_metadata_content_is_invalid_when_model_type_is_missing() -> None:
    # when
    result = model_metadata_content_is_invalid(
        content={
            "project_task_type": "object-detection",
        }
    )

    # then
    assert result is True


def test_model_metadata_content_is_invalid_when_task_type_is_missing() -> None:
    # when
    result = model_metadata_content_is_invalid(
        content={
            "model_type": "yolov8n",
        }
    )

    # then
    assert result is True


@pytest.mark.parametrize("is_lambda", [False, True])
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_save_model_metadata_in_cache(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
    is_lambda: bool,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path

    # when
    with mock.patch.object(roboflow, "LAMBDA", is_lambda):
        save_model_metadata_in_cache(
            dataset_id="some",
            version_id="1",
            project_task_type="instance-segmentation",
            model_type="yolov8l",
        )
    with open(metadata_path) as f:
        result = json.load(f)

    # then
    assert result["model_type"] == "yolov8l"
    assert result["project_task_type"] == "instance-segmentation"
    construct_model_type_cache_path_mock.assert_called_once_with(
        dataset_id="some", version_id="1"
    )


def test_save_and_load_model_metadata_in_cache_when_instant_model_slug_is_long(
    empty_local_dir: str,
) -> None:
    # given
    long_model_slug = "find-" + ("class-" * 60) + "instant-1"
    dataset_id = f"huizen/{long_model_slug}"

    # when
    with mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", empty_local_dir
    ), mock.patch.object(roboflow, "LAMBDA", True):
        save_model_metadata_in_cache(
            dataset_id=dataset_id,
            version_id=None,
            project_task_type="object-detection",
            model_type="yolov8n",
        )
        _in_process_metadata_cache.cache.clear()
        result = get_model_metadata_from_cache(dataset_id=dataset_id, version_id=None)
        cache_path = roboflow.construct_model_type_cache_path(
            dataset_id=dataset_id, version_id=None
        )

    # then
    assert result == ("object-detection", "yolov8n")
    assert os.path.isfile(cache_path)
    assert all(
        len(os.fsencode(path_segment)) <= 255
        for path_segment in cache_path.split(os.sep)
        if path_segment
    )


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_cache_is_utilised(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "project_task_type": "object-detection",
                    "model_type": "yolov8n",
                }
            )
        )

    # when
    result = get_model_type(model_id="some/1", api_key="my_api_key")

    # then
    construct_model_type_cache_path_mock.assert_called_once_with(
        dataset_id="some", version_id="1"
    )
    assert result == ("object-detection", "yolov8n")


@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_classification_subtype_is_cached(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "project_task_type": "multi-label-classification",
                    "model_type": "vit",
                }
            )
        )

    # when
    result = get_model_type(model_id="some/1", api_key="my_api_key")

    # then
    assert result == ("multi-label-classification", "vit")


@mock.patch.object(roboflow, "SAM3_FINE_TUNED_MODELS_ENABLED", False)
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_fine_tuned_sam3_is_cached_but_disabled(
    construct_model_type_cache_path_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    with open(metadata_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "project_task_type": "instance-segmentation",
                    "model_type": "sam3-large",
                }
            )
        )

    # when / then
    with pytest.raises(ModelDeploymentNotSupportedError) as error:
        get_model_type(model_id="workspace/123", api_key="my_api_key")

    assert str(error.value) == FINE_TUNED_SAM3_DEPLOYMENT_ERROR


@pytest.mark.parametrize(
    "model_id, expected_result",
    [
        ("clip/1", ("embed", "clip")),
        ("sam/1", ("embed", "sam")),
        ("gaze/1", ("gaze", "l2cs")),
    ],
)
def test_get_model_type_when_generic_model_is_utilised(
    model_id: str,
    expected_result: Tuple[TaskType, ModelType],
) -> None:
    # when
    result = get_model_type(model_id=model_id, api_key="my_api_key")

    # then
    assert result == expected_result


def test_model_pipelines_enumerate_all_coded_pp_ocr_ids() -> None:
    # given
    stage_variants = ("none", "tiny", "small", "medium")
    expected_combo_ids = {
        f"pp_ocr/{text_detection}-{text_recognition}"
        for text_detection in stage_variants
        for text_recognition in stage_variants
        if (text_detection, text_recognition) != ("none", "none")
    }
    expected_single_token_ids = {"pp_ocr/tiny", "pp_ocr/small", "pp_ocr/medium"}

    # then
    assert set(roboflow.MODEL_PIPELINES) == (
        expected_combo_ids | expected_single_token_ids | {"pp_ocr"}
    )
    for definition in roboflow.MODEL_PIPELINES.values():
        assert (definition.task_type, definition.model_type) == ("ocr", "pp_ocr")
        assert len(definition.downstream_model_ids) > 0
    assert "pp_ocr/none-none" not in roboflow.MODEL_PIPELINES
    assert "pp_ocr/none" not in roboflow.MODEL_PIPELINES
    # pipeline IDs must not leak into GENERIC_MODELS - auth treats them differently
    assert all(
        model_id not in roboflow.GENERIC_MODELS for model_id in roboflow.MODEL_PIPELINES
    )


@pytest.mark.parametrize(
    "model_id, expected_downstream",
    [
        ("pp_ocr/small-small", ("pp-ocrv6-det/small", "pp-ocrv6-rec/small")),
        ("pp_ocr/tiny-medium", ("pp-ocrv6-det/tiny", "pp-ocrv6-rec/medium")),
        ("pp_ocr/none-small", ("pp-ocrv6-rec/small",)),
        ("pp_ocr/medium-none", ("pp-ocrv6-det/medium",)),
        ("pp_ocr/tiny", ("pp-ocrv6-det/tiny", "pp-ocrv6-rec/tiny")),
        ("pp_ocr", ("pp-ocrv6-det/small", "pp-ocrv6-rec/small")),
    ],
)
def test_model_pipelines_map_to_expected_downstream_models(
    model_id: str, expected_downstream: Tuple[str, ...]
) -> None:
    assert (
        roboflow.MODEL_PIPELINES[model_id].downstream_model_ids == expected_downstream
    )


@pytest.mark.parametrize(
    "model_id, expected_downstream",
    [
        ("pp_ocr/tiny-medium", ("pp-ocrv6-det/tiny", "pp-ocrv6-rec/medium")),
        ("pp_ocr/none-small", ("pp-ocrv6-rec/small",)),
        ("pp_ocr/medium-none", ("pp-ocrv6-det/medium",)),
    ],
)
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
def test_check_api_key_for_pp_ocr_pipeline_authorizes_downstream_models(
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    model_id: str,
    expected_downstream: Tuple[str, ...],
) -> None:
    # when
    result = roboflow._check_if_api_key_has_access_to_model(
        api_key=f"my_api_key-{model_id}",
        model_id=model_id,
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    # then - the synthetic pipeline ID itself must never reach the remote registry,
    # but every downstream stage model must be authorized against it
    assert result is True
    checked_model_ids = [
        call.kwargs["model_id"]
        for call in get_model_metadata_from_inference_models_registry_mock.call_args_list
    ]
    assert checked_model_ids == list(expected_downstream)


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
def test_check_api_key_for_pp_ocr_pipeline_fails_when_downstream_model_not_authorized(
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
) -> None:
    # given - detection stage authorized, recognition stage not
    def _registry_response(api_key: str, model_id: str, **kwargs):
        if model_id == "pp-ocrv6-rec/medium":
            raise RoboflowAPINotAuthorizedError()
        return {"taskType": "ocr"}

    get_model_metadata_from_inference_models_registry_mock.side_effect = (
        _registry_response
    )

    # when
    result = roboflow._check_if_api_key_has_access_to_model(
        api_key="my_api_key",
        model_id="pp_ocr/tiny-medium",
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    # then
    assert result is False


@pytest.mark.parametrize("model_id", ["pp_ocr/small-small", "pp_ocr"])
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
@mock.patch.object(
    roboflow,
    "get_roboflow_instant_model_data",
    side_effect=RoboflowAPINotAuthorizedError,
)
@mock.patch.object(
    roboflow,
    "get_roboflow_model_data",
    side_effect=RoboflowAPINotAuthorizedError,
)
def test_check_api_key_for_pp_ocr_pipeline_not_recognized_without_inference_models(
    get_roboflow_model_data_mock: MagicMock,
    get_roboflow_instant_model_data_mock: MagicMock,
    model_id: str,
) -> None:
    # when - with USE_INFERENCE_MODELS disabled, pipeline IDs fall through to the
    # regular resolution and fail closed there
    result = roboflow._check_if_api_key_has_access_to_model(
        api_key="my_api_key",
        model_id=model_id,
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    # then
    assert result is False


@pytest.mark.parametrize("model_id", ["pp_ocr/none-none", "pp_ocr/huge-small"])
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
@mock.patch.object(
    roboflow,
    "get_model_metadata_from_inference_models_registry",
    side_effect=RoboflowAPINotAuthorizedError,
)
def test_check_api_key_for_invalid_pp_ocr_pipeline_fails_closed(
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    model_id: str,
) -> None:
    # when - IDs outside the coded pipeline set are not treated as pipelines
    result = roboflow._check_if_api_key_has_access_to_model(
        api_key=f"my_api_key-{model_id}",
        model_id=model_id,
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    # then
    assert result is False
    get_model_metadata_from_inference_models_registry_mock.assert_called_once_with(
        api_key=f"my_api_key-{model_id}",
        model_id=model_id,
        countinference=None,
        service_secret=None,
    )


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
@mock.patch.object(
    roboflow,
    "get_model_metadata_from_inference_models_registry",
    side_effect=RoboflowAPINotAuthorizedError,
)
def test_check_api_key_does_not_blanket_trust_generic_models(
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
) -> None:
    # when - full-ID GENERIC_MODELS entries (e.g. sam3/sam3_interactive) must still
    # be authorized remotely; regression guard against trusting GENERIC_MODELS as such
    result = roboflow._check_if_api_key_has_access_to_model(
        api_key="my_api_key",
        model_id="sam3/sam3_interactive",
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    # then
    assert result is False
    get_model_metadata_from_inference_models_registry_mock.assert_called_once()


@pytest.mark.parametrize(
    "model_id",
    ["pp_ocr", "pp_ocr/small", "pp_ocr/tiny-medium", "pp_ocr/none-small"],
)
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
@mock.patch.object(roboflow, "get_roboflow_instant_model_data")
@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
def test_get_model_type_for_pipeline_when_inference_models_enabled(
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    get_roboflow_instant_model_data_mock: MagicMock,
    model_id: str,
) -> None:
    # when
    result = get_model_type(model_id=model_id, api_key="my_api_key")

    # then - pipeline recognition is static and must not call any remote API
    assert result == ("ocr", "pp_ocr")
    get_model_metadata_from_inference_models_registry_mock.assert_not_called()
    get_roboflow_model_data_mock.assert_not_called()
    get_roboflow_instant_model_data_mock.assert_not_called()


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
@mock.patch.object(roboflow, "get_roboflow_model_data")
def test_get_model_type_for_pipeline_when_inference_models_disabled(
    get_roboflow_model_data_mock: MagicMock,
) -> None:
    # given - with the flag off, pipeline IDs are not recognized and resolution
    # falls through to the regular Roboflow API pathway
    get_roboflow_model_data_mock.side_effect = RoboflowAPINotAuthorizedError()

    # when / then
    with pytest.raises(RoboflowAPINotAuthorizedError):
        get_model_type(model_id="pp_ocr/small-small", api_key="my_api_key")
    get_roboflow_model_data_mock.assert_called_once()


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "get_roboflow_model_data")
def test_check_api_key_for_yolo_world_core_model_uses_legacy_core_model_endpoint(
    get_roboflow_model_data_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
) -> None:
    # when
    result = roboflow._check_if_api_key_has_access_to_model(
        api_key="my_api_key",
        model_id="yolo_world/l",
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    # then
    assert result is True
    get_model_metadata_from_inference_models_registry_mock.assert_not_called()
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="yolo_world/l",
        endpoint_type=ModelEndpointType.CORE_MODEL,
        device_id=GLOBAL_DEVICE_ID,
        countinference=None,
        service_secret=None,
    )


@mock.patch.object(roboflow, "SAM3_FINE_TUNED_MODELS_ENABLED", False)
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_fine_tuned_sam3_is_requested_and_disabled(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_data_mock.return_value = {
        "ort": {
            "type": "instance-segmentation",
            "modelType": "sam3-large",
        }
    }

    # when / then
    with pytest.raises(ModelDeploymentNotSupportedError) as error:
        get_model_type(
            model_id="workspace/123",
            api_key="my_api_key",
        )

    assert str(error.value) == FINE_TUNED_SAM3_DEPLOYMENT_ERROR
    assert not os.path.exists(metadata_path)
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="workspace/123",
        countinference=None,
        service_secret=None,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    )


@mock.patch.object(roboflow, "SAM3_FINE_TUNED_MODELS_ENABLED", False)
@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_when_sam3_from_new_model_registry_is_requested_and_disabled(
    construct_model_type_cache_path_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_model_metadata_from_inference_models_registry_mock.return_value = {
        "modelType": "sam3",
        "taskType": "instance-segmentation",
    }

    # when / then
    with pytest.raises(ModelDeploymentNotSupportedError) as error:
        get_model_type(
            model_id="workspace/123",
            api_key="my_api_key",
        )

    assert str(error.value) == FINE_TUNED_SAM3_DEPLOYMENT_ERROR
    assert not os.path.exists(metadata_path)
    get_model_metadata_from_inference_models_registry_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="workspace/123",
        countinference=None,
        service_secret=None,
    )


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called_for_specific_model(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_data_mock.return_value = {
        "ort": {
            "type": "object-detection",
            "modelType": "yolov8n",
        }
    }

    # when
    result = get_model_type(
        model_id="some/1",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "yolov8n")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "yolov8n"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="some/1",
        countinference=None,
        service_secret=None,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    )


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called_for_specific_model_and_model_type_specified_as_ort(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_data_mock.return_value = {
        "ort": {
            "type": "object-detection",
            "modelType": "ort",
        }
    }

    # when
    result = get_model_type(
        model_id="some/1",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "yolov5v2s")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "yolov5v2s"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="some/1",
        countinference=None,
        service_secret=None,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    )


@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_when_roboflow_api_is_called_for_model_from_new_model_registry(
    construct_model_type_cache_path_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_model_metadata_from_inference_models_registry_mock.return_value = {
        "modelType": "yolov8",
        "taskType": "object-detection",
    }

    # when
    result = get_model_type(
        model_id="dummy-model",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "yolov8")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "yolov8"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_model_metadata_from_inference_models_registry_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="dummy-model",
        countinference=None,
        service_secret=None,
    )


@mock.patch.object(
    roboflow, "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", True
)
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_for_local_inference_models_package_uses_declared_architecture(
    empty_local_dir: str,
) -> None:
    # given
    with open(os.path.join(empty_local_dir, "model_config.json"), "w") as f:
        json.dump(
            {
                "model_architecture": "depth-anything-v2",
                "task_type": "depth-estimation",
                "backend_type": "torch",
            },
            f,
        )

    # when
    result = get_model_type(model_id=empty_local_dir, api_key="my_api_key")

    # then
    assert result == ("depth-estimation", "depth-anything-v2")


@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_when_new_model_registry_returns_classification_subtype(
    construct_model_type_cache_path_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_model_metadata_from_inference_models_registry_mock.return_value = {
        "modelType": "vit",
        "taskType": "multi-label-classification",
    }

    # when
    result = get_model_type(
        model_id="animal-classification-9lufm/1",
        api_key="my_api_key",
    )

    # then
    assert result == ("multi-label-classification", "vit")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "vit"
    assert persisted_metadata["project_task_type"] == "multi-label-classification"


@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "get_model_metadata_from_inference_models_registry")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", True)
def test_get_model_type_when_versioned_model_from_new_model_registry_is_requested(
    construct_model_type_cache_path_mock: MagicMock,
    get_model_metadata_from_inference_models_registry_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_model_metadata_from_inference_models_registry_mock.return_value = {
        "modelType": "rfdetr",
        "taskType": "object-detection",
    }

    # when
    result = get_model_type(
        model_id="coco/38",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "rfdetr")
    get_model_metadata_from_inference_models_registry_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="coco/38",
        countinference=None,
        service_secret=None,
    )
    get_roboflow_model_data_mock.assert_not_called()


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called_for_specific_model_and_model_type_not_specified(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_data_mock.return_value = {
        "ort": {
            "type": "object-detection",
        }
    }

    # when
    result = get_model_type(
        model_id="some/1",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "yolov5v2s")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "yolov5v2s"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="some/1",
        countinference=None,
        service_secret=None,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    )


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called_for_specific_model_and_project_type_not_specified(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_data_mock.return_value = {"ort": {}}

    # when
    result = get_model_type(
        model_id="some/1",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "yolov5v2s")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "yolov5v2s"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key="my_api_key",
        model_id="some/1",
        countinference=None,
        service_secret=None,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    )


@mock.patch.object(roboflow, "USE_INFERENCE_MODELS", False)
@mock.patch.object(roboflow, "get_roboflow_model_data")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called_for_specific_model_without_api_key_for_public_model(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_model_data_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_model_data_mock.return_value = {
        "ort": {
            "type": "object-detection",
            "modelType": "yolov8n",
        }
    }

    # when
    result = get_model_type(
        model_id="some/1",
        api_key=None,
    )

    # then
    assert result == ("object-detection", "yolov8n")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "yolov8n"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_roboflow_model_data_mock.assert_called_once_with(
        api_key=None,
        model_id="some/1",
        countinference=None,
        service_secret=None,
        endpoint_type=ModelEndpointType.ORT,
        device_id=GLOBAL_DEVICE_ID,
    )


@mock.patch.object(roboflow, "get_roboflow_workspace")
@mock.patch.object(roboflow, "get_roboflow_dataset_type")
@mock.patch.object(roboflow, "construct_model_type_cache_path")
def test_get_model_type_when_roboflow_api_is_called_for_mock(
    construct_model_type_cache_path_mock: MagicMock,
    get_roboflow_dataset_type_mock: MagicMock,
    get_roboflow_workspace_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    metadata_path = os.path.join(empty_local_dir, "model_type.json")
    construct_model_type_cache_path_mock.return_value = metadata_path
    get_roboflow_dataset_type_mock.return_value = "object-detection"
    get_roboflow_workspace_mock.return_value = "my_workspace"

    # when
    result = get_model_type(
        model_id="some/0",
        api_key="my_api_key",
    )

    # then
    assert result == ("object-detection", "stub")
    with open(metadata_path) as f:
        persisted_metadata = json.load(f)
    assert persisted_metadata["model_type"] == "stub"
    assert persisted_metadata["project_task_type"] == "object-detection"
    get_roboflow_dataset_type_mock.assert_called_once_with(
        api_key="my_api_key",
        workspace_id="my_workspace",
        dataset_id="some",
    )
    get_roboflow_workspace_mock.assert_called_once_with(api_key="my_api_key")


def test_get_model_type_when_roboflow_api_is_called_for_mock_without_api_key() -> None:
    with pytest.raises(MissingApiKeyError):
        _ = get_model_type(
            model_id="some/0",
            api_key=None,
        )


@mock.patch.object(roboflow, "get_model_type")
def test_roboflow_model_registry_get_model_on_cache_miss(
    get_model_type_mock: MagicMock,
) -> None:
    # given
    get_model_type_mock.return_value = ("object-detection", "yolov8n")
    registry = RoboflowModelRegistry(registry_dict={})

    # when
    with pytest.raises(ModelNotRecognisedError):
        _ = registry.get_model(model_id="some/1", api_key="my_api_key")


@mock.patch.object(roboflow, "get_model_type")
def test_roboflow_model_registry_get_model_on_cache_ht(
    get_model_type_mock: MagicMock,
) -> None:
    # given
    get_model_type_mock.return_value = ("object-detection", "yolov8n")
    registry = RoboflowModelRegistry(
        registry_dict={("object-detection", "yolov8n"): "some"}
    )

    # when
    result = registry.get_model(model_id="some/1", api_key="my_api_key")

    # then
    assert result == "some"
