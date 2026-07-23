import json
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock

import pytest
from filelock import Timeout

from inference.core.exceptions import ModelArtefactError
from inference.core.models import roboflow
from inference.core.models.roboflow import (
    RoboflowCoreModel,
    RoboflowInferenceModel,
    acquire_model_download_lock,
    class_mapping_not_available_in_environment,
    color_mapping_available_in_environment,
    get_class_names_from_environment_file,
    get_color_mapping_from_environment,
    is_model_artefacts_bucket_available,
)


@mock.patch.object(roboflow, "AWS_ACCESS_KEY_ID", None)
def test_is_model_artefacts_bucket_available_when_access_key_not_set() -> None:
    # when
    result = is_model_artefacts_bucket_available()

    # then
    assert result is False


@mock.patch.object(roboflow, "AWS_SECRET_ACCESS_KEY", None)
def test_is_model_artefacts_bucket_available_when_secret_not_set() -> None:
    # when
    result = is_model_artefacts_bucket_available()

    # then
    assert result is False


@mock.patch.object(roboflow, "LAMBDA", False)
def test_is_model_artefacts_bucket_available_when_not_in_lambda_mode() -> None:
    # when
    result = is_model_artefacts_bucket_available()

    # then
    assert result is False


@mock.patch.object(roboflow, "S3_CLIENT", None)
def test_is_model_artefacts_bucket_available_when_s3_client_not_initialised() -> None:
    # when
    result = is_model_artefacts_bucket_available()

    # then
    assert result is False


@mock.patch.object(roboflow, "AWS_ACCESS_KEY_ID", "some")
@mock.patch.object(roboflow, "AWS_SECRET_ACCESS_KEY", "other")
@mock.patch.object(roboflow, "LAMBDA", True)
@mock.patch.object(roboflow, "S3_CLIENT", MagicMock())
def test_is_model_artefacts_bucket_available_when_availability_check_should_pass() -> (
    None
):
    # when
    result = is_model_artefacts_bucket_available()

    # then
    assert result is True


@mock.patch.object(roboflow, "OFFLINE_MODE", True)
@mock.patch.object(roboflow, "AWS_ACCESS_KEY_ID", "some")
@mock.patch.object(roboflow, "AWS_SECRET_ACCESS_KEY", "other")
@mock.patch.object(roboflow, "LAMBDA", True)
@mock.patch.object(roboflow, "S3_CLIENT", MagicMock())
def test_model_artefacts_bucket_is_unavailable_in_offline_mode() -> None:
    assert is_model_artefacts_bucket_available() is False


def test_cached_model_load_skips_api_authorization_in_offline_mode() -> None:
    model = object.__new__(RoboflowInferenceModel)
    model.api_key = None
    model.endpoint = "workspace/model/1"
    model.cache_model_artefacts = MagicMock()
    model.load_model_artifacts_from_cache = MagicMock()

    with mock.patch.object(roboflow, "OFFLINE_MODE", True), mock.patch.object(
        roboflow,
        "MODELS_CACHE_AUTH_ENABLED",
        True,
    ), mock.patch.object(
        roboflow,
        "_check_if_api_key_has_access_to_model",
    ) as access_check_mock:
        model.get_model_artifacts()

    access_check_mock.assert_not_called()
    model.cache_model_artefacts.assert_called_once()
    model.load_model_artifacts_from_cache.assert_called_once()


def test_missing_model_artifacts_do_not_use_network_in_offline_mode() -> None:
    model = object.__new__(RoboflowInferenceModel)
    model.endpoint = "workspace/model/1"
    model.get_all_required_infer_bucket_file = MagicMock(return_value=["weights.onnx"])
    model.download_model_artefacts_from_s3 = MagicMock()
    model.download_model_artifacts_from_roboflow_api = MagicMock()

    with mock.patch.object(roboflow, "OFFLINE_MODE", True), mock.patch.object(
        roboflow,
        "are_all_files_cached",
        return_value=False,
    ):
        with pytest.raises(ModelArtefactError, match="OFFLINE_MODE"):
            model.cache_model_artefacts()

    model.download_model_artefacts_from_s3.assert_not_called()
    model.download_model_artifacts_from_roboflow_api.assert_not_called()


def test_missing_core_model_weights_do_not_use_network_in_offline_mode() -> None:
    model = object.__new__(RoboflowCoreModel)
    model.endpoint = "core-model"
    model.get_infer_bucket_file_list = MagicMock(return_value=["weights.pt"])
    model.download_model_artefacts_from_s3 = MagicMock()
    model.download_model_from_roboflow_api = MagicMock()

    with mock.patch.object(roboflow, "OFFLINE_MODE", True), mock.patch.object(
        roboflow,
        "MODELS_CACHE_AUTH_ENABLED",
        True,
    ), mock.patch.object(
        roboflow,
        "_check_if_api_key_has_access_to_model",
    ) as access_check_mock, mock.patch.object(
        roboflow,
        "are_all_files_cached",
        return_value=False,
    ):
        with pytest.raises(ModelArtefactError, match="OFFLINE_MODE"):
            model.download_weights()

    access_check_mock.assert_not_called()
    model.download_model_artefacts_from_s3.assert_not_called()
    model.download_model_from_roboflow_api.assert_not_called()


@pytest.mark.parametrize(
    "environment, expected_result",
    [
        (None, False),
        ({}, False),
        ({"COLORS": json.dumps({"class_a": "#ffffff"})}, False),
        ({"COLORS": {"class_a": "#ffffff"}}, True),
    ],
)
def test_color_mapping_available_in_environment_when_environment(
    environment: Optional[dict], expected_result: bool
) -> None:
    # when
    result = color_mapping_available_in_environment(environment=environment)

    # then
    assert result is expected_result


def test_get_color_mapping_from_environment_when_color_mapping_in_environment() -> None:
    # given
    environment = {"COLORS": {"class_a": "#ffffff"}}

    # when
    result = get_color_mapping_from_environment(
        environment=environment, class_names=["class_a"]
    )

    # then
    assert result == {"class_a": "#ffffff"}


def test_get_color_mapping_from_environment_when_color_mapping_in_environment_as_json_string() -> (
    None
):
    # given
    environment = {"COLORS": json.dumps({"class_a": "#ffffff"})}

    # when
    result = get_color_mapping_from_environment(
        environment=environment, class_names=["class_a"]
    )

    # then
    assert result == {"class_a": "#4892EA"}


def test_get_color_mapping_from_environment_when_color_mapping_not_in_environment() -> (
    None
):
    # given
    environment = {}

    # when
    result = get_color_mapping_from_environment(
        environment=environment, class_names=["class_a", "class_b"]
    )

    # then
    assert result == {"class_a": "#4892EA", "class_b": "#00EEC3"}


@pytest.mark.parametrize(
    "environment, expected_result",
    [
        ({}, True),
        ({"CLASS_MAP": json.dumps({"0": "class_a"})}, True),
        ({"CLASS_MAP": {"0": "class_1"}}, False),
    ],
)
def test_class_mapping_not_available_in_environment(
    environment: dict, expected_result: bool
) -> None:
    # when
    result = class_mapping_not_available_in_environment(environment=environment)

    # then
    assert result is expected_result


@pytest.mark.parametrize(
    "environment", [None, {}, {"CLASS_MAP": json.dumps({"0": "class_a"})}]
)
def test_get_class_names_from_environment_file_when_procedure_should_fail(
    environment: Optional[dict],
) -> None:
    # when
    with pytest.raises(ModelArtefactError):
        _ = get_class_names_from_environment_file(environment=environment)


def test_get_class_names_from_environment_file() -> None:
    # given
    environment = {
        "CLASS_MAP": {
            "0": "class_a",
            "1": "class_b",
            "2": "class_c",
            "3": "class_d",
            "4": "class_e",
            "5": "class_f",
            "6": "class_g",
            "7": "class_h",
            "8": "class_i",
            "9": "class_j",
            "10": "class_k",
            "11": "class_l",
        }
    }

    # when
    result = get_class_names_from_environment_file(environment=environment)

    # then
    assert result == [
        "class_a",
        "class_b",
        "class_c",
        "class_d",
        "class_e",
        "class_f",
        "class_g",
        "class_h",
        "class_i",
        "class_j",
        "class_k",
        "class_l",
    ]


@mock.patch.object(roboflow, "MODEL_WEIGHTS_DOWNLOAD_LOCK_MAX_ATTEMPTS", 3)
@mock.patch.object(roboflow, "MODEL_WEIGHTS_DOWNLOAD_LOCK_TIMEOUT", 600.0)
@mock.patch.object(roboflow, "FileLock")
def test_acquire_model_download_lock_returns_held_lock_on_first_success(
    file_lock_mock: MagicMock,
) -> None:
    # given
    lock = MagicMock()
    file_lock_mock.return_value = lock

    # when
    result = acquire_model_download_lock("/tmp/model.lock", model_id="m/1")

    # then
    assert result is lock
    file_lock_mock.assert_called_once_with("/tmp/model.lock", timeout=600.0)
    lock.acquire.assert_called_once()


@mock.patch.object(roboflow, "MODEL_WEIGHTS_DOWNLOAD_LOCK_MAX_ATTEMPTS", 3)
@mock.patch.object(roboflow, "MODEL_WEIGHTS_DOWNLOAD_LOCK_TIMEOUT", 600.0)
@mock.patch.object(roboflow, "FileLock")
def test_acquire_model_download_lock_retries_on_timeout_then_succeeds(
    file_lock_mock: MagicMock,
) -> None:
    # given - first lock times out on acquire (holder still downloading), second succeeds
    timing_out_lock = MagicMock()
    timing_out_lock.acquire.side_effect = Timeout("/tmp/model.lock")
    succeeding_lock = MagicMock()
    file_lock_mock.side_effect = [timing_out_lock, succeeding_lock]

    # when
    result = acquire_model_download_lock("/tmp/model.lock", model_id="m/1")

    # then
    assert result is succeeding_lock
    assert file_lock_mock.call_count == 2


@mock.patch.object(roboflow, "MODEL_WEIGHTS_DOWNLOAD_LOCK_MAX_ATTEMPTS", 3)
@mock.patch.object(roboflow, "MODEL_WEIGHTS_DOWNLOAD_LOCK_TIMEOUT", 600.0)
@mock.patch.object(roboflow, "FileLock")
def test_acquire_model_download_lock_raises_after_exhausting_attempts(
    file_lock_mock: MagicMock,
) -> None:
    # given - every attempt times out
    always_timing_out_lock = MagicMock()
    always_timing_out_lock.acquire.side_effect = Timeout("/tmp/model.lock")
    file_lock_mock.return_value = always_timing_out_lock

    # when / then
    with pytest.raises(Timeout):
        acquire_model_download_lock("/tmp/model.lock", model_id="m/1")
    assert file_lock_mock.call_count == 3
