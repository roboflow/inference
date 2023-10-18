import json
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.exceptions import ModelArtefactError
from inference.core.models.roboflow import (
    is_model_artefacts_bucket_available,
    color_mapping_available_in_environment,
    get_color_mapping_from_environment,
    class_mapping_not_available_in_environment,
    get_class_names_from_environment_file,
)
from inference.core.models import roboflow


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
        ({"CLASS_MAP": json.dumps({"1": "class_a"})}, True),
        ({"CLASS_MAP": {"1": "class_1"}}, False),
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
    "environment", [None, {}, {"CLASS_MAP": json.dumps({"1": "class_a"})}]
)
def test_get_class_names_from_environment_file_when_procedure_should_fail(
    environment: Optional[dict],
) -> None:
    # when
    with pytest.raises(ModelArtefactError):
        _ = get_class_names_from_environment_file(environment=environment)


def test_get_class_names_from_environment_file() -> None:
    # given
    environment = {"CLASS_MAP": {"1": "class_a", "2": "class_b", "3": "class_c"}}

    # when
    result = get_class_names_from_environment_file(environment=environment)

    # then
    assert result == ["class_a", "class_b", "class_c"]
