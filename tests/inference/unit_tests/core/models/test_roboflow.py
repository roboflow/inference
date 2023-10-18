from unittest import mock
from unittest.mock import MagicMock

from inference.core.models.roboflow import is_model_artefacts_bucket_available
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
