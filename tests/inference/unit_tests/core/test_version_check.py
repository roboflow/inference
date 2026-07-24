"""Tests for the GitHub version check in inference/core/__init__.py."""

from unittest import mock

from inference import core as inference_core


def _reset_version_check_state() -> None:
    inference_core.latest_release = None
    inference_core.last_checked = 0


def test_get_latest_release_version_survives_unexpected_response_body() -> None:
    # given - a 200 response that is not the expected GitHub payload
    _reset_version_check_state()
    response = mock.MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"message": "rate limited"}

    # when - must swallow the KeyError like a network failure
    with mock.patch.object(
        inference_core, "DISABLE_VERSION_CHECK", False
    ), mock.patch.object(inference_core.requests, "get", return_value=response):
        inference_core.get_latest_release_version()

    # then
    assert inference_core.latest_release is None


def test_get_latest_release_version_is_guarded_when_version_check_disabled() -> None:
    # given
    _reset_version_check_state()

    # when
    with mock.patch.object(
        inference_core, "DISABLE_VERSION_CHECK", True
    ), mock.patch.object(inference_core.requests, "get") as get_mock:
        inference_core.get_latest_release_version()

    # then - no network attempt at all
    get_mock.assert_not_called()


def test_get_latest_release_version_parses_expected_payload() -> None:
    # given
    _reset_version_check_state()
    response = mock.MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"tag_name": "v9.9.9"}

    # when
    with mock.patch.object(
        inference_core, "DISABLE_VERSION_CHECK", False
    ), mock.patch.object(inference_core.requests, "get", return_value=response):
        inference_core.get_latest_release_version()

    # then
    assert inference_core.latest_release == "9.9.9"
    _reset_version_check_state()
