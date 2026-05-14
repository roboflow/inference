"""Behavioural tests for the deprecated detect_gazes SDK helpers."""

import pytest
from requests_mock import Mocker

from inference_sdk import InferenceHTTPClient
from inference_sdk.http.errors import FeatureDeprecatedError, HTTPClientError


def test_detect_gazes_raises_feature_deprecated_error_without_network_call(
    requests_mock: Mocker,
) -> None:
    # given — register a catch-all matcher that asserts no traffic
    requests_mock.register_uri(
        "POST",
        "http://some.com/gaze/gaze_detection",
        json={"unexpected": "should not be reached"},
    )
    client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")

    # when
    with pytest.raises(FeatureDeprecatedError) as captured:
        client.detect_gazes(inference_input="/some/image.jpg")

    # then
    assert captured.value.feature == "InferenceHTTPClient.detect_gazes"
    assert requests_mock.call_count == 0, "SDK helper must short-circuit before any HTTP traffic."


def test_detect_gazes_feature_deprecated_error_is_an_http_client_error() -> None:
    # given
    client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")

    # when / then — broad-catch consumers of HTTPClientError must catch this too
    with pytest.raises(HTTPClientError):
        client.detect_gazes(inference_input="/some/image.jpg")


@pytest.mark.asyncio
async def test_detect_gazes_async_raises_feature_deprecated_error_without_network_call() -> None:
    # given
    client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")

    # when
    with pytest.raises(FeatureDeprecatedError) as captured:
        await client.detect_gazes_async(inference_input="/some/image.jpg")

    # then
    assert captured.value.feature == "InferenceHTTPClient.detect_gazes_async"
