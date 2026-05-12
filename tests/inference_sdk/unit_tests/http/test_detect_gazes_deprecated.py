"""Behavioural tests for the deprecated detect_gazes SDK helpers."""

import warnings

import pytest
from requests_mock import Mocker

from inference_sdk import InferenceHTTPClient
from inference_sdk.config import InferenceSDKDeprecationWarning
from inference_sdk.http.errors import FeatureDeprecatedError


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


def test_detect_gazes_emits_deprecation_warning() -> None:
    # given
    client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")

    # when
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", InferenceSDKDeprecationWarning)
        with pytest.raises(FeatureDeprecatedError):
            client.detect_gazes(inference_input="/some/image.jpg")

    # then
    deprecation_warnings = [
        w for w in recorded if issubclass(w.category, InferenceSDKDeprecationWarning)
    ]
    assert any(
        "detect_gazes" in str(w.message) for w in deprecation_warnings
    ), "An InferenceSDKDeprecationWarning mentioning detect_gazes must be emitted."


@pytest.mark.asyncio
async def test_detect_gazes_async_raises_feature_deprecated_error_without_network_call() -> None:
    # given
    client = InferenceHTTPClient(api_key="my-api-key", api_url="http://some.com")

    # when
    with pytest.raises(FeatureDeprecatedError) as captured:
        await client.detect_gazes_async(inference_input="/some/image.jpg")

    # then
    assert captured.value.feature == "InferenceHTTPClient.detect_gazes_async"


def test_sdk_feature_deprecated_error_is_distinct_from_inference_core_class() -> None:
    """Catching one must NOT catch the other — they're intentionally separate."""
    from inference.core.exceptions import (
        FeatureDeprecatedError as CoreFeatureDeprecatedError,
    )

    assert FeatureDeprecatedError is not CoreFeatureDeprecatedError
    assert not issubclass(FeatureDeprecatedError, CoreFeatureDeprecatedError)
    assert not issubclass(CoreFeatureDeprecatedError, FeatureDeprecatedError)
