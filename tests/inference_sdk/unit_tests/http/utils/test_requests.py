import pytest
from requests import HTTPError, Response

from inference_sdk.http.utils.requests import (
    API_KEY_PATTERN,
    api_key_safe_raise_for_status,
    deduct_api_key,
    inject_images_into_payload,
    inject_nested_batches_of_images_into_payload,
)


def test_deduce_api_key_when_no_api_key_available() -> None:
    # given
    url = "https://some.com/endpoint?param_1=12098xx87s&param_2=2982s8x"

    # when
    result = API_KEY_PATTERN.sub(deduct_api_key, url)

    # then
    assert result == url


def test_deduce_api_key_when_short_api_key_available() -> None:
    # given
    url = "https://some.com/endpoint?api_key=my_key&param_2=some_value"

    # when
    result = API_KEY_PATTERN.sub(deduct_api_key, url)

    # then
    assert result == "https://some.com/endpoint?api_key=***&param_2=some_value"


def test_deduce_api_key_when_single_long_api_key_available() -> None:
    # given
    url = "https://some.com/endpoint?api_key=19xjs9-XSXSAos0s&param_2=some_value"

    # when
    result = API_KEY_PATTERN.sub(deduct_api_key, url)

    # then
    assert result == "https://some.com/endpoint?api_key=19***0s&param_2=some_value"


def test_deduce_api_key_when_multiple_api_key_available() -> None:
    # given
    url = "https://some.com/endpoint?api_key=19xjs9-XSXSAos0s&param_2=some_value&api_key=my_key"

    # when
    result = API_KEY_PATTERN.sub(deduct_api_key, url)

    # then
    assert (
        result
        == "https://some.com/endpoint?api_key=19***0s&param_2=some_value&api_key=***"
    )


@pytest.mark.parametrize("status_code", [200, 201, 300])
def test_api_key_safe_raise_for_status_when_no_error_occurs(status_code: int) -> None:
    # given
    response = Response()
    response.status_code = status_code

    # when
    api_key_safe_raise_for_status(response=response)

    # then no error happens


@pytest.mark.parametrize(
    "status_code", [400, 401, 402, 403, 404, 500, 501, 502, 503, 504]
)
def test_api_keysafe_raise_for_status_when_error_occurs(status_code: int) -> None:
    # given
    response = Response()
    response.status_code = status_code
    response.url = (
        "https://some.com/endpoint?api_key=19xjs9-XSXSAos0s&param_2=some_value"
    )

    # when
    with pytest.raises(HTTPError) as expected_error:
        api_key_safe_raise_for_status(response=response)

    # then
    assert "https://some.com/endpoint?api_key=19***0s&param_2=some_value" in str(
        expected_error.value
    )


def test_inject_images_into_payload_when_empty_list_of_images_is_given() -> None:
    # when
    result = inject_images_into_payload(
        payload={"my": "payload"},
        encoded_images=[],
    )

    # then
    assert result == {
        "my": "payload"
    }, "Payload is expected not to be modified when no image given to be injected"


def test_inject_images_into_payload_when_non_empty_list_of_images_is_given() -> None:
    # when
    result = inject_images_into_payload(
        payload={"my": "payload"},
        encoded_images=[("image_payload_1", 0.3), ("image_payload_2", 0.5)],
    )

    # then
    assert result == {
        "my": "payload",
        "image": [
            {"type": "base64", "value": "image_payload_1"},
            {"type": "base64", "value": "image_payload_2"},
        ],
    }, "Payload is expected to be extended with the content of all encoded images given"


def test_inject_images_into_payload_when_single_image_is_given() -> None:
    # when
    result = inject_images_into_payload(
        payload={"my": "payload"},
        encoded_images=[("image_payload_1", 0.3)],
    )

    # then
    assert result == {
        "my": "payload",
        "image": {"type": "base64", "value": "image_payload_1"},
    }, "Payload is expected to be extended with the content of only single image"


def test_inject_images_into_payload_when_payload_key_is_specified() -> None:
    # when
    result = inject_images_into_payload(
        payload={"my": "payload"},
        encoded_images=[("image_payload_1", 0.3)],
        key="prompt",
    )

    # then
    assert result == {
        "my": "payload",
        "prompt": {"type": "base64", "value": "image_payload_1"},
    }, "Payload is expected to be extended with the content of only single image under `prompt` key"


def test_inject_nested_batches_of_images_into_payload_when_single_image_given() -> None:
    # when
    result = inject_nested_batches_of_images_into_payload(
        payload={},
        encoded_images=("img1", None),
    )

    # then
    assert result == {"image": {"type": "base64", "value": "img1"}}


def test_inject_nested_batches_of_images_into_payload_when_1d_batch_of_images_given() -> None:
    # when
    result = inject_nested_batches_of_images_into_payload(
        payload={},
        encoded_images=[("img1", None), ("img2", None)],
    )

    # then
    assert result == {
        "image": [
            {"type": "base64", "value": "img1"},
            {"type": "base64", "value": "img2"},
        ]
    }


def test_inject_nested_batches_of_images_into_payload_when_nested_batch_of_images_given() -> None:
    # when
    result = inject_nested_batches_of_images_into_payload(
        payload={},
        encoded_images=[[("img1", None)], [("img2", None), ("img3", None)]],
    )

    # then
    assert result == {
        "image": [
            [{"type": "base64", "value": "img1"}],
            [
                {"type": "base64", "value": "img2"},
                {"type": "base64", "value": "img3"},
            ],
        ]
    }
