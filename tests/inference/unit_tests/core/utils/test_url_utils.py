from unittest import mock
from urllib.parse import parse_qs, urlparse

from inference.core.utils import url_utils
from inference.core.utils.url_utils import wrap_url


@mock.patch.object(url_utils, "LICENSE_SERVER", "licence-server.com")
def test_wrap_url_when_license_server_is_provided() -> None:
    # given
    original_url = "https://detection.roboflow.com/eye-detection/1?api_key=X"

    # when
    result = wrap_url(url=original_url)

    # then
    assert (
        result
        == "http://licence-server.com/proxy?url=https%3A%2F%2Fdetection.roboflow.com%2Feye-detection%2F1%3Fapi_key%3DX"
    )
    assert parse_qs(urlparse(result).query)["url"][0] == original_url


@mock.patch.object(url_utils, "LICENSE_SERVER", None)
def test_wrap_url_when_license_server_is_not_provided() -> None:
    # given
    original_url = "https://detection.roboflow.com/eye-detection/1?api_key=X"

    # when
    result = wrap_url(url=original_url)

    # then
    assert result == original_url
