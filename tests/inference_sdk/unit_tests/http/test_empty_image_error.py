from unittest.mock import AsyncMock, Mock

import pytest

from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.http.errors import EncodingError
from inference_sdk.http.utils import loaders


@pytest.mark.parametrize("payload", [b"", b"not an image"])
def test_infer_rejects_undecodable_download(payload, monkeypatch) -> None:
    fetch = Mock(return_value=payload)
    monkeypatch.setattr(loaders, "fetch_url_bytes", fetch)
    client = InferenceHTTPClient(
        api_key="test-key", api_url="https://detect.roboflow.com"
    )
    client.configure(InferenceConfiguration(client_downsizing_disabled=False))

    with pytest.raises(EncodingError):
        client.infer("https://example.com/image.png", model_id="example/1")

    fetch.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"", b"not an image"])
async def test_infer_async_rejects_undecodable_download(payload, monkeypatch) -> None:
    fetch = AsyncMock(return_value=payload)
    monkeypatch.setattr(loaders, "fetch_url_bytes_async", fetch)
    client = InferenceHTTPClient(
        api_key="test-key", api_url="https://detect.roboflow.com"
    )
    client.configure(InferenceConfiguration(client_downsizing_disabled=False))

    with pytest.raises(EncodingError):
        await client.infer_async("https://example.com/image.png", model_id="example/1")

    fetch.assert_awaited_once()
