import pytest

from inference_sdk import InferenceConfiguration, InferenceHTTPClient


@pytest.mark.parametrize("transport", ["legacy", "both", "header"])
@pytest.mark.parametrize("is_preview", [False, True])
def test_workflow_preview_is_sent_in_body_only(requests_mock, transport, is_preview):
    client = InferenceHTTPClient(api_url="http://localhost:9001", api_key="test")
    client.configure(InferenceConfiguration(api_key_transport=transport))
    requests_mock.post("http://localhost:9001/workflows/run", json={"outputs": []})

    def metadata_headers():
        return {
            key: value
            for key, value in requests_mock.last_request.headers.items()
            if key.lower() != "content-length"
        }

    client.run_workflow(specification={})
    normal_headers = metadata_headers()

    client.run_workflow(specification={}, is_preview=is_preview)
    payload = requests_mock.last_request.json()
    if is_preview:
        assert payload["is_preview"] is True
    else:
        assert "is_preview" not in payload
    assert metadata_headers() == normal_headers

    client.run_workflow(specification={})
    assert "is_preview" not in requests_mock.last_request.json()
    assert metadata_headers() == normal_headers
