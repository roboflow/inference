import numpy as np
import pytest
from aioresponses import aioresponses

from inference_sdk import InferenceConfiguration, InferenceHTTPClient

PAYLOAD = {
    "embeddings": [[2.0, 3.0], [4.0, 5.0]],
    "embedding_info": {"space_id": "space", "normalization": "none"},
    "time": 0.1,
}


@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
def test_sync_splits_batched_vectors_with_metadata(requests_mock, output_type):
    request = requests_mock.post("http://localhost:9001/infer/embeddings", json=PAYLOAD)
    client = InferenceHTTPClient("http://localhost:9001", api_key="key")
    client.configure(
        InferenceConfiguration(max_batch_size=2, source="workflow-execution")
    )
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    result = client.get_image_embeddings(
        [image, image], "resnet101", output_type=output_type
    )
    assert [item["embeddings"] for item in result] == [[[2.0, 3.0]], [[4.0, 5.0]]]
    assert all(item["embedding_info"] == PAYLOAD["embedding_info"] for item in result)
    assert request.last_request.json()["model_id"] == "resnet101"
    assert request.last_request.json()["api_key"] == "key"
    assert request.last_request.json()["source"] == "workflow-execution"
    assert request.last_request.json()["output_type"] == output_type


def test_v0_auth_is_forwarded_in_query_string(requests_mock):
    request = requests_mock.post(
        "http://localhost:9001/infer/embeddings?api_key=key",
        json={**PAYLOAD, "embeddings": [[2.0, 3.0]]},
    )
    client = InferenceHTTPClient("http://localhost:9001", api_key="key").select_api_v0()
    result = client.get_image_embeddings(
        np.zeros((16, 16, 3), dtype=np.uint8), "resnet101"
    )
    assert result["embeddings"] == [[2.0, 3.0]]
    assert request.last_request.qs["api_key"] == ["key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
async def test_async_preserves_metadata_and_single_image_unwraps(output_type):
    client = InferenceHTTPClient("http://localhost:9001", api_key="key")
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    payload = {**PAYLOAD, "embeddings": [[2.0, 3.0]]}
    with aioresponses() as responses:
        responses.post("http://localhost:9001/infer/embeddings", payload=payload)
        result = await client.get_image_embeddings_async(
            image, "resnet101", output_type=output_type
        )
        sent = next(iter(responses.requests.values()))[0]
        assert sent.kwargs["json"]["output_type"] == output_type
    assert result["embeddings"] == [[2.0, 3.0]]
    assert result["embedding_info"] == PAYLOAD["embedding_info"]
