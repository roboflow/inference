import json

import pytest

from inference_server.framework.entities import CommonRequestParams, InputParseError
from inference_server.framework.registry import DYNAMIC_MODELS_HANDLERS
from inference_server.handlers.embeddings.input_parser import parse_embeddings_input


def test_embedding_routes_registered():
    for action in ("embed_images", "embed_text", "compare"):
        assert ("embedding", action) in DYNAMIC_MODELS_HANDLERS


@pytest.mark.asyncio
async def test_compare_cosine():
    from inference_server.handlers.embeddings.handler import _cosine

    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


class _FormData(dict):
    def multi_items(self):
        for k, v in self.items():
            yield k, v


class _QueryParams(dict):
    def getlist(self, key):
        return []


class _Req:
    def __init__(self, form_data):
        self.headers = {"content-type": "multipart/form-data; boundary=x"}
        self._form_data = form_data
        self.query_params = _QueryParams()

    async def form(self):
        return self._form_data


class _JsonReq:
    def __init__(self, body):
        self.headers = {"content-type": "application/json"}
        self._body = body
        self.query_params = _QueryParams()

    async def json(self):
        return self._body


@pytest.mark.asyncio
async def test_compare_prompt_texts_without_subject_is_400():
    req = _Req(_FormData({"inputs": json.dumps({"prompt_texts": ["a cat"]})}))
    common = CommonRequestParams(model_id="clip/1", api_key="k", action="compare")
    with pytest.raises(InputParseError) as exc_info:
        await parse_embeddings_input(req, common)
    assert exc_info.value.response.status_code == 400
    body = json.loads(exc_info.value.response.body)
    assert body["error_code"] == "MISSING_PARAM"


@pytest.mark.asyncio
async def test_embed_text_json_body_without_image_parses():
    req = _JsonReq({"inputs": {"texts": ["a cat", "a dog"]}})
    common = CommonRequestParams(model_id="clip/1", api_key="k", action="embed_text")
    out = await parse_embeddings_input(req, common)
    assert out["images"] == []
    assert out["params"]["texts"] == ["a cat", "a dog"]


@pytest.mark.asyncio
async def test_compare_json_body_subject_and_prompt_texts_parses():
    req = _JsonReq({"inputs": {"subject_text": "a cat", "prompt_texts": ["a dog"]}})
    common = CommonRequestParams(model_id="clip/1", api_key="k", action="compare")
    out = await parse_embeddings_input(req, common)
    assert out["images"] == []
    assert out["params"]["subject_text"] == "a cat"
    assert out["params"]["prompt_texts"] == ["a dog"]
