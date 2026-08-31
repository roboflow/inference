"""Hash-only (image-less) input parsing for interactive instance segmentation."""

from __future__ import annotations

import json

import pytest

from inference_model_manager.hash_namespacing import (
    namespace_client_hash_id,
    tenant_namespace,
)
from inference_server.framework.entities import CommonRequestParams, InputParseError
from inference_server.handlers.interactive_instance_segmentation.input_parser import (
    parse_interactive_instance_segmentation_input,
)


class _FormData(dict):
    def multi_items(self):
        for k, v in self.items():
            yield k, v


class _QueryParams(dict):
    def getlist(self, key):
        return []


class _FilePart:
    def __init__(self, data):
        self._data = data

    async def read(self):
        return self._data


class _Req:
    def __init__(self, form_data):
        self.headers = {"content-type": "multipart/form-data; boundary=x"}
        self._form_data = form_data
        self.query_params = _QueryParams()

    async def form(self):
        return self._form_data


def _common() -> CommonRequestParams:
    return CommonRequestParams(model_id="sam3/sam3_final", api_key="k")


@pytest.mark.asyncio
async def test_hash_only_multipart_is_accepted():
    req = _Req(_FormData({"inputs": json.dumps({"image_hashes": ["h1"]})}))
    out = await parse_interactive_instance_segmentation_input(req, _common())
    assert out["images"] == []
    assert out["params"]["image_hashes"] == [namespace_client_hash_id("h1", "k")]


@pytest.mark.asyncio
async def test_client_hashes_are_namespaced_per_tenant():
    req_a = _Req(_FormData({"inputs": json.dumps({"image_hashes": ["h1"]})}))
    req_b = _Req(_FormData({"inputs": json.dumps({"image_hashes": ["h1"]})}))

    out_a = await parse_interactive_instance_segmentation_input(
        req_a, CommonRequestParams(model_id="sam3/sam3_final", api_key="key-a")
    )
    out_b = await parse_interactive_instance_segmentation_input(
        req_b, CommonRequestParams(model_id="sam3/sam3_final", api_key="key-b")
    )

    assert out_a["params"]["image_hashes"] != out_b["params"]["image_hashes"]
    assert out_a["params"]["image_hashes"] == [f"{tenant_namespace('key-a')}:h1"]
    assert out_b["params"]["image_hashes"] == [f"{tenant_namespace('key-b')}:h1"]


@pytest.mark.asyncio
async def test_string_shaped_client_hash_is_namespaced():
    req = _Req(_FormData({"inputs": json.dumps({"image_hashes": "h1"})}))
    out = await parse_interactive_instance_segmentation_input(req, _common())
    assert out["params"]["image_hashes"] == namespace_client_hash_id("h1", "k")


@pytest.mark.asyncio
async def test_empty_string_client_hash_with_image_is_namespaced():
    req = _Req(
        _FormData(
            {
                "image": _FilePart(b"\x93NUMPY" + b"\x00" * 16),
                "inputs": json.dumps({"image_hashes": ""}),
            }
        )
    )
    out = await parse_interactive_instance_segmentation_input(req, _common())
    assert len(out["images"]) == 1
    assert out["params"]["image_hashes"] == namespace_client_hash_id("", "k")


@pytest.mark.asyncio
async def test_no_image_and_no_hashes_still_400():
    req = _Req(_FormData({"inputs": json.dumps({"point_labels": [[1]]})}))
    with pytest.raises(InputParseError) as exc_info:
        await parse_interactive_instance_segmentation_input(req, _common())
    assert exc_info.value.response.status_code == 400


from unittest.mock import AsyncMock, MagicMock

from inference_server.framework.entities import ServerHooks
from inference_server.handlers.interactive_instance_segmentation.handler import (
    handle_interactive_instance_segmentation,
)


@pytest.mark.asyncio
async def test_empty_images_issues_single_empty_payload_infer():
    proxy = MagicMock()
    proxy.infer = AsyncMock(return_value={"ok": True})
    common = _common()
    result = await handle_interactive_instance_segmentation(
        "segment_with_visual_prompts",
        {"images": [], "params": {"image_hashes": ["h1"]}},
        proxy,
        ServerHooks(request=None, common=common),
    )
    assert result == {"ok": True}
    proxy.infer.assert_awaited_once()
    kwargs = proxy.infer.await_args.kwargs
    assert kwargs["image"] == b""
    assert kwargs["task"] == "segment_with_visual_prompts"
    assert kwargs["params"]["image_hashes"] == ["h1"]


from dataclasses import dataclass
from typing import Tuple

import torch

from inference_server.handlers.interactive_instance_segmentation.output_serializer import (
    serialize_sam_embeddings,
)


@dataclass(frozen=True)
class _FakeSAMEmbeddings:
    image_hash: str
    image_size_hw: Tuple[int, int]
    embeddings: torch.Tensor


def test_sam_embed_dataclass_labeled_sam_embeddings_v1():
    emb = _FakeSAMEmbeddings(
        image_hash="abc123",
        image_size_hw=(480, 640),
        embeddings=torch.zeros(1, 4),
    )
    resp = serialize_sam_embeddings(emb, _common())
    body = json.loads(resp.body)
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]
    assert p["type"] == "roboflow-sam-embeddings-v1"
    assert p["data"]["image_hash"] == "abc123"
    assert p["data"]["image_size_hw"] == [480, 640]
    assert p["data"]["embeddings"] == [[0.0, 0.0, 0.0, 0.0]]


def test_plain_tensor_embeddings_keep_compact_label():
    resp = serialize_sam_embeddings(torch.zeros(1, 4), _common())
    body = json.loads(resp.body)
    p = body["predictions"][0]
    assert p["type"] == "roboflow-embeddings-compact-v1"
    assert p["embeddings"] == [[0.0, 0.0, 0.0, 0.0]]


def test_sam_embed_response_strips_own_tenant_namespace():
    emb = _FakeSAMEmbeddings(
        image_hash=namespace_client_hash_id("h1", "k"),
        image_size_hw=(480, 640),
        embeddings=torch.zeros(1, 4),
    )
    resp = serialize_sam_embeddings(emb, _common())
    body = json.loads(resp.body)
    assert body["predictions"][0]["data"]["image_hash"] == "h1"
