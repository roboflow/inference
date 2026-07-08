"""Hash-only (image-less) input parsing for interactive instance segmentation."""

from __future__ import annotations

import json

import pytest

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
    assert out["params"]["image_hashes"] == ["h1"]


@pytest.mark.asyncio
async def test_no_image_and_no_hashes_still_400():
    req = _Req(_FormData({"inputs": json.dumps({"point_labels": [[1]]})}))
    with pytest.raises(InputParseError) as exc_info:
        await parse_interactive_instance_segmentation_input(req, _common())
    assert exc_info.value.response.status_code == 400
