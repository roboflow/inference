import logging

import pytest

from inference_server.hf_preload import hf_id_to_model_id, preload_hf_models


@pytest.mark.parametrize(
    "hf_id,model_id",
    [
        ("google/owlv2-base-patch16-ensemble", "owlv2/owlv2-base-patch16-ensemble"),
        ("owlv2-large-patch14-ensemble", "owlv2/owlv2-large-patch14-ensemble"),
        ("owlv2/owlv2-large-patch14", "owlv2/owlv2-large-patch14"),
    ],
)
def test_hf_id_maps_to_owlv2_registry_id(hf_id, model_id):
    assert hf_id_to_model_id(hf_id) == model_id


@pytest.mark.asyncio
async def test_failed_preload_log_leaves_out_the_failure_description(caplog):
    class _Gateway:
        async def load(self, model_id, api_key="", timeout_s=None, pinned=True):
            return ("error", 5, {"error_type": "RetryError", "message": "DETAIL-TEXT"})

    with caplog.at_level(logging.DEBUG, logger="inference_server.hf_preload"):
        await preload_hf_models(_Gateway(), ["google/owlv2-base"], api_key="k")

    assert [record.getMessage() for record in caplog.records] == [
        "Preload of HF model 'google/owlv2-base' failed: ('error', 5)"
    ]
    assert "DETAIL-TEXT" not in caplog.text
