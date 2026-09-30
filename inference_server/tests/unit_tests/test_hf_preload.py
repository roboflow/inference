import pytest

from inference_server.hf_preload import hf_id_to_model_id


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
