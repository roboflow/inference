from inference.core.entities.requests.owlv2 import OwlV2InferenceRequest
from inference.core.env import OWLV2_VERSION_ID


def test_owlv2_request_derives_model_id_from_version_id() -> None:
    request = OwlV2InferenceRequest(
        image={"type": "url", "value": "https://some/image.jpg"},
        training_data=[],
        owlv2_version_id="owlv2-base-patch16-ensemble",
    )

    assert request.model_id == "owlv2/owlv2-base-patch16-ensemble"


def test_owlv2_request_derives_model_id_from_default_version_id() -> None:
    request = OwlV2InferenceRequest(
        image={"type": "url", "value": "https://some/image.jpg"},
        training_data=[],
    )

    assert request.model_id == f"owlv2/{OWLV2_VERSION_ID}"


def test_owlv2_request_keeps_explicit_model_id() -> None:
    request = OwlV2InferenceRequest(
        image={"type": "url", "value": "https://some/image.jpg"},
        training_data=[],
        model_id="owlv2/owlv2-large-patch14-ensemble",
        owlv2_version_id="owlv2-base-patch16-ensemble",
    )

    assert request.model_id == "owlv2/owlv2-large-patch14-ensemble"
