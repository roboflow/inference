import pytest

from inference.core.entities.requests.inference import DepthEstimationRequest


@pytest.mark.parametrize(
    "fields, expected_model_id",
    [
        ({}, None),
        ({"depth_version_id": "large"}, "depth-anything-v2/large"),
        ({"model_id": None}, None),
        (
            {"model_id": None, "depth_version_id": "base"},
            "depth-anything-v2/base",
        ),
        (
            {"model_id": "depth-anything-v3/small", "depth_version_id": "large"},
            "depth-anything-v3/small",
        ),
        ({"depth_version_id": None}, None),
        (
            {"model_id": "depth-anything-v3/small", "depth_version_id": None},
            "depth-anything-v3/small",
        ),
    ],
)
def test_depth_request_resolves_model_id(fields, expected_model_id) -> None:
    request = DepthEstimationRequest(
        image={"type": "url", "value": "https://example.com/image.jpg"},
        **fields,
    )

    assert request.model_id == expected_model_id
    assert request.model_dump()["model_id"] == expected_model_id


@pytest.mark.parametrize(
    "fields",
    [{"depth_version_id": "base"}, {"model_id": None, "depth_version_id": "base"}],
)
def test_depth_request_derived_model_id_is_not_an_explicit_body_id(fields) -> None:
    request = DepthEstimationRequest(
        image={"type": "url", "value": "https://example.com/image.jpg"},
        **fields,
    )

    assert request.model_id == "depth-anything-v2/base"
    assert "model_id" not in request.model_fields_set
