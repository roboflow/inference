"""Workflows passes `endpoint_type` as a plain string; the server coerces it.

`get_roboflow_model_data` reads an api-data cache when
`MODELS_CACHE_AUTH_ENABLED` is FALSE (`roboflow_api.py:597-601`), so the
two-call comparison below patches it to TRUE - otherwise the second call is a
cache hit and never builds a URL.
"""

from unittest import mock

from inference.core.roboflow_api import ModelEndpointType
from inference.core.workflows.prototypes.models_provider import CORE_MODEL_ENDPOINT_TYPE


def test_constant_matches_the_server_enum_value() -> None:
    assert CORE_MODEL_ENDPOINT_TYPE == ModelEndpointType.CORE_MODEL.value


def test_enum_construction_from_the_constant_is_the_core_member() -> None:
    assert ModelEndpointType(CORE_MODEL_ENDPOINT_TYPE) is ModelEndpointType.CORE_MODEL


def test_enum_construction_is_idempotent_for_real_members() -> None:
    for member in ModelEndpointType:
        assert ModelEndpointType(member) is member


def test_get_roboflow_model_data_builds_the_same_url_for_string_and_enum() -> None:
    import inference.core.roboflow_api as roboflow_api

    urls = []

    def fake_get_from_url(url: str, json_response: bool = True):
        urls.append(url)
        return {"ok": True}

    with mock.patch.object(
        roboflow_api, "_get_from_url", fake_get_from_url
    ), mock.patch.object(roboflow_api, "MODELS_CACHE_AUTH_ENABLED", True):
        roboflow_api.get_roboflow_model_data(
            api_key="k",
            model_id="clip/ViT-B-16",
            endpoint_type=CORE_MODEL_ENDPOINT_TYPE,
            device_id="d",
        )
        roboflow_api.get_roboflow_model_data(
            api_key="k",
            model_id="clip/ViT-B-16",
            endpoint_type=ModelEndpointType.CORE_MODEL,
            device_id="d",
        )

    assert len(urls) == 2, "the api-data cache must be bypassed for this comparison"
    assert urls[0] == urls[1]
    assert "/core_model/" in urls[0]


def test_access_check_treats_the_string_like_the_enum() -> None:
    import inference.core.registries.roboflow as registries

    registries._check_if_api_key_has_access_to_model.cache_clear()

    seen = []

    def fake_get_roboflow_model_data(**kwargs):
        seen.append(kwargs["endpoint_type"])
        return {}

    with mock.patch.object(
        registries, "get_roboflow_model_data", fake_get_roboflow_model_data
    ), mock.patch.object(registries, "USE_INFERENCE_MODELS", False):
        registries._check_if_api_key_has_access_to_model(
            api_key="k",
            model_id="yolo_world/l",
            endpoint_type=CORE_MODEL_ENDPOINT_TYPE,
        )

    # The `yolo_world` legacy-auth branch is selected by
    # `endpoint_type == ModelEndpointType.CORE_MODEL`, so reaching
    # get_roboflow_model_data at all proves the coercion happened.
    assert seen == [ModelEndpointType.CORE_MODEL]


def test_access_check_string_and_enum_share_one_cache_entry() -> None:
    import inference.core.registries.roboflow as registries

    registries._check_if_api_key_has_access_to_model.cache_clear()

    seen = []

    def fake_get_roboflow_model_data(**kwargs):
        seen.append(kwargs["endpoint_type"])
        return {}

    with mock.patch.object(
        registries, "get_roboflow_model_data", fake_get_roboflow_model_data
    ), mock.patch.object(registries, "USE_INFERENCE_MODELS", False):
        # First call with the string constant
        registries._check_if_api_key_has_access_to_model(
            api_key="k",
            model_id="yolo_world/m",
            endpoint_type=CORE_MODEL_ENDPOINT_TYPE,
        )
        # Second call with the enum should be a cache hit
        registries._check_if_api_key_has_access_to_model(
            api_key="k",
            model_id="yolo_world/m",
            endpoint_type=ModelEndpointType.CORE_MODEL,
        )

    # Both calls should use the same cache entry, so get_roboflow_model_data
    # is only called once (the second is a cache hit).
    assert seen == [ModelEndpointType.CORE_MODEL]
