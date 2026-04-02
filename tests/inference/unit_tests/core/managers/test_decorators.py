from unittest.mock import MagicMock

from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.base import ModelManagerDecorator
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.managers.model_load_collector import (
    RequestModelIds,
    current_request_path,
    request_model_ids,
)


def test_model_manager_decorator_records_request_metadata_for_warm_model() -> None:
    model_manager = ModelManager(model_registry=MagicMock())
    model_manager._models = {"some/1": MagicMock()}
    decorator = ModelManagerDecorator(model_manager)
    request_path_token = current_request_path.set("/sam3/concept_segment")
    ids = RequestModelIds()
    ids_token = request_model_ids.set(ids)

    try:
        decorator.add_model(
            model_id="some/1",
            api_key="key",
            model_id_alias="sam3/sam3_final",
        )
    finally:
        request_model_ids.reset(ids_token)
        current_request_path.reset(request_path_token)

    [description] = model_manager.describe_models()
    assert description.model_id == "some/1"
    assert description.request_aliases == ["sam3/sam3_final"]
    assert description.request_paths == ["/sam3/concept_segment"]
    assert ids.get_ids() == {"some/1"}


def test_fixed_size_cache_records_request_metadata_for_warm_model() -> None:
    model_manager = ModelManager(model_registry=MagicMock())
    model_manager._models = {"sam3/sam3_interactive": MagicMock()}
    decorator = WithFixedSizeCache(model_manager, max_size=8)
    token = current_request_path.set("/sam3/embed_image")

    try:
        decorator.add_model(
            model_id="sam3/sam3_final",
            api_key="key",
            model_id_alias="sam3/sam3_interactive",
        )
        decorator.add_model(
            model_id="sam3/sam3_final",
            api_key="key",
            model_id_alias="sam3/sam3_interactive",
        )
    finally:
        current_request_path.reset(token)

    [description] = model_manager.describe_models()
    assert description.model_id == "sam3/sam3_interactive"
    assert description.request_aliases == ["sam3/sam3_final"]
    assert description.request_paths == ["/sam3/embed_image"]
