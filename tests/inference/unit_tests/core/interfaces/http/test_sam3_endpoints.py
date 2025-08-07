import os
import pytest


@pytest.mark.asyncio
async def test_sam3_routes_present(monkeypatch):
    if os.getenv("CORE_MODEL_SAM3_ENABLED", "True").lower() in ("0", "false"): 
        pytest.skip("SAM3 disabled")

    from inference.core.interfaces.http.http_api import HttpInterface
    from inference.core.managers.base import ModelManager
    from inference.core.registries.roboflow import RoboflowModelRegistry
    from inference.models.utils import ROBOFLOW_MODEL_TYPES

    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    manager = ModelManager(model_registry)
    http = HttpInterface(manager)
    app = http.get_app()

    routes = {r.path for r in app.routes}
    # Ensure endpoints are registered
    assert "/sam3/embed_image" in routes
    assert "/sam3/segment_image" in routes


