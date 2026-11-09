import importlib
import importlib.util
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def yolo_world(monkeypatch):
    # clip and ultralytics are optional extras; stub them when they are absent.
    for name in ("clip", "ultralytics"):
        if name not in sys.modules and importlib.util.find_spec(name) is None:
            stub = ModuleType(name)
            stub.YOLO = MagicMock()
            stub.settings = MagicMock()
            monkeypatch.setitem(sys.modules, name, stub)
    return importlib.import_module("inference.models.yolo_world.yolo_world")


def test_yolo_world_passes_the_request_key_to_its_inner_clip(
    yolo_world, monkeypatch
) -> None:
    # given
    def _core_model_init(self, *args, model_id, api_key=None, **kwargs):
        self.api_key = api_key

    monkeypatch.setattr(yolo_world.RoboflowCoreModel, "__init__", _core_model_init)
    monkeypatch.setattr(yolo_world.YOLOWorld, "cache_file", lambda self, f: f)
    monkeypatch.setattr(yolo_world, "YOLO", MagicMock())
    clip_class = MagicMock()
    monkeypatch.setattr(yolo_world, "Clip", clip_class)

    # when
    model = yolo_world.YOLOWorld(model_id="yolo_world/l", api_key="request-key")

    # then
    clip_class.assert_called_once_with(model_id="clip/ViT-B-32", api_key="request-key")
    assert model.clip_model is clip_class.return_value
