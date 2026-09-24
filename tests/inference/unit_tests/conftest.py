import base64
import io
import os.path
import pickle
import sys
import tempfile
import types
from typing import Generator
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from _pytest.fixtures import fixture
from PIL import Image

ASSETS_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))


@fixture(scope="function")
def image_as_numpy() -> np.ndarray:
    return np.zeros((128, 128, 3), dtype=np.uint8)


@fixture(scope="function")
def image_as_jpeg_bytes() -> bytes:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    return np.array(encoded_image).tobytes()


@fixture(scope="function")
def image_as_png_bytes() -> bytes:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".png", image)
    return np.array(encoded_image).tobytes()


@fixture(scope="function")
def image_as_jpeg_base64_bytes() -> bytes:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    payload = np.array(encoded_image).tobytes()
    return base64.b64encode(payload)


@fixture(scope="function")
def image_as_jpeg_base64_string() -> str:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    payload = np.array(encoded_image).tobytes()
    return base64.b64encode(payload).decode("utf-8")


@fixture(scope="function")
def image_as_buffer() -> Generator[io.BytesIO, None, None]:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    payload = np.array(encoded_image).tobytes()
    with io.BytesIO() as buffer:
        buffer.write(payload)
        yield buffer


@fixture(scope="function")
def image_as_rgba_buffer() -> Generator[io.BytesIO, None, None]:
    image = np.zeros((128, 128, 4), dtype=np.uint8)
    image[:, :, -1] = 255
    _, encoded_image = cv2.imencode(".png", image)
    payload = np.array(encoded_image).tobytes()
    with io.BytesIO() as buffer:
        buffer.write(payload)
        yield buffer


@fixture(scope="function")
def image_as_gray_buffer() -> Generator[io.BytesIO, None, None]:
    image = np.zeros((128, 128, 1), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    payload = np.array(encoded_image).tobytes()
    with io.BytesIO() as buffer:
        buffer.write(payload)
        yield buffer


@fixture(scope="function")
def image_as_pickled_bytes() -> bytes:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    return pickle.dumps(image)


@fixture(scope="function")
def image_as_base64_encoded_pickled_bytes() -> bytes:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    bytes = pickle.dumps(image)
    return base64.b64encode(bytes).decode()


@fixture(scope="function")
def image_as_pickled_bytes_rgba() -> bytes:
    image = np.zeros((128, 128, 4), dtype=np.uint8)
    image[:, :, -1] = 255
    return pickle.dumps(image)


@fixture(scope="function")
def image_as_pickled_bytes_gray() -> bytes:
    image = np.zeros((128, 128, 1), dtype=np.uint8)
    return pickle.dumps(image)


@fixture(scope="function")
def image_as_pillow() -> Image.Image:
    return Image.new(mode="RGB", size=(128, 128), color=(0, 0, 0))


@fixture(scope="function")
def empty_local_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@fixture(scope="function")
def image_as_local_path() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "some_image.jpg")
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.imwrite(file_path, image)
        yield file_path


@fixture(scope="function")
def example_text_file() -> str:
    return os.path.join(ASSETS_DIR_PATH, "example_text_file.txt")


# Everything whose import needs `ultralytics`: the yolo_world model and the
# stream handler built on it, under its legacy and historical (facade) names.
_ULTRALYTICS_DEPENDENT_MODULES = (
    "ultralytics",
    "inference.models.yolo_world.yolo_world",
    "inference.models.yolo_world",
    "inference.core.interfaces.legacy_stream.model_handlers.yolo_world",
    "inference.core.interfaces.stream.model_handlers.yolo_world",
)


@fixture(scope="function")
def stub_ultralytics_if_missing() -> Generator[None, None, None]:
    """Let yolo_world's module-level `from ultralytics import YOLO, settings`
    succeed when the optional `ultralytics` dependency (requirements.yolo_world.txt)
    is not installed, so tests can exercise the real import chain around it
    without requiring the heavy extra.

    On teardown every stub-backed module is dropped from `sys.modules` and from
    its parent package, and the `YOLOWorld` class cached by
    `inference.models.get_model_class` is forgotten, so later imports see the
    dependency as missing again. Other modules imported meanwhile are kept.
    """
    try:
        import ultralytics  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        yield
        return

    preloaded = {
        name: sys.modules[name]
        for name in _ULTRALYTICS_DEPENDENT_MODULES
        if name in sys.modules
    }
    models = sys.modules.get("inference.models")
    registry_had_yolo_world = models is not None and (
        "YOLOWorld" in models._MODEL_REGISTRY
    )
    stub = types.ModuleType("ultralytics")
    stub.YOLO = MagicMock(name="ultralytics.YOLO")
    stub.settings = MagicMock(name="ultralytics.settings")
    sys.modules["ultralytics"] = stub
    try:
        yield
    finally:
        for name in _ULTRALYTICS_DEPENDENT_MODULES:
            if name in preloaded:
                sys.modules[name] = preloaded[name]
                continue
            module = sys.modules.pop(name, None)
            parent_name, _, child_name = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if module is not None and parent is not None:
                if vars(parent).get(child_name) is module:
                    delattr(parent, child_name)
        models = sys.modules.get("inference.models")
        if models is not None and not registry_had_yolo_world:
            models._MODEL_REGISTRY.pop("YOLOWorld", None)


# --- aiohttp 3.14 / aioresponses compatibility shim --------------------------
# aiohttp 3.14.0 made `stream_writer` a required keyword-only argument of
# ClientResponse.__init__. aioresponses (<=0.7.8) builds its mock response
# without it, raising at construction time:
#   TypeError: ClientResponse.__init__() missing 1 required keyword-only
#   argument: 'stream_writer'
# The upstream fix is unmerged (pnuckowski/aioresponses#288) and unreleased, so
# we replicate it locally: default aioresponses' response class to a subclass
# that injects Mock(output_size=0) (the only attribute aiohttp reads at init).
# Signature-guarded, so it is a no-op on aiohttp < 3.14 and can be deleted once
# a fixed aioresponses ships.
import inspect as _inspect
from unittest.mock import Mock as _Mock

import aioresponses.core as _aioresponses_core
from aiohttp.client_reqrep import ClientResponse as _ClientResponse

_AIOHTTP_NEEDS_STREAM_WRITER = (
    "stream_writer" in _inspect.signature(_ClientResponse).parameters
)


class _CompatClientResponse(_ClientResponse):
    def __init__(self, *args, **kwargs):
        if _AIOHTTP_NEEDS_STREAM_WRITER and "stream_writer" not in kwargs:
            kwargs["stream_writer"] = _Mock(output_size=0)
        super().__init__(*args, **kwargs)


@pytest.fixture(autouse=True)
def _patch_aioresponses_stream_writer(monkeypatch):
    # aioresponses._build_response defaults response_class to the module-global
    # ClientResponse; swap it for the compat subclass for the duration of each test.
    monkeypatch.setattr(_aioresponses_core, "ClientResponse", _CompatClientResponse)
