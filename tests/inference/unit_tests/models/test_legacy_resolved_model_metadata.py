import importlib
import sys
from pathlib import Path
from threading import Lock
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

import inference.models
from inference.core.entities.requests.clip import ClipTextEmbeddingRequest
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.sam import (
    SamEmbeddingRequest,
    SamSegmentationRequest,
)
from inference.core.entities.responses.ocr import OCRInferenceResponse

IMAGE = InferenceRequestImage(type="base64", value="unused")
MODEL_ID = "foundation/model"
METADATA = {
    "model_id": MODEL_ID,
    "model_package_id": None,
    "backend": None,
    "quantization": None,
}


@pytest.fixture(
    params=[False, True], ids=["legacy-default", "inference-models-default"]
)
def load_legacy_model(monkeypatch, request):
    base_module = importlib.import_module("inference.core.models.base")
    monkeypatch.setattr(base_module, "USE_INFERENCE_MODELS", request.param)
    imported_modules = []

    def load(module_name, class_name):
        package_name = module_name.split(".")[0]
        package = ModuleType(f"inference.models.{package_name}")
        package.__path__ = [str(Path(inference.models.__file__).parent / package_name)]
        monkeypatch.setitem(sys.modules, package.__name__, package)
        module_name = f"inference.models.{module_name}"
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        module = importlib.import_module(module_name)
        imported_modules.append(module_name)
        model_class = getattr(module, class_name)
        model = model_class.__new__(model_class)
        model.endpoint = MODEL_ID
        return model

    yield load
    for module_name in imported_modules:
        sys.modules.pop(module_name, None)


def test_legacy_clip_request_metadata(load_legacy_model, monkeypatch):
    monkeypatch.setitem(sys.modules, "clip", ModuleType("clip"))
    model = load_legacy_model("clip.clip_model", "Clip")
    model.embed_text = Mock(return_value=np.array([[0.1, 0.2]]))

    response = model.infer_from_request(
        ClipTextEmbeddingRequest(id="test-request", model_id=None, text="cat")
    )

    assert response.embeddings == [[0.1, 0.2]]
    assert response.model_dump()["resolved_model"] == METADATA
    assert response.time >= 0


@pytest.mark.parametrize(
    "module_name,class_name,request_class,batched",
    [
        ("doctr.doctr_model", "DocTR", DoctrOCRInferenceRequest, False),
        ("easy_ocr.easy_ocr", "EasyOCR", EasyOCRInferenceRequest, False),
        ("easy_ocr.easy_ocr", "EasyOCR", EasyOCRInferenceRequest, True),
    ],
)
def test_legacy_ocr_request_metadata(
    load_legacy_model, monkeypatch, module_name, class_name, request_class, batched
):
    for module_name_to_stub, attributes in [
        ("doctr.io", ["DocumentFile"]),
        ("doctr.models", ["crnn_vgg16_bn", "db_resnet50", "ocr_predictor"]),
        ("easyocr", []),
    ]:
        module = ModuleType(module_name_to_stub)
        for attribute in attributes:
            setattr(module, attribute, Mock())
        monkeypatch.setitem(sys.modules, module_name_to_stub, module)
    model = load_legacy_model(module_name, class_name)
    model.single_request = Mock(
        side_effect=lambda request: OCRInferenceResponse(result="text", time=0.1)
    )

    response = model.infer_from_request(
        request_class(image=[IMAGE, IMAGE] if batched else IMAGE)
    )

    responses = response if isinstance(response, list) else [response]
    assert len(responses) == (2 if batched else 1)
    for result in responses:
        assert result.result == "text"
        assert result.model_dump()["resolved_model"] == METADATA


@pytest.mark.parametrize("response_format", ["json", "binary"])
def test_legacy_sam_embedding_metadata(load_legacy_model, monkeypatch, response_format):
    for module_name in ["rasterio", "rasterio.features", "segment_anything"]:
        module = ModuleType(module_name)
        if module_name == "segment_anything":
            setattr(module, "SamPredictor", Mock())
            setattr(module, "sam_model_registry", {})
        monkeypatch.setitem(sys.modules, module_name, module)
    model = load_legacy_model("sam.segment_anything", "SegmentAnything")
    model._state_lock = Lock()
    model.embed_image = Mock(return_value=(np.array([[0.1, 0.2]]), None))

    response = model.infer_from_request(
        SamEmbeddingRequest(
            id="test-request", model_id=None, image=IMAGE, format=response_format
        )
    )

    assert response.resolved_model.model_dump() == METADATA
    if response_format == "binary":
        assert isinstance(response.embeddings, bytes)
    else:
        assert response.embeddings == [[0.1, 0.2]]


def test_legacy_sam_binary_segmentation_preserves_payload(
    load_legacy_model, monkeypatch
):
    for module_name in ["rasterio", "rasterio.features", "segment_anything"]:
        module = ModuleType(module_name)
        if module_name == "segment_anything":
            setattr(module, "SamPredictor", Mock())
            setattr(module, "sam_model_registry", {})
        monkeypatch.setitem(sys.modules, module_name, module)
    model = load_legacy_model("sam.segment_anything", "SegmentAnything")
    model._state_lock = Lock()
    model.segment_image = Mock(return_value=(np.zeros((1, 2, 2)), np.zeros((1, 2, 2))))

    response = model.infer_from_request(
        SamSegmentationRequest(
            id="test-request",
            model_id=None,
            embeddings=None,
            image=IMAGE,
            format="binary",
        )
    )

    assert isinstance(response, bytes)
