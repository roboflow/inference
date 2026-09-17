import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import inference.models
from inference.core.entities.requests.clip import ClipTextEmbeddingRequest
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.requests.groundingdino import GroundingDINOInferenceRequest
from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.perception_encoder import (
    PerceptionEncoderTextEmbeddingRequest,
)
from inference.core.entities.requests.sam import (
    SamEmbeddingRequest,
    SamSegmentationRequest,
)
from inference.core.entities.requests.sam2 import (
    Sam2EmbeddingRequest,
    Sam2SegmentationRequest,
)
from inference.core.entities.requests.sam3 import Sam3SegmentationRequest
from inference.core.entities.requests.trocr import TrOCRInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
)
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.entities.responses.sam3 import Sam3SegmentationResponse

METADATA = {
    "model_id": "foundation/model",
    "model_package_id": "package-id",
    "backend": "torch",
    "quantization": "fp32",
}
IMAGE = InferenceRequestImage(type="base64", value="unused")


@pytest.fixture
def load_adapter(monkeypatch):
    base_module = importlib.import_module("inference.core.models.base")
    monkeypatch.setattr(base_module, "USE_INFERENCE_MODELS", True)
    imported_modules = []

    def load(module_name, class_name):
        module_name = f"inference.models.{module_name}_inference_models"
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        module = importlib.import_module(module_name)
        imported_modules.append(module_name)
        adapter_class = getattr(module, class_name)
        adapter = adapter_class.__new__(adapter_class)
        adapter._model = SimpleNamespace(resolved_model=SimpleNamespace(**METADATA))
        if hasattr(module, "record_fixed_model_input_for_request"):
            monkeypatch.setattr(module, "record_fixed_model_input_for_request", Mock())
        return adapter

    yield load
    for module_name in imported_modules:
        sys.modules.pop(module_name, None)


@pytest.fixture
def segmentation_backends(monkeypatch):
    # Request handling does not need the optional GPU model implementations.
    for name in ["sam", "sam2", "sam3"]:
        package = ModuleType(f"inference.models.{name}")
        package.__path__ = [str(Path(inference.models.__file__).parent / name)]
        monkeypatch.setitem(sys.modules, package.__name__, package)
    for module_name, class_name in [
        ("inference_models.models.sam.sam_torch", "SAMTorch"),
        ("inference_models.models.sam2.sam2_torch", "SAM2Torch"),
        ("inference_models.models.sam3.sam3_torch", "SAM3Torch"),
    ]:
        module = ModuleType(module_name)
        setattr(module, class_name, type(class_name, (), {}))
        setattr(module, "compute_image_hash", Mock())
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(sys.modules, "rasterio", ModuleType("rasterio"))
    sam2 = ModuleType("sam2")
    sam2_utils = ModuleType("sam2.utils")
    sam2_misc = ModuleType("sam2.utils.misc")
    setattr(sam2, "utils", sam2_utils)
    setattr(sam2_utils, "misc", sam2_misc)
    for module in [sam2, sam2_utils, sam2_misc]:
        monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize(
    "module_name,class_name,request_class",
    [
        ("clip.clip", "InferenceModelsClipAdapter", ClipTextEmbeddingRequest),
        (
            "perception_encoder.perception_encoder",
            "InferenceModelsPerceptionEncoderAdapter",
            PerceptionEncoderTextEmbeddingRequest,
        ),
    ],
)
def test_embedding_request_metadata(
    load_adapter, module_name, class_name, request_class
):
    adapter = load_adapter(module_name, class_name)
    adapter.embed_text = Mock(return_value=np.array([[0.1, 0.2]]))

    response = adapter.infer_from_request(request_class(text="cat"))

    assert response.embeddings == [[0.1, 0.2]]
    assert response.model_dump()["resolved_model"] == METADATA


@pytest.mark.parametrize(
    "module_name,class_name,request_class",
    [
        ("doctr.doctr_model", "InferenceModelsDocTRAdapter", DoctrOCRInferenceRequest),
        ("easy_ocr.easy_ocr", "InferenceModelsEasyOCRAdapter", EasyOCRInferenceRequest),
    ],
)
@pytest.mark.parametrize("batched", [True, False])
def test_ocr_request_metadata(
    load_adapter, module_name, class_name, request_class, batched
):
    adapter = load_adapter(module_name, class_name)
    adapter.single_request = Mock(
        side_effect=lambda request: OCRInferenceResponse(result="text", time=0.1)
    )

    response = adapter.infer_from_request(
        request_class(image=[IMAGE, IMAGE] if batched else IMAGE)
    )

    responses = response if batched else [response]
    assert len(responses) == (2 if batched else 1)
    assert all(item.model_dump()["resolved_model"] == METADATA for item in responses)


def test_trocr_request_metadata(load_adapter):
    adapter = load_adapter("trocr.trocr", "InferenceModelsTrOCRAdapter")
    adapter.infer = Mock(return_value="text")

    response = adapter.infer_from_request(
        TrOCRInferenceRequest.model_validate({"image": IMAGE})
    )

    assert response.result == "text"
    assert response.model_dump()["resolved_model"] == METADATA


def test_grounding_dino_request_metadata(load_adapter):
    adapter = load_adapter(
        "grounding_dino.grounding_dino", "InferenceModelsGroundingDINOAdapter"
    )
    adapter.infer = Mock(
        return_value=[
            ObjectDetectionInferenceResponse(
                predictions=[], image=InferenceResponseImage(width=8, height=8)
            )
        ]
    )

    response = adapter.infer_from_request(
        GroundingDINOInferenceRequest.model_validate({"image": IMAGE, "text": ["cat"]})
    )

    assert response[0].model_dump()["resolved_model"] == METADATA


def test_ocr_request_without_backend_metadata(load_adapter):
    adapter = load_adapter("trocr.trocr", "InferenceModelsTrOCRAdapter")
    adapter._model = SimpleNamespace()
    adapter.infer = Mock(return_value="text")

    response = adapter.infer_from_request(
        TrOCRInferenceRequest.model_validate({"image": IMAGE})
    )

    assert "resolved_model" not in response.model_dump(exclude_none=True)


@pytest.mark.usefixtures("segmentation_backends")
@pytest.mark.parametrize(
    "module_name,class_name,request_class,embedding_result",
    [
        (
            "sam.segment_anything",
            "InferenceModelsSAMAdapter",
            SamEmbeddingRequest,
            (np.zeros((1, 1, 1, 1)), None),
        ),
        (
            "sam2.segment_anything2",
            "InferenceModelsSAM2Adapter",
            Sam2EmbeddingRequest,
            (None, None, "image-id"),
        ),
        (
            "sam3.visual_segmentation",
            "InferenceModelsSAM3InteractiveAdapter",
            Sam2EmbeddingRequest,
            (None, None, "image-id"),
        ),
    ],
)
def test_sam_embedding_request_metadata(
    load_adapter, module_name, class_name, request_class, embedding_result
):
    adapter = load_adapter(module_name, class_name)
    adapter.embed_image = Mock(return_value=embedding_result)

    response = adapter.infer_from_request(request_class(image=IMAGE))

    assert response.model_dump()["resolved_model"] == METADATA


@pytest.mark.usefixtures("segmentation_backends")
@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("sam2.segment_anything2", "InferenceModelsSAM2Adapter"),
        ("sam3.visual_segmentation", "InferenceModelsSAM3InteractiveAdapter"),
    ],
)
@pytest.mark.parametrize("format", ["json", "rle", "binary"])
def test_sam_segmentation_response_formats(
    load_adapter, module_name, class_name, format
):
    adapter = load_adapter(module_name, class_name)
    masks = np.zeros((1, 8, 8), dtype=np.float32)
    adapter.segment_image = Mock(return_value=(masks, np.array([0.9]), masks))
    request = Sam2SegmentationRequest.model_construct(image=IMAGE, format=format)

    response = adapter.infer_from_request(request)

    if format == "binary":
        assert isinstance(response, bytes)
    else:
        assert response.model_dump()["resolved_model"] == METADATA


@pytest.mark.usefixtures("segmentation_backends")
@pytest.mark.parametrize("format", ["json", "binary"])
def test_sam_segmentation_metadata(load_adapter, format):
    adapter = load_adapter("sam.segment_anything", "InferenceModelsSAMAdapter")
    masks = np.zeros((1, 8, 8), dtype=np.float32)
    adapter.segment_image = Mock(return_value=(masks, masks))

    response = adapter.infer_from_request(
        SamSegmentationRequest.model_validate({"format": format})
    )

    if format == "binary":
        assert isinstance(response, bytes)
    else:
        assert response.model_dump()["resolved_model"] == METADATA


@pytest.mark.usefixtures("segmentation_backends")
def test_sam3_segmentation_request_metadata(load_adapter):
    adapter = load_adapter("sam3.segment_anything3", "InferenceModelsSAM3Adapter")
    adapter.segment_image = Mock(
        return_value=Sam3SegmentationResponse(prompt_results=[], time=0.1)
    )
    request = Sam3SegmentationRequest.model_validate(
        {"image": IMAGE, "prompts": [{"text": "cat"}]}
    )

    response = adapter.infer_from_request(request)

    assert response.model_dump()["resolved_model"] == METADATA
