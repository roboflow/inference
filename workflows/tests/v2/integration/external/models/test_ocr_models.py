"""The OCR blocks on real local ``inference_models`` packages (opt-in).

Each test runs only when its package directories are named by environment
variables and the backend's library is installed; nothing is downloaded::

    OCR_DOCTR_PACKAGE          DocTR package directory          (needs doctr)
    OCR_EASY_OCR_PACKAGE       EasyOCR English package          (needs easyocr)
    OCR_PP_OCR_DET_PACKAGE     PP-OCRv6 detection package       (needs onnxruntime)
    OCR_PP_OCR_REC_PACKAGE     PP-OCRv6 recognition package

The image is black text "OCR 2026" rendered on white with OpenCV.
"""

import importlib.util
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.ocr import (
    DOCTR_MODEL_ID,
    EasyOCRBlock,
    OCRModelBlock,
    PPOCRBlock,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND

TEXT = "OCR 2026"


def _package(variable: str) -> Optional[Path]:
    value = os.environ.get(variable)
    package = Path(value) if value else None

    return package if package is not None and package.is_dir() else None


def _skip_unless(*variables: str, library: str) -> Any:
    missing = [variable for variable in variables if _package(variable) is None]
    if importlib.util.find_spec(library) is None:
        missing.append(f"library {library}")
    marker = pytest.mark.skipif(bool(missing), reason=f"absent: {missing}")

    return marker


def _load(variable: str) -> Any:
    from inference_models import AutoModel

    model = AutoModel.from_pretrained(
        str(_package(variable)),
        device=torch.device("cpu"),
        allow_untrusted_packages=True,
    )

    return model


def _text_image() -> Batch:
    canvas = np.full((96, 360, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, TEXT, (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    pixels = torch.from_numpy(canvas).permute(2, 0, 1).contiguous()
    batch = Batch([ImageData.from_tensor(pixels, image_id="text")])

    return batch


def _check(result: dict) -> None:
    OBJECT_DETECTION_PREDICTION_KIND.validate(result["predictions"])
    assert len(result["predictions"]) > 0
    assert "2026" in result["result"].replace(" ", "")


@_skip_unless("OCR_DOCTR_PACKAGE", library="doctr")
def test_doctr_reads_rendered_text() -> None:
    models = ModelCache(supplied={DOCTR_MODEL_ID: _load("OCR_DOCTR_PACKAGE")})

    (result,) = OCRModelBlock(models=models).run(image=_text_image())

    _check(result)


@_skip_unless("OCR_EASY_OCR_PACKAGE", library="easyocr")
def test_easy_ocr_reads_rendered_text() -> None:
    model = _load("OCR_EASY_OCR_PACKAGE")
    models = ModelCache(supplied={"easy_ocr/english_g2": model})

    (result,) = EasyOCRBlock(models=models).run(image=_text_image())

    _check(result)


@_skip_unless("OCR_PP_OCR_DET_PACKAGE", "OCR_PP_OCR_REC_PACKAGE", library="onnxruntime")
def test_pp_ocr_reads_rendered_text() -> None:
    models = ModelCache(
        supplied={
            "pp-ocrv6-det/small": _load("OCR_PP_OCR_DET_PACKAGE"),
            "pp-ocrv6-rec/small": _load("OCR_PP_OCR_REC_PACKAGE"),
        }
    )

    (result,) = PPOCRBlock(models=models).run(image=_text_image())

    _check(result)
