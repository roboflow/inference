"""Regression tests for Moondream2 model registration."""

import importlib
import os
import sys

import pytest

OPTIONAL_MODEL_FLAGS = [
    "CORE_MODEL_CLIP_ENABLED",
    "CORE_MODEL_DOCTR_ENABLED",
    "CORE_MODEL_EASYOCR_ENABLED",
    "CORE_MODEL_GAZE_ENABLED",
    "CORE_MODEL_GROUNDINGDINO_ENABLED",
    "CORE_MODEL_OWLV2_ENABLED",
    "CORE_MODEL_PE_ENABLED",
    "CORE_MODEL_SAM2_ENABLED",
    "CORE_MODEL_SAM3_ENABLED",
    "CORE_MODEL_SAM_ENABLED",
    "CORE_MODEL_TROCR_ENABLED",
    "CORE_MODEL_YOLO_WORLD_ENABLED",
    "DEPTH_ESTIMATION_ENABLED",
    "FLORENCE2_ENABLED",
    "GLM_OCR_ENABLED",
    "PALIGEMMA_ENABLED",
    "QWEN_2_5_ENABLED",
    "QWEN_3_5_ENABLED",
    "QWEN_3_ENABLED",
    "SAM3_3D_OBJECTS_ENABLED",
    "SMOLVLM2_ENABLED",
]


def test_moondream2_lmm_registry_entry_resolves_to_inference_models_adapter() -> None:
    env_names = [
        *OPTIONAL_MODEL_FLAGS,
        "MOONDREAM2_ENABLED",
        "USE_INFERENCE_MODELS",
    ]
    original_env = {name: os.environ.get(name) for name in env_names}
    env_module = None
    try:
        for name in OPTIONAL_MODEL_FLAGS:
            os.environ[name] = "False"
        os.environ["MOONDREAM2_ENABLED"] = "True"
        os.environ["USE_INFERENCE_MODELS"] = "True"

        env_module = importlib.import_module("inference.core.env")
        importlib.reload(env_module)
        sys.modules.pop("inference.models.utils", None)

        from inference.models.moondream2.moondream2_inference_models import (
            InferenceModelsMoondream2Adapter,
        )
        from inference.models.utils import ROBOFLOW_MODEL_TYPES

        assert (
            ROBOFLOW_MODEL_TYPES.get(("lmm", "moondream2"))
            is InferenceModelsMoondream2Adapter
        )
        assert (
            ROBOFLOW_MODEL_TYPES.get(("vlm", "moondream2"))
            is InferenceModelsMoondream2Adapter
        )
    finally:
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if env_module is not None:
            importlib.reload(env_module)
        sys.modules.pop("inference.models.utils", None)


def test_moondream2_inference_models_adapter_reports_lmm_task_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from inference.models.moondream2 import moondream2_inference_models
    from inference.models.moondream2.moondream2_inference_models import (
        InferenceModelsMoondream2Adapter,
    )

    monkeypatch.setattr(
        moondream2_inference_models.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    adapter = InferenceModelsMoondream2Adapter("moondream2/moondream2_2b_jul24")

    assert adapter.task_type == "lmm"
