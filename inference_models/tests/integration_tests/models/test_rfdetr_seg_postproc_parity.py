import importlib
import os
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask

pytest.importorskip("triton")

if not torch.cuda.is_available():  # pragma: no cover - host-dependent
    pytest.skip(
        "CUDA required for RF-DETR Triton post-processing parity",
        allow_module_level=True,
    )


BackendSpec = Dict[str, Any]


def _decode_masks(prediction) -> torch.Tensor:
    if isinstance(prediction.mask, torch.Tensor):
        return prediction.mask.detach().to(dtype=torch.bool).cpu()
    return coco_rle_masks_to_torch_mask(
        instances_masks=prediction.mask,
        device=torch.device("cpu"),
    )


def _reload_backend_modules(
    *,
    backend_spec: BackendSpec,
    triton_enabled: bool,
):
    os.environ["RFDETR_TRITON_POSTPROC"] = "true" if triton_enabled else "false"
    configuration_module = importlib.import_module("inference_models.configuration")
    common_module = importlib.import_module("inference_models.models.rfdetr.common")
    backend_module = importlib.import_module(backend_spec["module"])
    importlib.reload(configuration_module)
    common_module = importlib.reload(common_module)
    backend_module = importlib.reload(backend_module)
    return common_module, backend_module


def _run_backend_once(
    *,
    backend_spec: BackendSpec,
    package_path: str,
    image: np.ndarray,
    confidence: float,
    mask_format: str,
) -> Tuple[Any, int]:
    previous_env = os.environ.get("RFDETR_TRITON_POSTPROC")
    common_module, backend_module = _reload_backend_modules(
        backend_spec=backend_spec,
        triton_enabled=backend_spec["triton_enabled"],
    )
    call_count = {"value": 0}
    original_postproc = getattr(common_module, "rfdetr_triton_postproc", None)
    if original_postproc is not None:
        def counting_postproc(*args, **kwargs):
            call_count["value"] += 1
            return original_postproc(*args, **kwargs)

        common_module.rfdetr_triton_postproc = counting_postproc

    try:
        model_class = getattr(backend_module, backend_spec["class_name"])
        model = model_class.from_pretrained(package_path, **backend_spec["init_kwargs"])
        prediction = model(
            image,
            confidence=confidence,
            mask_format=mask_format,
        )[0]
        return prediction, call_count["value"]
    finally:
        if original_postproc is not None:
            common_module.rfdetr_triton_postproc = original_postproc
        if previous_env is None:
            os.environ.pop("RFDETR_TRITON_POSTPROC", None)
        else:
            os.environ["RFDETR_TRITON_POSTPROC"] = previous_env
        configuration_module = importlib.import_module("inference_models.configuration")
        importlib.reload(configuration_module)
        importlib.reload(common_module)
        importlib.reload(backend_module)


def _assert_predictions_match(reference, candidate) -> None:
    torch.testing.assert_close(
        candidate.xyxy.cpu(),
        reference.xyxy.cpu(),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        candidate.confidence.cpu(),
        reference.confidence.cpu(),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        candidate.class_id.cpu(),
        reference.class_id.cpu(),
        atol=0,
        rtol=0,
    )
    assert torch.equal(_decode_masks(candidate), _decode_masks(reference))


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend_spec",
    [
        pytest.param(
            {
                "name": "torch",
                "module": (
                    "inference_models.models.rfdetr."
                    "rfdetr_instance_segmentation_pytorch"
                ),
                "class_name": "RFDetrForInstanceSegmentationTorch",
                "package_fixture": "snakes_rfdetr_seg_torch_stretch_package",
                "image_fixture": "snake_image_numpy",
                "init_kwargs": {},
                "triton_enabled": True,
            },
            id="torch",
            marks=pytest.mark.torch_models,
        ),
        pytest.param(
            {
                "name": "onnx",
                "module": (
                    "inference_models.models.rfdetr."
                    "rfdetr_instance_segmentation_onnx"
                ),
                "class_name": "RFDetrForInstanceSegmentationOnnx",
                "package_fixture": "snakes_rfdetr_seg_onnx_static_bs_stretch_package",
                "image_fixture": "snake_image_numpy",
                "init_kwargs": {
                    "onnx_execution_providers": [
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ]
                },
                "triton_enabled": True,
            },
            id="onnx",
            marks=pytest.mark.onnx_extras,
        ),
        pytest.param(
            {
                "name": "trt",
                "module": (
                    "inference_models.models.rfdetr."
                    "rfdetr_instance_segmentation_trt"
                ),
                "class_name": "RFDetrForInstanceSegmentationTRT",
                "package_fixture": "rfdetr_seg_asl_trt_package",
                "image_fixture": "asl_image_numpy",
                "init_kwargs": {"engine_host_code_allowed": True},
                "triton_enabled": True,
            },
            id="trt",
            marks=pytest.mark.trt_extras,
        ),
    ],
)
@pytest.mark.parametrize("mask_format", ["dense", "rle"])
def test_rfdetr_seg_predictions_match_when_triton_postproc_is_toggled(
    backend_spec: BackendSpec,
    mask_format: str,
    request: pytest.FixtureRequest,
) -> None:
    package_path = request.getfixturevalue(backend_spec["package_fixture"])
    image = request.getfixturevalue(backend_spec["image_fixture"])

    enabled_spec = dict(backend_spec)
    enabled_spec["triton_enabled"] = True
    disabled_spec = dict(backend_spec)
    disabled_spec["triton_enabled"] = False

    enabled_prediction, enabled_calls = _run_backend_once(
        backend_spec=enabled_spec,
        package_path=package_path,
        image=image,
        confidence=0.5,
        mask_format=mask_format,
    )
    disabled_prediction, disabled_calls = _run_backend_once(
        backend_spec=disabled_spec,
        package_path=package_path,
        image=image,
        confidence=0.5,
        mask_format=mask_format,
    )

    assert enabled_calls == 1
    assert disabled_calls == 0
    _assert_predictions_match(
        reference=disabled_prediction,
        candidate=enabled_prediction,
    )
