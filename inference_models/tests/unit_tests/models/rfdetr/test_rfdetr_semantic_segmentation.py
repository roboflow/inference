import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.rfdetr.rfdetr_base_pytorch import (
    RFDETRSemNanoConfig,
    build_model,
)
from inference_models.models.rfdetr.rfdetr_semantic_segmentation_pytorch import (
    RFDetrForSemanticSegmentationTorch,
)
from inference_models.models.rfdetr.semantic_segmentation_common import (
    post_process_semantic_segmentation_results,
)

CPU = torch.device("cpu")
RESOLUTION = 96
CLASS_NAMES = ["background", "road", "sky"]


@pytest.fixture(scope="module")
def network() -> torch.nn.Module:
    torch.manual_seed(0)
    config = RFDETRSemNanoConfig(
        device=CPU, resolution=RESOLUTION, num_classes=len(CLASS_NAMES)
    )
    model = build_model(config=config)
    return model.eval()


@pytest.fixture(scope="module")
def model_package(network: torch.nn.Module, tmp_path_factory) -> Path:
    package_dir = tmp_path_factory.mktemp("rfdetr-sem-nano")
    (package_dir / "class_names.txt").write_text("\n".join(CLASS_NAMES))
    (package_dir / "model_type.json").write_text(
        json.dumps({"model_type": "rfdetr-sem-nano"})
    )
    inference_config = {
        "network_input": {
            "training_input_size": {"height": RESOLUTION, "width": RESOLUTION},
            "dynamic_spatial_size_supported": False,
            "color_mode": "rgb",
            "resize_mode": "stretch",
            "input_channels": 3,
            "scaling_factor": 255,
            "normalization": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        },
        "class_names_operations": [],
    }
    (package_dir / "inference_config.json").write_text(json.dumps(inference_config))
    torch.save({"model": network.state_dict()}, package_dir / "weights.pth")
    return package_dir


def test_torch_package_predicts_a_class_map_at_the_original_image_size(
    model_package: Path,
) -> None:
    # given
    model = RFDetrForSemanticSegmentationTorch.from_pretrained(
        str(model_package), device=CPU
    )
    image = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)

    # when
    (result,) = model(image)

    # then
    assert model.class_names == CLASS_NAMES
    assert tuple(result.segmentation_map.shape) == (60, 80)
    assert set(result.segmentation_map.unique().tolist()) <= {0, 1, 2}


def test_onnx_package_matches_the_torch_network(
    network: torch.nn.Module, model_package: Path, tmp_path: Path
) -> None:
    # given
    pytest.importorskip("onnxruntime")
    from inference_models.models.rfdetr.rfdetr_semantic_segmentation_onnx import (
        RFDetrForSemanticSegmentationOnnx,
    )

    package_dir = tmp_path / "package"
    package_dir.mkdir()
    for name in ("class_names.txt", "inference_config.json"):
        (package_dir / name).write_text((model_package / name).read_text())
    images = torch.randn(1, 3, RESOLUTION, RESOLUTION)
    torch.onnx.export(
        _in_export_mode(network),
        (images,),
        str(package_dir / "weights.onnx"),
        input_names=["input"],
        output_names=["masks"],
        opset_version=17,
        dynamo=False,
    )
    model = RFDetrForSemanticSegmentationOnnx.from_pretrained(
        str(package_dir), device=CPU, onnx_execution_providers=["CPUExecutionProvider"]
    )

    # when
    logits = model.forward(images)

    # then
    with torch.no_grad():
        expected_logits = network(images)
    assert torch.allclose(logits, expected_logits, atol=1e-4)


def _in_export_mode(network: torch.nn.Module) -> torch.nn.Module:
    # As the training export does: modules with an export mode switch to their
    # ONNX-friendly forward.
    exported = copy.deepcopy(network)
    for module in exported.modules():
        if getattr(module, "_export", True) is False and callable(
            getattr(module, "export", None)
        ):
            module.export()

    return exported


def test_post_processing_crops_letterbox_padding_of_the_intermediate_image() -> None:
    # given: a 32x16 intermediate image, letterboxed by 8 px on each side,
    # stretched to the 16x16 network input; logits at stride 4 are 4x4.
    logits = torch.full((1, 3, 4, 4), -10.0)
    logits[0, 1, :, 1:3] = 2.0
    logits[0, 2, :, [0, 3]] = 2.0
    metadata = PreProcessingMetadata(
        pad_left=8,
        pad_top=0,
        pad_right=8,
        pad_bottom=0,
        original_size=ImageDimensions(height=16, width=16),
        size_after_pre_processing=ImageDimensions(height=16, width=16),
        inference_size=ImageDimensions(height=16, width=16),
        scale_width=1.0,
        scale_height=1.0,
        static_crop_offset=StaticCropOffset(
            offset_x=0, offset_y=0, crop_width=16, crop_height=16
        ),
        nonsquare_intermediate_size=ImageDimensions(height=16, width=32),
    )

    # when
    (result,) = post_process_semantic_segmentation_results(
        logits,
        [metadata],
        class_names=CLASS_NAMES,
        background_class_id=0,
        device=CPU,
        confidence=0.5,
        recommended_parameters=None,
    )

    # then: only the road columns between the pads remain
    assert result.segmentation_map.unique().tolist() == [1]
    assert torch.allclose(result.confidence, torch.sigmoid(torch.tensor(2.0)))
