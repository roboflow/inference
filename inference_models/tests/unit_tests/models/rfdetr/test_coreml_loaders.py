import json
from typing import Any, Dict, Mapping

import pytest

from inference_models.errors import ModelPackageRestrictedError
from inference_models.models.common.coreml import CoreMLModel, CoreMLModelSignature
from inference_models.models.rfdetr import (
    rfdetr_instance_segmentation_coreml,
    rfdetr_object_detection_coreml,
)

LOADERS = [
    (
        rfdetr_object_detection_coreml,
        rfdetr_object_detection_coreml.RFDetrForObjectDetectionCoreML,
    ),
    (
        rfdetr_instance_segmentation_coreml,
        rfdetr_instance_segmentation_coreml.RFDetrForInstanceSegmentationCoreML,
    ),
]


class _NoPredictions:
    def predict(self, feed: Mapping[str, Any]) -> Dict[str, Any]:
        raise AssertionError("not expected to run")


def _package(tmp_path, configured_size: int) -> str:
    (tmp_path / "class_names.txt").write_text("egg\nbackground_class83422\n")
    (tmp_path / "inference_config.json").write_text(
        json.dumps(
            {
                "network_input": {
                    "training_input_size": {
                        "height": configured_size,
                        "width": configured_size,
                    },
                    "dynamic_spatial_size_supported": False,
                    "color_mode": "rgb",
                    "resize_mode": "stretch",
                    "input_channels": 3,
                    "scaling_factor": 255,
                    "normalization": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
                },
                "class_names_operations": [
                    {
                        "type": "class_name_removal",
                        "class_name": "background_class83422",
                    }
                ],
            }
        )
    )
    return str(tmp_path)


def _fake_load(model_size: int):
    def load(model_package_dir: str) -> CoreMLModel:
        return CoreMLModel(
            model=_NoPredictions(),
            signature=CoreMLModelSignature(
                input_name="image_input",
                image_input=True,
                input_height=model_size,
                input_width=model_size,
                output_names=("boxes", "scores", "labels", "masks"),
            ),
        )

    return load


@pytest.mark.parametrize("module, model_class", LOADERS)
def test_coreml_loader_limits_the_model_input_size_not_the_config(
    tmp_path, monkeypatch, module, model_class
) -> None:
    monkeypatch.setattr(module, "load_coreml_package", _fake_load(model_size=384))

    model = model_class.from_pretrained(
        _package(tmp_path, configured_size=640), rf_detr_max_input_resolution=512
    )

    size = model._inference_config.network_input.training_input_size
    assert (size.height, size.width) == (384, 384)
    assert model.class_names == ["egg"]


@pytest.mark.parametrize("module, model_class", LOADERS)
def test_coreml_loader_rejects_a_model_input_over_the_limit(
    tmp_path, monkeypatch, module, model_class
) -> None:
    monkeypatch.setattr(module, "load_coreml_package", _fake_load(model_size=1024))

    with pytest.raises(ModelPackageRestrictedError):
        model_class.from_pretrained(
            _package(tmp_path, configured_size=384), rf_detr_max_input_resolution=800
        )
