import json
import sys
import zipfile
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from filelock import FileLock, Timeout

from inference_models.errors import ModelPackageRestrictedError
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


@pytest.mark.parametrize("module, model_class", LOADERS)
@pytest.mark.parametrize("zipped", [False, True])
@pytest.mark.parametrize("input_type", ["imageType", "multiArrayType"])
@pytest.mark.parametrize(
    "configured_size, model_size, limit, rejected",
    [
        (640, 384, 512, False),
        (384, 512, 512, False),
        (384, 1024, 800, True),
        (1024, 1024, 800, True),
        (640, 384, None, False),
    ],
)
def test_coreml_loader_validates_actual_dimensions_before_runtime_loading(
    tmp_path,
    monkeypatch,
    module,
    model_class,
    zipped,
    input_type,
    configured_size,
    model_size,
    limit,
    rejected,
) -> None:
    """Check authoritative dimensions before any native compilation can occur.

    Args:
        tmp_path (Path): Temporary package directory.
        monkeypatch (pytest.MonkeyPatch): Replace the native Core ML constructor.
        module (ModuleType): RF-DETR loader module under test.
        model_class (type): Detection or segmentation loader.
        zipped (bool): Exercise the archive path and its cache lock when true.
        input_type (str): Image or tensor model signature.
        configured_size (int): Dimensions declared by package metadata.
        model_size (int): Dimensions declared by the model specification.
        limit (int | None): Maximum allowed dimensions.
        rejected (bool): Whether the actual model exceeds the limit.
    """
    package_dir = _package(tmp_path, configured_size=configured_size)
    if zipped:
        with zipfile.ZipFile(tmp_path / "weights.mlpackage.zip", "w") as archive:
            archive.writestr("Manifest.json", "{}")
    else:
        bundle = tmp_path / "weights.mlpackage"
        bundle.mkdir()
        (bundle / "Manifest.json").write_text("{}")

    input_spec = SimpleNamespace(
        WhichOneof=lambda _: input_type,
        imageType=SimpleNamespace(height=model_size, width=model_size),
        multiArrayType=SimpleNamespace(shape=[1, 3, model_size, model_size]),
    )
    spec = SimpleNamespace(
        description=SimpleNamespace(
            input=[SimpleNamespace(name="image_input", type=input_spec)],
            output=[
                SimpleNamespace(name=name)
                for name in ("boxes", "scores", "labels", "masks")
            ],
        )
    )

    def create_model(path, *, skip_model_load=False, **kwargs):
        if zipped:
            # Both inspection and runtime loading must retain the watchdog lock.
            with pytest.raises(Timeout):
                with FileLock(tmp_path / ".coreml_cache.lock", timeout=0):
                    pass

        if not skip_model_load and rejected:
            pytest.fail("Oversized model reached native compilation/loading")

        return SimpleNamespace(get_spec=lambda: spec)

    ml_model = Mock(side_effect=create_model)
    monkeypatch.setitem(
        sys.modules,
        "coremltools",
        SimpleNamespace(
            models=SimpleNamespace(MLModel=ml_model),
            ComputeUnit=SimpleNamespace(CPU_AND_GPU="cpu-and-gpu"),
        ),
    )
    # Keep these tests independent of compute-unit environment overrides.
    monkeypatch.setattr(
        module,
        "load_coreml_package",
        partial(module.load_coreml_package, compute_units="CPUAndGPU"),
    )

    if rejected:
        with pytest.raises(ModelPackageRestrictedError):
            model_class.from_pretrained(package_dir, rf_detr_max_input_resolution=limit)

        assert ml_model.call_count == 1
        assert ml_model.call_args.kwargs["skip_model_load"] is True
        return

    model = model_class.from_pretrained(package_dir, rf_detr_max_input_resolution=limit)

    assert ml_model.call_count == 2
    inspection_call, runtime_call = ml_model.call_args_list
    assert inspection_call.kwargs["skip_model_load"] is True
    assert not runtime_call.kwargs.get("skip_model_load", False)
    assert inspection_call.args == runtime_call.args
    size = model._inference_config.network_input.training_input_size
    assert (size.height, size.width) == (model_size, model_size)
    assert model.class_names == ["egg"]
