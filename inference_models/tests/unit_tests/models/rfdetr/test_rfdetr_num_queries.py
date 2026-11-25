"""RF-DETR Torch packages trained with a non-default decoder query count.

The query embeddings in `weights.pth` are sized by the count the model was
trained with, so a package whose count differs from its variant's default
records it as `num_queries` in `model_type.json`, and the Torch loaders build
the network with it.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from inference_models.errors import CorruptedModelPackageError
from inference_models.models.rfdetr.common import parse_num_queries
from inference_models.models.rfdetr.rfdetr_base_pytorch import (
    RFDETRNanoConfig,
    RFDETRSegNanoConfig,
    build_model,
)
from inference_models.models.rfdetr.rfdetr_instance_segmentation_pytorch import (
    RFDetrForInstanceSegmentationTorch,
)
from inference_models.models.rfdetr.rfdetr_object_detection_pytorch import (
    RFDetrForObjectDetectionTorch,
)

CPU = torch.device("cpu")
RESOLUTION = 96
CLASS_NAMES = ["cat", "dog"]
NUM_QUERIES = 7

VARIANTS = [
    pytest.param(
        RFDETRNanoConfig, "rfdetr-nano", RFDetrForObjectDetectionTorch, id="detection"
    ),
    pytest.param(
        RFDETRSegNanoConfig,
        "rfdetr-seg-nano",
        RFDetrForInstanceSegmentationTorch,
        id="instance-segmentation",
    ),
]


def _write_package(
    package_dir: Path, config_class, model_type: str, model_type_json: dict
) -> torch.nn.Module:
    """A package laid out as the training platform registers it."""
    torch.manual_seed(0)
    config = config_class(
        device=CPU,
        resolution=RESOLUTION,
        num_classes=len(CLASS_NAMES),
        num_queries=NUM_QUERIES,
        num_select=NUM_QUERIES,
    )
    network = build_model(config=config).eval()
    (package_dir / "class_names.txt").write_text("\n".join(CLASS_NAMES))
    (package_dir / "model_type.json").write_text(
        json.dumps({"model_type": model_type, **model_type_json})
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
    return network


@pytest.mark.parametrize(("config_class", "model_type", "model_class"), VARIANTS)
def test_package_with_non_default_num_queries_loads_and_predicts(
    tmp_path: Path, config_class, model_type: str, model_class
) -> None:
    # given
    network = _write_package(
        tmp_path, config_class, model_type, {"num_queries": NUM_QUERIES}
    )

    # when
    model = model_class.from_pretrained(str(tmp_path), device=CPU)
    image = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    model(image, confidence=0.0)

    # then
    loaded = model._model
    assert loaded.num_queries == NUM_QUERIES
    for name, weights in network.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], weights), name


@pytest.mark.parametrize(("config_class", "model_type", "model_class"), VARIANTS)
def test_package_without_num_queries_is_built_with_the_variant_default(
    tmp_path: Path, config_class, model_type: str, model_class
) -> None:
    # given: the query embeddings say NUM_QUERIES, model_type.json says nothing
    _write_package(tmp_path, config_class, model_type, {})

    # when / then: the variant default cannot hold these weights
    with pytest.raises(RuntimeError, match="query_feat"):
        model_class.from_pretrained(str(tmp_path), device=CPU)


def test_parse_num_queries_is_none_when_absent(tmp_path: Path) -> None:
    config_path = tmp_path / "model_type.json"
    config_path.write_text(json.dumps({"model_type": "rfdetr-nano"}))

    assert parse_num_queries(config_path=str(config_path)) is None


def test_parse_num_queries_returns_the_recorded_count(tmp_path: Path) -> None:
    config_path = tmp_path / "model_type.json"
    config_path.write_text(
        json.dumps({"model_type": "rfdetr-nano", "num_queries": 450})
    )

    assert parse_num_queries(config_path=str(config_path)) == 450


@pytest.mark.parametrize("num_queries", [0, -3, 4.5, "450", True])
def test_parse_num_queries_rejects_malformed_values(
    tmp_path: Path, num_queries
) -> None:
    config_path = tmp_path / "model_type.json"
    config_path.write_text(
        json.dumps({"model_type": "rfdetr-nano", "num_queries": num_queries})
    )

    with pytest.raises(CorruptedModelPackageError):
        parse_num_queries(config_path=str(config_path))
