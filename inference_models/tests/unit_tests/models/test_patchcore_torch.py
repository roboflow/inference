import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import ndimage
from torch import nn

from inference_models.errors import CorruptedModelPackageError
from inference_models.models.patchcore import patchcore_anomaly_detection_torch
from inference_models.models.patchcore.patchcore_anomaly_detection_torch import (
    EMBEDDING_DIMENSION,
    PatchCoreForAnomalyDetectionTorch,
    PatchCoreModel,
)

IMAGE_SIZE = 64


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.layer1 = nn.Conv2d(8, 8, 3, padding=1)
        self.layer2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.layer3 = nn.Conv2d(16, 32, 3, stride=2, padding=1)


@pytest.fixture
def tiny_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        patchcore_anomaly_detection_torch,
        "wide_resnet50_2",
        lambda weights: TinyBackbone(),
    )


def manifest(architecture: str = "patchcore", neighbors: int = 1) -> dict:
    return {
        "schema_version": 1,
        "config": {
            "architecture": architecture,
            "image_size": IMAGE_SIZE,
            "seed": 0,
            "batch_size": 2,
            "target_fpr": 0.01,
            "coreset_fraction": 0.1,
            "neighbors": neighbors,
            "epochs": 2000,
            "lr": 0.001,
            "feature_layer": 3,
            "top_k": 10,
        },
        "calibration": {
            "threshold": 1.0,
            "scale": 0.5,
            "method": "normal_quantile",
            "normal_count": 1,
            "anomalous_count": 0,
            "target_fpr": 0.01,
        },
        "class_names": ["normal", "anomalous"],
        "upstream_revision": "fcaa92f124fb1ad74a7acf56726decd4b27cbcad",
        "preprocessing": "rgb-stretch-bilinear-imagenet-v1",
    }


def save_package(
    package_dir: Path, memory_bank: torch.Tensor, package_manifest: dict
) -> str:
    torch.manual_seed(0)
    (package_dir / "inference_config.json").write_text(json.dumps(package_manifest))
    torch.save(
        {
            "manifest": package_manifest,
            "state": {
                "backbone": TinyBackbone().state_dict(),
                "memory_bank": memory_bank,
            },
        },
        package_dir / "weights.pth",
    )
    return str(package_dir)


def test_patch_scores_are_mean_squared_distances_to_nearest_memory_entries(
    tiny_backbone: None,
) -> None:
    torch.manual_seed(0)
    memory_bank = torch.randn(50, EMBEDDING_DIMENSION)
    embeddings = torch.randn(7, EMBEDDING_DIMENSION)
    model = PatchCoreModel(memory_bank=memory_bank, neighbors=3, image_size=IMAGE_SIZE)

    scores = model.score_patches(embeddings)

    squared_distances = (
        (embeddings[:, None].double() - memory_bank[None].double()) ** 2
    ).sum(dim=2)
    expected = squared_distances.sort(dim=1).values[:, :3].mean(dim=1)
    assert torch.allclose(scores.double(), expected, rtol=1e-4)


def test_map_smoothing_matches_scipy_gaussian_filter(tiny_backbone: None) -> None:
    torch.manual_seed(0)
    maps = torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE)
    model = PatchCoreModel(
        memory_bank=torch.zeros(1, EMBEDDING_DIMENSION),
        neighbors=1,
        image_size=IMAGE_SIZE,
    )

    smoothed = model.smooth(maps)

    for index in range(2):
        expected = ndimage.gaussian_filter(maps[index, 0].numpy(), sigma=4)
        assert np.allclose(smoothed[index, 0].numpy(), expected, atol=1e-5)


def test_memory_bank_images_score_zero_and_other_images_are_anomalous(
    tiny_backbone: None, tmp_path: Path
) -> None:
    torch.manual_seed(1)
    normal_image = torch.randint(0, 255, (48, 80, 3), dtype=torch.uint8).numpy()
    other_image = torch.randint(0, 255, (48, 80, 3), dtype=torch.uint8).numpy()
    # A placeholder memory bank lets the package load so the real one can be
    # computed with exactly the pre-processing and embedding used for inference.
    package = save_package(tmp_path, torch.zeros(1, EMBEDDING_DIMENSION), manifest())
    model = PatchCoreForAnomalyDetectionTorch.from_pretrained(package, device="cpu")
    with torch.inference_mode():
        memory_bank, _ = model._model.embed(
            model.pre_process(normal_image).network_input
        )
    package = save_package(tmp_path, memory_bank.clone(), manifest())
    model = PatchCoreForAnomalyDetectionTorch.from_pretrained(package, device="cpu")

    predictions = model([normal_image, other_image, normal_image])
    single = model(other_image, include_anomaly_map=True)

    assert model.class_names == ["normal", "anomalous"]
    assert predictions.class_id.tolist() == [0, 1, 0]
    assert predictions.confidence.shape == (3, 2)
    scores = [metadata["anomaly_score"] for metadata in predictions.images_metadata]
    assert scores[0] == pytest.approx(0, abs=1e-3)
    assert scores[1] > 1.0
    assert "anomaly_map" not in predictions.images_metadata[1]
    # Scores do not depend on which other images share the request.
    assert single.images_metadata[0]["anomaly_score"] == pytest.approx(
        scores[1], rel=1e-5
    )
    assert single.images_metadata[0]["anomaly_map"].shape == (48, 80)
    assert single.images_metadata[0]["anomaly_threshold"] == 1.0


def test_package_trained_for_other_architecture_is_rejected(
    tiny_backbone: None, tmp_path: Path
) -> None:
    package = save_package(
        tmp_path, torch.zeros(1, EMBEDDING_DIMENSION), manifest(architecture="foundad")
    )

    with pytest.raises(CorruptedModelPackageError, match="foundad"):
        PatchCoreForAnomalyDetectionTorch.from_pretrained(package, device="cpu")


def test_memory_bank_smaller_than_neighbors_is_rejected(
    tiny_backbone: None, tmp_path: Path
) -> None:
    package = save_package(
        tmp_path, torch.zeros(2, EMBEDDING_DIMENSION), manifest(neighbors=3)
    )

    with pytest.raises(CorruptedModelPackageError, match="memory bank"):
        PatchCoreForAnomalyDetectionTorch.from_pretrained(package, device="cpu")
