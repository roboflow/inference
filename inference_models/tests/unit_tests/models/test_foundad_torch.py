import json
from pathlib import Path

import pytest
import torch
from torch import nn

from inference_models.errors import CorruptedModelPackageError
from inference_models.models.foundad import foundad_anomaly_detection_torch
from inference_models.models.foundad.foundad_anomaly_detection_torch import (
    ENCODER_NAME,
    FoundADForAnomalyDetectionTorch,
    FoundADModel,
    PredictorAttention,
)

IMAGE_SIZE = 64
EMBED_DIM = 24


class TinyEncoder(nn.Module):
    embed_dim = EMBED_DIM

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, EMBED_DIM, kernel_size=16, stride=16)
        self.requested_indices = None

    def forward_intermediates(
        self, images, indices, norm, output_fmt, intermediates_only
    ):
        assert norm and intermediates_only and output_fmt == "NLC"
        self.requested_indices = indices
        return [self.patch_embed(images).flatten(2).transpose(1, 2)]


@pytest.fixture
def tiny_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    def create_model(model_name: str, pretrained: bool) -> TinyEncoder:
        assert model_name == ENCODER_NAME
        assert pretrained is False
        return TinyEncoder()

    monkeypatch.setattr(
        foundad_anomaly_detection_torch.timm, "create_model", create_model
    )


def manifest(image_size: int = IMAGE_SIZE, top_k: int = 4) -> dict:
    return {
        "schema_version": 1,
        "config": {
            "architecture": "foundad",
            "image_size": image_size,
            "seed": 0,
            "batch_size": 2,
            "target_fpr": 0.01,
            "coreset_fraction": 0.1,
            "neighbors": 1,
            "epochs": 2000,
            "lr": 0.001,
            "feature_layer": 3,
            "top_k": top_k,
        },
        "calibration": {
            "threshold": 0.0,
            "scale": 1.0,
            "method": "normal_quantile",
            "normal_count": 1,
            "anomalous_count": 0,
            "target_fpr": 0.01,
        },
        "class_names": ["normal", "anomalous"],
        "upstream_revision": "a590587d96184249eeb59e403cf06dc097fdf347",
        "preprocessing": "rgb-stretch-bilinear-imagenet-v1",
    }


def save_package(package_dir: Path, package_manifest: dict) -> str:
    torch.manual_seed(0)
    state = FoundADModel(image_size=IMAGE_SIZE, feature_layer=3, top_k=4).state_dict()
    (package_dir / "inference_config.json").write_text(json.dumps(package_manifest))
    torch.save(
        {"manifest": package_manifest, "state": state}, package_dir / "weights.pth"
    )
    return str(package_dir)


def test_checkpoint_layout_matches_foundad_training_checkpoints(
    tiny_encoder: None,
) -> None:
    keys = set(
        FoundADModel(image_size=IMAGE_SIZE, feature_layer=3, top_k=4).state_dict()
    )

    assert {
        "encoder.backbone.patch_embed.weight",
        "predictor.predictor_embed.weight",
        "predictor.mask_token",
        "predictor.predictor_blocks.0.norm1.weight",
        "predictor.predictor_blocks.5.attn.qkv.bias",
        "predictor.predictor_blocks.5.attn.proj.weight",
        "predictor.predictor_blocks.5.mlp.fc1.weight",
        "predictor.predictor_blocks.5.mlp.fc2.bias",
        "predictor.predictor_norm.weight",
        "predictor.predictor_proj.bias",
    } <= keys


def test_predictor_attention_matches_explicit_softmax_attention() -> None:
    torch.manual_seed(0)
    attention = PredictorAttention(dim=24, num_heads=4).eval()
    x = torch.randn(2, 5, 24)

    result = attention(x)

    qkv = attention.qkv(x).reshape(2, 5, 3, 4, 6).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0] * 6**-0.5, qkv[1], qkv[2]
    expected = ((q @ k.transpose(-2, -1)).softmax(dim=-1) @ v).transpose(1, 2)
    expected = attention.proj(expected.reshape(2, 5, 24))
    assert torch.allclose(result, expected, atol=1e-6)


def test_score_is_mean_of_largest_patch_residuals(
    tiny_encoder: None, tmp_path: Path
) -> None:
    package = save_package(tmp_path, manifest())
    model = FoundADForAnomalyDetectionTorch.from_pretrained(package, device="cpu")
    torch.manual_seed(2)
    image = torch.randint(0, 255, (40, 72, 3), dtype=torch.uint8).numpy()

    prediction = model(image, include_anomaly_map=True)

    network = model._model
    with torch.inference_mode():
        features = network.encoder.backbone.patch_embed(
            model.pre_process(image).network_input
        )
        features = features.flatten(2).transpose(1, 2)
        residuals = ((features - network.predictor(features)) ** 2).mean(dim=2)
    expected = residuals[0].sort(descending=True).values[:4].mean().item()
    metadata = prediction.images_metadata[0]
    assert metadata["anomaly_score"] == pytest.approx(expected, rel=1e-5)
    assert metadata["is_anomalous"] is True
    assert metadata["anomaly_map"].shape == (40, 72)
    assert prediction.class_id.tolist() == [1]
    assert network.encoder.backbone.requested_indices == [-3]


def test_top_k_larger_than_patch_grid_is_rejected(
    tiny_encoder: None, tmp_path: Path
) -> None:
    package = save_package(tmp_path, manifest(top_k=17))

    with pytest.raises(CorruptedModelPackageError, match="patch grid"):
        FoundADForAnomalyDetectionTorch.from_pretrained(package, device="cpu")
