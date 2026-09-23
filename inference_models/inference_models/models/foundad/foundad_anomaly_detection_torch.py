from typing import List, Optional, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn

from inference_models import ClassificationModel, ClassificationPrediction
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.anomaly_detection import (
    ANOMALY_CLASS_NAMES,
    CORRUPTED_PACKAGE_HELP_URL,
    AnomalyModelManifest,
    AnomalyPreProcessedImages,
    AnomalyRawPrediction,
    parse_anomaly_model_manifest,
    post_process_anomaly_scores,
    pre_process_anomaly_images,
)
from inference_models.models.common.model_packages import get_model_package_contents

# Settings of the FoundAD recipe used by Roboflow training
# (https://github.com/ymxlzgy/FoundAD).
ENCODER_NAME = "vit_base_patch16_dinov3.lvd1689m"
ENCODER_PATCH_SIZE = 16
PREDICTOR_EMBED_DIM = 384
PREDICTOR_DEPTH = 6
PREDICTOR_NUM_HEADS = 12
PREDICTOR_MLP_RATIO = 4


class PredictorAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, tokens, 3, self.num_heads, dim // self.num_heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(batch_size, tokens, dim))


class PredictorMlp(nn.Module):

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PredictorBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = PredictorAttention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = PredictorMlp(dim=dim, hidden_dim=dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class FoundADPredictor(nn.Module):
    """Predicts normal-looking encoder features; module names follow the FoundAD
    checkpoint layout so trained weights load without key remapping."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, PREDICTOR_EMBED_DIM)
        # Unused by the forward pass, but present in every trained checkpoint.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, PREDICTOR_EMBED_DIM))
        self.predictor_blocks = nn.ModuleList(
            [
                PredictorBlock(
                    dim=PREDICTOR_EMBED_DIM,
                    num_heads=PREDICTOR_NUM_HEADS,
                    mlp_ratio=PREDICTOR_MLP_RATIO,
                )
                for _ in range(PREDICTOR_DEPTH)
            ]
        )
        self.predictor_norm = nn.LayerNorm(PREDICTOR_EMBED_DIM, eps=1e-6)
        self.predictor_proj = nn.Linear(PREDICTOR_EMBED_DIM, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.predictor_embed(x)
        embedded = x
        for block in self.predictor_blocks:
            x = block(x) + embedded
        return self.predictor_proj(self.predictor_norm(x))


class FoundADEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(ENCODER_NAME, pretrained=False)


class FoundADModel(nn.Module):

    def __init__(self, image_size: int, feature_layer: int, top_k: int):
        super().__init__()
        self.encoder = FoundADEncoder()
        self.predictor = FoundADPredictor(embed_dim=self.encoder.backbone.embed_dim)
        self.image_size = image_size
        self.feature_layer = feature_layer
        self.top_k = top_k

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder.backbone.forward_intermediates(
            images,
            indices=[-self.feature_layer],
            norm=True,
            output_fmt="NLC",
            intermediates_only=True,
        )[0]
        residuals = ((features - self.predictor(features)) ** 2).mean(dim=2)
        scores = residuals.topk(self.top_k, dim=1).values.mean(dim=1)
        side = self.image_size // ENCODER_PATCH_SIZE
        maps = F.interpolate(
            residuals.reshape(-1, 1, side, side),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return scores, maps[:, 0]


class FoundADForAnomalyDetectionTorch(
    ClassificationModel[AnomalyPreProcessedImages, AnomalyRawPrediction]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "FoundADForAnomalyDetectionTorch":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["inference_config.json", "weights.pth"],
        )
        manifest = parse_anomaly_model_manifest(
            config_path=model_package_content["inference_config.json"],
            expected_architecture="foundad",
        )
        image_size = manifest.config.image_size
        if (
            image_size % ENCODER_PATCH_SIZE
            or manifest.config.top_k > (image_size // ENCODER_PATCH_SIZE) ** 2
        ):
            raise CorruptedModelPackageError(
                message=f"FoundAD image size {image_size} and top_k "
                f"{manifest.config.top_k} do not fit the encoder patch grid.",
                help_url=CORRUPTED_PACKAGE_HELP_URL,
            )
        model = FoundADModel(
            image_size=image_size,
            feature_layer=manifest.config.feature_layer,
            top_k=manifest.config.top_k,
        )
        state = torch.load(
            model_package_content["weights.pth"],
            map_location="cpu",
            weights_only=True,
        )["state"]
        model.load_state_dict(state)
        return cls(model=model.to(device).eval(), manifest=manifest, device=device)

    def __init__(
        self,
        model: FoundADModel,
        manifest: AnomalyModelManifest,
        device: torch.device,
    ):
        self._model = model
        self._manifest = manifest
        self._device = device

    @property
    def class_names(self) -> List[str]:
        return ANOMALY_CLASS_NAMES

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> AnomalyPreProcessedImages:
        return pre_process_anomaly_images(
            images=images,
            manifest=self._manifest,
            target_device=self._device,
            input_color_format=input_color_format,
        )

    def forward(
        self, pre_processed_images: AnomalyPreProcessedImages, **kwargs
    ) -> AnomalyRawPrediction:
        scores, maps = [], []
        with torch.inference_mode():
            for batch in pre_processed_images.network_input.split(
                self._manifest.config.batch_size
            ):
                batch_scores, batch_maps = self._model(batch)
                scores.append(batch_scores)
                maps.append(batch_maps)
        return AnomalyRawPrediction(
            scores=torch.cat(scores),
            maps=torch.cat(maps),
            original_sizes_hw=pre_processed_images.original_sizes_hw,
        )

    def post_process(
        self,
        model_results: AnomalyRawPrediction,
        include_anomaly_map: bool = False,
        **kwargs,
    ) -> ClassificationPrediction:
        return post_process_anomaly_scores(
            model_results=model_results,
            calibration=self._manifest.calibration,
            include_anomaly_map=include_anomaly_map,
        )
