from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import wide_resnet50_2

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

# Settings of the PatchCore recipe used by Roboflow training
# (https://github.com/amazon-science/patchcore-inspection).
PATCH_SIZE = 3
EMBEDDING_DIMENSION = 1024
MAP_SMOOTHING_SIGMA = 4.0
MAP_SMOOTHING_RADIUS = 16  # scipy.ndimage.gaussian_filter default: truncate=4.0
# Bounds the (patches x memory bank) distance matrix held in memory at once.
NEAREST_NEIGHBOURS_CHUNK_SIZE = 4096


class PatchCoreModel(nn.Module):

    def __init__(self, memory_bank: torch.Tensor, neighbors: int, image_size: int):
        super().__init__()
        self.backbone = wide_resnet50_2(weights=None)
        self.register_buffer("memory_bank", memory_bank)
        self.register_buffer(
            "memory_bank_squared_norms", (memory_bank**2).sum(dim=1), persistent=False
        )
        self.neighbors = neighbors
        self.image_size = image_size
        offsets = torch.arange(
            -MAP_SMOOTHING_RADIUS, MAP_SMOOTHING_RADIUS + 1, dtype=torch.float32
        )
        kernel = torch.exp(-0.5 * (offsets / MAP_SMOOTHING_SIGMA) ** 2)
        self.register_buffer(
            "smoothing_kernel", kernel / kernel.sum(), persistent=False
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]
        embeddings, grid_hw = self.embed(images)
        patch_scores = self.score_patches(embeddings).reshape(batch_size, *grid_hw)
        image_scores = patch_scores.flatten(start_dim=1).max(dim=1).values
        maps = F.interpolate(
            patch_scores[:, None],
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return image_scores, self.smooth(maps)[:, 0]

    def embed(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(x)
        layer3 = self.backbone.layer3(layer2)
        grid_hw = layer2.shape[-2:]
        # Deeper features are brought to the layer2 patch grid before pooling.
        layers = [
            patchify(layer2),
            resize_patch_grid(
                patchify(layer3), source_hw=layer3.shape[-2:], target_hw=grid_hw
            ),
        ]
        pooled = torch.stack(
            [
                F.adaptive_avg_pool1d(
                    patches.reshape(-1, 1, patches[0, 0].numel()), EMBEDDING_DIMENSION
                ).squeeze(1)
                for patches in layers
            ],
            dim=1,
        )
        embeddings = F.adaptive_avg_pool1d(
            pooled.reshape(len(pooled), 1, -1), EMBEDDING_DIMENSION
        ).squeeze(1)
        return embeddings, (int(grid_hw[0]), int(grid_hw[1]))

    def score_patches(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Mean squared L2 distance to the nearest memory bank entries."""
        scores = []
        for chunk in embeddings.split(NEAREST_NEIGHBOURS_CHUNK_SIZE):
            distances = (
                (chunk**2).sum(dim=1, keepdim=True)
                - 2 * chunk @ self.memory_bank.T
                + self.memory_bank_squared_norms[None]
            ).clamp(min=0)
            nearest = distances.topk(self.neighbors, dim=1, largest=False).values
            scores.append(nearest.mean(dim=1))
        return torch.cat(scores)

    def smooth(self, maps: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian filter matching scipy.ndimage.gaussian_filter(sigma=4)."""
        radius = MAP_SMOOTHING_RADIUS
        for dim in (-1, -2):
            # scipy's default `reflect` boundary repeats the edge sample.
            maps = torch.cat(
                [
                    maps.narrow(dim, 0, radius).flip(dim),
                    maps,
                    maps.narrow(dim, maps.shape[dim] - radius, radius).flip(dim),
                ],
                dim=dim,
            )
            kernel_shape = [1, 1, 1, 1]
            kernel_shape[dim] = -1
            maps = F.conv2d(maps, self.smoothing_kernel.reshape(kernel_shape))
        return maps


def patchify(features: torch.Tensor) -> torch.Tensor:
    """(bs, c, h, w) -> (bs, h * w, c, PATCH_SIZE, PATCH_SIZE)"""
    unfolded = F.unfold(
        features, kernel_size=PATCH_SIZE, stride=1, padding=(PATCH_SIZE - 1) // 2
    )
    unfolded = unfolded.reshape(*features.shape[:2], PATCH_SIZE, PATCH_SIZE, -1)
    return unfolded.permute(0, 4, 1, 2, 3)


def resize_patch_grid(
    patches: torch.Tensor, source_hw: Tuple[int, int], target_hw: Tuple[int, int]
) -> torch.Tensor:
    batch_size, _, channels = patches.shape[:3]
    patches = patches.reshape(batch_size, *source_hw, *patches.shape[2:])
    patches = patches.permute(0, 3, 4, 5, 1, 2)
    permuted_shape = patches.shape
    patches = F.interpolate(
        patches.reshape(-1, 1, *source_hw),
        size=tuple(target_hw),
        mode="bilinear",
        align_corners=False,
    )
    patches = patches.reshape(*permuted_shape[:-2], *target_hw)
    patches = patches.permute(0, 4, 5, 1, 2, 3)
    return patches.reshape(batch_size, -1, channels, PATCH_SIZE, PATCH_SIZE)


class PatchCoreForAnomalyDetectionTorch(
    ClassificationModel[AnomalyPreProcessedImages, AnomalyRawPrediction]
):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "PatchCoreForAnomalyDetectionTorch":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["inference_config.json", "weights.pth"],
        )
        manifest = parse_anomaly_model_manifest(
            config_path=model_package_content["inference_config.json"],
            expected_architecture="patchcore",
        )
        state = torch.load(
            model_package_content["weights.pth"],
            map_location="cpu",
            weights_only=True,
        )["state"]
        memory_bank = state["memory_bank"]
        if (
            memory_bank.ndim != 2
            or memory_bank.shape[1] != EMBEDDING_DIMENSION
            or memory_bank.shape[0] < manifest.config.neighbors
        ):
            raise CorruptedModelPackageError(
                message=f"Invalid PatchCore memory bank of shape {tuple(memory_bank.shape)}.",
                help_url=CORRUPTED_PACKAGE_HELP_URL,
            )
        model = PatchCoreModel(
            memory_bank=memory_bank.float(),
            neighbors=manifest.config.neighbors,
            image_size=manifest.config.image_size,
        )
        model.backbone.load_state_dict(state["backbone"])
        return cls(model=model.to(device).eval(), manifest=manifest, device=device)

    def __init__(
        self,
        model: PatchCoreModel,
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
