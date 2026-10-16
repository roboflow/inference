"""Shared contract for Roboflow anomaly detection model packages (PatchCore, FoundAD).

The training job saves `inference_config.json` (the manifest parsed here) next to
`weights.pth`. Scores are only comparable with the saved threshold when the input pixels
match the ones seen in training, so the pre-processing below reproduces the platform
export (OpenCV AREA resize, Pillow JPEG quality 75, optional OpenCV JPEG quality 95)
before the network resize.
"""

import json
import math
from dataclasses import dataclass
from io import BytesIO
from typing import List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

from inference_models.entities import ColorFormat
from inference_models.errors import CorruptedModelPackageError, ModelInputError
from inference_models.models.base.anomaly_detection import AnomalyDetectionPrediction

ANOMALY_CLASS_NAMES = ["normal", "anomalous"]
CORRUPTED_PACKAGE_HELP_URL = "https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror"
MODEL_INPUT_HELP_URL = (
    "https://inference-models.roboflow.com/errors/input-validation/#modelinputerror"
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ExportPreProcessing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["roboflow-folder-export-v1"]
    source_resize: Optional[Tuple[PositiveInt, PositiveInt]] = None
    online_resize: Optional[Tuple[PositiveInt, PositiveInt]] = None
    online_reencode: bool = False


class AnomalyModelConfig(BaseModel):
    # Training-only settings (epochs, lr, ...) are stored in the same object.
    model_config = ConfigDict(extra="ignore", frozen=True)

    architecture: Literal["patchcore", "foundad"]
    image_size: int = Field(ge=32, le=1024)
    batch_size: int = Field(ge=1, le=256)
    neighbors: int = Field(ge=1, le=20)
    feature_layer: int = Field(ge=1, le=12)
    top_k: int = Field(ge=1)


class AnomalyCalibration(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    threshold: float = Field(
        allow_inf_nan=False,
        description="Decision threshold on the raw anomaly score, fitted at training time.",
    )
    # Older packages carry the scale of the confidence sigmoid that used to be
    # served. Nothing reads it any more: the models speak score and threshold only.
    scale: Optional[float] = Field(
        default=None,
        description="Legacy field written by older trainers; never read.",
    )


class AnomalyModelManifest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_version: Literal[1, 2]
    config: AnomalyModelConfig
    calibration: AnomalyCalibration
    class_names: Tuple[Literal["normal"], Literal["anomalous"]]
    preprocessing: Union[
        Literal["rgb-stretch-bilinear-imagenet-v1"], ExportPreProcessing
    ]

    @property
    def export_pre_processing(self) -> Optional[ExportPreProcessing]:
        if isinstance(self.preprocessing, ExportPreProcessing):
            return self.preprocessing
        return None


def parse_anomaly_model_manifest(
    config_path: str, expected_architecture: str
) -> AnomalyModelManifest:
    try:
        with open(config_path) as f:
            manifest = AnomalyModelManifest.model_validate(json.load(f))
    except (OSError, ValueError, ValidationError) as error:
        raise CorruptedModelPackageError(
            message=f"Could not parse anomaly detection model manifest: {error}",
            help_url=CORRUPTED_PACKAGE_HELP_URL,
        ) from error
    if (manifest.schema_version == 2) != (manifest.export_pre_processing is not None):
        raise CorruptedModelPackageError(
            message="Anomaly detection manifest schema version does not match its "
            "pre-processing contract.",
            help_url=CORRUPTED_PACKAGE_HELP_URL,
        )
    if manifest.config.architecture != expected_architecture:
        raise CorruptedModelPackageError(
            message=f"Model package was trained as `{manifest.config.architecture}` but "
            f"is being loaded as `{expected_architecture}`.",
            help_url=CORRUPTED_PACKAGE_HELP_URL,
        )
    return manifest


@dataclass
class AnomalyPreProcessedImages:
    network_input: torch.Tensor  # (bs, 3, image_size, image_size)
    original_sizes_hw: List[Tuple[int, int]]


@dataclass
class AnomalyRawPrediction:
    scores: torch.Tensor  # (bs, ) raw image-level anomaly scores
    maps: torch.Tensor  # (bs, image_size, image_size) raw local anomaly evidence
    original_sizes_hw: List[Tuple[int, int]]


def pre_process_anomaly_images(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    manifest: AnomalyModelManifest,
    target_device: torch.device,
    input_color_format: Optional[ColorFormat] = None,
) -> AnomalyPreProcessedImages:
    rgb_images = _to_rgb_uint8_images(
        images=images, input_color_format=input_color_format
    )
    image_size = manifest.config.image_size
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    network_inputs = []
    for rgb_image in rgb_images:
        pil_image = Image.fromarray(rgb_image)
        if manifest.export_pre_processing is not None:
            pil_image = _materialize_export(
                image=pil_image, contract=manifest.export_pre_processing
            )
        pil_image = pil_image.resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        tensor = torch.from_numpy(np.asarray(pil_image).copy())
        tensor = tensor.permute(2, 0, 1).float() / 255
        network_inputs.append((tensor - mean) / std)
    return AnomalyPreProcessedImages(
        network_input=torch.stack(network_inputs).to(target_device),
        original_sizes_hw=[image.shape[:2] for image in rgb_images],
    )


def _to_rgb_uint8_images(
    images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
    input_color_format: Optional[ColorFormat],
) -> List[np.ndarray]:
    if isinstance(images, np.ndarray):
        images = [images]
    elif isinstance(images, torch.Tensor):
        images = [images] if images.ndim == 3 else list(images)
    if not images:
        raise ModelInputError(
            message="Anomaly detection model received an empty list of images.",
            help_url=MODEL_INPUT_HELP_URL,
        )
    results = []
    for image in images:
        if isinstance(image, torch.Tensor):
            # Tensors follow the package convention: CHW, RGB unless stated otherwise.
            color_format = input_color_format or "rgb"
            image = image.permute(1, 2, 0).cpu().numpy()
        elif isinstance(image, np.ndarray):
            color_format = input_color_format or "bgr"
        else:
            raise ModelInputError(
                message=f"Unsupported image type for anomaly detection: {type(image)}.",
                help_url=MODEL_INPUT_HELP_URL,
            )
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ModelInputError(
                message="Anomaly detection models require 3-channel uint8 images; got "
                f"shape {tuple(image.shape)} and dtype {image.dtype}.",
                help_url=MODEL_INPUT_HELP_URL,
            )
        if color_format == "bgr":
            image = image[:, :, ::-1]
        results.append(np.ascontiguousarray(image))
    return results


def _materialize_export(
    image: Image.Image, contract: ExportPreProcessing
) -> Image.Image:
    image = _resize_area(image=image, size_wh=contract.source_resize)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=75)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    if contract.online_reencode:
        image = _resize_area(image=image, size_wh=contract.online_resize)
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise ModelInputError(
                message="Could not re-encode image for anomaly detection.",
                help_url=MODEL_INPUT_HELP_URL,
            )
        image = Image.open(BytesIO(encoded.tobytes())).convert("RGB")
    return image


def _resize_area(image: Image.Image, size_wh: Optional[Tuple[int, int]]) -> Image.Image:
    if size_wh is None:
        return image
    return Image.fromarray(
        cv2.resize(np.asarray(image), size_wh, interpolation=cv2.INTER_AREA)
    )


def resolve_anomaly_threshold(
    calibration: AnomalyCalibration, anomaly_threshold: Optional[float]
) -> float:
    """Pick the threshold in effect: the caller's override or the saved one.

    Args:
        calibration (AnomalyCalibration): Calibration saved with the model.
        anomaly_threshold (Optional[float]): Caller override on the raw anomaly
            score; ``None`` keeps the threshold saved with the model.

    Returns:
        The threshold to decide ``is_anomalous`` against.

    Raises:
        ModelInputError: If the override is not a finite number.
    """
    if anomaly_threshold is None:
        return calibration.threshold

    if isinstance(anomaly_threshold, bool) or not isinstance(
        anomaly_threshold, (int, float)
    ):
        raise ModelInputError(
            message="anomaly_threshold must be a number, got "
            f"{type(anomaly_threshold).__name__}.",
            help_url=MODEL_INPUT_HELP_URL,
        )
    threshold = float(anomaly_threshold)
    if not math.isfinite(threshold):
        raise ModelInputError(
            message=f"anomaly_threshold must be finite, got {threshold}.",
            help_url=MODEL_INPUT_HELP_URL,
        )

    return threshold


def post_process_anomaly_scores(
    model_results: AnomalyRawPrediction,
    calibration: AnomalyCalibration,
    include_anomaly_map: bool,
    anomaly_threshold: Optional[float] = None,
) -> AnomalyDetectionPrediction:
    """Decide the verdict of each image from its raw anomaly score.

    Args:
        model_results (AnomalyRawPrediction): Raw image scores and local anomaly
            maps for one batch, as produced by the model forward pass.
        calibration (AnomalyCalibration): Calibration saved with the model.
        include_anomaly_map (bool): If True, the prediction carries the maps.
        anomaly_threshold (Optional[float]): Decision threshold replacing the
            saved one for this call; ``None`` keeps the saved threshold.

    Returns:
        AnomalyDetectionPrediction with the raw scores, the threshold in effect,
        ``is_anomalous = anomaly_score >= anomaly_threshold`` and, when requested,
        the float32 maps of shape ``(bs, image_size, image_size)`` in network input
        coordinates, covering the whole (square-resized) image.

    Raises:
        CorruptedModelPackageError: If the model produced non-finite values.
        ModelInputError: If ``anomaly_threshold`` is not a finite number.
    """
    scores = model_results.scores.double()
    if not torch.isfinite(scores).all() or not torch.isfinite(model_results.maps).all():
        raise CorruptedModelPackageError(
            message="Anomaly detection model produced non-finite scores.",
            help_url=CORRUPTED_PACKAGE_HELP_URL,
        )
    threshold = resolve_anomaly_threshold(
        calibration=calibration, anomaly_threshold=anomaly_threshold
    )

    anomaly_maps = None
    if include_anomaly_map:
        # Network coordinates: the map covers the whole (square-resized) input, so
        # resizing it to the image is a plain stretch left to the consumer. The
        # patch grid is the map's real resolution; anything finer is interpolation.
        anomaly_maps = model_results.maps.float()
    prediction = AnomalyDetectionPrediction(
        anomaly_scores=scores,
        anomaly_threshold=threshold,
        is_anomalous=scores >= threshold,
        anomaly_maps=anomaly_maps,
    )

    return prediction
