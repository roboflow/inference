"""Run RF-DETR Core ML packages and return the raw outputs the shared RF-DETR post-processing consumes.

Two package layouts are supported, told apart by the model's input type:

- Tensor input (public rfdetr's ``format="coreml"`` export): takes the same normalized NCHW tensor as the
  ONNX model and returns the same raw ``(boxes, logits[, masks])``, so it is a drop-in forward pass.
- Image input (Train's ``weights.mlpackage``, also used by the iOS SDKs): takes an RGB image and applies the
  normalization and RF-DETR's top-k selection itself, returning the selected ``boxes``, ``scores``, ``labels``
  (and gathered ``masks``). Those selections are expanded back into raw-format tensors, one query per selection,
  so the shared post-processing re-selects exactly them and every downstream step runs unchanged.
"""

from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from inference_models.models.common.coreml import CoreMLModel
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    NetworkInputDefinition,
)

# Logit for classes a selected query was not selected for: sigmoid(-1e4) is exactly 0 in float32, so the
# shared top-k never prefers one of them over a real selection.
UNSELECTED_CLASS_LOGIT = -1e4
# Keeps logit(score) finite for scores that round to exactly 0 or 1 in the package's float16 output.
SCORE_EPSILON = 1e-7


def run_rfdetr_coreml(
    model: CoreMLModel,
    pre_processed_images: torch.Tensor,
    network_input: NetworkInputDefinition,
    num_logit_classes: int,
    with_masks: bool,
) -> Tuple[torch.Tensor, ...]:
    """Run each image through the package and stack raw ``(boxes, logits[, masks])`` for the batch."""
    per_image = [
        _run_single_image(
            model=model,
            image=image,
            network_input=network_input,
            num_logit_classes=num_logit_classes,
            with_masks=with_masks,
        )
        for image in pre_processed_images.cpu()
    ]
    return tuple(torch.stack(parts) for parts in zip(*per_image))


def _run_single_image(
    model: CoreMLModel,
    image: torch.Tensor,
    network_input: NetworkInputDefinition,
    num_logit_classes: int,
    with_masks: bool,
) -> List[torch.Tensor]:
    signature = model.signature
    if not signature.image_input:
        outputs = model.predict({signature.input_name: image[None].float().numpy()})
        names = signature.output_names[: 3 if with_masks else 2]
        return [_as_tensor(outputs[name]) for name in names]
    outputs = model.predict(
        {
            signature.input_name: to_package_image(
                image=image, network_input=network_input
            )
        }
    )
    labels = _as_tensor(outputs["labels"]).round().long()
    raw = [
        _as_tensor(outputs["boxes"]),
        selections_to_logits(
            scores=_as_tensor(outputs["scores"]),
            labels=labels,
            num_logit_classes=num_logit_classes,
        ),
    ]
    if with_masks:
        raw.append(_as_tensor(outputs["masks"]))
    return raw


def selections_to_logits(
    scores: torch.Tensor, labels: torch.Tensor, num_logit_classes: int
) -> torch.Tensor:
    """Build ``[K, C]`` logits whose flat top-K is exactly the package's ``K`` (score, label) selections."""
    num_classes = max(
        num_logit_classes, int(labels.max().item()) + 1 if labels.numel() else 0
    )
    logits = torch.full(
        (scores.shape[0], num_classes), UNSELECTED_CLASS_LOGIT, dtype=torch.float32
    )
    clamped = scores.float().clamp(SCORE_EPSILON, 1 - SCORE_EPSILON)
    logits[torch.arange(scores.shape[0]), labels] = torch.log(clamped) - torch.log1p(
        -clamped
    )
    return logits


def to_package_image(
    image: torch.Tensor, network_input: NetworkInputDefinition
) -> Image.Image:
    """Undo the network-input scaling and normalization to recover the 8-bit RGB image the package expects."""
    pixels = image.float()
    if network_input.normalization is not None:
        mean = torch.tensor(network_input.normalization[0], dtype=torch.float32)[
            :, None, None
        ]
        std = torch.tensor(network_input.normalization[1], dtype=torch.float32)[
            :, None, None
        ]
        pixels = pixels * std + mean
    if network_input.scaling_factor is not None:
        pixels = pixels * network_input.scaling_factor
    if network_input.color_mode is ColorMode.BGR:
        pixels = pixels.flip(0)
    array = pixels.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(np.ascontiguousarray(array))


def _as_tensor(value) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32))[0]
