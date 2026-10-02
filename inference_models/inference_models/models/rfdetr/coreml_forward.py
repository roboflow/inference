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

from inference_models.logger import LOGGER
from inference_models.models.common.coreml import CoreMLModel, CoreMLModelSignature
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    InferenceConfig,
    NetworkInputDefinition,
    TrainingInputSize,
)

# Logit for classes a selected query was not selected for: sigmoid(-1e4) is exactly 0 in float32, so the
# shared top-k never prefers one of them over a real selection.
UNSELECTED_CLASS_LOGIT = -1e4
# Keeps logit(score) finite for scores that round to exactly 0 or 1 in the package's float16 output.
SCORE_EPSILON = 1e-7


def align_network_input_with_model(
    inference_config: InferenceConfig, signature: CoreMLModelSignature
) -> InferenceConfig:
    """Use the Core ML model's own input size when ``inference_config.json`` declares a different one.

    Some registered packages carry an inference config whose training input size does not match the exported
    weights. The model's spec is authoritative: RF-DETR's outputs are relative to the input, so pre-processing
    to the size the model takes gives correct results, where feeding the configured size fails every call.
    The model's input is fixed, so spatial size overrides are disabled as well.

    Args:
        inference_config (InferenceConfig): The package's parsed inference config.
        signature (CoreMLModelSignature): The loaded model's signature.

    Returns:
        InferenceConfig: ``inference_config`` itself when it already declares the model's fixed input size,
        otherwise a copy that does.
    """
    network_input = inference_config.network_input
    size = network_input.training_input_size
    model_size = (signature.input_height, signature.input_width)
    size_matches = size is not None and (size.height, size.width) == model_size
    if size_matches and not network_input.dynamic_spatial_size_supported:
        return inference_config
    if size is not None and not size_matches:
        LOGGER.warning(
            "Core ML model takes %sx%s input, but the package's inference_config.json declares %sx%s; using "
            "the model's input size.",
            signature.input_width,
            signature.input_height,
            size.width,
            size.height,
        )
    aligned_network_input = network_input.model_copy(
        update={
            "training_input_size": TrainingInputSize(
                height=signature.input_height, width=signature.input_width
            ),
            "dynamic_spatial_size_supported": False,
            "dynamic_spatial_size_mode": None,
        }
    )
    return inference_config.model_copy(update={"network_input": aligned_network_input})


def run_rfdetr_coreml(
    model: CoreMLModel,
    pre_processed_images: torch.Tensor,
    network_input: NetworkInputDefinition,
    num_logit_classes: int,
    with_masks: bool,
) -> Tuple[torch.Tensor, ...]:
    """Run each image through the package and stack raw ``(boxes, logits[, masks])`` for the batch.

    Args:
        model (CoreMLModel): The loaded Core ML model.
        pre_processed_images (torch.Tensor): Normalized NCHW batch from the shared RF-DETR pre-processing.
        network_input (NetworkInputDefinition): The package's network input definition, used to recover the
            8-bit image for image-input packages.
        num_logit_classes (int): Logit width the shared post-processing expects (class names + 1).
        with_masks (bool): Whether to return mask logits as well.

    Returns:
        tuple[torch.Tensor, ...]: ``(boxes, logits)`` or ``(boxes, logits, masks)`` stacked over the batch,
        in the layout of the ONNX model's raw outputs.
    """
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
    """Build ``[K, C]`` logits whose flat top-K is exactly the package's ``K`` (score, label) selections.

    Args:
        scores (torch.Tensor): ``[K]`` selection scores (sigmoid probabilities).
        labels (torch.Tensor): ``[K]`` selected class ids.
        num_logit_classes (int): Minimum logit width; widened if a label does not fit.

    Returns:
        torch.Tensor: ``[K, C]`` float32 logits with ``logit(score)`` at each selection's label and a large
        negative value elsewhere.
    """
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
    """Undo the network-input scaling and normalization to recover the 8-bit RGB image the package expects.

    Mirrors the shared pre-processing, which divides pixels by 255 and then either normalizes (ignoring
    ``scaling_factor``) or rescales to ``pixel / scaling_factor``.

    Args:
        image (torch.Tensor): One normalized CHW image from the shared pre-processing.
        network_input (NetworkInputDefinition): Normalization, scaling factor and color mode it was built with.

    Returns:
        PIL.Image.Image: The RGB image, rounded to 8 bits.
    """
    pixels = image.float()
    if network_input.normalization:
        mean = torch.tensor(network_input.normalization[0], dtype=torch.float32)[
            :, None, None
        ]
        std = torch.tensor(network_input.normalization[1], dtype=torch.float32)[
            :, None, None
        ]
        pixels = (pixels * std + mean) * 255
    elif network_input.scaling_factor is not None:
        pixels = pixels * network_input.scaling_factor
    else:
        pixels = pixels * 255
    if network_input.color_mode is ColorMode.BGR:
        pixels = pixels.flip(0)
    array = pixels.round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(np.ascontiguousarray(array))


def _as_tensor(value) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32))[0]
