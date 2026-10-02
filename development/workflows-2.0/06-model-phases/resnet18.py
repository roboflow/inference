"""ResNet-18 primitives shared by every classifier implementation.

Each function is one step of the flip-averaged classifier. The implementations
in ``classifier.py`` compose them in ``run`` and expose them as phases, so the
same code runs in both execution modes.
"""

from pathlib import Path
from typing import List, Mapping, Tuple

import torch
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from torchvision.models import ResNet18_Weights, resnet18

from inference_models.models.base.classification import ClassificationPrediction

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
CATEGORIES: Tuple[str, ...] = tuple(WEIGHTS.meta["categories"])
NETWORK_INPUT = WEIGHTS.transforms()
"""Resize 256 (bilinear, antialiased), center crop 224, ImageNet normalization."""


def load_state_dict(weights_path: Path) -> Mapping[str, torch.Tensor]:
    """Read a ResNet-18 state dict onto the CPU without executing pickled code.

    Args:
        weights_path: Verified ``.pth`` file (see ``assets.locate_weights``).

    Returns:
        Parameter and buffer tensors by name.
    """
    state_dict = torch.load(weights_path, weights_only=True, map_location="cpu")

    return state_dict


def build_network(
    state_dict: Mapping[str, torch.Tensor], *, device: torch.device
) -> torch.nn.Module:
    """Create ResNet-18 in eval mode on ``device`` with the given weights.

    Args:
        state_dict: Trained parameters; never initialized randomly here.
        device: Device of the network.

    Returns:
        The network, ready for inference.
    """
    network = resnet18(weights=None)
    network.load_state_dict(state_dict)
    network = network.eval().to(device)

    return network


def network_input(image: ImageData, *, device: torch.device) -> torch.Tensor:
    """Turn one RGB CHW ``uint8`` image into a normalized ``[1, 3, 224, 224]`` batch.

    Args:
        image: Workflow image; its tensor is read, never modified.
        device: Device the batch is prepared on.

    Returns:
        A new float tensor.
    """
    pixels = image.tensor_image.to(device)
    if image.channels == 1:
        pixels = pixels.expand(3, -1, -1)

    batch = NETWORK_INPUT(pixels).unsqueeze(0)

    return batch


def flip(batch: torch.Tensor) -> torch.Tensor:
    """Flip a batch horizontally into a new tensor.

    Args:
        batch: ``[N, C, H, W]`` network input.

    Returns:
        The flipped copy; ``batch`` is untouched.
    """
    flipped = torch.flip(batch, dims=[-1])

    return flipped


def forward(network: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Run the network forward.

    Args:
        network: ResNet-18 from ``build_network``.
        batch: Network input on the network's device.

    Returns:
        ``[N, 1000]`` class logits.
    """
    with torch.inference_mode():
        scores = network(batch)

    return scores


def mean_probabilities(*views: torch.Tensor) -> torch.Tensor:
    """Average the softmax distributions of several views of one image.

    Args:
        *views: ``[1, 1000]`` logits per view.

    Returns:
        ``[1, 1000]`` averaged class probabilities.
    """
    probabilities = sum(view.softmax(dim=-1) for view in views) / len(views)

    return probabilities


def to_prediction(
    probabilities: torch.Tensor, image: ImageData
) -> ClassificationPrediction:
    """Wrap probabilities as the native single-image classification payload.

    Args:
        probabilities: ``[1, 1000]`` class distribution.
        image: The classified image; its provenance becomes the metadata.

    Returns:
        ``ClassificationPrediction`` with ``class_id`` ``[1]``, ``confidence``
        ``[1, 1000]`` and one ``images_metadata`` row.
    """
    prediction = ClassificationPrediction(
        class_id=probabilities.argmax(dim=-1),
        confidence=probabilities,
        images_metadata=[image.prediction_metadata()],
    )

    return prediction


def top_classes(prediction: ClassificationPrediction, k: int = 5) -> List[dict]:
    """List the ``k`` most likely classes of a prediction.

    Args:
        prediction: Single-image classification prediction.
        k: Number of classes.

    Returns:
        ``{"class_id", "class_name", "confidence"}`` records, most likely first.
    """
    values, indices = prediction.confidence[0].detach().cpu().topk(k)
    classes = [
        {
            "class_id": int(index),
            "class_name": CATEGORIES[int(index)],
            "confidence": float(value),
        }
        for value, index in zip(values, indices)
    ]

    return classes
