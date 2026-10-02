"""Flip-averaged ResNet-18 classifier: one logical block, three implementations.

Phase graph of the phased implementations (a diamond)::

    image ─ tensor ─┬──────────── logits ─────────┐
                    └─ flipped ─ flipped_logits ──┴─ probabilities ─ result

``run`` composes the same phase methods, so explicit and phased execution
compute identical tensors. The compiler picks the first implementation whose
requirements the target meets, in the order of ``implementations``.
"""

from typing import Any, ClassVar, Dict, List, Mapping

import torch
from pydantic import Field
from resnet18 import (
    WEIGHTS,
    build_network,
    flip,
    forward,
    mean_probabilities,
    network_input,
    to_prediction,
    top_classes,
)
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.implementations import Implementation
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.execution_engine.v2.phases import phase

from inference_models.models.base.classification import ClassificationPrediction


def _outputs(prediction: ClassificationPrediction) -> Dict[str, Any]:
    (best,) = top_classes(prediction, k=1)
    result = {
        "predictions": prediction,
        "top_class": best["class_name"],
        "confidence": best["confidence"],
    }

    return result


class _FlipAveragedResNet18(Implementation):
    """Phases shared by the CPU and MPS implementations; not listed itself.

    The network and every intermediate live on ``device``. The ``result``
    phase copies the probabilities back to the input image's device (the
    CPU); that copy waits for the device work, so output readiness stays
    local to this implementation and the engine never synchronizes a device.
    """

    device: ClassVar[str]

    def __init__(self, *, resnet18_state_dict: Mapping[str, torch.Tensor]):
        self.network = build_network(
            resnet18_state_dict, device=torch.device(self.device)
        )

    @phase
    def tensor(self, image: ImageData) -> torch.Tensor:
        """Resize, crop and normalize the image on the network's device."""
        batch = network_input(image, device=torch.device(self.device))

        return batch

    @phase
    def logits(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward the original view."""
        scores = forward(self.network, tensor)

        return scores

    @phase
    def flipped(self, tensor: torch.Tensor) -> torch.Tensor:
        """Flip the network input horizontally (a new tensor)."""
        flipped = flip(tensor)

        return flipped

    @phase
    def flipped_logits(self, flipped: torch.Tensor) -> torch.Tensor:
        """Forward the flipped view."""
        scores = forward(self.network, flipped)

        return scores

    @phase
    def probabilities(
        self, logits: torch.Tensor, flipped_logits: torch.Tensor
    ) -> torch.Tensor:
        """Join both branches: the mean of their softmax distributions."""
        averaged = mean_probabilities(logits, flipped_logits)

        return averaged

    @phase
    def result(self, probabilities: torch.Tensor, image: ImageData) -> Dict[str, Any]:
        """Build the native prediction on the image's device, plus top-1 values."""
        prediction = to_prediction(probabilities.to(image.device), image)
        result = _outputs(prediction)

        return result

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        """Classify one image by averaging it with its mirror image.

        Args:
            image: RGB image of any size.

        Returns:
            ``predictions`` (native ``ClassificationPrediction`` on the image's
            device, with the image's provenance), ``top_class`` and
            ``confidence``.
        """
        tensor = self.tensor(image)
        logits = self.logits(tensor)
        flipped_logits = self.flipped_logits(self.flipped(tensor))
        probabilities = self.probabilities(logits, flipped_logits)
        result = self.result(probabilities, image)

        return result


class CpuResNet18(_FlipAveragedResNet18):
    """Both views forwarded separately on the CPU (phased)."""

    name = "cpu"
    requires = ("cpu",)
    device = "cpu"


class MpsResNet18(_FlipAveragedResNet18):
    """Both views forwarded separately on Apple's Metal device (phased)."""

    name = "mps"
    requires = ("mps",)
    device = "mps"


class CpuBatchedViews(Implementation):
    """Both views in one ``[2, 3, 224, 224]`` forward on the CPU; no phases.

    A fused backend does not pretend to have the diamond. Batching changes
    floating-point summation order, so its confidences match the phased
    implementations within about 1e-6, not bitwise.
    """

    name = "cpu-batched-views"
    requires = ("cpu", "batched_views")

    def __init__(self, *, resnet18_state_dict: Mapping[str, torch.Tensor]):
        self.network = build_network(resnet18_state_dict, device=torch.device("cpu"))

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        """Classify one image with a single two-view forward.

        Args:
            image: RGB image of any size.

        Returns:
            Same outputs as the phased implementations.
        """
        batch = network_input(image, device=torch.device("cpu"))
        both = forward(self.network, torch.cat([batch, flip(batch)]))
        probabilities = mean_probabilities(both[0:1], both[1:2])
        prediction = to_prediction(probabilities, image)
        result = _outputs(prediction)

        return result


class FlipAveragedClassifier(Block):
    """ImageNet classification of an image averaged with its mirror image.

    Uses torchvision ResNet-18 ``IMAGENET1K_V1`` trained weights, passed by
    the host as the ``resnet18_state_dict`` resource; nothing is downloaded.
    """

    type = "model_demo/flip_averaged_classifier@v1"
    implementations = (MpsResNet18, CpuBatchedViews, CpuResNet18)
    outputs = {
        "predictions": Output(
            CLASSIFICATION_PREDICTION_KIND,
            source="image",
            description="Native prediction over the 1000 ImageNet classes.",
        ),
        "top_class": Output(STRING_KIND, source="image", description="Top-1 name."),
        "confidence": Output(
            FLOAT_KIND, source="image", description="Top-1 averaged probability."
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to classify.")

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Name the trained weights without reading them.

        Args:
            params: Validated step parameters.

        Returns:
            The pinned weights file every implementation needs.
        """
        resources = [
            DependentResource(
                resource_type="model_weights",
                identifier="torchvision/resnet18/IMAGENET1K_V1",
                details={"url": WEIGHTS.url, "sha256_prefix": "f37072fd"},
            )
        ]

        return resources
