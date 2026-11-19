from inference_models.configuration import INFERENCE_MODELS_CONVNEXT_DEFAULT_CONFIDENCE
from inference_models.models.resnet.resnet_classification_torch import (
    ResNetForClassificationTorch,
    ResNetForMultiLabelClassificationTorch,
)


class ConvNeXtForClassificationTorch(ResNetForClassificationTorch):
    """Single-label ConvNeXt classifier on the Torch backend.

    Roboflow packages timm ConvNeXt models like timm ResNet models, so loading,
    pre-processing and inference are inherited from the ResNet class.
    """


class ConvNeXtForMultiLabelClassificationTorch(ResNetForMultiLabelClassificationTorch):
    """Multi-label ConvNeXt classifier on the Torch backend.

    Roboflow packages timm ConvNeXt models like timm ResNet models, so loading,
    pre-processing and inference are inherited from the ResNet class. The
    default confidence threshold comes from
    ``INFERENCE_MODELS_CONVNEXT_DEFAULT_CONFIDENCE``.
    """

    _default_confidence = INFERENCE_MODELS_CONVNEXT_DEFAULT_CONFIDENCE
