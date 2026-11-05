from inference.core.models.classification_base import (
    ClassificationBaseOnnxRoboflowInferenceModel,
)


class VitClassification(ClassificationBaseOnnxRoboflowInferenceModel):
    """VitClassification handles classification inference
    for Vision Transformer (ViT) models using ONNX.

    Inherits:
        ClassificationBaseOnnxRoboflowInferenceModel: Base class for ONNX Roboflow Inference.
        ClassificationMixin: Mixin class providing classification-specific methods.

    Attributes:
        multiclass (bool): A flag that specifies if the model should handle multiclass classification.
    """

    preprocess_means = [0.5, 0.5, 0.5]
    preprocess_stds = [0.5, 0.5, 0.5]

    def __init__(self, *args, **kwargs):
        """Initializes the VitClassification instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.multiclass = self.environment.get("MULTICLASS", False)

    @property
    def weights_file(self) -> str:
        """Names the file holding the model weights.

        Returns:
            str: Path to the weights file.
        """
        return "best.onnx"
