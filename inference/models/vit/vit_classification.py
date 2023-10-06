from inference.core.env import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, LAMBDA
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
        """Determines the weights file to be used based on the availability of AWS keys.

        If AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set, it returns the path to 'weights.onnx'.
        Otherwise, it returns the path to 'best.onnx'.

        Returns:
            str: Path to the weights file.
        """
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and LAMBDA:
            return "weights.onnx"
        else:
            return "best.onnx"
