from inference.core.models.classification_base import (
    ClassificationBaseOnnxRoboflowInferenceModel,
)
from inference.core.models.mixins import ClassificationMixin


class YOLOv8Classification(
    ClassificationBaseOnnxRoboflowInferenceModel, ClassificationMixin
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiclass = self.environment.get("MULTICLASS", False)

    @property
    def weights_file(self) -> str:
        return "weights.onnx"
