from inference.core.models.classification_base import (
    ClassificationBaseOnnxRoboflowInferenceModel,
)


class YOLOv8Classification(ClassificationBaseOnnxRoboflowInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiclass = self.environment.get("MULTICLASS", False)

    @property
    def weights_file(self) -> str:
        return "weights.onnx"
